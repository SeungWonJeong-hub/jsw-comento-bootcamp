"""달 지형 스테레오 실험 — 사진 두 장으로 고도를 복원한다.

무엇을 하는가
    측정된 고도 모델(DTM)을 3D 로 펴고, 자세를 아는 카메라 두 대로 다시
    찍는다. 그 두 장으로 스테레오를 돌려 고도를 복원하고, 원래 고도와
    비교한다. 착륙선이 하강하며 두 번 찍어 지형을 판단하는 상황이다.

    기하는 실제 달이고 밝기는 렌더링이다. 그림자와 표면 반사 특성은 실제
    영상과 다르므로, 이 결과를 "실제 영상에서도 이만큼 나온다" 로 읽으면
    안 된다. 알고리즘과 촬영 기하가 맞는지를 보는 실험이다.

사용법
    py -3 run_moon_experiment.py            # 합성 지형 (데이터 없이 실행)
    py -3 run_moon_experiment.py DTM.tif    # 실제 고도 모델
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figures  # noqa: E402
from src import metrics, pointcloud, stereo, terrain  # noqa: E402
from src.camera import PinholeCamera, Pose  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")

# 촬영 고도를 상수로 박지 않고 초점거리에서 유도한다. 고도 모델의 화소
# 크기가 데이터마다 다르므로(합성 5 m, LOLA 118 m), 고도를 고정하면 한쪽에서
# 시차가 화면을 넘거나 몇 픽셀로 쪼그라든다. "영상 화소 = 지형 화소" 가
# 되도록 고도 = 초점거리 x 화소 크기로 맞춘다.
FOCAL_PX = 500.0        # 정렬 후 초점거리 [px]
CONVERGENCE = 15.0      # 두 시선이 벌어진 각 [도]
BLOCK = 5               # 정합 블록. 아래 [4] 에서 재서 고른다.
MIN_CONTRAST = 2.0      # 이보다 무늬가 옅은 곳은 값을 내지 않는다.
                        # 근거는 stereo.texture_mask 의 설명에 있다.
SUBPIXEL = 2            # 정합 전에 가로로 이만큼 늘린다. 아래 설명 참고.

_ALT = None             # 촬영 고도 [m]. main() 이 화소 크기에서 정한다.
_log = []


def log(msg=""):
    print(msg, flush=True)
    _log.append(msg)


def build_terrain(path=None):
    """고도 모델을 준비한다. 파일을 주면 읽고, 없으면 합성 지형을 만든다."""
    if path:
        elev, gsd = terrain.load_dtm(path)
        source = os.path.basename(path)
    else:
        elev, gsd = terrain.synthetic_dtm(size=512, gsd=5.0, relief=300.0,
                                          seed=0)
        source = "합성 지형 (크레이터 25개)"
    return elev, gsd, source


def make_views(elev, gsd, camera, convergence):
    """지형을 두 시점에서 렌더링한다 (화소마다 광선을 쏘는 방식).

    점을 화소에 흩뿌리는 방식은 쓰지 않는다. 반올림 오차가 표면 기울기를
    따라 쏠리는데 수렴 촬영의 두 카메라는 반대로 기울어 있어, 시차가 한쪽으로
    치우친다. 티코에서 실측하니 +2.6 px(깊이 1.2 km)였다. terrain.render_
    heightfield 의 설명에 자세히 적었다.
    """
    shading = terrain.shade(elev, gsd)
    # 대비를 0~1 로 편다. 실제 영상도 방사 보정 뒤 이렇게 다룬다.
    shading = (shading - shading.min()) / np.ptp(shading)

    left_pose, right_pose, baseline = terrain.stereo_cameras(
        _ALT, convergence)
    left, depth_left = terrain.render_heightfield(elev, gsd, shading, camera,
                                                  left_pose)
    right, _ = terrain.render_heightfield(elev, gsd, shading, camera,
                                          right_pose)
    return {"elev": elev, "gsd": gsd, "shading": shading,
            "points": terrain.surface_points(elev, gsd),
            "left": left, "right": right,
            "depth_left": depth_left, "pose": left_pose,
            "pose_right": right_pose, "baseline": baseline}


def _rectified_depth_range(pair, view, margin=0.02):
    """지형이 정렬된 왼쪽 카메라에서 차지하는 Z 구간 [m].

    정답 깊이를 보고 정하면 채점에 쓸 값을 미리 훔쳐 보는 셈이 된다. 대신
    촬영을 설계할 때 이미 아는 것 - 지형의 가로세로 범위와 고도 범위 - 만
    쓴다. 경계 상자의 여덟 꼭짓점이면 볼록성 덕분에 전체를 감싼다.
    """
    p = view["points"]
    lo3, hi3 = p.min(axis=0), p.max(axis=0)
    corners = np.array([[x, y, z] for x in (lo3[0], hi3[0])
                        for y in (lo3[1], hi3[1]) for z in (lo3[2], hi3[2])])
    z = (view["pose"].apply(corners) @ pair.R1.T)[:, 2]
    span = z.max() - z.min()
    return float(z.min() - margin * span), float(z.max() + margin * span)


def reconstruct(view, camera, relief, block_size=BLOCK):
    """정렬 → 시차 → 깊이. 시차 탐색 구간은 지형 기복에서 유도한다."""
    pair = stereo.RectifiedPair(
        camera, *stereo.relative_pose(view["pose"], view["pose_right"]),
        alpha=-1.0)

    # 깊이 구간을 "고도 ± 기복" 으로 잡으면 안 된다. 정렬된 카메라는 수렴각의
    # 절반만큼 기울어 있어, 지형이 넓으면 Z 가 화면을 가로질러 크게 변한다.
    # 티코(75 km 폭, 고도 59 km)에서 Z 가 ±5 km 흔들려, 옳은 시차가 구간
    # 밖으로 잘려 나가고 결과가 한쪽으로 1.2 km 치우쳤다. 지형의 경계 상자
    # 여덟 꼭짓점을 정렬 카메라로 옮겨 실제 Z 범위를 구한다.
    lo, hi = _rectified_depth_range(pair, view)
    d_lo = pair.focal * pair.baseline / hi
    d_hi = pair.focal * pair.baseline / lo
    min_disp = max(0, int(np.floor(d_lo / 16)) * 16)
    n_disp = max(16, int(np.ceil((d_hi - min_disp) / 16)) * 16)

    L, R, _ = pair.remap(view["left"], view["right"])

    # 가로로 SUBPIXEL 배 늘려 정합한 뒤 시차를 되돌린다.
    #
    # 왜 — SGBM 의 부화소 보간은 시차를 정수 쪽으로 끌어당긴다(픽셀 락킹).
    # 시차 소수부를 세어 보면 정수 부근(±0.1 px)에 49.3% 가 몰려 있다. 균일
    # 하면 20% 여야 한다. 그 쏠림이 깊이를 +0.043 px 치우치게 만든다.
    #
    # 가로만 늘리는 이유는 시차가 가로 방향이기 때문이다. 늘려서 정합하면
    # 시차를 반 픽셀 단위로 표현할 수 있어 양자화가 절반이 된다. 실측:
    #
    #     배율   값이 나온 화소   Z 오차 중앙값   정수 부근   시차 편향
    #      1x        69.4%         64.9 m        49.3%     +0.043 px
    #      2x        69.3%         34.6 m        35.9%     +0.024 px
    #      4x        69.6%         43.0 m        40.9%     +0.028 px
    #
    # 2배에서 오차가 거의 반이 되고 화소 손실은 없다. 4배는 오히려 나빠진다 -
    # 보간이 없는 정보를 만들어 내지는 못하기 때문이다.
    if SUBPIXEL > 1:
        up = (cv2.resize(L, None, fx=SUBPIXEL, fy=1, interpolation=cv2.INTER_CUBIC),
              cv2.resize(R, None, fx=SUBPIXEL, fy=1, interpolation=cv2.INTER_CUBIC))
        raw = stereo.compute_disparity(
            up[0], up[1], num_disparities=n_disp * SUBPIXEL,
            min_disparity=min_disp * SUBPIXEL, block_size=block_size)
        # 원래 격자로 되돌린 뒤 필터를 건다. 이상치 필터의 크기 기준(덩어리
        # 화소 수)이 늘린 영상에서는 다른 넓이를 뜻하기 때문이다.
        raw = cv2.resize(raw, (L.shape[1], L.shape[0]),
                         interpolation=cv2.INTER_NEAREST) / SUBPIXEL
    else:
        raw = stereo.compute_disparity(L, R, num_disparities=n_disp,
                                       min_disparity=min_disp,
                                       block_size=block_size)
    disparity = stereo.filter_disparity(raw)
    depth = stereo.disparity_to_depth(disparity, pair.focal, pair.baseline)
    depth = np.where((depth >= lo) & (depth <= hi), depth, np.nan)
    # 무늬가 없는 곳에서는 매처가 무엇을 고르든 믿을 수 없다. 값을 내지 않는다.
    depth = np.where(stereo.texture_mask(L, MIN_CONTRAST), depth, np.nan)

    # 기준 깊이도 같은 광선 방식으로 만든다. 정렬된 왼쪽 카메라에 직접
    # 쏘므로 흩뿌리기의 반올림이 끼어들지 않는다.
    rect_pose = Pose(pair.R1 @ view["pose"].R, pair.R1 @ view["pose"].t)
    reference = terrain.render_heightfield(
        view["elev"], view["gsd"], view["shading"], pair.camera, rect_pose)[1]
    return {"pair": pair, "left_rect": L, "right_rect": R,
            "disparity": disparity, "depth": depth, "reference": reference,
            "min_disparity": min_disp, "num_disparities": n_disp}


#: 허용오차를 미터로 못 박지 않고 **시차 몇 픽셀**로 잡는다. 시차 1픽셀이
#: 바꾸는 깊이가 곧 이 촬영 기하의 분해능 한계이므로, 그보다 훨씬 작은 값을
#: 기준으로 삼으면 알고리즘이 아니라 운을 재게 된다. 촬영 조건이 바뀌어도
#: 같은 뜻을 갖는다는 장점도 있다.
TOLERANCE_PX = (0.25, 0.5, 1.0, 2.0)


def score(depth, reference, resolution):
    """복원 깊이를 기준 깊이와 견준다. 허용오차는 분해능의 배수다."""
    domain = np.isfinite(reference)
    m = metrics.depth_metrics(depth, reference, mask=domain)
    err = np.abs(depth - reference)
    valid = np.isfinite(err) & domain
    out = {"valid_ratio": m["valid_ratio"], "median_abs": m["median_abs"],
           "rmse": m["rmse"], "n_valid": m["n_valid"],
           "median_abs_px": m["median_abs"] / resolution,
           "depth_resolution_m_per_px": resolution}
    for tol in TOLERANCE_PX:
        key = f"within_{str(tol).replace('.', 'p')}px"
        out[key] = float((err[valid] < tol * resolution).mean()) if valid.any() else 0.0
    return out


def to_elevation(rec, view, camera):
    """복원한 깊이 맵을 지형 좌표계의 고도로 되돌린다."""
    pair = rec["pair"]
    points_cam = pair.camera.unproject(rec["depth"])
    world = pair.to_body(points_cam, view["pose"])
    return world


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    dtm_path = sys.argv[1] if len(sys.argv) > 1 else None

    log("=" * 70)
    log("달 지형 스테레오 — 사진 두 장으로 고도를 복원한다")
    log("=" * 70)

    global _ALT
    elev, gsd, source = build_terrain(dtm_path)
    relief = float(elev.max() - elev.min())
    focal = FOCAL_PX
    _ALT = FOCAL_PX * gsd
    camera = PinholeCamera(elev.shape[1], elev.shape[0], focal)

    log("\n[1] 촬영 조건")
    log(f"  지형        {source}")
    log(f"  격자        {elev.shape[1]}x{elev.shape[0]} · 화소 {gsd:.1f} m")
    log(f"  기복        {relief:.0f} m  (고도 {elev.min():.0f} ~ {elev.max():.0f} m)")
    log(f"  촬영 고도   {_ALT:.0f} m")
    log(f"  초점거리    {focal:.0f} px  (고도 = 초점거리 x 화소 크기)")

    view = make_views(elev, gsd, camera, CONVERGENCE)
    rec = reconstruct(view, camera, relief)
    pair = rec["pair"]
    log(f"  수렴각      {CONVERGENCE:.0f}도 · 베이스라인 {view['baseline']:.0f} m")
    log(f"  기대 시차   {pair.expected_disparity(_ALT):.0f} px "
        f"(탐색 {rec['min_disparity']}~{rec['min_disparity']+rec['num_disparities']})")
    log(f"  깊이 분해능 {stereo.depth_resolution(_ALT, pair.focal, pair.baseline):.1f} m/px")
    log(f"  빈 화소     {np.isnan(view['depth_left']).mean()*100:.1f}%")

    resolution = stereo.depth_resolution(_ALT, pair.focal, pair.baseline)
    best = score(rec["depth"], rec["reference"], resolution)
    log("\n[2] 복원 결과")
    log(f"  유효화소    {best['valid_ratio']*100:.1f}%")
    log(f"  Z 오차 중앙값 {best['median_abs']:.1f} m "
        f"= 시차 {best['median_abs_px']:.2f} px · RMSE {best['rmse']:.1f} m")
    log(f"  기복 {relief:,.0f} m 대비 {best['median_abs']/relief*100:.2f}%")
    log("  허용오차는 시차 몇 픽셀에 해당하는지로 잡는다. 미터로 못 박으면")
    log("  촬영 기하가 바뀔 때 뜻이 달라진다.")
    for tol in TOLERANCE_PX:
        key = f"within_{str(tol).replace('.', 'p')}px"
        log(f"  {tol:4.2f} px ({tol*resolution:6.0f} m) 이내   "
            f"{best[key]*100:5.1f}%")

    log("\n[3] 수렴각을 바꾸면 — 정밀도와 정합 가능성의 맞바꿈")
    log("  각이 크면 고도를 정밀하게 얻지만 두 영상이 달라 보여 정합이 어렵다.")
    log(f"  {'수렴각':>6s}{'베이스라인':>11s}{'분해능':>9s}{'유효화소':>10s}"
        f"{'Z오차중앙':>11s}{'= 시차':>9s}")
    conv_rows = []
    for conv in (10.0, 15.0, 20.0, 30.0):
        v = make_views(elev, gsd, camera, conv)
        r = reconstruct(v, camera, relief)
        res = stereo.depth_resolution(_ALT, r["pair"].focal,
                                      r["pair"].baseline)
        s = score(r["depth"], r["reference"], res)
        conv_rows.append({"convergence_deg": conv, "baseline_m": v["baseline"],
                          "depth_resolution_m_per_px": res, **s})
        log(f"  {conv:5.0f}도{v['baseline']:10.0f}m{res:9.1f}"
            f"{s['valid_ratio']*100:9.1f}%{s['median_abs']:11.1f}"
            f"{s['median_abs_px']:9.2f}")

    log("\n[4] 정합 블록 크기 — 무늬가 넉넉하면 작은 쪽이 정확하다")
    log("  달 지형은 크레이터와 그림자로 무늬가 넉넉해 작은 블록이 유리하다.")
    log(f"  {'블록':>5s}{'유효화소':>10s}{'Z오차중앙':>11s}{'= 시차':>9s}")
    block_rows = []
    for bs in (3, 5, 7, 9, 11, 15):
        r = reconstruct(view, camera, relief, block_size=bs)
        s = score(r["depth"], r["reference"], resolution)
        block_rows.append({"block_size": bs, **s})
        log(f"  {bs:5d}{s['valid_ratio']*100:9.1f}%{s['median_abs']:11.1f}"
            f"{s['median_abs_px']:9.2f}")
    log(f"  채택: 블록 {BLOCK}. 정확도를 우선하고 유효화소는 조금 내준다.")

    log("\n[5] 3D 로 변환")
    world = to_elevation(rec, view, camera)
    ok = np.isfinite(world).all(axis=1)
    cloud = world[ok]
    log(f"  포인트 클라우드 {len(cloud):,}점")
    log(f"  복원 고도 범위 {cloud[:, 2].min():.0f} ~ {cloud[:, 2].max():.0f} m "
        f"(정답 {elev.min():.0f} ~ {elev.max():.0f} m)")

    log("\n그림 생성")
    figures.figure_overview(elev, gsd, view, rec, best,
                                 os.path.join(OUT, "00_overview.png"))
    figures.figure_tradeoff(conv_rows, block_rows,
                                 os.path.join(OUT, "01_tradeoff.png"))
    figures.figure_cloud(view["points"], cloud,
                              os.path.join(OUT, "02_pointcloud.png"))

    pointcloud.write_ply(os.path.join(OUT, "pointcloud_stereo.ply"), cloud[::7])
    summary = {
        "scene": {"source": source, "grid": list(elev.shape), "gsd_m": gsd,
                  "relief_m": relief, "altitude_m": _ALT,
                  "focal_px": focal, "convergence_deg": CONVERGENCE,
                  "baseline_m": view["baseline"], "block_size": BLOCK,
                  "min_contrast": MIN_CONTRAST, "subpixel": SUBPIXEL},
        "best": best,
        "convergence_sweep": conv_rows,
        "block_size_sweep": block_rows,
        "n_points": int(len(cloud)),
    }
    with open(os.path.join(OUT, "metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "run_log.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    log(f"\n완료 -> {os.path.relpath(OUT, ROOT)}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
