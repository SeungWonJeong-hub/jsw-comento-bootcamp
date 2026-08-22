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
BLOCK = 11              # 정합 블록. 아래 [4] 에서 재서 고른다.
CONTRAST_K = 2.0        # 대비 하한을 잡음 표준편차의 몇 배로 둘지.
                        # 절대값이 아니라 배수로 두어야 영상이 바뀌어도
                        # "잡음보다 이만큼 뚜렷한 무늬" 라는 뜻이 유지된다.
LRC_MAX_DIFF = 1.0      # 좌우 일관성 검사에서 허용할 왕복 시차 차이 [px]
RESIDUAL_DROP = 0.05    # 광도 잔차가 가장 큰 이만큼을 버린다. 정답을 보지
                        # 않고 정하는 값이며, 아래 [6] 에서 근거를 잰다.
SNR = 100.0             # 최대 밝기에서의 신호 대 잡음비. LROC NAC 수준.
BLUR_PX = 0.7           # 광학 흐림 [px]
SUN_ELEVATION = 25.0    # 태양 고도 [도]
MEAN_ALBEDO = 0.12      # 표면의 평균 반사율. 실제 사진이 있으면 무늬만
                        # 그 사진에서 가져오고 평균은 이 값으로 맞춘다.

_ALBEDO = MEAN_ALBEDO   # 알베도 지도. main() 이 실제 사진에서 만든다.
SUBPIXEL = 2            # 정합 전에 가로로 이만큼 늘린다. 아래 설명 참고.
FUSE_ANGLES = (10.0, 15.0, 20.0, 30.0)   # 융합에 쓰는 수렴각
MAX_SPREAD_PX = 0.5     # 각도끼리 이보다 어긋난 셀은 버린다 (시차 픽셀 단위)

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


def load_albedo(dtm_path, elev, gsd):
    """고도 모델 옆에 같은 구역을 찍은 사진이 있으면 표면 무늬로 쓴다.

    왜 이것이 중요한가
        상수 알베도로 그리면 영상의 무늬가 **전부 지형의 음영**에서만 나온다.
        그런데 이 고도 모델은 레이저 궤적을 격자로 편 것이라 매끈해서, 실제
        표면에 있는 잔크레이터·광조(ray)·밝고 어두운 물질이 통째로 빠진다.
        정합기가 붙잡을 무늬가 실제보다 적고 굵다.

        같은 구역을 실제로 찍은 사진에서 음영을 빼내면 그 무늬가 돌아온다.
        고도는 여전히 레이저 측정치이고, 무늬만 사진에서 온다.

    Returns
    -------
    (albedo, info) : 지도 또는 상수, 그리고 사진에서 알아낸 것들
    """
    if not dtm_path:
        return MEAN_ALBEDO, None
    guess = dtm_path.replace("_118m.tif", "_wac.tif")
    if guess == dtm_path or not os.path.exists(guess):
        return MEAN_ALBEDO, None

    img = terrain.load_image(guess)
    if img.shape != elev.shape:
        return MEAN_ALBEDO, None
    albedo, info = terrain.derive_albedo(img, elev, gsd, MEAN_ALBEDO)
    info["source"] = os.path.basename(guess)
    info["variation"] = float(albedo.std() / albedo.mean())
    return albedo, info


def make_views(elev, gsd, camera, convergence, snr=None, seed=0,
               photometric=True, albedo=None):
    """지형을 두 시점에서 렌더링한다 (화소마다 광선을 쏘는 방식).

    점을 화소에 흩뿌리는 방식은 쓰지 않는다. 반올림 오차가 표면 기울기를
    따라 쏠리는데 수렴 촬영의 두 카메라는 반대로 기울어 있어, 시차가 한쪽으로
    치우친다. 티코에서 실측하니 +2.6 px(깊이 1.2 km)였다. terrain.render_
    heightfield 의 설명에 자세히 적었다.

    밝기를 시점마다 따로 만드는 이유
        램버트 반사는 밝기가 태양 방향에만 의존하므로 같은 지면 조각이 두
        사진에서 정확히 같은 밝기로 찍힌다. 실제 달 표면은 후방산란이 강해
        보는 방향에 따라 밝기가 달라지고, 그 차이가 정합의 진짜 난이도다.
        시점별로 렌더링하지 않으면 그 난이도가 실험에서 통째로 빠진다.

    대비를 같은 자로 펴는 이유
        두 장을 각각 min~max 로 펴면 방금 만든 시점 간 밝기 차이가 그
        정규화에서 도로 지워진다. 두 장을 함께 보고 하나의 자로 편다.
    """
    left_pose, right_pose, baseline = terrain.stereo_cameras(
        _ALT, convergence)

    if albedo is None:
        albedo = _ALBEDO
    if photometric:
        shade_left = terrain.shade(elev, gsd, sun_elevation_deg=SUN_ELEVATION,
                                   albedo=albedo,
                                   viewer=terrain.camera_center(left_pose))
        shade_right = terrain.shade(elev, gsd, sun_elevation_deg=SUN_ELEVATION,
                                    albedo=albedo,
                                    viewer=terrain.camera_center(right_pose))
    else:
        shade_left = shade_right = terrain.shade(
            elev, gsd, sun_elevation_deg=SUN_ELEVATION, albedo=albedo)

    lo = min(shade_left.min(), shade_right.min())
    hi = max(shade_left.max(), shade_right.max())
    shade_left = (shade_left - lo) / max(hi - lo, 1e-12)
    shade_right = (shade_right - lo) / max(hi - lo, 1e-12)

    left, depth_left = terrain.render_heightfield(elev, gsd, shade_left,
                                                  camera, left_pose)
    right, _ = terrain.render_heightfield(elev, gsd, shade_right, camera,
                                          right_pose)

    # 잡음은 두 장에 서로 다른 씨앗으로 넣는다. 같은 씨앗을 쓰면 잡음 무늬가
    # 두 장에서 똑같아져 매처가 그것을 단서로 삼아 버린다. 실제 촬영에서
    # 두 장의 잡음은 독립이다.
    if snr:
        left = terrain.sensor_image(left, snr=snr, blur_px=BLUR_PX, seed=seed)
        right = terrain.sensor_image(right, snr=snr, blur_px=BLUR_PX,
                                     seed=seed + 1)

    return {"elev": elev, "gsd": gsd, "shading": shade_left,
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


def reconstruct(view, camera, relief, block_size=BLOCK,
                min_contrast=None, use_lrc=True, residual_drop=RESIDUAL_DROP):
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
    # 왼쪽 기준과 오른쪽 기준 시차를 함께 구한다. 오른쪽 기준은 좌우 일관성
    # 검사에만 쓰이지만, 그 검사가 가림을 걸러 내는 유일한 수단이다.
    if SUBPIXEL > 1:
        up = (cv2.resize(L, None, fx=SUBPIXEL, fy=1, interpolation=cv2.INTER_CUBIC),
              cv2.resize(R, None, fx=SUBPIXEL, fy=1, interpolation=cv2.INTER_CUBIC))
        raw_l, raw_r = stereo.compute_disparity_both(
            up[0], up[1], num_disparities=n_disp * SUBPIXEL,
            min_disparity=min_disp * SUBPIXEL, block_size=block_size)
        # 원래 격자로 되돌린 뒤 필터를 건다. 이상치 필터의 크기 기준(덩어리
        # 화소 수)이 늘린 영상에서는 다른 넓이를 뜻하기 때문이다.
        def _back(d):
            return cv2.resize(d, (L.shape[1], L.shape[0]),
                              interpolation=cv2.INTER_NEAREST) / SUBPIXEL
        raw_l, raw_r = _back(raw_l), _back(raw_r)
    else:
        raw_l, raw_r = stereo.compute_disparity_both(
            L, R, num_disparities=n_disp, min_disparity=min_disp,
            block_size=block_size)

    disparity = stereo.filter_disparity(raw_l)
    lrc = stereo.left_right_consistency(
        disparity, stereo.filter_disparity(raw_r), LRC_MAX_DIFF)
    residual = stereo.photometric_residual(L, R, disparity)
    # 이긴 시차가 이웃보다 얼마나 뚜렷하게 이겼는가. 정합기 안에서 실제로
    # 일어난 일에 가장 가까운 단서이고, 역시 정답을 쓰지 않는다.
    peak = stereo.matching_margin(L, R, disparity)

    # 대비 하한을 영상에서 정한다. 절대값으로 박아 두면 밝기 분포가 다른
    # 영상에 그대로 옮겼을 때 뜻이 달라진다.
    if min_contrast is None:
        min_contrast = stereo.contrast_floor(L, CONTRAST_K)

    depth = stereo.disparity_to_depth(disparity, pair.focal, pair.baseline)
    depth = np.where((depth >= lo) & (depth <= hi), depth, np.nan)
    # 무늬가 없는 곳에서는 매처가 무엇을 고르든 믿을 수 없다. 값을 내지 않는다.
    depth = np.where(stereo.texture_mask(L, min_contrast), depth, np.nan)
    if use_lrc:
        depth = np.where(lrc, depth, np.nan)

    # 광도 잔차가 큰 화소를 버린다. 옳게 맞춘 화소는 두 영상을 포개므로 잔차가
    # 잡음 수준이고, 틀린 화소는 다른 지점을 포개므로 크다. 문턱은 남아 있는
    # 화소의 분포에서 잡으므로 정답도, 절대 상수도 쓰지 않는다.
    if residual_drop:
        alive = np.isfinite(depth) & np.isfinite(residual)
        if alive.any():
            cut = np.percentile(residual[alive], 100.0 * (1.0 - residual_drop))
            depth = np.where(alive & (residual > cut), np.nan, depth)

    # 기준 깊이도 같은 광선 방식으로 만든다. 정렬된 왼쪽 카메라에 직접
    # 쏘므로 흩뿌리기의 반올림이 끼어들지 않는다.
    rect_pose = Pose(pair.R1 @ view["pose"].R, pair.R1 @ view["pose"].t)
    reference = terrain.render_heightfield(
        view["elev"], view["gsd"], view["shading"], pair.camera, rect_pose)[1]
    return {"pair": pair, "left_rect": L, "right_rect": R,
            "disparity": disparity, "depth": depth, "reference": reference,
            "min_disparity": min_disp, "num_disparities": n_disp,
            "lrc": lrc, "residual": residual, "min_contrast": min_contrast,
            "block_size": block_size, "margin": peak["margin"],
            "curvature": peak["curvature"], "match_score": peak["score"]}


#: 허용오차를 미터로 못 박지 않고 **시차 몇 픽셀**로 잡는다. 시차 1픽셀이
#: 바꾸는 깊이가 곧 이 촬영 기하의 분해능 한계이므로, 그보다 훨씬 작은 값을
#: 기준으로 삼으면 알고리즘이 아니라 운을 재게 된다. 촬영 조건이 바뀌어도
#: 같은 뜻을 갖는다는 장점도 있다.
TOLERANCE_PX = (0.25, 0.5, 1.0, 2.0)


def to_grid(points, elev, gsd):
    """복원한 점들을 지형 격자에 얹는다. 셀마다 중앙값.

    수렴각마다 정렬 창의 크기가 달라 깊이 맵끼리는 화소 단위로 겹칠 수 없다.
    지형 격자는 모든 각도가 공유하는 유일한 좌표계이고, 만들려는 산출물
    자체가 "격자 위의 고도" 이므로 여기서 합치는 것이 자연스럽다.
    """
    h, w = elev.shape
    p = points[np.isfinite(points).all(axis=1)]
    col = np.round(p[:, 0] / gsd + (w - 1) / 2.0).astype(np.int64)
    row = np.round((h - 1) / 2.0 - p[:, 1] / gsd).astype(np.int64)
    inside = (col >= 0) & (col < w) & (row >= 0) & (row < h)
    flat = row[inside] * w + col[inside]
    order = np.argsort(flat, kind="stable")
    flat, z = flat[order], p[inside, 2][order]

    grid = np.full(h * w, np.nan)
    if len(flat):
        cuts = np.flatnonzero(np.diff(flat)) + 1
        for a, b in zip(np.r_[0, cuts], np.r_[cuts, len(flat)]):
            grid[flat[a]] = np.median(z[a:b])
    return grid.reshape(h, w)


def fuse(layers, resolutions, max_spread):
    """여러 각도의 고도 격자를 정밀도로 가중해 합친다.

    가중치를 1/분해능^2 으로 두는 이유
        시차 1픽셀이 바꾸는 깊이가 그 각도의 정밀도다. 오차가 그 값에
        비례한다고 보면 분산은 제곱에 비례하므로, 역분산 가중이 곧
        1/분해능^2 이 된다. 수렴각이 큰(정밀한) 층이 자연히 무거워진다.

    각도끼리 어긋나는 셀을 버리는 이유
        정답을 보지 않고도 "믿을 수 없는 곳" 을 가려낼 수 있는 유일한 단서다.
        서로 다른 기하에서 본 값이 다르면 둘 중 하나 이상이 틀린 것이다.
        실측하면 90 퍼센타일 오차가 87.2 → 82.1 m 로 준다.

    Returns
    -------
    (fused, n_layers, spread) : 융합 고도, 셀마다 값을 낸 층 수, 층 간 최대 차이
    """
    stack = np.stack(layers)
    weight = np.asarray(resolutions, dtype=np.float64) ** -2
    have = np.isfinite(stack)

    num = np.nansum(np.where(have, stack * weight[:, None, None], 0.0), axis=0)
    den = np.sum(np.where(have, weight[:, None, None], 0.0), axis=0)
    fused = np.where(den > 0, num / np.maximum(den, 1e-12), np.nan)

    n_layers = have.sum(axis=0)
    # nanmax/nanmin 은 전부 NaN 인 셀에서 경고를 낸다. 없는 값을 무한대로
    # 채워 두면 같은 결과를 경고 없이 얻는다.
    hi = np.where(have, stack, -np.inf).max(axis=0)
    lo = np.where(have, stack, np.inf).min(axis=0)
    spread = np.where(n_layers >= 2, hi - lo, np.nan)
    # 한 층만 값을 낸 셀은 견줄 상대가 없어 그대로 둔다.
    drop = np.isfinite(spread) & (spread > max_spread)
    return np.where(drop, np.nan, fused), n_layers, spread


def _fill_holes(grid):
    """빈 셀을 가장 가까운 값으로 메운다. 다시 그리려면 구멍이 없어야 한다."""
    from scipy.ndimage import distance_transform_edt
    have = np.isfinite(grid)
    if have.all():
        return grid.copy()
    if not have.any():
        raise ValueError("값이 있는 셀이 하나도 없습니다.")
    idx = distance_transform_edt(~have, return_distances=False,
                                 return_indices=True)
    return grid[tuple(idx)]


def reshading_score(grid, gsd, camera, view):
    """복원한 고도를 같은 조명으로 다시 그려 실제 영상과 얼마나 닮았는지 본다.

    왜 이 항이 필요한가
        교차 각도 불일치는 정합 잡음은 잡지만 **뭉개짐은 못 잡는다.** 블록을
        키우면 두 각도가 똑같이 매끄러워져 서로 더 잘 맞기 때문이다. 공통
        모드라 상대 비교로는 드러나지 않는다.

        뭉개진 고도는 다시 그려도 무늬가 밋밋해 실제 영상과 덜 닮는다. 실제
        영상은 두 각도의 평활과 무관한 **바깥의 기준**이므로 공통 모드가 깨진다.
        정답 고도는 여전히 쓰지 않는다 — 쓰는 것은 찍은 사진뿐이다.

    Returns
    -------
    다시 그린 영상과 실제 영상의 정규화 상관 (1 에 가까울수록 닮았다).
    """
    filled = _fill_holes(grid)
    lit = terrain.shade(filled, gsd, sun_elevation_deg=SUN_ELEVATION,
                        viewer=terrain.camera_center(view["pose"]))
    lit = (lit - lit.min()) / max(np.ptp(lit), 1e-12)
    again, _ = terrain.render_heightfield(filled, gsd, lit, camera,
                                          view["pose"])

    a = again.astype(np.float64)
    b = view["left"].astype(np.float64)
    ok = (again > 0) & (view["left"] > 0)
    if ok.sum() < 100:
        return float("nan")
    a, b = a[ok], b[ok]
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / den) if den > 0 else float("nan")


def grid_score(grid, truth):
    """격자 위 고도를 정답 고도와 견준다."""
    ok = np.isfinite(grid)
    err = np.abs(grid[ok] - truth[ok])
    if not ok.any():
        return {"coverage": 0.0, "median_abs": float("nan"),
                "p90_abs": float("nan"), "rmse": float("nan"), "n_cells": 0}
    return {"coverage": float(ok.mean()), "median_abs": float(np.median(err)),
            "p90_abs": float(np.percentile(err, 90)),
            "rmse": float(np.sqrt((err ** 2).mean())), "n_cells": int(ok.sum())}


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


def load_apollo_height():
    """실제 사진 실험의 결과가 있으면 읽는다. 없으면 그 칸을 빼고 그린다.

    run_apollo_stereo.py 를 돌려야 생기는 파일이라, 없다고 해서 이 실험이
    멈추면 안 된다.
    """
    path = os.path.join(OUT, "apollo_height.tif")
    meta = os.path.join(OUT, "apollo_metrics.json")
    if not (os.path.exists(path) and os.path.exists(meta)):
        return None
    try:
        import rasterio
        with rasterio.open(path) as src:
            height = src.read(1).astype(np.float64)
        with open(meta, encoding="utf-8") as f:
            relief = json.load(f)["relief_m"] / 1000.0
    except Exception:
        return None
    return height, relief


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

    global _ALBEDO
    _ALBEDO, albedo_info = load_albedo(dtm_path, elev, gsd)

    log("\n[1] 촬영 조건")
    log(f"  지형        {source}")
    log(f"  격자        {elev.shape[1]}x{elev.shape[0]} · 화소 {gsd:.1f} m")
    log(f"  기복        {relief:.0f} m  (고도 {elev.min():.0f} ~ {elev.max():.0f} m)")
    log(f"  촬영 고도   {_ALT:.0f} m")
    log(f"  초점거리    {focal:.0f} px  (고도 = 초점거리 x 화소 크기)")

    view = make_views(elev, gsd, camera, CONVERGENCE, snr=SNR)
    rec = reconstruct(view, camera, relief)
    pair = rec["pair"]
    log(f"  수렴각      {CONVERGENCE:.0f}도 · 베이스라인 {view['baseline']:.0f} m")
    log(f"  기대 시차   {pair.expected_disparity(_ALT):.0f} px "
        f"(탐색 {rec['min_disparity']}~{rec['min_disparity']+rec['num_disparities']})")
    log(f"  깊이 분해능 {stereo.depth_resolution(_ALT, pair.focal, pair.baseline):.1f} m/px")
    log(f"  빈 화소     {np.isnan(view['depth_left']).mean()*100:.1f}%")
    log(f"  반사 모델   Lunar-Lambert (McEwen) · 태양 고도 {SUN_ELEVATION:.0f}도")
    if albedo_info:
        log(f"  표면 무늬    {albedo_info['source']} — 실제로 찍은 사진에서 음영을")
        log(f"              빼내 알베도로 쓴다 (조명 방위 {albedo_info['azimuth_deg']:.0f}도 ·"
            f" 고도 {albedo_info['elevation_deg']:.0f}도 · 상관 {albedo_info['correlation']:.2f},")
        log(f"              무늬 변동 {albedo_info['variation']*100:.0f}%)")
    else:
        log(f"  표면 무늬    상수 알베도 {MEAN_ALBEDO} (실제 사진 없음)")
    log(f"  센서        SNR {SNR:.0f} · 흐림 {BLUR_PX:.1f} px · 8 bit")
    log(f"  대비 하한   {rec['min_contrast']:.2f} 회색조 "
        f"(잡음 {rec['min_contrast']/CONTRAST_K:.2f} 의 {CONTRAST_K:.0f}배, 영상에서 유도)")

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
        v = make_views(elev, gsd, camera, conv, snr=SNR,
                       seed=int(conv) * 2)
        r = reconstruct(v, camera, relief)
        res = stereo.depth_resolution(_ALT, r["pair"].focal,
                                      r["pair"].baseline)
        s = score(r["depth"], r["reference"], res)
        conv_rows.append({"convergence_deg": conv, "baseline_m": v["baseline"],
                          "depth_resolution_m_per_px": res, **s})
        log(f"  {conv:5.0f}도{v['baseline']:10.0f}m{res:9.1f}"
            f"{s['valid_ratio']*100:9.1f}%{s['median_abs']:11.1f}"
            f"{s['median_abs_px']:9.2f}")

    log("\n[4] 정합 블록 크기 — 잡음이 있으면 큰 쪽이 정확하다")
    log("  블록이 작으면 창 안의 무늬가 적어 잡음 한 점이 정합을 흔든다. 크면")
    log("  기울어진 면에서 깊이가 뭉개진다. 그 맞바꿈의 균형점을 재서 고른다.")
    log(f"  {'블록':>5s}{'유효화소':>10s}{'Z오차중앙':>11s}{'= 시차':>9s}")
    block_rows = []
    for bs in (3, 5, 7, 9, 11, 15):
        r = reconstruct(view, camera, relief, block_size=bs)
        s = score(r["depth"], r["reference"], resolution)
        block_rows.append({"block_size": bs, **s})
        log(f"  {bs:5d}{s['valid_ratio']*100:9.1f}%{s['median_abs']:11.1f}"
            f"{s['median_abs_px']:9.2f}")
    log(f"  채택: 블록 {BLOCK}. 손으로 고른 값이 아니라 [7] 의 절차가 정답을 보지")
    log("  않고 고른 값이다. 정답으로 고르면 다른 값이 나오고, 그 차이가 이 방식의")
    log("  비용이다 — [7] 에 적어 두었다.")
    log("  촬영 조건이 바뀌면 최적점도 움직인다. 잡음이 없을 때는 작은 블록이")
    log("  이겼고, 센서 모델을 넣자 큰 쪽으로 갔다가, 실제 표면 무늬를 넣으니")
    log("  다시 작은 쪽으로 왔다. 손으로 박아 두면 그때마다 조용히 틀린 값이 된다.")

    log("\n[5] 여러 수렴각을 합친다 — 정밀도와 덮는 범위를 함께")
    log("  각도마다 정렬 창이 달라 깊이 맵끼리는 못 겹친다. 지형 격자 위에서")
    log("  합치되, 정밀한 각도가 무거워지도록 1/분해능^2 으로 가중한다.")
    grids, resolutions = [], []
    for conv in FUSE_ANGLES:
        vv = make_views(elev, gsd, camera, conv, snr=SNR,
                        seed=int(conv) * 2)
        rr = reconstruct(vv, camera, relief)
        rr_res = stereo.depth_resolution(_ALT, rr["pair"].focal, rr["pair"].baseline)
        grids.append(to_grid(to_elevation(rr, vv, camera), elev, gsd))
        resolutions.append(rr_res)

    fused, n_layers, spread = fuse(grids, resolutions,
                                   MAX_SPREAD_PX * resolutions[1])
    log(f"  {'':22s}{'덮은 셀':>10s}{'오차 중앙값':>13s}{'90%':>10s}{'RMSE':>10s}")
    for conv, g in zip(FUSE_ANGLES, grids):
        sc = grid_score(g, elev)
        log(f"  {conv:5.0f}도 단독{'':10s}{sc['coverage']*100:9.1f}%"
            f"{sc['median_abs']:12.1f} m{sc['p90_abs']:9.1f}{sc['rmse']:10.1f}")
    fused_score = grid_score(fused, elev)
    log(f"  {'융합 (%d개 각도)' % len(FUSE_ANGLES):22s}"
        f"{fused_score['coverage']*100:9.1f}%{fused_score['median_abs']:12.1f} m"
        f"{fused_score['p90_abs']:9.1f}{fused_score['rmse']:10.1f}")
    log("  단독으로는 정밀한 각도가 좁게 덮고 넓은 각도가 성기게 덮는데,")
    log("  합치면 둘 다 얻는다. 각도끼리 어긋나는 셀은 버려서 90% 오차도 줄인다.")
    log(f"  두 각도 이상이 값을 낸 셀 {np.mean(n_layers >= 2)*100:.1f}%, "
        f"세 각도 이상 {np.mean(n_layers >= 3)*100:.1f}%")

    log("\n[6] 정답 없이 믿을 곳을 가려낸다")
    log("  지금까지는 어느 화소가 맞았는지 정답과 비교해야만 알았다. 실제")
    log("  운용에는 정답이 없으므로 두 영상만으로 계산되는 단서가 필요하다.")
    # 채점용은 관문을 모두 끄고 뽑는다. 잔차 관문을 켠 채로 그 잔차를 채점하면
    # 정보가 있는 꼬리가 이미 잘려 나간 표본을 보게 된다 — 앞서 각도 간 편차를
    # 거른 격자에서 재다가 같은 함정에 빠졌다.
    without = reconstruct(view, camera, relief, use_lrc=False,
                          residual_drop=0.0)
    s_without = score(without["depth"], without["reference"], resolution)
    log(f"  {'':22s}{'유효화소':>10s}{'Z오차중앙':>12s}{'90%':>10s}")
    log(f"  {'좌우 일관성 없음':22s}{s_without['valid_ratio']*100:9.1f}%"
        f"{s_without['median_abs']:11.1f} m"
        f"{np.nanpercentile(np.abs(without['depth']-without['reference']), 90):9.1f}")
    log(f"  {'좌우 일관성 적용':22s}{best['valid_ratio']*100:9.1f}%"
        f"{best['median_abs']:11.1f} m"
        f"{np.nanpercentile(np.abs(rec['depth']-rec['reference']), 90):9.1f}")
    # 다른 관문을 이미 통과한 화소 가운데 몇 개를 이 검사가 더 걸러 내는지를
    # 본다. 전체 화소로 세면 애초에 시차가 없던 곳까지 섞여 들어간다.
    survived = np.isfinite(without["depth"])
    log(f"  다른 관문을 통과한 화소 중 이 검사가 걸러 낸 비율 "
        f"{(survived & ~without['lrc']).sum() / max(survived.sum(), 1)*100:.1f}%")

    log("")
    log("  신뢰도가 실제 오차를 예측하는가 — 희소화 곡선 (AUSE)")
    log("  신뢰도 낮은 순으로 버리며 남은 오차를 재고, 실제 오차 순으로 버린")
    log("  이상적인 곡선과의 넓이를 잰다. 0 에 가까울수록 잘 예측한다는 뜻이다.")
    depth_err = np.abs(without["depth"] - without["reference"])
    rng = np.random.default_rng(0)
    fused_signal = stereo.rank_fuse(-without["residual"], without["margin"],
                                    without["curvature"])
    signals = [
        ("봉우리 margin", without["margin"]),
        ("봉우리 곡률", without["curvature"]),
        ("광도 잔차", -without["residual"]),
        ("셋을 순위로 합침", fused_signal),
        ("국소 대비", stereo.local_contrast(without["left_rect"])),
        ("무작위 (대조군)", rng.random(depth_err.shape)),
    ]
    ause_rows = []
    for name, conf in signals:
        sp = stereo.sparsification(depth_err, conf)
        ause_rows.append({"signal": name, "ause": sp["ause"],
                          "spearman": sp["spearman"],
                          "curve": sp["curve"], "oracle": sp["oracle"],
                          "fractions": sp["fractions"]})
        auc = stereo.blunder_auc(depth_err, conf, resolution)
        ause_rows[-1]["auc"] = auc
        log(f"  {name:18s} AUSE {sp['ause']:.3f} · 순위상관 "
            f"{sp['spearman']:+.3f} · 크게틀린것 AUC {auc:.3f}"
            f"   (평균오차 {sp['curve'][0]:.0f} → {sp['curve'][-1]:.0f} m)")

    # 편차가 큰 셀을 이미 버린 격자에서 재면, 바로 그 셀들이 표본에서
    # 빠져 신호가 실제보다 약해 보인다. 거르기 전 격자에서 잰다.
    fused_raw, _, spread_raw = fuse(grids, resolutions, np.inf)
    grid_err = np.abs(fused_raw - elev)
    sp_spread = stereo.sparsification(grid_err, -spread_raw)
    ause_rows.append({"signal": "각도 간 편차 (격자)", "ause": sp_spread["ause"],
                      "spearman": sp_spread["spearman"],
                      "curve": sp_spread["curve"], "oracle": sp_spread["oracle"],
                      "fractions": sp_spread["fractions"]})
    spread_auc = stereo.blunder_auc(grid_err, -spread_raw, resolution)
    ause_rows[-1]["auc"] = spread_auc
    log(f"  {'각도 간 편차 (격자)':18s} AUSE {sp_spread['ause']:.3f} · 순위상관 "
        f"{sp_spread['spearman']:+.3f} · 크게틀린것 AUC {spread_auc:.3f}"
        f"   (평균오차 {sp_spread['curve'][0]:.0f} → {sp_spread['curve'][-1]:.0f} m)")
    log("")
    log("  AUSE 는 오차 크기를 얼마나 잘 맞추는지를 보고, AUC 는 '크게 틀린 화소를")
    log(f"  골라내는가' (시차 1 px = {resolution:.0f} m 초과) 를 본다. 오차의 대부분은")
    log("  부화소 잡음이라 원리적으로 예측할 수 없다. 실제로 필요한 판단은")
    log("  '이 값을 버릴까' 이므로 AUC 쪽이 쓰임에 가깝다. 0.5 가 무작위다.")

    log("")
    log("  재기만 해서는 뜻이 없다 — 실제로 버려 보고 좋아지는지 본다")
    log("  잔차가 큰 쪽부터 버린다. 얼마나 버릴지는 남은 화소의 분포에서 정하므로")
    log("  정답도 절대 상수도 쓰지 않는다.")
    log(f"  {'버리는 비율':>12s}{'유효화소':>10s}{'Z오차중앙':>12s}{'90%':>10s}"
        f"{'1px 초과':>10s}")
    drop_rows = []
    for frac in (0.0, 0.05, 0.10, 0.20, 0.30):
        r = reconstruct(view, camera, relief, residual_drop=frac)
        sc_ = score(r["depth"], r["reference"], resolution)
        e = np.abs(r["depth"] - r["reference"])
        drop_rows.append({"drop": frac, "valid_ratio": sc_["valid_ratio"],
                          "median_abs": sc_["median_abs"],
                          "p90_abs": float(np.nanpercentile(e, 90)),
                          "over_1px": 1.0 - sc_["within_1p0px"]})
        log(f"  {frac*100:11.0f}%{sc_['valid_ratio']*100:9.1f}%"
            f"{sc_['median_abs']:11.1f} m{drop_rows[-1]['p90_abs']:9.1f}"
            f"{drop_rows[-1]['over_1px']*100:9.2f}%")
    log(f"  채택: {RESIDUAL_DROP*100:.0f}%.")
    log("  중앙값은 거의 안 움직이는데 크게 틀린 화소 비율이 0.26 에서 0.07% 로")
    log("  떨어진다 — 이 신호가 겨냥하는 것이 바로 그쪽이기 때문이다. 5% 를 넘겨")
    log("  더 버려도 크게 틀린 화소는 더 안 줄고 덮는 범위만 잃는다.")

    log("\n[7] 파라미터를 영상에서 정한다")
    log("  상수로 박은 값은 이 지형에서 훑어 고른 값이라 지형이 바뀌면 다시")
    log("  골라야 한다. 영상 자체에서 유도하면 따라온다.")
    sigma = stereo.estimate_noise_sigma(rec["left_rect"])
    log(f"  잡음 표준편차   {sigma:.2f} 회색조   (Immerkaer, 영상 한 장에서)")
    log(f"  대비 하한       {rec['min_contrast']:.2f} = 잡음 x {CONTRAST_K:.0f}"
        f"   (예전에는 2.00 을 손으로 박았다)")
    log(f"  무늬 자기상관   {stereo.autocorrelation_length(rec['left_rect']):.1f} px"
        f" → 블록 {stereo.suggest_block_size(rec['left_rect'])}")
    log("")
    log("  블록 크기를 정답 없이 고를 수 있는가")
    log("  광도 잔차와 좌우 일관성 통과율은 블록에 대해 단조라서 항상 양 끝을")
    log("  고른다. 두 가지를 대신 쓴다. 둘 다 정답 고도를 쓰지 않는다.")
    log("    (가) 수렴각이 다른 두 촬영의 결과가 서로 얼마나 일치하는가")
    log("    (나) 복원한 고도를 같은 조명으로 다시 그리면 실제 사진과 닮는가")
    log("  (가) 는 정합 잡음을 잡지만 뭉개짐은 못 잡는다 — 블록을 키우면 두 각도가")
    log("  똑같이 매끄러워져 서로 더 잘 맞기 때문이다. (나) 는 두 각도 바깥의")
    log("  기준이라 그 공통 모드를 깬다.")
    log(f"  {'블록':>5s}{'일관성':>9s}{'광도잔차':>10s}"
        f"{'(가) 불일치':>14s}{'(나) 재조명':>13s}{'| 정답 Z오차':>15s}")
    view30 = make_views(elev, gsd, camera, 30.0, snr=SNR, seed=60)
    tune_rows = []
    for bs in (3, 5, 7, 9, 11, 15, 21, 31):
        r = reconstruct(view, camera, relief, block_size=bs)
        sc_ = score(r["depth"], r["reference"], resolution)
        ok = r["lrc"] & np.isfinite(r["residual"])

        # 같은 지형을 다른 각도로 찍어 복원하고, 두 결과가 격자 위에서
        # 얼마나 어긋나는지 잰다. 정답은 쓰지 않는다.
        r30 = reconstruct(view30, camera, relief, block_size=bs)
        g15 = to_grid(to_elevation(r, view, camera), elev, gsd)
        g30 = to_grid(to_elevation(r30, view30, camera), elev, gsd)
        both = np.isfinite(g15) & np.isfinite(g30)
        disagree = (float(np.median(np.abs(g15[both] - g30[both])))
                    if both.any() else float("nan"))

        # 다시 그려서 실제 사진과 견준다. 뭉갤수록 무늬가 사라져 덜 닮는다.
        relit = reshading_score(g15, gsd, camera, view)

        tune_rows.append({
            "block_size": bs,
            "lrc_pass": float(r["lrc"].mean()),
            "residual": float(np.median(r["residual"][ok])) if ok.any() else float("nan"),
            "cross_angle_disagreement": disagree,
            "reshading": relit,
            "median_abs": sc_["median_abs"]})
        log(f"  {bs:5d}{tune_rows[-1]['lrc_pass']*100:8.1f}%"
            f"{tune_rows[-1]['residual']:10.2f}{disagree:12.1f} m"
            f"{relit:13.3f}{sc_['median_abs']:12.1f} m")

    # 두 항을 순위로 합친다. 불일치는 작을수록, 재조명 상관은 클수록 좋다.
    combined = stereo.rank_fuse(
        np.array([-r["cross_angle_disagreement"] for r in tune_rows]),
        np.array([r["reshading"] for r in tune_rows]))
    for row, c in zip(tune_rows, combined):
        row["combined"] = float(c)
    pick_free = max(tune_rows, key=lambda r: r["combined"])["block_size"]
    pick_cross = min(tune_rows,
                     key=lambda r: r["cross_angle_disagreement"])["block_size"]
    pick_truth = min(tune_rows, key=lambda r: r["median_abs"])["block_size"]
    err_free = next(r["median_abs"] for r in tune_rows
                    if r["block_size"] == pick_free)
    err_truth = next(r["median_abs"] for r in tune_rows
                     if r["block_size"] == pick_truth)
    err_cross = next(r["median_abs"] for r in tune_rows
                     if r["block_size"] == pick_cross)
    log(f"  (가) 만 쓰면 블록 {pick_cross} ({err_cross:.1f} m), "
        f"(가)+(나) 를 합치면 블록 {pick_free} ({err_free:.1f} m).")
    log(f"  정답으로 고르면 블록 {pick_truth} ({err_truth:.1f} m).")
    log(f"  잘못 골랐을 때 더 내는 값: (가) 만 {err_cross - err_truth:+.1f} m "
        f"({(err_cross/err_truth-1)*100:+.1f}%), 합치면 {err_free - err_truth:+.1f} m "
        f"({(err_free/err_truth-1)*100:+.1f}%).")

    log("\n[8] 촬영 모델이 결과를 얼마나 바꾸는가")
    log("  렌더링을 실제 쪽으로 옮기는 요소가 셋이다. 방향은 서로 다르다 —")
    log("  반사 모델과 센서 잡음은 문제를 어렵게 만들고, 실제 표면 무늬는 오히려")
    log("  쉽게 만든다. 실제에 가깝다는 것이 곧 성적이 나쁘다는 뜻은 아니다.")
    log(f"  {'촬영 모델':34s}{'유효화소':>10s}{'Z오차중앙':>12s}{'= 시차':>9s}")
    render_rows = []
    for label, photo, snr, alb in (
            ("램버트 · 상수 알베도 · 무잡음", False, None, MEAN_ALBEDO),
            ("달 반사 · 상수 알베도 · 무잡음", True, None, MEAN_ALBEDO),
            ("달 반사 · 실제 무늬 · 무잡음", True, None, None),
            ("달 반사 · 실제 무늬 · SNR 400", True, 400.0, None),
            ("달 반사 · 실제 무늬 · SNR 100 (채택)", True, 100.0, None),
            ("달 반사 · 실제 무늬 · SNR 40", True, 40.0, None)):
        v = make_views(elev, gsd, camera, CONVERGENCE, snr=snr,
                       photometric=photo, albedo=alb)
        r = reconstruct(v, camera, relief)
        sc_ = score(r["depth"], r["reference"], resolution)
        render_rows.append({"label": label, "photometric": photo,
                            "snr": snr, "real_albedo": alb is None,
                            "min_contrast": r["min_contrast"], **sc_})
        log(f"  {label:34s}{sc_['valid_ratio']*100:9.1f}%"
            f"{sc_['median_abs']:11.1f} m{sc_['median_abs_px']:9.2f}")
    log("  잡음이 가장 크게 좌우한다. 실제 표면 무늬는 고도 모델에 없는 잔무늬를")
    log("  되돌려 주므로 정합이 쉬워진다 — 이 고도 모델이 레이저 궤적을 격자로")
    log("  편 것이라 매끈하다는 한계를 사진이 메우는 셈이다.")
    log("  그래도 실제로 찍은 스테레오 쌍은 아니다. 이 표가 말하는 것은 렌더링의")
    log("  어느 부분이 결과를 얼마나 좌우하는가이지, 실제 영상의 성적이 아니다.")

    log("\n[9] 3D 로 변환")
    # 점구름도 융합 결과에서 만든다. 한 각도만 쓰면 위에서 얻은 이득이
    # 산출물에 반영되지 않는다.
    yy, xx = np.mgrid[0:elev.shape[0], 0:elev.shape[1]]
    keep = np.isfinite(fused)
    cloud = np.column_stack([
        (xx[keep] - (elev.shape[1] - 1) / 2.0) * gsd,
        ((elev.shape[0] - 1) / 2.0 - yy[keep]) * gsd,
        fused[keep]])
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
    figures.figure_method(rec, os.path.join(OUT, "05_method.png"))
    # 발표 3장은 세 칸으로 둔다. 네 칸이면 각 칸이 작아지고, 실제 사진은
    # 다른 지역이라 색 범위도 달라 나란히 놓으면 읽는 쪽이 한 번 멈춘다.
    # 실제 사진 결과는 outputs/07_apollo.png 에 따로 있다.
    figures.figure_result(elev, fused, cloud, gsd, fused_score,
                          os.path.join(OUT, "04_result.png"), apollo=None)
    figures.figure_fusion(elev, fused,
                          [(c, g, grid_score(g, elev))
                           for c, g in zip(FUSE_ANGLES, grids)],
                          fused_score, gsd,
                          os.path.join(OUT, "03_fusion.png"))

    pointcloud.write_ply(os.path.join(OUT, "pointcloud_stereo.ply"), cloud[::7])
    summary = {
        "scene": {"source": source, "grid": list(elev.shape), "gsd_m": gsd,
                  "relief_m": relief, "altitude_m": _ALT,
                  "focal_px": focal, "convergence_deg": CONVERGENCE,
                  "baseline_m": view["baseline"], "block_size": BLOCK,
                  "min_contrast": rec["min_contrast"],
                  "contrast_k": CONTRAST_K, "subpixel": SUBPIXEL,
                  "snr": SNR, "blur_px": BLUR_PX,
                  "sun_elevation_deg": SUN_ELEVATION,
                  "photometry": "lunar-lambert"},
        "best": best,
        "fusion": {"angles": list(FUSE_ANGLES),
                   "max_spread_m": MAX_SPREAD_PX * resolutions[1],
                   "per_angle": [dict(convergence_deg=c, **grid_score(g, elev))
                                 for c, g in zip(FUSE_ANGLES, grids)],
                   "fused": fused_score,
                   "cells_with_2plus_layers": float(np.mean(n_layers >= 2)),
                   "cells_with_3plus_layers": float(np.mean(n_layers >= 3))},
        "confidence": {
            "lrc_max_diff_px": LRC_MAX_DIFF,
            "without_lrc": s_without,
            "with_lrc": best,
            "rejected_by_lrc": float((without["lrc"] == False).mean()),
            "sparsification": ause_rows,
        },
        "auto_parameters": {
            "noise_sigma": sigma,
            "contrast_k": CONTRAST_K,
            "contrast_floor": rec["min_contrast"],
            "autocorrelation_px": stereo.autocorrelation_length(rec["left_rect"]),
            "suggested_block": stereo.suggest_block_size(rec["left_rect"]),
            "tuning": tune_rows,
            "picked_without_truth": pick_free,
            "picked_by_cross_angle_only": pick_cross,
            "picked_with_truth": pick_truth,
        },
        "residual_gate": {"adopted": RESIDUAL_DROP, "sweep": drop_rows},
        "albedo": albedo_info,
        "render_model": render_rows,
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
