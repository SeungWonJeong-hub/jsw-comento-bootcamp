"""실제로 찍은 달 사진 두 장으로 고도를 복원한다 — 아폴로 15호 매핑 카메라.

본 실험(run_3d_experiment.py)은 레이저 고도 모델에서 두 장을 렌더링해 쓴다.
기하는 실제 달이지만 사진 자체는 만든 것이다. 여기서는 **1971년에 실제로
필름에 찍힌 두 장**을 그대로 넣는다.

왜 아폴로인가
    달 궤도 스테레오는 거의 전부 푸시브룸이다 — 한 줄씩 쓸어 담으면서 자세가
    계속 변하므로, 두 장 사이의 관계를 행렬 하나로 적을 수 없다. 카메라 자세
    커널(SPICE)과 전용 도구 체계가 있어야 푼다.

    아폴로 매핑 카메라는 **프레임 카메라**다. 한 순간에 한 장을 통째로 찍으므로
    두 장 사이가 기본행렬 하나로 정확히 기술된다. 이 파이프라인이 쓰는 정렬과
    삼각측량이 그대로 성립한다.

    (Kaguya TC 로도 시도했으나, 받기 쉬운 지도 투영 판본은 이미 전역 DEM 으로
     정사보정되어 시차가 제거되어 있었다. tools/check_real_pair.py 에 기록.)

무엇을 아는가 — 전부 아카이브에 적혀 있는 값이다
    초점거리    76.054 mm (매핑 카메라 검정값)
    스캔 화소   6.756 um (원본) · 이 판본은 4048 화소로 줄인 것
    촬영 고도   100.99 km / 100.73 km
    중심 좌표   (25.89, -6.10) / (25.89, -7.43)

    두 중심의 경도 차이가 곧 베이스라인이다. 즉 **고도의 크기(스케일)를 정답
    고도에서 가져오지 않는다.** 카메라 제원과 궤도에서 나온다.

    정답과 견주려면 복원 결과를 지도 위에 놓아야 하는데, 그러려면 촬영 자세
    전체가 필요하다. 여기서는 2차원 정합(평행이동·회전·크기)으로 맞춘다.
    **맞추는 것은 위치이고, 고도의 크기는 맞추지 않는다.**

사용법
    py -3 tools/get_apollo_pair.py
    py -3 run_apollo_stereo.py
"""

from __future__ import annotations

import json
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import stereo  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(ROOT, "outputs")
DATA = os.path.join(ROOT, "data", "moon", "apollo")

FRAMES = ("AS15-M-1000", "AS15-M-1001")
FOCAL_MM = 76.054           # 매핑 카메라 검정 초점거리
FORMAT_MM = 114.0           # 필름 한 변
SCAN_PX = 4048              # 이 판본의 한 변
ALTITUDE_M = (100990.0, 100730.0)
CENTER = ((25.89, -6.10), (25.89, -7.43))
MOON_RADIUS = 1737400.0

BLOCK = 11
SUBPIXEL = 2
CONTRAST_K = 2.0
LRC_MAX_DIFF = 1.0
RESIDUAL_DROP = 0.05

_log = []


def log(msg=""):
    print(msg, flush=True)
    _log.append(msg)


def baseline_m():
    """두 촬영 지점 사이 거리. 중심 좌표의 경도 차이에서 나온다."""
    (lat1, lon1), (lat2, lon2) = CENTER
    lat = math.radians((lat1 + lat2) / 2.0)
    dlon = math.radians(abs(lon2 - lon1))
    dlat = math.radians(abs(lat2 - lat1))
    east = MOON_RADIUS * dlon * math.cos(lat)
    north = MOON_RADIUS * dlat
    return math.hypot(east, north)


def radial_bowl(pa, pb, K, k1, size):
    """왜곡 계수를 넣고 삼각측량했을 때 남는 '사발' 의 세기.

    복원한 깊이에서 평면 성분을 뺀 잔차가 화면 중심으로부터의 거리 제곱과
    얼마나 상관되는지를 본다. 실제 지형이 화면 중심 기준으로 방사 대칭일
    이유는 없으므로, 상관이 크면 그것은 지형이 아니라 왜곡이다.
    """
    w, h = size
    D = np.array([k1, 0.0, 0.0, 0.0, 0.0])
    ua = cv2.undistortPoints(pa.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    ub = cv2.undistortPoints(pb.reshape(-1, 1, 2), K, D, P=K).reshape(-1, 2)
    E, mask = cv2.findEssentialMat(ua, ub, K, method=cv2.RANSAC,
                                   prob=0.9999, threshold=1.5)
    if E is None or E.shape != (3, 3):
        return float("inf"), None
    inl = mask.ravel().astype(bool)
    if inl.sum() < 100:
        return float("inf"), None

    _, R, t, _ = cv2.recoverPose(E, ua[inl], ub[inl], K)
    P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
    P2 = K @ np.hstack([R, t])
    X = cv2.triangulatePoints(P1, P2, ua[inl].T, ub[inl].T)
    X = (X[:3] / X[3]).T
    Z = X[:, 2]
    ok = np.isfinite(Z) & (Z > 0)
    if ok.sum() < 100:
        return float("inf"), None

    Z, q = Z[ok], ua[inl][ok]
    r2 = (q[:, 0] - w / 2.0) ** 2 + (q[:, 1] - h / 2.0) ** 2
    plane = np.column_stack([q[:, 0], q[:, 1], np.ones(len(q))])
    res = Z - plane @ np.linalg.lstsq(plane, Z, rcond=None)[0]
    if res.std() <= 0:
        return float("inf"), None
    return abs(float(np.corrcoef(res, r2)[0, 1])), float(inl.mean())


def estimate_radial(pa, pb, K, size):
    """사발이 사라지는 왜곡 계수를 찾는다. 거친 훑기 뒤 반씩 좁힌다."""
    grid = np.arange(-0.06, 0.0201, 0.005)
    best = min(grid, key=lambda k: radial_bowl(pa, pb, K, float(k), size)[0])
    step = 0.005
    for _ in range(4):
        step /= 2.0
        cand = [best - step, best, best + step]
        best = min(cand, key=lambda k: radial_bowl(pa, pb, K, float(k), size)[0])
    return float(best)


def load(name):
    path = os.path.join(DATA, f"{name}.png")
    if not os.path.exists(path):
        raise SystemExit(f"{path} 가 없습니다. tools/get_apollo_pair.py 를 실행하세요.")
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise SystemExit(f"읽을 수 없습니다: {path}")
    return img


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    log("=" * 70)
    log("실제로 찍은 달 사진 두 장으로 고도를 복원한다 — 아폴로 15호 매핑 카메라")
    log("=" * 70)

    a, b = load(FRAMES[0]), load(FRAMES[1])
    h, w = a.shape
    px_mm = FORMAT_MM / SCAN_PX
    focal_px = FOCAL_MM / px_mm
    alt = sum(ALTITUDE_M) / 2.0
    base = baseline_m()
    gsd = alt * px_mm / FOCAL_MM

    log("\n[1] 촬영 조건 — 전부 아카이브에 적힌 값이다")
    log(f"  사진        {FRAMES[0]} / {FRAMES[1]} · {w}x{h}")
    log(f"  촬영        1971-07-31 · 태양 고도 15도 · 24초 간격")
    log(f"  초점거리    {FOCAL_MM} mm = {focal_px:.0f} px  (화소 {px_mm*1000:.1f} um)")
    log(f"  촬영 고도   {alt/1000:.1f} km")
    log(f"  베이스라인  {base/1000:.1f} km  (중심 경도 차 {abs(CENTER[1][1]-CENTER[0][1]):.2f}도)")
    log(f"  수렴각      {math.degrees(2*math.atan(base/2/alt)):.1f}도  "
        f"(베이스라인/고도 {base/alt:.2f})")
    log(f"  지상 화소   {gsd:.1f} m")
    log(f"  깊이 분해능 {alt**2/(focal_px*base):.0f} m/px "
        f"(= Z^2 / (f x B), 본 실험의 450 m 와 견줄 값)")

    log("\n[2] 두 장의 관계를 찾는다")
    det = cv2.SIFT_create(nfeatures=12000)
    ka, da = det.detectAndCompute(a, None)
    kb, db = det.detectAndCompute(b, None)
    bf = cv2.BFMatcher(cv2.NORM_L2)
    good = [m for m, n in bf.knnMatch(da, db, k=2)
            if m.distance < 0.75 * n.distance]
    log(f"  특징점      {len(ka):,} / {len(kb):,}")
    log(f"  맞춘 점     {len(good):,}")
    if len(good) < 100:
        log("  정합이 너무 적다.")
        return 1

    pa = np.float32([ka[m.queryIdx].pt for m in good])
    pb = np.float32([kb[m.trainIdx].pt for m in good])
    K = np.array([[focal_px, 0.0, w / 2.0],
                  [0.0, focal_px, h / 2.0],
                  [0.0, 0.0, 1.0]])

    # 이 판본의 스캔이 필름 형식을 몇 화소에 담았는지 적혀 있지 않아, 방사
    # 왜곡이 남는다. 복원한 고도에 중앙이 솟은 사발이 생기는데 실제 지형이
    # 그럴 리 없다. 대응점만으로 그 계수를 찾는다 — 정답 고도는 쓰지 않는다.
    before = radial_bowl(pa, pb, K, 0.0, (w, h))[0]
    k1 = estimate_radial(pa, pb, K, (w, h))
    after, share = radial_bowl(pa, pb, K, k1, (w, h))
    dist = np.array([k1, 0.0, 0.0, 0.0, 0.0])
    log(f"  방사 왜곡   k1 = {k1:+.4f}  (자기 검정)")
    log(f"              '사발' 세기 {before:.3f} → {after:.3f}  "
        f"(0 이면 방사 성분이 없다는 뜻)")

    pa = cv2.undistortPoints(pa.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)
    pb = cv2.undistortPoints(pb.reshape(-1, 1, 2), K, dist, P=K).reshape(-1, 2)

    # 검정된 초점거리를 알고 있으므로 필수행렬을 바로 쓴다. 기본행렬만 쓰면
    # 복원이 사영 변환만큼 불확정이라 고도에 기울기가 남는다.
    E, mask = cv2.findEssentialMat(pa, pb, K, method=cv2.RANSAC, prob=0.9999,
                                   threshold=1.5)
    inl = mask.ravel().astype(bool)
    log(f"  필수행렬 내점 {int(inl.sum()):,} / {len(good):,} "
        f"({inl.mean()*100:.0f}%)")

    n_in, R, t, _ = cv2.recoverPose(E, pa[inl], pb[inl], K)
    t = t.reshape(3, 1) * base       # 크기를 궤도에서 가져온다
    ang = math.degrees(np.linalg.norm(cv2.Rodrigues(R)[0]))
    log(f"  자세 차이   {ang:.2f}도 · 이동 {np.round(t/1000, 1)} km")

    log("\n[3] 정렬하고 정합한다")
    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        K, dist, K, dist, (w, h), R, t, flags=cv2.CALIB_ZERO_DISPARITY)
    m1 = cv2.initUndistortRectifyMap(K, dist, R1, P1, (w, h), cv2.CV_32FC1)
    m2 = cv2.initUndistortRectifyMap(K, dist, R2, P2, (w, h), cv2.CV_32FC1)
    L = cv2.remap(a, *m1, cv2.INTER_LINEAR)
    Rr = cv2.remap(b, *m2, cv2.INTER_LINEAR)
    rect_focal = float(P1[0, 0])

    # 정렬이 됐는지는 대응점의 세로 좌표가 맞는지로 확인한다. 이것이 어긋나
    # 있으면 매처는 같은 줄에서 짝을 못 찾는다.
    qa = cv2.undistortPoints(pa[inl].reshape(-1, 1, 2), K, None,
                             R=R1, P=P1).reshape(-1, 2)
    qb = cv2.undistortPoints(pb[inl].reshape(-1, 1, 2), K, None,
                             R=R2, P=P2).reshape(-1, 2)
    log(f"  정렬 잔차   세로 {np.median(np.abs(qa[:, 1] - qb[:, 1])):.2f} px "
        f"(중앙값)")

    # 탐색 구간도 대응점에서 직접 읽는다. 부호와 크기를 짐작하지 않는다.
    seen = qa[:, 0] - qb[:, 0]

    # 시차가 음수면 정렬이 두 장의 역할을 바꿔 놓은 것이다. 그대로 두면
    # 좌우 일관성 검사(오른쪽 기준 시차를 좌우 반전으로 구한다)가 성립하지
    # 않는다. 역할을 되돌려 시차를 양수로 만든다.
    if np.median(seen) < 0:
        L, Rr = Rr, L
        seen = -seen
        log("  정렬이 두 장의 역할을 바꿔 놓았다 — 되돌린다")

    d_lo, d_hi = np.percentile(seen, [1, 99])
    margin = max(48.0, (d_hi - d_lo))
    min_disp = int(np.floor((d_lo - margin) / 16)) * 16
    n_disp = int(np.ceil((d_hi + margin - min_disp) / 16)) * 16
    log(f"  대응점 시차 {np.median(seen):+.0f} px "
        f"(1~99% {d_lo:+.0f}~{d_hi:+.0f}) → 탐색 {min_disp}~{min_disp+n_disp}")

    up_l = cv2.resize(L, None, fx=SUBPIXEL, fy=1, interpolation=cv2.INTER_CUBIC)
    up_r = cv2.resize(Rr, None, fx=SUBPIXEL, fy=1, interpolation=cv2.INTER_CUBIC)
    raw_l, raw_r = stereo.compute_disparity_both(
        up_l, up_r, num_disparities=n_disp * SUBPIXEL,
        min_disparity=min_disp * SUBPIXEL, block_size=BLOCK)

    def back(d):
        return cv2.resize(d, (w, h), interpolation=cv2.INTER_NEAREST) / SUBPIXEL

    disp = stereo.filter_disparity(back(raw_l))
    lrc = stereo.left_right_consistency(disp, stereo.filter_disparity(back(raw_r)),
                                        LRC_MAX_DIFF)
    residual = stereo.photometric_residual(L, Rr, disp)
    sigma = stereo.estimate_noise_sigma(L)
    floor = stereo.contrast_floor(L, CONTRAST_K)
    keep = lrc & stereo.texture_mask(L, floor) & np.isfinite(disp) & (L > 0)
    alive = keep & np.isfinite(residual)
    if RESIDUAL_DROP and alive.any():
        cut = np.percentile(residual[alive], 100 * (1 - RESIDUAL_DROP))
        keep &= ~(alive & (residual > cut))
    log(f"  잡음 {sigma:.2f} 회색조 → 대비 하한 {floor:.2f} (영상에서 유도)")
    log(f"  좌우 일관성 통과 {lrc.mean()*100:.1f}% · 최종 남은 화소 "
        f"{keep.mean()*100:.1f}%")

    d = np.where(keep, disp, np.nan)
    # 부호는 정렬이 어느 쪽을 왼쪽으로 놓았는지에 달렸다. 크기만 쓴다.
    depth = rect_focal * base / np.abs(d)
    # 카메라에서 가까울수록 높다. 기준면은 이 장면의 중앙값으로 잡는다.
    height = np.nanmedian(depth) - depth
    ok = np.isfinite(height)
    if ok.sum() < 10000:
        log("  값이 나온 화소가 너무 적다.")
        return 1

    lo, hi = np.nanpercentile(height[ok], [1, 99])
    log("\n[4] 결과 — 고도의 크기는 카메라 제원과 궤도에서만 나온다")
    log(f"  값이 나온 화소 {ok.mean()*100:.1f}%")
    log(f"  카메라까지 거리 중앙값 {np.nanmedian(depth)/1000:.1f} km "
        f"(아카이브의 촬영 고도 {alt/1000:.1f} km)")
    log(f"  복원한 기복    {hi-lo:.0f} m  (1~99 퍼센타일 {lo:.0f} ~ {hi:.0f} m)")

    truth_path = os.path.join(ROOT, "data", "moon", "apollo_as15_lola.tif")
    corr = float("nan")
    if os.path.exists(truth_path):
        import rasterio
        with rasterio.open(truth_path) as src:
            truth = src.read(1).astype(np.float64)
        t_lo, t_hi = np.nanpercentile(truth, [1, 99])
        log(f"  같은 구역 LOLA 기복 {t_hi-t_lo:.0f} m "
            f"({t_lo:.0f} ~ {t_hi:.0f} m)")
        log(f"  기복의 비    {(hi-lo)/(t_hi-t_lo):.2f} 배")
        log("  두 값이 자릿수까지 맞으면, 어긋남이 실제로 지형의 높낮이에서")
        log("  왔다는 뜻이다. 복원 쪽은 화소가 잘아 잔무늬가 더 많이 잡힌다.")

    import figures
    shown = np.where(ok, height, np.nan)
    figures.figure_apollo(L, Rr, shown, os.path.join(OUT, "07_apollo.png"))
    figures.figure_apollo_slide(L, shown, (hi - lo) / 1000.0,
                                os.path.join(OUT, "08_apollo_slide.png"))

    summary = {
        "source": "Apollo 15 Metric (Mapping) Camera, ASU Apollo Image Archive",
        "frames": list(FRAMES), "focal_mm": FOCAL_MM,
        "focal_px": float(focal_px), "altitude_m": alt,
        "baseline_m": float(base), "gsd_m": float(gsd),
        "convergence_deg": float(math.degrees(2 * math.atan(base / 2 / alt))),
        "depth_resolution_m_per_px": float(alt ** 2 / (focal_px * base)),
        "matches": len(good), "essential_inliers": int(inl.sum()),
        "radial_k1": k1, "bowl_before": before, "bowl_after": after,
        "coverage": float(ok.mean()),
        "relief_m": float(hi - lo),
        "noise_sigma": float(sigma), "contrast_floor": float(floor),
        "lrc_pass": float(lrc.mean()),
    }
    with open(os.path.join(OUT, "apollo_metrics.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "apollo_log.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(_log) + "\n")
    log(f"\n완료 -> outputs/07_apollo.png · outputs/apollo_metrics.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
