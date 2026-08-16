"""2차 업무 실험 실행기 — 스테레오 삼각측량으로 깊이 맵과 3D 포인트 클라우드 생성.

실행
    py -3 tools/get_spe3r_aqua.py      # 데이터 준비 (최초 1회)
    py -3 run_3d_experiment.py         # 결과는 outputs/ 에 저장

구성
    [1] 합성 장면 검증
        광선-도형 교차를 해석적으로 풀어 정답 깊이를 오차 없이 만든 뒤,
        스테레오 파이프라인과 과제 예시 코드(밝기->깊이)를 같은 조건에서 비교한다.
        정답이 정확하므로 남는 오차는 전부 정합 알고리즘에서 온 것이다.

    [2] SPE3R 실제 데이터 적용
        카메라는 옆으로 움직이지 않지만 타겟이 회전하므로, 두 뷰의 상대 자세를
        계산하면 유효 베이스라인이 생긴다. 쓸 만한 쌍을 골라 같은 파이프라인을
        돌리고, 메시를 z-buffer 로 투영해 만든 기준 깊이와 비교한다.
"""

from __future__ import annotations

import json
import os
import sys
import time

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src import baseline, depth as depth_mod, metrics, pointcloud, scene, stereo  # noqa: E402
from src.camera import PinholeCamera, Pose, quaternion_to_rotation  # noqa: E402
from src.spe3r import SPE3RModel  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "spe3r")
OUT = os.path.join(ROOT, "outputs")
MODEL_NAME = "aqua"

SYNTH_BASELINE = 0.40
MESH_SAMPLES = 400_000
MAX_ROTATION_DEG = 8.0
MIN_LATERAL_RATIO = 2.0

# 발표 화면에서 읽히도록 글꼴을 키우고, PPT 와 같은 한글 글꼴을 쓴다.
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": ["Pretendard", "Malgun Gothic", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.unicode_minus": False,
})


def log(msg=""):
    print(msg, flush=True)


def within(pred, ref, mask, tol=0.05):
    v = np.isfinite(pred) & np.isfinite(ref) & mask
    if v.sum() == 0:
        return 0.0
    return float((np.abs(pred[v] - ref[v]) < tol).mean())


def span(depth, mask):
    v = np.isfinite(depth) & mask
    return float(depth[v].max() - depth[v].min()) if v.sum() else 0.0


# ---------------------------------------------------------------------------
# [1] 합성 장면 검증
# ---------------------------------------------------------------------------

def run_synthetic():
    """정답 깊이가 오차 없이 주어지는 조건에서 파이프라인을 검증한다."""
    cam = PinholeCamera(640, 512, 1277.37226, cx=320.0, cy=256.0)
    prims = scene.default_satellite()
    pose_l = Pose(quaternion_to_rotation([0.94, 0.0, 0.342, 0.0]), (0.0, 0.0, 5.0))
    pose_r = Pose(pose_l.R, pose_l.t - np.array([SYNTH_BASELINE, 0.0, 0.0]))

    left = scene.render(cam, pose_l, prims, texture_strength=0.35)
    right = scene.render(cam, pose_r, prims, texture_strength=0.35)
    mask, gt = left["mask"], left["depth"]

    disparity = stereo.compute_disparity(left["image"], right["image"],
                                         num_disparities=144)
    disparity = stereo.filter_disparity(disparity)
    z_stereo = np.where(mask, stereo.disparity_to_depth(disparity, cam.fx,
                                                       SYNTH_BASELINE), np.nan)

    z_bright = depth_mod.brightness_depth(left["image"], mask=mask)
    z_bright, a, b = depth_mod.align_scale_shift(z_bright, gt, mask=mask)

    m_s = metrics.depth_metrics(z_stereo, gt, mask=mask)
    m_b = metrics.depth_metrics(z_bright, gt, mask=mask)

    points = cam.unproject(z_stereo, mask=mask)
    result = {
        "camera": [cam.width, cam.height, cam.fx],
        "baseline_m": SYNTH_BASELINE,
        "expected_disparity_px": cam.fx * SYNTH_BASELINE / 5.0,
        "depth_resolution_m_per_px": stereo.depth_resolution(5.0, cam.fx, SYNTH_BASELINE),
        "gt_span_m": span(gt, mask),
        "stereo": {**m_s, "within_5cm": within(z_stereo, gt, mask),
                   "span_m": span(z_stereo, mask), "n_points": int(len(points))},
        "example_code": {**m_b, "within_5cm": within(z_bright, gt, mask),
                         "span_m": span(z_bright, mask),
                         "affine_scale": a, "affine_shift": b},
    }
    return result, {"cam": cam, "left": left, "right": right,
                    "disparity": disparity, "z_stereo": z_stereo,
                    "z_bright": z_bright, "points": points}


# ---------------------------------------------------------------------------
# [2] SPE3R 적용
# ---------------------------------------------------------------------------

def reference_depth(mesh_points_body, pose, pair):
    """메시를 정렬된 왼쪽 카메라로 투영해 기준 깊이 맵을 만든다.

    SPE3R 은 화소 단위 정답 깊이를 제공하지 않는다. 대신 watertight 메시를
    조밀하게 샘플링해 z-buffer 로 투영하면 기준값을 만들 수 있다.
    샘플링 밀도에서 오는 오차가 있으므로 '정답'이 아니라 '기준'으로 부른다.
    """
    p_left = pose.apply(mesh_points_body)
    p_rect = p_left @ pair.R1.T
    d = pointcloud.zbuffer_depth(p_rect, pair.camera, splat=1, fill_holes=True)
    return pair.unrotate(d) if not pair.horizontal else d


def evaluate_pair(model, geo, mesh_points, target_radius=0.8, erode_px=2):
    """뷰 쌍 하나에 대해 스테레오 복원을 수행하고 기준 깊이와 비교한다.

    두 가지 표준 후처리를 적용하고, 적용 전후를 모두 기록한다.

    1) 실루엣 경계 침식 (erode_px)
       경계 화소는 전경과 배경이 섞여 있어 정합이 신뢰할 수 없다.
    2) 물리적 깊이 범위 제한 (target_radius)
       타겟의 경계 반지름을 알고 있으므로 [거리-R, 거리+R] 밖의 값은
       정합 실패다. 이 정보는 정답 깊이가 아니라 물체 크기에서 나온다.
    """
    i, j = geo["i"], geo["j"]
    try:
        pair = stereo.RectifiedPair(model.camera, geo["R_ij"], geo["t_ij"], alpha=-1.0)
    except (ValueError, cv2.error):
        return None

    expected = pair.expected_disparity(geo["distance"])
    if not (6.0 < expected < 0.85 * pair.size[0]):
        return None

    silhouette = model.load_mask(i).astype(np.uint8)
    if erode_px > 0:
        silhouette = cv2.erode(silhouette, np.ones((3, 3), np.uint8),
                               iterations=erode_px)

    d0 = geo["distance"]
    out = stereo.reconstruct(pair, model.load_image(i, grayscale=True),
                             model.load_image(j, grayscale=True),
                             mask=silhouette > 0, distance=d0,
                             depth_range=(d0 - target_radius, d0 + target_radius))
    raw = stereo.reconstruct(pair, model.load_image(i, grayscale=True),
                             model.load_image(j, grayscale=True),
                             mask=model.load_mask(i), distance=d0,
                             postfilter=False)

    ref = reference_depth(mesh_points, model.pose(i), pair)
    mask = out["mask"] if out["mask"] is not None else np.isfinite(ref)
    raw_mask = raw["mask"] if raw["mask"] is not None else np.isfinite(ref)

    m = metrics.depth_metrics(out["depth"], ref, mask=mask)
    if m["n_valid"] < 50:
        return None
    m_raw = metrics.depth_metrics(raw["depth"], ref, mask=raw_mask)

    return {
        "i": i, "j": j,
        "rotation_deg": geo["rotation_deg"],
        "baseline_m": pair.baseline,
        "lateral_ratio": geo["lateral_ratio"],
        "focal_px": pair.focal,
        "horizontal": bool(pair.horizontal),
        "expected_disparity_px": expected,
        "num_disparities": out["num_disparities"],
        "distance_m": d0,
        "rmse": m["rmse"], "median_abs": m["median_abs"],
        "valid_ratio": m["valid_ratio"],
        "within_5cm": within(out["depth"], ref, mask),
        "rmse_unfiltered": m_raw["rmse"],
        "median_abs_unfiltered": m_raw["median_abs"],
        "within_5cm_unfiltered": within(raw["depth"], ref, raw_mask),
        "depth_resolution_m_per_px": stereo.depth_resolution(
            d0, pair.focal, pair.baseline),
        "_pair": pair, "_out": out, "_ref": ref, "_mask": mask,
    }


def run_spe3r(model, mesh_points):
    geos = stereo.find_pairs(model, max_rotation_deg=MAX_ROTATION_DEG,
                             min_lateral_ratio=MIN_LATERAL_RATIO)
    log(f"  회전 {MAX_ROTATION_DEG:.0f}도 이내 · 횡방향비 {MIN_LATERAL_RATIO:.0f} 이상 "
        f"후보 {len(geos)}쌍")

    t0 = time.perf_counter()
    results = [r for r in (evaluate_pair(model, g, mesh_points) for g in geos)
               if r is not None]
    log(f"  복원 성공 {len(results)}쌍  ({time.perf_counter() - t0:.1f}s)")
    if not results:
        return [], None

    results.sort(key=lambda r: r["median_abs"])
    good = [r for r in results if r["median_abs"] < 0.10]
    log(f"  깊이 오차 중앙값 10 cm 이내 {len(good)}쌍")
    for r in results[:5]:
        log(f"    img{r['i']+1:06d}/img{r['j']+1:06d}  {r['rotation_deg']:5.2f}deg  "
            f"B={r['baseline_m']:.3f} m  med={r['median_abs']:.4f} m  "
            f"<5cm {r['within_5cm']*100:5.1f}%  cov {r['valid_ratio']*100:5.1f}%")
    return results, results[0]


def run_example_code(model, best):
    """과제 예시 코드를 같은 정렬 좌표계에서 돌려 동일 기준으로 비교한다."""
    pair, out, ref, mask = best["_pair"], best["_out"], best["_ref"], best["_mask"]
    left_rect = pair.unrotate(out["left"]) if not pair.horizontal else out["left"]

    z = depth_mod.brightness_depth(left_rect, mask=mask)
    aligned, a, b = depth_mod.align_scale_shift(z, ref, mask=mask)
    m = metrics.depth_metrics(aligned, ref, mask=mask)

    grid = baseline.image_to_points_3d(cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR))
    cloud = baseline.points_3d_to_cloud(grid, mask)
    return {**m, "within_5cm": within(aligned, ref, mask),
            "span_m": span(aligned, mask),
            "affine_scale": a, "affine_shift": b,
            "n_points": int(len(cloud))}, aligned, cloud


# ---------------------------------------------------------------------------
# 그림
# ---------------------------------------------------------------------------

def figure_concept(best, path):
    """방법의 핵심을 그림 한 장으로 설명한다 (발표용 개념도).

    ① 데이터가 주는 것 : 카메라는 제자리, 타겟만 회전
    ② 같은 상황을 타겟 기준으로 : 카메라가 궤도를 돈 것과 같다 -> 베이스라인
    ③ 삼각측량 : Z = f·B/d
    """
    from matplotlib.patches import Arc, FancyArrow, Rectangle

    dist = best["distance_m"]
    ang = best["rotation_deg"]
    base = best["baseline_m"]

    # 패널 하나가 데이터 좌표 10 x 7 (비율 0.7) 이므로, 셋을 나란히 놓으면
    # 그림 전체 비율이 슬라이드 카드와 맞도록 캔버스 크기를 정한다.
    fig, ax = plt.subplots(1, 3, figsize=(15.0, 3.9))
    # 세 패널의 좌표 범위를 똑같이 맞춰야 크기와 제목 높이가 나란히 정렬된다.
    for a in ax:
        a.set_aspect("equal")
        a.axis("off")
        a.set_xlim(-5.0, 5.0)
        a.set_ylim(-3.5, 3.5)

    def camera(a, x, y, angle_deg=0, size=0.5, color="#1f77b4", label=None):
        """카메라를 삼각형으로 그린다 (시야각 표현)."""
        t = np.deg2rad(angle_deg)
        pts = np.array([[0, 0], [size * 1.6, size * 0.6], [size * 1.6, -size * 0.6]])
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        pts = pts @ R.T + np.array([x, y])
        a.fill(pts[:, 0], pts[:, 1], color=color, alpha=0.85, zorder=3)
        if label:
            a.text(x, y - size * 1.15, label, ha="center", va="top",
                   fontsize=10, color=color)

    def satellite(a, x, y, rot_deg, scale=1.0, color="#444444", alpha=1.0):
        """위성을 본체 + 태양전지판으로 간단히 그린다."""
        t = np.deg2rad(rot_deg)
        R = np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
        parts = [np.array([[-.28, -.28], [.28, -.28], [.28, .28], [-.28, .28]]),
                 np.array([[-1.15, -.07], [-.3, -.07], [-.3, .07], [-1.15, .07]]),
                 np.array([[1.15, -.07], [.3, -.07], [.3, .07], [1.15, .07]])]
        for k, p in enumerate(parts):
            q = (p * scale) @ R.T + np.array([x, y])
            a.fill(q[:, 0], q[:, 1], color=color if k == 0 else "#2b5f9e",
                   alpha=alpha, zorder=2)

    # ---- ① 카메라는 제자리, 타겟만 회전 ----
    a = ax[0]
    a.set_title("① 데이터가 주는 것", pad=12)
    a.text(0, 3.0, "카메라 위치는 고정 · 병진이 항상 (0, 0, Z)",
           ha="center", fontsize=11.5, color="#171717")
    camera(a, -4.0, 0, 0, 0.55, "#1f77b4", "카메라")
    a.plot([-2.9, 0.2], [0, 0], ls="--", color="#aaaaaa", lw=1.2)
    a.text(-1.4, -0.5, f"거리 {dist:.2f} m", ha="center", fontsize=10.5,
           color="#555555")
    satellite(a, 1.9, 0, 14, 0.85, alpha=0.30)
    satellite(a, 1.9, 0, 72, 0.85, alpha=1.0)
    a.text(1.9, -1.55, "프레임 i / 프레임 j", ha="center", fontsize=10.5,
           color="#555555")
    a.annotate("", xy=(1.35, 1.75), xytext=(3.3, 1.30),
               arrowprops=dict(arrowstyle="->", color="#d62728", lw=2.0,
                               connectionstyle="arc3,rad=0.45"))
    a.text(3.4, 1.55, "타겟만\n회전", color="#d62728", fontsize=11, ha="left",
           va="center")
    a.text(0, -3.15, "카메라 좌표계만 보면 베이스라인이 없다", ha="center",
           fontsize=11, color="#7a7a7a")

    # ---- ② 타겟 기준으로 보면 ----
    a = ax[1]
    a.set_title("② 타겟 기준으로 보면", pad=12)
    r, show = 3.25, 30.0        # 실제 4.5°는 너무 작아 그림에서는 과장한다
    cy0 = -1.35
    satellite(a, 0, cy0, 20, 0.85)
    for sgn, alpha in ((-1, 0.5), (1, 1.0)):
        th = np.deg2rad(90 + sgn * show / 2)
        cx, cyy = r * np.cos(th), cy0 + r * np.sin(th)
        camera(a, cx, cyy, np.rad2deg(th) + 180, 0.5, "#1f77b4")
        a.plot([cx, 0], [cyy, cy0], ls=":", color="#bbbbbb", lw=1.1)
    a.add_patch(Arc((0, cy0), 2 * r, 2 * r, theta1=90 - show / 2,
                    theta2=90 + show / 2, color="#1f77b4", lw=1.4, ls="--"))
    th1, th2 = np.deg2rad(90 - show / 2), np.deg2rad(90 + show / 2)
    a.plot([r * np.cos(th1), r * np.cos(th2)],
           [cy0 + r * np.sin(th1), cy0 + r * np.sin(th2)],
           color="#d62728", lw=3.2, zorder=4)
    a.text(0, cy0 + r + 0.35, f"유효 베이스라인  B = {base:.3f} m",
           ha="center", fontsize=12.5, color="#d62728")
    a.text(0, 3.0, f"타겟이 {ang:.2f}° 회전 = 카메라가 궤도를 돈 것",
           ha="center", fontsize=11.5, color="#171717")
    a.text(0, -3.15, "R_ij = R_j·R_iᵀ      t_ij = t_j − R_ij·t_i",
           ha="center", fontsize=10.5, color="#555555", family="monospace")
    a.text(3.2, 0.4, "(각도는\n 과장)", fontsize=9.5, color="#aaaaaa", ha="center")

    # ---- ③ 삼각측량 ----
    a = ax[2]
    a.set_title("③ 삼각측량으로 깊이 계산", pad=12)
    xl, xr, yc = -1.25, 1.25, -1.9
    px, py = 0.45, 1.55
    plane_y = yc + 0.80
    crossings = []
    for x in (xl, xr):
        camera(a, x, yc, 90, 0.5, "#1f77b4")
        a.plot([x, px], [yc, py], color="#999999", lw=1.2, ls=":")
        a.add_patch(Rectangle((x - 0.9, plane_y), 1.8, 0.15, color="#333333"))
        # 광선이 상면을 지나는 지점 = 그 영상에서 점이 맺히는 위치
        s = (plane_y - yc) / (py - yc)
        crossings.append(x + s * (px - x))
    a.plot(crossings, [plane_y + 0.08] * 2, "o", color="#7f2fbf", ms=7, zorder=6)

    a.plot([xl, xr], [yc - 0.42, yc - 0.42], color="#d62728", lw=3.2)
    a.text(xr + 0.28, yc - 0.42, "B", ha="left", va="center", fontsize=13,
           color="#d62728")
    a.plot(px, py, "o", color="#2ca02c", ms=10, zorder=5)
    a.text(px + 0.3, py + 0.1, "표면의 한 점", fontsize=10.5, color="#2ca02c",
           va="center")
    a.annotate("", xy=(crossings[0], plane_y + 0.62), xytext=(crossings[1], plane_y + 0.62),
               arrowprops=dict(arrowstyle="<->", color="#7f2fbf", lw=1.8))
    for c in crossings:
        a.plot([c, c], [plane_y + 0.12, plane_y + 0.58], color="#7f2fbf",
               lw=0.9, ls=":")
    a.text(np.mean(crossings), plane_y + 0.85, "시차 d", ha="center",
           fontsize=11.5, color="#7f2fbf")
    a.text(0, 3.0, "Z  =  f · B / d", ha="center", fontsize=18, color="#171717")
    a.text(0, 2.35, f"f = {best['focal_px']:.0f} px      B = {base:.3f} m",
           ha="center", fontsize=10.5, color="#555555")
    a.text(0, -3.15, f"시차 1픽셀이 깊이 "
           f"{best['depth_resolution_m_per_px']*100:.1f} cm 에 해당", ha="center",
           fontsize=11, color="#7a7a7a")

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def crop_box(mask, margin=0.18):
    """마스크를 감싸는 정사각 영역을 구한다. 물체가 화면에서 작을 때 쓴다."""
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return slice(None), slice(None)
    cy, cx = (ys.min() + ys.max()) / 2, (xs.min() + xs.max()) / 2
    half = max(ys.max() - ys.min(), xs.max() - xs.min()) / 2 * (1 + margin)
    y0 = max(0, int(cy - half)); y1 = min(mask.shape[0], int(cy + half) + 1)
    x0 = max(0, int(cx - half)); x1 = min(mask.shape[1], int(cx + half) + 1)
    return slice(y0, y1), slice(x0, x1)


def figure_synthetic(art, res, path):
    left, right = art["left"], art["right"]
    mask = left["mask"]
    cy, cx = crop_box(mask)
    m = mask[cy, cx]

    gt = left["depth"][cy, cx]
    lo, hi = np.nanmin(gt[m]), np.nanmax(gt[m])

    fig, ax = plt.subplots(1, 6, figsize=(16.2, 3.6))
    ax[0].imshow(left["image"][cy, cx], cmap="gray")
    ax[0].set_title("왼쪽 영상")
    ax[1].imshow(right["image"][cy, cx], cmap="gray")
    ax[1].set_title(f"오른쪽 영상\n베이스라인 {SYNTH_BASELINE:.2f} m")
    d = ax[2].imshow(np.where(m, art["disparity"][cy, cx], np.nan), cmap="magma")
    ax[2].set_title(f"시차 [px]\n예상 {res['expected_disparity_px']:.0f} px")
    fig.colorbar(d, ax=ax[2], fraction=0.046)
    g = ax[3].imshow(gt, cmap="viridis", vmin=lo, vmax=hi)
    ax[3].set_title(f"정답 깊이 [m]\n깊이 폭 {res['gt_span_m']:.3f} m")
    fig.colorbar(g, ax=ax[3], fraction=0.046)
    s = ax[4].imshow(art["z_stereo"][cy, cx], cmap="viridis", vmin=lo, vmax=hi)
    ax[4].set_title(f"스테레오 복원  Z = f·B/d\n오차 중앙값 "
                    f"{res['stereo']['median_abs']*100:.1f} cm · 깊이 폭 "
                    f"{res['stereo']['span_m']:.3f} m")
    fig.colorbar(s, ax=ax[4], fraction=0.046)
    e = ax[5].imshow(art["z_bright"][cy, cx], cmap="viridis", vmin=lo, vmax=hi)
    ax[5].set_title(f"과제 예시 코드 (최적 정렬)\n오차 중앙값 "
                    f"{res['example_code']['median_abs']*100:.1f} cm · 깊이 폭 "
                    f"{res['example_code']['span_m']:.3f} m")
    fig.colorbar(e, ax=ax[5], fraction=0.046)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("합성 장면 — 정답 깊이에 오차가 없는 조건에서 비교", fontsize=13)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)


def figure_survey(results, path):
    rot = np.array([r["rotation_deg"] for r in results])
    med = np.array([r["median_abs"] for r in results])
    base = np.array([r["baseline_m"] for r in results])
    cov = np.array([r["valid_ratio"] * 100 for r in results])
    w5 = np.array([r["within_5cm"] * 100 for r in results])

    fig, ax = plt.subplots(1, 3, figsize=(12.6, 3.6))
    sc = ax[0].scatter(base, med * 100, c=rot, cmap="viridis", s=60)
    ax[0].set_xlabel("유효 베이스라인 [m]"); ax[0].set_ylabel("깊이 오차 중앙값 [cm]")
    ax[0].set_title("베이스라인과 정확도"); ax[0].set_yscale("log")
    fig.colorbar(sc, ax=ax[0], label="회전각 [도]", fraction=0.046)

    ax[1].scatter(cov, w5, s=60, color="#2ca02c")
    ax[1].set_xlabel("유효 화소 비율 [%]"); ax[1].set_ylabel("5 cm 이내 화소 [%]")
    ax[1].set_title("복원 범위와 정확도")

    order = np.argsort(med)
    ax[2].plot(np.arange(1, len(med) + 1), med[order] * 100, "o-", ms=8,
               color="#9467bd")
    ax[2].axhline(10, ls="--", c="#333", lw=1.2)
    ax[2].text(len(med) * 0.55, 11, "10 cm 기준", fontsize=10, color="#333")
    ax[2].set_xlabel("쌍 순위"); ax[2].set_ylabel("깊이 오차 중앙값 [cm]")
    ax[2].set_title(f"정확도 순 정렬 ({len(results)}쌍)"); ax[2].set_yscale("log")

    fig.suptitle("SPE3R 뷰 쌍별 복원 결과", fontsize=13)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def figure_spe3r(best, example_depth, path):
    pair, out = best["_pair"], best["_out"]
    unrot = (lambda a: pair.unrotate(a)) if not pair.horizontal else (lambda a: a)
    L, R = unrot(out["left"]), unrot(out["right"])
    disp, dep = unrot(out["disparity"]), unrot(out["depth"])
    ref, mask = best["_ref"], best["_mask"]

    finite = np.isfinite(ref) & mask
    lo, hi = np.nanpercentile(ref[finite], [1, 99])
    cy, cx = crop_box(finite)
    m = mask[cy, cx]

    fig, ax = plt.subplots(1, 6, figsize=(16.2, 3.6))
    ax[0].imshow(L[cy, cx], cmap="gray")
    ax[0].set_title(f"정렬된 왼쪽 영상\nimg{best['i']+1:06d}")
    ax[1].imshow(R[cy, cx], cmap="gray")
    ax[1].set_title(f"정렬된 오른쪽 영상  img{best['j']+1:06d}\n"
                    f"회전 {best['rotation_deg']:.2f}°, B = {best['baseline_m']:.3f} m")
    d = ax[2].imshow(np.where(m, disp[cy, cx], np.nan), cmap="magma")
    ax[2].set_title(f"시차 [px]\n탐색 범위 {best['num_disparities']}")
    fig.colorbar(d, ax=ax[2], fraction=0.046)
    r = ax[3].imshow(np.where(m, ref[cy, cx], np.nan), cmap="viridis", vmin=lo, vmax=hi)
    ax[3].set_title("기준 깊이 [m]\n(메시 z-buffer)")
    fig.colorbar(r, ax=ax[3], fraction=0.046)
    s = ax[4].imshow(dep[cy, cx], cmap="viridis", vmin=lo, vmax=hi)
    ax[4].set_title(f"스테레오 복원  Z = f·B/d\n오차 중앙값 "
                    f"{best['median_abs']*100:.1f} cm · 5cm 이내 "
                    f"{best['within_5cm']*100:.0f}%")
    fig.colorbar(s, ax=ax[4], fraction=0.046)
    e = ax[5].imshow(np.where(m, example_depth[cy, cx], np.nan), cmap="viridis",
                     vmin=lo, vmax=hi)
    ax[5].set_title("과제 예시 코드\n(밝기, 최적 정렬)")
    fig.colorbar(e, ax=ax[5], fraction=0.046)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("SPE3R aqua — 회전하는 타겟을 찍은 두 시점에서 복원", fontsize=13)
    fig.tight_layout(); fig.savefig(path); plt.close(fig)


def _scatter3d(ax, pts, color, title, lim, size=0.5, title_size=18):
    step = max(1, len(pts) // 12000)
    p = pts[::step]
    ax.scatter(p[:, 0], p[:, 2], -p[:, 1], s=size, c=color, linewidths=0,
               depthshade=False)
    # 이 그림은 슬라이드에서 작게 들어가므로 글자를 크게 잡는다.
    ax.set_title(title, fontsize=title_size)
    ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    ax.set_xlim(*lim[0]); ax.set_ylim(*lim[1]); ax.set_zlim(*lim[2])
    ax.set_box_aspect((1, 1, 1))


def figure_pointclouds(mesh_points, stereo_body, example_cloud, path):
    def limits(p):
        c = (p.min(axis=0) + p.max(axis=0)) / 2
        h = float((p.max(axis=0) - p.min(axis=0)).max()) / 2 * 1.1
        return ((c[0] - h, c[0] + h), (c[2] - h, c[2] + h), (-c[1] - h, -c[1] + h))

    lim = limits(mesh_points)
    fig = plt.figure(figsize=(11.4, 3.0))
    a1 = fig.add_subplot(1, 3, 1, projection="3d")
    _scatter3d(a1, mesh_points, "#1f77b4", "정답\n(메시 표면)", lim, 0.3)
    a2 = fig.add_subplot(1, 3, 2, projection="3d")
    _scatter3d(a2, stereo_body, "#d62728",
               f"스테레오 깊이 → 3D\n{len(stereo_body):,}점 · 단일 시점", lim)
    a3 = fig.add_subplot(1, 3, 3, projection="3d")
    norm, _, _ = pointcloud.normalize_scale(example_cloud)
    _scatter3d(a3, norm, "#7f7f7f", "과제 예시 코드\n(픽셀, 픽셀, 밝기)",
               limits(norm), 0.2)
    # 3D 축은 제목이 축 상자보다 위로 크게 벗어나므로 전체 제목을 두지 않는다.
    # 그림이 무엇인지는 슬라이드와 README 의 설명이 맡는다.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.0, wspace=0.05)
    fig.savefig(path); plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    log("=" * 70)
    log("2차 업무 — 스테레오 삼각측량으로 깊이 맵과 3D 포인트 클라우드 생성")
    log("=" * 70)

    log("\n[1] 합성 장면 검증 (정답 깊이 오차 0)")
    synth, art = run_synthetic()
    s, e = synth["stereo"], synth["example_code"]
    log(f"  베이스라인 {SYNTH_BASELINE} m · 기대 시차 "
        f"{synth['expected_disparity_px']:.1f} px · 깊이 분해능 "
        f"{synth['depth_resolution_m_per_px']*100:.1f} cm/px")
    log(f"  {'':22s}{'RMSE':>9s}{'중앙값':>10s}{'<5cm':>9s}{'깊이폭':>9s}")
    log(f"  {'스테레오':22s}{s['rmse']:9.4f}{s['median_abs']:10.4f}"
        f"{s['within_5cm']*100:8.1f}%{s['span_m']:9.3f}")
    log(f"  {'과제 예시 (최적정렬)':22s}{e['rmse']:9.4f}{e['median_abs']:10.4f}"
        f"{e['within_5cm']*100:8.1f}%{e['span_m']:9.3f}")
    log(f"  정답 깊이폭 {synth['gt_span_m']:.3f} m  →  중앙값 오차 "
        f"{e['median_abs']/s['median_abs']:.1f}배 개선")

    log("\n[2] SPE3R 실제 데이터")
    model = SPE3RModel(DATA, MODEL_NAME)
    log(f"  {model}")
    vertices, faces = model.load_mesh()
    mesh_points = pointcloud.sample_mesh_surface(vertices, faces, MESH_SAMPLES, seed=0)
    log(f"  메시 표면 샘플 {len(mesh_points):,}점 → 기준 깊이 맵 생성용")

    results, best = run_spe3r(model, mesh_points)
    if best is None:
        log("복원 가능한 쌍이 없습니다.")
        return 1

    ex, ex_depth, ex_cloud = run_example_code(model, best)
    log(f"\n  최적 쌍 img{best['i']+1:06d}/img{best['j']+1:06d}")
    log(f"  {'':22s}{'RMSE':>9s}{'중앙값':>10s}{'<5cm':>9s}")
    log(f"  {'스테레오':22s}{best['rmse']:9.4f}{best['median_abs']:10.4f}"
        f"{best['within_5cm']*100:8.1f}%")
    log(f"  {'과제 예시 (최적정렬)':22s}{ex['rmse']:9.4f}{ex['median_abs']:10.4f}"
        f"{ex['within_5cm']*100:8.1f}%")

    pair = best["_pair"]
    stereo_body = pair.to_body(best["_out"]["points"], model.pose(best["i"]))
    chamfer = metrics.chamfer_distance(stereo_body, mesh_points, norm=1)
    log(f"  포인트 클라우드 {len(stereo_body):,}점 · 단방향 Chamfer(pred→GT) "
        f"{chamfer['pred_to_target']:.4f}")

    log("\n그림 생성")
    figure_concept(best, os.path.join(OUT, "00_concept.png"))
    figure_synthetic(art, synth, os.path.join(OUT, "01_synthetic_validation.png"))
    figure_survey(results, os.path.join(OUT, "02_pair_survey.png"))
    figure_spe3r(best, ex_depth, os.path.join(OUT, "03_spe3r_stereo.png"))
    figure_pointclouds(mesh_points, stereo_body, ex_cloud,
                       os.path.join(OUT, "04_pointclouds.png"))

    log("PLY 저장")
    pointcloud.write_ply(os.path.join(OUT, "pointcloud_ground_truth.ply"),
                         mesh_points[::13])
    pointcloud.write_ply(os.path.join(OUT, "pointcloud_stereo.ply"), stereo_body)
    pointcloud.write_ply(os.path.join(OUT, "pointcloud_example_code.ply"),
                         pointcloud.normalize_scale(ex_cloud)[0])

    med = np.array([r["median_abs"] for r in results])
    summary = {
        "dataset": {
            "name": "SPE3R", "model": MODEL_NAME, "views": len(model),
            "image_size": [model.camera.width, model.camera.height],
            "fx": model.camera.fx,
            "camera_translation": "always (0, 0, Z); Z sweeps 5.000 to 6.000 m",
            "note": ("카메라는 옆으로 움직이지 않지만 타겟이 회전하므로 두 뷰의 "
                     "상대 자세에서 유효 베이스라인이 생긴다"),
            "license": "CC BY-NC-SA 4.0",
            "source": "https://purl.stanford.edu/pk719hm4806",
        },
        "synthetic_validation": synth,
        "spe3r_pair_survey": {
            "max_rotation_deg": MAX_ROTATION_DEG,
            "min_lateral_ratio": MIN_LATERAL_RATIO,
            "pairs_reconstructed": len(results),
            "pairs_within_10cm": int((med < 0.10).sum()),
            "median_error_m": {"best": float(med.min()),
                               "median": float(np.median(med)),
                               "worst": float(med.max())},
            "pairs": [{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in results],
        },
        "best_pair": {k: v for k, v in best.items() if not k.startswith("_")},
        "best_pair_example_code": ex,
        "best_pair_chamfer_pred_to_gt": chamfer["pred_to_target"],
        "best_pair_points": int(len(stereo_body)),
    }
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    log(f"\n완료 -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
