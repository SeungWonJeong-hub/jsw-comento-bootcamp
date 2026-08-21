"""결과 그림 생성 — 실험 로직과 분리해 둔다.

run_3d_experiment.py 가 900 줄을 넘었고 그 절반 가까이가 matplotlib 코드였다.
실험 로직과 그리기 코드가 한 파일에 섞여 있으면 실험 쪽을 읽기 어렵다. 이
저장소에서 좌표 규약 버그가 실제로 그 파일에서 났고, 그 파일에는 테스트가 없다.
얇게 유지한다.

상세 그림(6패널)과 발표용 그림(3패널)을 함께 만든다. 상세본은 README 와
outputs/ 에 남고, 3패널은 슬라이드에 쓴다.
"""

from __future__ import annotations

import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src import pointcloud  # noqa: E402

SYNTH_BASELINE = 0.40

# 발표 화면에서 읽히도록 글꼴을 키우고, PPT 와 같은 한글 글꼴을 쓴다.
plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 150,
    "font.family": ["Pretendard", "Malgun Gothic", "DejaVu Sans"],
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
    "xtick.labelsize": 10, "ytick.labelsize": 10,
    "axes.grid": True, "grid.alpha": 0.25, "axes.unicode_minus": False,
})


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
                    f"{res['example_code']['full']['median_abs']*100:.1f} cm · 깊이 폭 "
                    f"{res['example_code']['full']['span_m']:.3f} m")
    fig.colorbar(e, ax=ax[5], fraction=0.046)

    for a in ax.ravel():
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.suptitle("합성 장면 — 정답 깊이에 오차가 없는 조건에서 비교", fontsize=13)
    fig.tight_layout()
    fig.savefig(path); plt.close(fig)

    figure_depth_triptych(
        [(f"정답 깊이 [m]\n깊이 폭 {res['gt_span_m']:.3f} m", gt, "viridis", lo, hi),
         (f"스테레오  Z = f·B/d\n깊이 폭 {res['stereo']['span_m']:.3f} m",
          art["z_stereo"][cy, cx], "viridis", lo, hi),
         (f"과제 예시 코드 (최적 정렬)\n깊이 폭 "
          f"{res['example_code']['full']['span_m']:.3f} m",
          art["z_bright"][cy, cx], "viridis", lo, hi)],
        path.replace(".png", "_slide.png"))


def figure_depth_triptych(panels, path):
    """발표 슬라이드용 3패널 비교 그림.

    상세 그림은 6패널이라 슬라이드에서 하나하나가 너무 작아진다. 논증에 실제로
    쓰이는 것은 '기준 깊이 / 스테레오 / 과제 예시' 세 장뿐이고, 원본 좌우 영상과
    시차 맵은 근거를 따라가려는 사람에게 필요한 자료다. 슬라이드에는 세 장만
    싣고 여섯 장짜리는 README 와 outputs/ 에 남긴다.

    panels : (제목, 배열, 색상표, vmin, vmax) 3개
    """
    # 가로로 길게 잡는다. 슬라이드의 그림 자리가 가로로 넓어서, 세로로 긴
    # 그림을 넣으면 높이에 걸려 카드의 절반만 차고 작아 보인다. suptitle 은
    # 빼고 슬라이드 부제에 맡긴다 - 같은 말을 두 번 적을 자리가 아니다.
    fig, ax = plt.subplots(1, 3, figsize=(13.6, 3.5))
    for a, (title, arr, cmap, lo, hi) in zip(ax, panels):
        im = a.imshow(arr, cmap=cmap, vmin=lo, vmax=hi)
        a.set_title(title, fontsize=13)
        cb = fig.colorbar(im, ax=a, fraction=0.040, pad=0.02)
        cb.ax.tick_params(labelsize=10)
        a.set_xticks([]); a.set_yticks([]); a.grid(False)
    fig.tight_layout(pad=0.4)
    fig.savefig(path, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


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
    out = best["_out"]
    # reconstruct() 가 정렬된 왼쪽 카메라 좌표계로 되돌려 주므로 여기서 방향을
    # 다시 맞출 필요가 없다.
    L, R = out["left"], out["right"]
    disp, dep = out["disparity"], out["depth"]
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

    figure_depth_triptych(
        [("기준 깊이 [m]\n(동봉 메시 z-buffer)",
          np.where(m, ref[cy, cx], np.nan), "viridis", lo, hi),
         (f"스테레오  Z = f·B/d\n오차 중앙값 {best['median_abs']*100:.1f} cm · "
          f"5cm 이내 {best['within_5cm']*100:.0f}%",
          dep[cy, cx], "viridis", lo, hi),
         (f"과제 예시 코드 (밝기, 최적 정렬)\n오차 중앙값 "
          f"{best['_example_median']*100:.1f} cm",
          np.where(m, example_depth[cy, cx], np.nan), "viridis", lo, hi)],
        path.replace(".png", "_slide.png"))


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


def figure_pointclouds(mesh_points, stereo_body, carved_body, example_cloud, path,
                       coverage=None):
    """정답 · 스테레오(단일 시점) · 카빙(전방위) · 과제 예시를 나란히 놓는다.

    스테레오만 보이면 "3D 복원했다" 로 읽히는데 실제로 덮은 것은 앞면뿐이다.
    전방위 복원을 옆에 두어야 무엇이 빠졌는지 눈으로 보인다.
    """
    def limits(p):
        c = (p.min(axis=0) + p.max(axis=0)) / 2
        h = float((p.max(axis=0) - p.min(axis=0)).max()) / 2 * 1.1
        return ((c[0] - h, c[0] + h), (c[2] - h, c[2] + h), (-c[1] - h, -c[1] + h))

    cov = coverage or {}
    lim = limits(mesh_points)
    fig = plt.figure(figsize=(12.6, 3.3))
    a1 = fig.add_subplot(1, 4, 1, projection="3d")
    _scatter3d(a1, mesh_points, "#1f77b4", "정답\n(메시 표면)", lim, 0.3)
    a2 = fig.add_subplot(1, 4, 2, projection="3d")
    _scatter3d(a2, stereo_body, "#d62728",
               f"스테레오 (단일 시점)\n{len(stereo_body):,}점 · 표면 "
               f"{cov.get('stereo', 0) * 100:.0f}%", lim)
    a3 = fig.add_subplot(1, 4, 3, projection="3d")
    _scatter3d(a3, carved_body, "#2ca02c",
               f"실루엣 카빙 (전방위)\n{len(carved_body):,}점 · 표면 "
               f"{cov.get('carving', 0) * 100:.0f}%", lim, 0.4)
    a4 = fig.add_subplot(1, 4, 4, projection="3d")
    norm, _, _ = pointcloud.normalize_scale(example_cloud)
    _scatter3d(a4, norm, "#7f7f7f", "과제 예시 코드\n(픽셀, 픽셀, 밝기)",
               limits(norm), 0.2)
    # 3D 축은 제목이 축 상자보다 위로 크게 벗어나므로 전체 제목을 두지 않는다.
    # 그림이 무엇인지는 슬라이드와 README 의 설명이 맡는다.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.80, bottom=0.0, wspace=0.05)
    fig.savefig(path); plt.close(fig)
