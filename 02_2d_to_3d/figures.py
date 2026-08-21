"""달 지형 스테레오 실험의 결과 그림."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.bbox"] = "tight"

__all__ = ["figure_overview", "figure_tradeoff", "figure_cloud",
           "figure_fusion"]


def figure_overview(elev, gsd, view, rec, scored, path):
    """왼쪽 영상부터 오차 지도까지 한 줄로 보여 준다.

    깊이 맵을 색으로만 보여 주면 "그럴듯한 그림" 과 구분이 안 된다. 정답과
    복원을 같은 색 범위로 놓고 그 차이를 따로 그려야 얼마나 맞았는지 보인다.
    """
    ref, dep = rec["reference"], rec["depth"]
    finite = np.isfinite(ref)
    lo, hi = np.nanpercentile(ref[finite], [2, 98])
    err = np.abs(dep - ref)

    fig, ax = plt.subplots(1, 5, figsize=(15.4, 3.2))
    ax[0].imshow(rec["left_rect"], cmap="gray")
    ax[0].set_title(f"왼쪽 영상 (정렬 후)\n태양 고도 25도")
    ax[1].imshow(rec["right_rect"], cmap="gray")
    ax[1].set_title("오른쪽 영상 (정렬 후)")

    d = ax[2].imshow(rec["disparity"], cmap="magma")
    ax[2].set_title(f"시차 d\n탐색 {rec['min_disparity']}~"
                    f"{rec['min_disparity'] + rec['num_disparities']} px")
    fig.colorbar(d, ax=ax[2], fraction=0.046, label="[px]")

    r = ax[3].imshow(np.where(finite, ref, np.nan), cmap="terrain",
                     vmin=lo, vmax=hi)
    ax[3].set_title("정답 깊이 (측정 고도)")
    fig.colorbar(r, ax=ax[3], fraction=0.046, label="[m]")

    s = ax[4].imshow(dep, cmap="terrain", vmin=lo, vmax=hi)
    ax[4].set_title(f"복원 깊이\n중앙값 오차 {scored['median_abs']:.0f} m "
                    f"= 시차 {scored['median_abs_px']:.2f} px")
    fig.colorbar(s, ax=ax[4], fraction=0.046, label="[m]")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("달 지형 스테레오 — 두 장에서 고도까지", fontsize=13)
    fig.savefig(path); plt.close(fig)
    return err


def figure_tradeoff(conv_rows, block_rows, path):
    """수렴각과 블록 크기의 맞바꿈을 나란히 그린다.

    둘 다 "하나를 얻으면 하나를 내주는" 관계라, 표로 적으면 어느 쪽이
    무엇과 맞바뀌는지 눈에 안 들어온다.
    """
    fig, ax = plt.subplots(1, 2, figsize=(11.0, 3.6))

    conv = [r["convergence_deg"] for r in conv_rows]
    med = [r["median_abs"] for r in conv_rows]
    val = [r["valid_ratio"] * 100 for r in conv_rows]
    a = ax[0]
    a.plot(conv, med, "o-", color="#d62728", label="Z 오차 중앙값 [m]")
    a.set_xlabel("수렴각 [도]"); a.set_ylabel("Z 오차 중앙값 [m]", color="#d62728")
    a.tick_params(axis="y", labelcolor="#d62728")
    b = a.twinx()
    b.plot(conv, val, "s--", color="#1f77b4", label="유효화소 [%]")
    b.set_ylabel("유효화소 [%]", color="#1f77b4")
    b.tick_params(axis="y", labelcolor="#1f77b4")
    a.set_title("수렴각 — 각을 키우면 정밀해지고 정합은 어려워진다")

    bs = [r["block_size"] for r in block_rows]
    med2 = [r["median_abs"] for r in block_rows]
    val2 = [r["valid_ratio"] * 100 for r in block_rows]
    a = ax[1]
    a.plot(bs, med2, "o-", color="#d62728")
    a.set_xlabel("정합 블록 크기 [px]"); a.set_ylabel("Z 오차 중앙값 [m]",
                                                  color="#d62728")
    a.tick_params(axis="y", labelcolor="#d62728")
    b = a.twinx()
    b.plot(bs, val2, "s--", color="#1f77b4")
    b.set_ylabel("유효화소 [%]", color="#1f77b4")
    b.tick_params(axis="y", labelcolor="#1f77b4")
    a.set_title("블록 크기 — 무늬가 넉넉하면 작은 쪽이 정확하다")

    fig.savefig(path); plt.close(fig)


def figure_cloud(truth_points, cloud, path):
    """정답 지형과 복원한 점구름을 나란히 놓는다."""
    def draw(a, pts, color, title, step):
        p = pts[::step]
        a.scatter(p[:, 0], p[:, 1], c=p[:, 2] if color is None else color,
                  s=0.4, cmap="terrain", linewidths=0)
        a.set_aspect("equal"); a.set_xticks([]); a.set_yticks([])
        a.set_title(title, fontsize=12)

    fig, ax = plt.subplots(1, 2, figsize=(9.2, 4.4))
    draw(ax[0], truth_points, None,
         f"정답 지형 ({len(truth_points):,}점)", max(1, len(truth_points) // 60000))
    draw(ax[1], cloud, None,
         f"복원 포인트 클라우드 ({len(cloud):,}점)", max(1, len(cloud) // 60000))
    fig.suptitle("색은 고도 [m]", fontsize=11)
    fig.savefig(path); plt.close(fig)


def figure_fusion(truth, fused, per_angle, fused_score, gsd, path):
    """수렴각별 결과와 융합 결과를 나란히 놓는다.

    정밀한 각도는 좁게, 넓은 각도는 성기게 덮는다. 표로 적으면 그 상보성이
    눈에 안 들어오므로, 어디가 비었는지를 그림으로 보인다.
    """
    lo, hi = np.nanpercentile(truth, [2, 98])
    n = len(per_angle)
    fig, ax = plt.subplots(1, n + 2, figsize=(3.1 * (n + 2), 3.4))

    t = ax[0].imshow(truth, cmap="terrain", vmin=lo, vmax=hi)
    ax[0].set_title("정답 고도\n(레이저 측정)")
    fig.colorbar(t, ax=ax[0], fraction=0.046, label="[m]")

    for k, (conv, grid, sc) in enumerate(per_angle, start=1):
        ax[k].imshow(grid, cmap="terrain", vmin=lo, vmax=hi)
        ax[k].set_title(f"수렴각 {conv:.0f}도\n덮은 셀 {sc['coverage']*100:.0f}% · "
                        f"오차 {sc['median_abs']:.0f} m")

    f = ax[n + 1].imshow(fused, cmap="terrain", vmin=lo, vmax=hi)
    ax[n + 1].set_title(f"융합\n덮은 셀 {fused_score['coverage']*100:.0f}% · "
                        f"오차 {fused_score['median_abs']:.0f} m")
    fig.colorbar(f, ax=ax[n + 1], fraction=0.046, label="[m]")

    for a in ax:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle("수렴각마다 덮는 곳이 다르다 — 합치면 둘 다 얻는다", fontsize=13)
    fig.savefig(path); plt.close(fig)
