"""AI Hub 회전 박스 규약 확정 — 코멘토 3차 업무 / 정승원

왜 필요한가
-----------
형식과 규약은 다르다. 핀란드 S2Ships 는 8좌표 OBB '형식' 으로 배포되지만
실제 좌표를 13,069개 전수 측정하니 100% 축 정렬이었다. 형식만 보고 회전
박스라고 단정하는 바람에 학습·평가가 통째로 어긋나 있었다. 같은 실수를
AI Hub 에서 반복하지 않기 위한 스크립트다.

AI Hub 공개 설명에는 [중심 x, 중심 y, 박스 높이 H, 박스 너비 W, 회전각 theta]
라고만 적혀 있고, 다음이 명시돼 있지 않다. 이 스크립트가 실측으로 확정한다.

  - 좌표가 픽셀인가 지도좌표인가
  - 픽셀 원점이 좌상단인가
  - H, W 단위가 픽셀인가 미터인가
  - theta 가 도인가 라디안인가
  - 시계방향이 양수인가 반시계방향이 양수인가
  - 범위가 0~180 인가 -90~90 인가
  - W/H 교환에 따라 각도가 도는가

방법
----
후보 규약을 모두 만들어 각각 박스를 그리고, 박스 안 밝은 화소의 주축(PCA)과
박스 장변이 이루는 각을 잰다. 규약이 맞으면 이 각이 0 에 가깝다.
사람 눈으로 확인할 그림도 같이 낸다. 자동 점수만 믿지 않는다.

주석 규약의 기준
----------------
DOTA (Ding et al., 2021) 가 문서화한 표준을 따른다.
  네 꼭짓점을 시계방향으로, 첫 점은 HEAD (선박의 뱃머리).
  시각 단서가 없으면 좌상단을 시작점으로 한다.
회귀 표현은 (x, y, w, h, theta) 를 쓴다.

사용법
------
  py verify_aihub_convention.py --geojson <라벨> --image <영상> --gsd 0.5
"""
import os
import json
import math
import argparse
import itertools

import numpy as np


# 설명서가 답하지 않는 자유도들. 곱집합으로 전부 시험한다.
ANGLE_MAPS = {
    "theta": lambda t: t,
    "-theta": lambda t: -t,
    "90-theta": lambda t: 90.0 - t,
    "theta+90": lambda t: t + 90.0,
}

# 좌표 필드 이름 후보 — AI Hub 배포본마다 다르다
COORD_KEYS = ["object_imcoords", "object_angle", "bbox", "coords",
              "obb", "rbox", "geometry"]


def corners(cx, cy, w, h, theta_deg, cw_positive=True):
    """중심·크기·각도 -> 네 꼭짓점. 영상 좌표계라 y 는 아래로 증가한다."""
    t = math.radians(theta_deg if cw_positive else -theta_deg)
    c, s = math.cos(t), math.sin(t)
    hw, hh = w / 2.0, h / 2.0
    pts = [(-hw, -hh), (hw, -hh), (hw, hh), (-hw, hh)]
    return np.array([[cx + x * c - y * s, cy + x * s + y * c] for x, y in pts],
                    np.float32)


def principal_angle(img, box, pad=2.0):
    """박스 주변 밝은 화소의 주축 각도(도, x 축 기준). 못 재면 None."""
    x0 = int(max(0, box[:, 0].min() - pad))
    x1 = int(min(img.shape[1], box[:, 0].max() + pad))
    y0 = int(max(0, box[:, 1].min() - pad))
    y1 = int(min(img.shape[0], box[:, 1].max() + pad))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    patch = img[y0:y1, x0:x1].astype(np.float64)
    if patch.ndim == 3:
        patch = patch.mean(2)
    ys, xs = np.nonzero(patch > patch.mean() + patch.std())
    if len(xs) < 4:
        return None
    P = np.stack([xs - xs.mean(), ys - ys.mean()])
    ev, evec = np.linalg.eigh(np.cov(P))
    v = evec[:, int(np.argmax(ev))]
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def box_angle(box):
    e0 = np.linalg.norm(box[1] - box[0])
    e1 = np.linalg.norm(box[2] - box[1])
    v = (box[1] - box[0]) if e0 >= e1 else (box[2] - box[1])
    return math.degrees(math.atan2(v[1], v[0])) % 180.0


def angdiff(a, b):
    d = abs(a - b) % 180.0
    return min(d, 180.0 - d)


def read_records(feats, limit):
    """GeoJSON 에서 [cx, cy, H, W, theta] 를 뽑는다. 키 이름은 배포본마다 다르다."""
    out = []
    for f in feats[:limit]:
        p = f.get("properties", f) if isinstance(f, dict) else f
        v = None
        for k in COORD_KEYS:
            if k in p:
                v = p[k]
                break
        if isinstance(v, str):
            v = [float(x) for x in v.replace(",", " ").split()]
        if isinstance(v, dict):                      # geometry 형태
            v = np.array(v.get("coordinates", []), object).ravel().tolist()
        if v is None or len(v) < 5:
            continue
        out.append([float(x) for x in v[:5]])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--geojson", required=True, help="AI Hub 라벨 파일")
    ap.add_argument("--image", required=True, help="같은 장면 영상")
    ap.add_argument("--gsd", type=float, default=0.5, help="원본 GSD (m)")
    ap.add_argument("--max-obj", type=int, default=200)
    ap.add_argument("--out", default="outputs/aihub_convention.json")
    ap.add_argument("--fig", default="outputs/aihub_convention.png")
    a = ap.parse_args()

    from PIL import Image
    img = np.array(Image.open(a.image))
    H_img, W_img = img.shape[:2]
    print(f"영상 {W_img} x {H_img}, 원본 GSD {a.gsd} m")

    gj = json.load(open(a.geojson, encoding="utf-8"))
    feats = gj["features"] if isinstance(gj, dict) and "features" in gj else gj
    print(f"객체 {len(feats)}개")

    raw = read_records(feats, a.max_obj)
    if not raw:
        first = feats[0].get("properties", feats[0])
        print("좌표 필드를 못 찾음. 실제 키 목록:")
        print(" ", list(first.keys()))
        return
    R = np.array(raw)

    # --- 1) 좌표가 픽셀인가 지도좌표인가 ---
    in_px = (R[:, 0].min() >= -10 and R[:, 0].max() <= W_img + 10
             and R[:, 1].min() >= -10 and R[:, 1].max() <= H_img + 10)
    print(f"\n[좌표계] x {R[:, 0].min():.1f}~{R[:, 0].max():.1f}, "
          f"y {R[:, 1].min():.1f}~{R[:, 1].max():.1f}")
    print(f"         -> {'픽셀 좌표' if in_px else '지도좌표 (affine 재투영 필요)'}")
    if not in_px:
        print("         지도좌표면 10 m 영상의 좌표계·affine 으로 네 꼭짓점을")
        print("         재투영해야 한다. 해상도 비율만 곱하면 틀린다.")

    # --- 2) 각도 단위와 범위 ---
    th = R[:, 4].copy()
    unit = "라디안" if np.abs(th).max() <= 2 * math.pi + 0.1 else "도"
    print(f"[각도]   {th.min():.3f}~{th.max():.3f} -> {unit}, "
          f"음수 {'있음 (-90~90 계열)' if th.min() < -0.01 else '없음 (0~180 계열)'}")
    if unit == "라디안":
        th = np.degrees(th)

    # --- 3) H, W 단위 ---
    print(f"[크기]   H {R[:, 2].min():.1f}~{R[:, 2].max():.1f}, "
          f"W {R[:, 3].min():.1f}~{R[:, 3].max():.1f}")
    print(f"         픽셀이면 실제 {R[:, 2:4].min() * a.gsd:.1f}~"
          f"{R[:, 2:4].max() * a.gsd:.1f} m")

    # --- 4) 후보 규약 전수 시험 ---
    print(f"\n{'규약':<30}{'중앙 오차(도)':>14}{'15도 이내':>11}")
    results = {}
    for (amap, af), swap, cw in itertools.product(
            ANGLE_MAPS.items(), [False, True], [True, False]):
        errs = []
        for r, t in zip(R, th):
            w, h = (r[2], r[3]) if swap else (r[3], r[2])
            b = corners(r[0], r[1], w, h, af(t), cw)
            pa = principal_angle(img, b)
            if pa is not None:
                errs.append(angdiff(box_angle(b), pa))
        if not errs:
            continue
        e = np.array(errs)
        key = f"{amap}{' +swap' if swap else ''}{' ccw' if not cw else ''}"
        results[key] = dict(median_err=float(np.median(e)),
                            frac_under15=float((e < 15).mean()), n=len(e))
        print(f"{key:<30}{np.median(e):>14.2f}{100 * (e < 15).mean():>10.1f}%")

    best = min(results.items(), key=lambda kv: kv[1]["median_err"])
    print(f"\n확정 후보: {best[0]}  "
          f"(중앙 오차 {best[1]['median_err']:.2f}도, "
          f"{100 * best[1]['frac_under15']:.1f}% 가 15도 이내)")
    if best[1]["median_err"] > 20:
        print("경고: 최선 후보도 오차가 크다. 그림을 눈으로 확인하고,")
        print("      그래도 안 맞으면 한국항공우주연구원에 좌표계·각도 정의·")
        print("      W/H 기준을 서면 문의해 확정할 것. 규약이 확정되기 전에는")
        print("      학습·평가 데이터를 만들지 않는다.")

    # --- 5) 눈으로 확인 ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.patches import Polygon
        ks = sorted(results, key=lambda k: results[k]["median_err"])[:6]
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        for ax, k in zip(axes.ravel(), ks):
            ax.imshow(img, cmap="gray" if img.ndim == 2 else None)
            amap = k.split()[0]
            swap, cw = "+swap" in k, "ccw" not in k
            for r, t in list(zip(R, th))[:60]:
                w, h = (r[2], r[3]) if swap else (r[3], r[2])
                b = corners(r[0], r[1], w, h, ANGLE_MAPS[amap](t), cw)
                ax.add_patch(Polygon(b, fill=False, ec="lime", lw=1.2))
            ax.set_title(f"{k}\n중앙 오차 {results[k]['median_err']:.1f}도",
                         fontsize=10)
            ax.axis("off")
        plt.tight_layout()
        os.makedirs(os.path.dirname(a.fig) or ".", exist_ok=True)
        plt.savefig(a.fig, dpi=110)
        print(f"그림 저장: {a.fig}  — 반드시 눈으로 확인할 것")
    except Exception as e:
        print(f"그림 실패: {type(e).__name__}: {e}")

    # --- 6) 규약이 확정되면 종횡비 통계 ---
    scale = a.gsd if in_px else 1.0
    Lm = np.maximum(R[:, 2], R[:, 3]) * scale
    Wm = np.minimum(R[:, 2], R[:, 3]) * scale
    ratio = Wm / np.maximum(Lm, 1e-6)
    print(f"\n[종횡비] 폭/길이 중앙값 {np.median(ratio):.3f} "
          f"(p10 {np.percentile(ratio, 10):.3f}, "
          f"p90 {np.percentile(ratio, 90):.3f})")
    print(f"[길이]   중앙값 {np.median(Lm):.1f} m "
          f"-> 10 m 영상에서 {np.median(Lm) / 10:.2f} px")
    print(f"[폭]     중앙값 {np.median(Wm):.1f} m "
          f"-> 10 m 영상에서 {np.median(Wm) / 10:.2f} px")
    if np.median(Wm) / 10 < 2.0:
        print("         폭이 2 px 미만이다. 이 크기에서는 회전 박스보다")
        print("         중심점 또는 점 주변 고정 크기 박스가 안정적이다.")
        print("         xView3-SAR 이 10 m 에서 점 기반 F1 을 쓴 이유와 같다.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(coords="pixel" if in_px else "map",
                   angle_unit=unit,
                   best_convention=best[0],
                   candidates=results,
                   aspect_ratio_median=float(np.median(ratio)),
                   length_m_median=float(np.median(Lm)),
                   width_m_median=float(np.median(Wm))),
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"저장: {a.out}")


if __name__ == "__main__":
    main()
