"""라벨의 박스 규약을 실측한다 — 코멘토 3차 업무 / 정승원

왜 만들었나
-----------
YOLO-OBB 는 여덟 개 좌표로 배포된다. 그래서 나는 이 데이터가 회전 박스라고
믿고 학습·평가를 짰다. 그런데 좌표를 직접 찍어 보니 그렇지 않았다.

  핀란드 S2Ships    13,069개 중 100.00% 가 축 정렬 (회전이 전혀 없음)
  GFW 합성 라벨     16,929개 중  0.20% 만 축 정렬 (0~90도 균일 분포)

학습은 회전이 없는 라벨로, 평가는 회전이 균일한 정답으로 했다.
IoU 가 0 에 가까웠던 것은 모델이 못해서가 아니라 규약이 달라서다.
형식(8좌표)과 규약(실제 각도 분포)은 다르다. 믿지 말고 재야 한다.

무엇을 재나
-----------
  1. 축 정렬 비율 — 회전 박스인 척하는 HBB 를 잡아낸다
  2. 장변 각도 분포 — {0, 90} 에 몰려 있으면 회전을 못 배운다
  3. 크기 구간별 종횡비 — 커질수록 얇아지면 라벨러의 붓 자국이지 배 모양이 아니다

3번이 왜 판별이 되나
--------------------
실제 선박의 폭/길이는 0.1~0.25 이고 크기와 무관하다. 파나막스든 어선이든
비율은 비슷하다. 그런데 측정값이 크기에 따라 단조 감소하면, 작은 배를
뭉툭하게 칠했다는 뜻이다. 즉 물리가 아니라 주석 습관이다.

사용법
------
  py measure_box_convention.py --labels <YOLO 라벨 폴더> --tile 320
"""
import os
import glob
import math
import json
import argparse

import numpy as np

BINS = [(0, 3), (3, 5), (5, 8), (8, 15), (15, 30), (30, 1e9)]


def read_labels(root, tile):
    """YOLO-OBB 라벨을 읽어 (장변, 단변, 종횡비, 축정렬여부, 장변각도) 로."""
    rows = []
    splits = [d for d in ("train", "val", "test")
              if os.path.isdir(os.path.join(root, d))] or [""]
    for split in splits:
        for fp in glob.glob(os.path.join(root, split, "*.txt")):
            for line in open(fp):
                p = line.split()
                if len(p) != 9:
                    continue
                q = np.array([float(v) for v in p[1:]], np.float64).reshape(4, 2) * tile
                e = [np.linalg.norm(q[(i + 1) % 4] - q[i]) for i in range(4)]
                L, W = max(e[0], e[1]), min(e[0], e[1])
                if L <= 0:
                    continue
                # 네 변이 모두 축과 나란하면 축 정렬이다
                aa = all(abs(q[(i + 1) % 4][0] - q[i][0]) < 1e-6
                         or abs(q[(i + 1) % 4][1] - q[i][1]) < 1e-6
                         for i in range(4))
                li = 0 if e[0] >= e[1] else 1
                v = q[(li + 1) % 4] - q[li]
                ang = abs(math.degrees(math.atan2(v[1], v[0]))) % 180.0
                rows.append((L, W, W / L, aa, min(ang, 180.0 - ang)))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", required=True, help="YOLO 라벨 최상위 폴더")
    ap.add_argument("--tile", type=float, default=320.0)
    ap.add_argument("--name", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    rows = read_labels(a.labels, a.tile)
    if not rows:
        print("라벨을 못 찾음:", a.labels)
        return
    name = a.name or os.path.basename(os.path.dirname(a.labels.rstrip("/\\")))

    n = len(rows)
    L = np.array([r[0] for r in rows])
    W = np.array([r[1] for r in rows])
    ar = np.array([r[2] for r in rows])
    ang = np.array([r[4] for r in rows])
    aa = sum(1 for r in rows if r[3])
    near_axis = float(np.mean((ang < 1) | (ang > 89)))

    print(f"=== {name} ===")
    print(f"인스턴스        {n}")
    print(f"축 정렬         {aa} ({100 * aa / n:.2f}%)")
    print(f"축에서 1도 이내 {100 * near_axis:.2f}%")
    print(f"종횡비 W/L      중앙 {np.median(ar):.3f}  "
          f"p10 {np.percentile(ar, 10):.3f}  p90 {np.percentile(ar, 90):.3f}")
    print(f"장변 L (px)     중앙 {np.median(L):.2f}  "
          f"p10 {np.percentile(L, 10):.2f}  p90 {np.percentile(L, 90):.2f}")

    print(f"\n장변 각도 분포 (x 축 기준 0~90도)")
    h, _ = np.histogram(ang, bins=9, range=(0, 90))
    for i, c in enumerate(h):
        print(f"  {i * 10:>2}-{i * 10 + 10:>2}  {c:>6}  {'#' * int(50 * c / max(h.max(), 1))}")

    print(f"\n크기 구간별 종횡비 — 붓 자국 판별")
    print(f"{'L (px)':<12}{'개수':>8}{'W/L 중앙':>11}{'W 중앙(px)':>13}")
    strata = {}
    for lo, hi in BINS:
        m = (L >= lo) & (L < hi)
        if not m.any():
            continue
        lab = f"{lo}-{hi:.0f}" if hi < 1e8 else f"{lo}+"
        strata[lab] = dict(n=int(m.sum()), ar_median=float(np.median(ar[m])),
                           w_median_px=float(np.median(W[m])))
        print(f"{lab:<12}{m.sum():>8}{np.median(ar[m]):>11.3f}{np.median(W[m]):>13.2f}")

    keys = list(strata)
    if len(keys) >= 3:
        first, last = strata[keys[0]]["ar_median"], strata[keys[-1]]["ar_median"]
        if first - last > 0.15:
            print(f"\n판정: 종횡비가 {first:.3f} -> {last:.3f} 로 단조 감소한다.")
            print("      실제 선박 비율은 크기와 무관하므로, 이것은 배의 모양이")
            print("      아니라 주석 습관이다. 작은 배를 뭉툭하게 칠한 결과다.")

    if near_axis > 0.9:
        print(f"\n판정: {100 * near_axis:.1f}% 가 축에 붙어 있다. 8좌표 OBB 형식이지만")
        print("      실질은 HBB 다. 이 라벨로 학습한 모델은 회전을 배우지 못한다.")
        print("      회전된 정답과 IoU 로 비교하면 원리적으로 무너진다.")

    out = a.out or f"outputs/box_convention_{name}.json"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    json.dump(dict(name=name, n=n, axis_aligned_frac=aa / n,
                   near_axis_frac=near_axis,
                   ar_median=float(np.median(ar)),
                   L_median_px=float(np.median(L)),
                   W_median_px=float(np.median(W)),
                   strata=strata),
              open(out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
