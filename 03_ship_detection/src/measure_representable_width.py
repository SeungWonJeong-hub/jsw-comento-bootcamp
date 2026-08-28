"""10 m 영상에서 선박의 폭이 표현 가능한가 — 코멘토 3차 업무 / 정승원

물음
----
회전 박스를 쓰려면 장변과 단변이 둘 다 있어야 한다. 10 m 해상도에서
한국 선박의 단변(폭)이 몇 픽셀인지 재면, 박스 규약 논쟁이 끝난다.

왜 이 물음이 규약 논쟁을 끝내나
--------------------------------
GFW 는 폭을 주지 않는다. 길이와 방향만 준다. 그래서 나는 폭을
width_ratio 로 만들어 냈고, 그 값이 정답 노릇을 했다. 그런데 어떤 비율을
고르는 것이 옳은가를 따지기 전에, 애초에 폭이 화소로 표현되는지를
먼저 물어야 했다. 표현되지 않으면 어떤 비율도 옳지 않다.

실제 선박의 폭/길이는 대략 0.12~0.25 범위다. 상선일수록 작고 어선일수록
크다. 이 범위 전체를 넣어 보고, 결과가 범위와 무관하게 같은 결론이면
비율을 고를 필요 자체가 사라진다.

사용법
------
  py measure_representable_width.py --gfw C:/Users/seung/datasets/GFW/korea.csv
"""
import os
import csv
import json
import argparse

import numpy as np

# 실제 선박의 폭/길이 범위. 상선 쪽이 작고 어선 쪽이 크다.
RATIOS = (0.12, 0.15, 0.20, 0.25)


def load_lengths(path, min_presence=0.8):
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if str(r.get("likely_infrastructure", "")).lower() == "true":
                continue
            if str(r.get("potential_ice", "")).lower() == "true":
                continue
            try:
                if float(r["presence_score"]) < min_presence:
                    continue
                v = float(r["length_m_inferred"])
            except Exception:
                continue
            if v > 0:
                out.append(v)
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfw", default="C:/Users/seung/datasets/GFW/korea.csv")
    ap.add_argument("--gsd", type=float, default=10.0)
    ap.add_argument("--out", default="outputs/representable_width.json")
    a = ap.parse_args()

    L = load_lengths(a.gfw)
    if not len(L):
        print("탐지를 못 읽음:", a.gfw)
        return

    print(f"GFW 탐지 {len(L):,}건 ({os.path.basename(a.gfw)})")
    print(f"길이 (m)   중앙 {np.median(L):.1f}  "
          f"p10 {np.percentile(L, 10):.1f}  p90 {np.percentile(L, 90):.1f}")
    print(f"길이 (px)  중앙 {np.median(L) / a.gsd:.2f}  "
          f"({a.gsd:.0f} m 해상도)")

    print(f"\n폭이 몇 픽셀이 되나 — GFW 는 폭을 주지 않으므로 비율을 가정한다")
    print(f"{'W/L':<8}{'폭 중앙(px)':>13}{'2px 미만':>11}{'1px 미만':>11}")
    rows = {}
    for r in RATIOS:
        W = L * r / a.gsd
        rows[f"{r:.2f}"] = dict(w_median_px=float(np.median(W)),
                                frac_under2=float((W < 2).mean()),
                                frac_under1=float((W < 1).mean()))
        print(f"{r:<8.2f}{np.median(W):>13.2f}"
              f"{100 * (W < 2).mean():>10.1f}%{100 * (W < 1).mean():>10.1f}%")

    worst = max(RATIOS)
    frac1 = float((L * worst / a.gsd < 1).mean())
    print(f"\n판정")
    print(f"  가장 관대한 비율 {worst:.2f} 로 잡아도 "
          f"{100 * frac1:.1f}% 가 폭 1 픽셀 미만이다.")
    print(f"  선박의 폭은 이 해상도에 물리적으로 존재하지 않는다.")
    print(f"  따라서 회전 박스의 단변은 어떤 규약을 골라도 허구다.")
    print(f"  측정 가능한 것은 중심점, 길이, 방향 셋뿐이다.")
    print(f"  평가는 점 기반으로 한다. IoU 는 없는 축을 재게 된다.")
    print(f"  (xView3-SAR 이 10 m SAR 에서 점 기반 F1 을 쓴 이유와 같다)")
    print(f"\n  더 높은 해상도의 정답을 구해도 이 결론은 바뀌지 않는다.")
    print(f"  0.5 m 라벨에서 실측 폭을 얻어도 10 m 로 내리면 다시 서브픽셀이다.")
    print(f"  고해상도 정답의 값어치는 박스를 가능하게 하는 것이 아니라")
    print(f"  이 숫자를 확인해 주는 데 있다.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(source=os.path.basename(a.gfw), n=int(len(L)), gsd_m=a.gsd,
                   length_m_median=float(np.median(L)),
                   length_px_median=float(np.median(L) / a.gsd),
                   by_ratio=rows),
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
