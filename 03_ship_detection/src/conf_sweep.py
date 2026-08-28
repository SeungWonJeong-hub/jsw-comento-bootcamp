"""신뢰도 임계값과 물 게이트 — 코멘토 3차 업무 / 정승원

왜
--
한국 항만 결과를 눈으로 보니 항적이 뚜렷한 배가 박스 없이 지나갑니다.
특히 인천이 심합니다. 모델이 못 보는 것인지, 보고도 점수가 낮아 잘린
것인지를 가릅니다. 후자면 임계값 한 줄로 해결됩니다.

임계값만 낮추면 갯벌에 오탐이 붙습니다. 밝고 얼룩덜룩해서 배처럼 보입니다.
그래서 낮춘 다음 물 위에 있는 것만 남깁니다. 두 조치는 짝입니다.

  임계값을 낮춘다   -> 놓친 배를 되찾는다 (재현율)
  물 게이트를 건다  -> 새로 생긴 육지 오탐을 버린다 (정밀도)

같은 장면을 한 번만 받아 임계값만 바꿔가며 돌립니다. 장면이 달라지면
비교가 안 되기 때문입니다.

사용법
------
  py conf_sweep.py --weights weights/yolo11s_dota.pt --port incheon
"""
import os
import json
import argparse

import numpy as np
import cv2


def draw(rgb, dets, color=(0, 255, 255)):
    v = rgb.copy()
    for poly, _ in dets:
        cv2.polylines(v, [poly.astype(np.int32)], True, color, 1, cv2.LINE_AA)
    return v


def water_gate(gray, dets, on_water):
    """탐지 중 주변이 물처럼 보이는 것만 남깁니다. 판정 불가는 살려 둡니다."""
    keep, drop = [], []
    for poly, cf in dets:
        cx, cy = float(poly[:, 0].mean()), float(poly[:, 1].mean())
        v = on_water(gray, cx, cy)
        (keep if v is not False else drop).append((poly, cf))
    return keep, drop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--port", default="incheon")
    ap.add_argument("--confs", nargs="+", type=float,
                    default=[0.25, 0.15, 0.10, 0.05, 0.03])
    ap.add_argument("--max-cloud", type=float, default=10)
    ap.add_argument("--outdir", default="outputs/conf_sweep")
    a = ap.parse_args()

    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from korea_ports import PORTS, find_scene, fetch_window, detect, on_water
    from ultralytics import YOLO

    label, *bbox = PORTS[a.port]
    print(f"{label} — 장면 찾는 중...")
    sc = find_scene(bbox, a.max_cloud)
    if not sc:
        print("장면 없음")
        return
    print(f"장면 {sc['id']}  {sc['datetime'][:10]}")

    img, affine, crs = fetch_window(sc["visual"], bbox)
    print(f"창 {img.shape[1]}x{img.shape[0]} px\n")
    gray = cv2.cvtColor(img[:, :, :3].astype(np.uint8), cv2.COLOR_RGB2GRAY)

    model = YOLO(a.weights)
    os.makedirs(a.outdir, exist_ok=True)

    print(f"{'conf':>7}{'원본':>7}{'물 위':>7}{'버림':>7}{'유지율':>9}")
    rows = {}
    for c in sorted(a.confs, reverse=True):
        dets = detect(img, model, conf=c)
        keep, drop = water_gate(gray, dets, on_water)
        frac = len(keep) / len(dets) if dets else 0.0
        print(f"{c:>7.2f}{len(dets):>7}{len(keep):>7}{len(drop):>7}"
              f"{100 * frac:>8.1f}%")
        rows[f"{c:.2f}"] = dict(raw=len(dets), water=len(keep),
                                dropped=len(drop), keep_frac=frac)

        bgr = img[:, :, ::-1]
        cv2.imwrite(os.path.join(a.outdir, f"{a.port}_conf{int(c*100):03d}.jpg"),
                    draw(bgr, keep), [cv2.IMWRITE_JPEG_QUALITY, 90])
        if drop:
            # 버린 것을 빨강으로 겹쳐 그려, 게이트가 무엇을 지웠는지 봅니다
            v = draw(bgr, keep)
            v = draw(v, drop, (0, 0, 255))
            cv2.imwrite(os.path.join(a.outdir,
                                     f"{a.port}_conf{int(c*100):03d}_gate.jpg"),
                        v, [cv2.IMWRITE_JPEG_QUALITY, 90])

    js = os.path.join(a.outdir, f"{a.port}.json")
    json.dump(dict(port=a.port, label=label, scene=sc["id"],
                   datetime=sc["datetime"], sweep=rows),
              open(js, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {a.outdir}/{a.port}_conf*.jpg (노랑=유지, 빨강=게이트가 버림)")
    print(f"      {js}")


if __name__ == "__main__":
    main()
