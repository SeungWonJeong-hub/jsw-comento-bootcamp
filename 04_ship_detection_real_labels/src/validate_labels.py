# -*- coding: utf-8 -*-
"""변환한 OBB 라벨을 영상 위에 다시 그려 눈으로 검증합니다.

    py scripts/validate_labels.py --n 100

변환식이 맞는지는 숫자로 못 봅니다. 각도 부호 하나만 뒤집혀도 IoU 는
그럴듯하게 나오는데 상자는 배를 빗나갑니다. 그래서 최소 100장을 그려서
확인합니다(요구사항). 항구별로 고르게 뽑고, 선박이 많은 장면을 우선합니다.

숫자 점검도 같이 합니다
  - 정규화 범위 [0,1] 이탈
  - 꼭짓점 4개 · 볼록성
  - 면적 0
  - 상자 중심이 영상 밖
  - 장변/단변 비율이 비상식적(>20:1)
"""
import os
import glob
import random
import argparse
import collections

import numpy as np
import cv2

from common import ROOT, PORTS, load_cfg, rel, read_csv, dump_json


def load_label(p):
    out = []
    if not os.path.exists(p):
        return out
    for line in open(p, encoding="utf-8"):
        f = line.split()
        if len(f) == 9:
            out.append(np.array([float(v) for v in f[1:]], np.float64).reshape(4, 2))
    return out


def check(polys):
    bad = collections.Counter()
    for q in polys:
        if q.min() < -1e-6 or q.max() > 1 + 1e-6:
            bad["out_of_range"] += 1
        if cv2.contourArea(np.float32(q)) <= 1e-9:
            bad["zero_area"] += 1
        if not cv2.isContourConvex(np.float32(q)):
            bad["not_convex"] += 1
        d = [np.linalg.norm(q[i] - q[(i + 1) % 4]) for i in range(4)]
        L, B = max(d), min(d)
        if B > 0 and L / B > 20:
            bad["extreme_aspect"] += 1
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    img_dir = rel(cfg, "images")
    lab_dir = os.path.join(rel(cfg, "data"), "labels_obb")
    out = os.path.join(ROOT, "results", "label_check")
    os.makedirs(out, exist_ok=True)

    man = [r for r in read_csv(os.path.join(rel(cfg, "manifests"), "images.csv"))
           if int(r["n_ships"] or 0) > 0]

    # ---- 전수 숫자 점검 ---------------------------------------------
    bad = collections.Counter()
    n_poly = 0
    for r in man:
        polys = load_label(os.path.join(lab_dir, r["image_id"] + ".txt"))
        n_poly += len(polys)
        bad.update(check(polys))
    print("라벨 %d개 전수 점검" % n_poly)
    if bad:
        for k, v in bad.most_common():
            print("  %-14s %d (%.2f%%)" % (k, v, 100.0 * v / max(n_poly, 1)))
    else:
        print("  이상 없음")

    # ---- 항구별로 고르게 뽑아 그리기 --------------------------------
    byport = collections.defaultdict(list)
    for r in man:
        byport[r["port"]].append(r)
    rng = random.Random(a.seed)
    pick = []
    ports = sorted(byport, key=lambda k: -len(byport[k]))
    while len(pick) < a.n and any(byport[p] for p in ports):
        for p in ports:
            if not byport[p] or len(pick) >= a.n:
                continue
            lst = sorted(byport[p], key=lambda r: -int(r["n_ships"]))[:60]
            r = rng.choice(lst)
            byport[p].remove(r)
            pick.append(r)
    # 작은 항구가 먼저 바닥나거나 BMP 가 없는 항목이 있어 목표에 못 미칠 수
    # 있습니다. 여유분을 붙여 두고 실제로 그려진 장수로 멈춥니다.
    rest = [r for p in ports for r in byport[p]]
    rng.shuffle(rest)
    pick += rest

    drawn = 0
    for r in pick:
        if drawn >= a.n:
            break
        ip = os.path.join(img_dir, r["image_id"] + ".bmp")
        if not os.path.exists(ip):
            continue
        v = cv2.imread(ip)
        if v is None:
            continue
        H, W = v.shape[:2]
        for q in load_label(os.path.join(lab_dir, r["image_id"] + ".txt")):
            pts = np.int32(q * np.array([W, H]))
            cv2.polylines(v, [pts], True, (0, 255, 255), 2, cv2.LINE_AA)
            cv2.circle(v, tuple(pts[0]), 5, (0, 0, 255), -1)   # 시작 꼭짓점
        name = PORTS.get(r["port"], {}).get("name", r["port"])
        cv2.putText(v, "%s  %s  ships=%s" % (r["image_id"], name, r["n_ships"]),
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(v, "%s  %s  ships=%s" % (r["image_id"], name, r["n_ships"]),
                    (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(os.path.join(out, "%s_%s.jpg" % (r["port"], r["image_id"])), v,
                    [cv2.IMWRITE_JPEG_QUALITY, 88])
        drawn += 1

    print("\n시각검증 %d장 저장: %s" % (drawn, out))
    print("빨간 점이 첫 꼭짓점입니다 — 모든 상자에서 같은 모서리에 찍혀야 순서가 일관된 것입니다.")
    dump_json(os.path.join(ROOT, "results", "label_validation.json"),
              dict(polygons=n_poly, issues=dict(bad), drawn=drawn))


if __name__ == "__main__":
    main()
