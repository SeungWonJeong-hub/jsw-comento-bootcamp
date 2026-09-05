# -*- coding: utf-8 -*-
"""HRSC2016 XML -> YOLO OBB 라벨.

    py scripts/convert_hrsc_to_yolo_obb.py

출력 형식 (Ultralytics OBB)
    class_id x1 y1 x2 y2 x3 y3 x4 y4      모두 0~1 정규화

왜 정규화 좌표가 비교군 전부에 그대로 쓰이는가
----------------------------------------------
정규화 좌표는 **균일 리사이즈에 불변**입니다. HR(1174x834) -> LR(53x38) ->
bicubic x4(212x152) -> Real-ESRGAN x4(212x152) 어디서든 배의 상대 위치는
같습니다. 그래서 라벨 파일 하나를 모든 비교군이 공유하고, "SR 영상의
예측을 원좌표계로 역변환" 하는 문제가 애초에 생기지 않습니다.

역변환이 필요한 경우는 (a) 타일 추론, (b) 비대칭 letterbox 인데, 이
프로젝트는 타일을 쓰지 않고 Ultralytics 가 letterbox 를 스스로 되돌려
원영상 화소좌표로 예측을 돌려줍니다. evaluate_yolo.py 가 그 좌표를 영상
크기로 나눠 정규화 공간에서 정답과 맞댑니다 — 즉 모든 비교군이 같은
좌표계에서 채점됩니다.

각도 규약
---------
XML 은 `mbox_cx, mbox_cy, mbox_w, mbox_h, mbox_ang` 이고 ang 은 **라디안**
입니다. cv2.boxPoints 는 도(degree)를 받으므로 변환합니다. 이 변환이 맞는지는
라벨을 영상 위에 다시 그려 확인했습니다(validate_labels.py).

꼭짓점 순서
-----------
boxPoints 의 출력 순서는 각도에 따라 회전하므로, **가장 왼쪽-위 점에서
시작해 시계방향**으로 고정합니다. 순서가 들쭉날쭉하면 OBB 손실이 불안정해집니다.
"""
import os
import glob
import math
import argparse
import collections
import xml.etree.ElementTree as ET

import numpy as np
import cv2

from common import ROOT, load_cfg, rel, read_csv, dump_json


def _f(node, tag):
    e = node.find(tag)
    return float(e.text) if e is not None and e.text else 0.0


def order_corners(pts):
    """왼쪽-위에서 시작하는 시계방향으로 정렬합니다."""
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    idx = np.argsort(ang)              # 반시계
    pts = pts[idx][::-1]               # 시계
    start = int(np.argmin(pts[:, 0] + pts[:, 1]))
    return np.roll(pts, -start, axis=0)


def convert(xml_path, W, H, clip=True):
    """XML 한 장 -> (라벨 줄 목록, 통계)"""
    root = ET.parse(xml_path).getroot()
    lines, stat = [], collections.Counter()
    for o in root.findall("./HRSC_Objects/HRSC_Object"):
        cx, cy = _f(o, "mbox_cx"), _f(o, "mbox_cy")
        w, h = _f(o, "mbox_w"), _f(o, "mbox_h")
        ang = _f(o, "mbox_ang")                    # 라디안
        if w <= 1 or h <= 1:
            stat["degenerate"] += 1
            continue
        pts = cv2.boxPoints(((cx, cy), (w, h), math.degrees(ang)))
        pts = order_corners(np.asarray(pts, np.float64))
        out = pts[:, 0].min() < -1 or pts[:, 1].min() < -1 or \
            pts[:, 0].max() > W + 1 or pts[:, 1].max() > H + 1
        if out:
            stat["out_of_bounds"] += 1
        if clip:
            pts[:, 0] = np.clip(pts[:, 0], 0, W)
            pts[:, 1] = np.clip(pts[:, 1], 0, H)
        n = pts / np.array([W, H], np.float64)
        if not np.isfinite(n).all():
            stat["nan"] += 1
            continue
        area = cv2.contourArea(np.float32(n))
        if area <= 1e-8:
            stat["zero_area"] += 1
            continue
        lines.append("0 " + " ".join("%.6f" % v for v in n.reshape(-1)))
        stat["ok"] += 1
    return lines, stat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--no-clip", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    ann_dir = rel(cfg, "annotations")
    out = a.out or os.path.join(rel(cfg, "data"), "labels_obb")
    os.makedirs(out, exist_ok=True)

    man = {r["image_id"]: r for r in
           read_csv(os.path.join(rel(cfg, "manifests"), "images.csv"))}

    total = collections.Counter()
    n_img = 0
    for p in sorted(glob.glob(os.path.join(ann_dir, "*.xml"))):
        stem = os.path.splitext(os.path.basename(p))[0]
        m = man.get(stem)
        if not m or not m["width"]:
            continue
        W, H = int(m["width"]), int(m["height"])
        lines, st = convert(p, W, H, clip=not a.no_clip)
        with open(os.path.join(out, stem + ".txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        total.update(st)
        n_img += 1

    print("영상 %d장 · 라벨 %d개" % (n_img, total["ok"]))
    for k in ("out_of_bounds", "degenerate", "zero_area", "nan"):
        if total[k]:
            print("  %-14s %d" % (k, total[k]))
    print("  (out_of_bounds 는 %s)" %
          ("영상 경계로 잘라냈습니다" if not a.no_clip else "그대로 두었습니다"))
    dump_json(os.path.join(ROOT, "results", "label_conversion.json"),
              dict(images=n_img, **dict(total)))
    print("저장:", out)


if __name__ == "__main__":
    main()
