# -*- coding: utf-8 -*-
"""항만 5곳 x 1장 — HR 모델(0.45 m)의 탐지를 정답과 겹쳐 한 줄로."""
import os, csv, io, collections, numpy as np, cv2
from common import ROOT, PORTS, load_cfg, rel, read_csv
from ultralytics import YOLO
cfg = load_cfg(); W_ = os.path.join(ROOT, "weights", "hrsc_hr045_seed0.pt")
man = {r["image_id"]: r for r in read_csv(os.path.join(rel(cfg, "manifests"), "images.csv"))}
by = collections.defaultdict(list)
for r in read_csv(os.path.join(rel(cfg, "manifests"), "split.csv")):
    if r["official"] == "test" and r["port"] in PORTS and r["port"] != "murmansk":
        by[r["port"]].append(r["image_id"])
m = YOLO(W_); rows = []; PH = 420
for port in ["san_diego", "norfolk", "mayport", "everett", "newport"]:
    ids = sorted(by[port], key=lambda i: -int(man[i]["n_ships"] or 0))
    i = ids[min(2, len(ids) - 1)]
    p = os.path.join(rel(cfg, "images"), i + ".bmp"); v = cv2.imread(p); H, W = v.shape[:2]
    gt = [np.array([float(x) for x in l.split()[1:]]).reshape(4, 2) * [W, H]
          for l in open(os.path.join(rel(cfg, "data"), "labels_obb", i + ".txt"), encoding="utf-8") if len(l.split()) == 9]
    r = m.predict(p, imgsz=640, conf=0.25, iou=0.7, verbose=False, device="cpu")[0]
    pr = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2) if r.obb is not None and len(r.obb) else []
    for g in gt: cv2.polylines(v, [np.int32(g)], True, (0, 230, 255), 2, cv2.LINE_AA)
    for q in pr: cv2.polylines(v, [np.int32(q)], True, (60, 200, 90), 3, cv2.LINE_AA)
    s = PH / H; v = cv2.resize(v, (int(W * s), PH))
    cv2.rectangle(v, (0, 0), (v.shape[1], 30), (255, 255, 255), -1)
    cv2.putText(v, "%s   GT %d / pred %d" % (PORTS[port]["name"], len(gt), len(pr)), (8, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (20, 20, 20), 2, cv2.LINE_AA)
    rows.append(np.pad(v, ((0, 0), (0, 8), (0, 0)), constant_values=255)); print(port, i, len(gt), len(pr))
out = os.path.join(ROOT, "results", "ports_hr_detections.png"); cv2.imwrite(out, np.hstack(rows)); print("saved", out)
