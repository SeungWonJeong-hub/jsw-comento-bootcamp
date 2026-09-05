# -*- coding: utf-8 -*-
"""항만별 precision / recall / F1 / AP50 — HR 모델(seed0), 공식 test, conf 0.25, IoU 0.5.
웹앱 사이드바가 '그 해역에서 잰 값' 을 보여주기 위한 표입니다. 추측값 없음."""
import os, json, collections, numpy as np, cv2
from common import ROOT, PORTS, load_cfg, rel, read_csv, dump_json
from evaluate_yolo import match, ap_from
from ultralytics import YOLO
cfg = load_cfg(); m = YOLO(os.path.join(ROOT, "weights", "hrsc_hr045_seed0.pt"))
man = {r["image_id"]: r for r in read_csv(os.path.join(rel(cfg, "manifests"), "images.csv"))}
test = collections.defaultdict(list)
for r in read_csv(os.path.join(rel(cfg, "manifests"), "split.csv")):
    if r["official"] == "test" and r["port"] in PORTS and r["port"] != "murmansk": test[r["port"]].append(r["image_id"])
out = {}
for port, ids in test.items():
    tps, scs, n_gt, tp25, det25 = [], [], 0, 0, 0
    for i in ids:
        p = os.path.join(rel(cfg, "images"), i + ".bmp"); v = cv2.imread(p); H, W = v.shape[:2]
        gts = [dict(poly=np.array([float(x) for x in l.split()[1:]]).reshape(4, 2)) for l in open(os.path.join(rel(cfg, "data"), "labels_obb", i + ".txt"), encoding="utf-8") if len(l.split()) == 9]
        r = m.predict(p, imgsz=640, conf=0.001, iou=0.7, verbose=False, device="cpu")[0]
        dets = []
        if r.obb is not None and len(r.obb):
            for q, c in zip(r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2) / [W, H], r.obb.conf.cpu().numpy()): dets.append(dict(poly=q, score=float(c)))
        tp, fp, _ = match(gts, dets, 0.5); tps += list(tp); scs += [d["score"] for d in dets]; n_gt += len(gts)
        keep = [k for k, d in enumerate(dets) if d["score"] >= 0.25]; det25 += len(keep); tp25 += int(sum(tp[k] for k in keep))
    ap50 = ap_from(tps, scs, n_gt)[0]; P = tp25 / max(det25, 1); R = tp25 / max(n_gt, 1)
    out[port] = dict(precision=round(P, 3), recall=round(R, 3), f1=round(2 * P * R / max(P + R, 1e-9), 3), AP50=round(ap50, 3), n_images=len(ids), n_ships=n_gt, TP=tp25, FP=det25 - tp25, FN=n_gt - tp25)
    print("%-10s img %3d ships %4d  P %.3f R %.3f F1 %.3f AP50 %.3f" % (port, len(ids), n_gt, P, R, out[port]["f1"], ap50))
dump_json(os.path.join(ROOT, "results", "port_metrics_seed0.json"), out)
