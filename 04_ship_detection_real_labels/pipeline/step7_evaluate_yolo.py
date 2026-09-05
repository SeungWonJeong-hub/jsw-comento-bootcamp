# -*- coding: utf-8 -*-
"""OBB 탐지 평가 — 요구된 지표를 전부 직접 계산합니다.

    py pipeline/evaluate_yolo.py --split official

Ultralytics 의 val() 대신 자체 평가기를 씁니다. TP/FP/FN, PR curve,
최대 F1 임계값, 고정 임계값, 항구별·크기별·장면유형별, FP/km2, 환각 감사를
한 좌표계에서 일관되게 내야 하기 때문입니다.

좌표계
------
예측은 Ultralytics 가 letterbox 를 되돌려 **원영상 화소좌표**로 줍니다.
영상 크기로 나눠 정규화하면 비교군(LR·bicubic·SR)마다 영상 크기가 달라도
같은 공간에서 맞댈 수 있습니다. 정답도 같은 정규화 좌표입니다.
타일 추론을 쓰지 않으므로 타일 오프셋·오버랩 역변환은 없습니다.

IoU
---
회전 사각형이라 cv2.intersectConvexConvex 로 다각형 교집합을 씁니다.

세 종류의 P/R (요구사항)
------------------------
    1. 전체 PR curve
    2. F1 최대 지점의 P/R (비교군마다 다른 임계값)
    3. 고정 임계값 conf_fixed 의 P/R (모든 비교군 공통)
2번은 각 방법의 최선을, 3번은 운용 조건이 같을 때의 실제 차이를 봅니다.
"""
import os
import json
import time
import argparse
import collections

import numpy as np
import cv2

from common import ROOT, PORTS, load_cfg, rel, read_csv, dump_json, results_dir, smoke_cap


# ------------------------------------------------------------------ 기하
def poly_iou(a, b):
    ia, _ = cv2.intersectConvexConvex(np.float32(a), np.float32(b))
    if ia <= 0:
        return 0.0
    aa = cv2.contourArea(np.float32(a))
    ab = cv2.contourArea(np.float32(b))
    u = aa + ab - ia
    return float(ia / u) if u > 0 else 0.0


def long_side_m(poly, gw, gh):
    """정규화 다각형의 장변을 미터로."""
    p = poly * np.array([gw, gh])
    d = [np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)]
    return max(d)


# ------------------------------------------------------------------ 매칭
def match(gts, dets, iou_thr):
    """점수 내림차순 그리디 매칭. 반환: (tp flags, fp flags, 매칭된 gt 인덱스)"""
    order = np.argsort([-d["score"] for d in dets]) if dets else []
    used = set()
    tp = np.zeros(len(dets), bool)
    gidx = -np.ones(len(dets), int)
    for k in order:
        best, bi = iou_thr, -1
        for j, g in enumerate(gts):
            if j in used:
                continue
            v = poly_iou(dets[k]["poly"], g["poly"])
            if v >= best:
                best, bi = v, j
        if bi >= 0:
            used.add(bi)
            tp[k] = True
            gidx[k] = bi
    return tp, ~tp, gidx


def ap_from(tp, scores, n_gt):
    """101점 보간 AP 와 PR curve."""
    if n_gt == 0:
        return float("nan"), np.zeros(0), np.zeros(0), np.zeros(0)
    if len(scores) == 0:
        return 0.0, np.zeros(0), np.zeros(0), np.zeros(0)
    o = np.argsort(-np.asarray(scores))
    tp = np.asarray(tp)[o].astype(float)
    sc = np.asarray(scores)[o]
    ctp = np.cumsum(tp)
    cfp = np.cumsum(1 - tp)
    rec = ctp / n_gt
    pre = ctp / np.maximum(ctp + cfp, 1e-9)
    mpre = np.concatenate([[1.0], pre, [0.0]])
    mrec = np.concatenate([[0.0], rec, [1.0]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    ap = float(np.mean(np.interp(x, mrec, mpre)))
    return ap, pre, rec, sc


# ------------------------------------------------------------------ 평가
def evaluate(gt_by_img, det_by_img, cfg, meta, iou_thr=0.5):
    """한 (비교군, 시드) 결과를 채점합니다."""
    all_tp, all_sc, n_gt = [], [], 0
    per = collections.defaultdict(lambda: dict(tp=[], sc=[], n=0))
    hit_size, tot_size = collections.Counter(), collections.Counter()
    bins = cfg["eval"]["size_bins_m"]
    for img, gts in gt_by_img.items():
        dets = det_by_img.get(img, [])
        tp, fp, gidx = match(gts, dets, iou_thr)
        all_tp += list(tp)
        all_sc += [d["score"] for d in dets]
        n_gt += len(gts)
        m = meta[img]
        for key in (("port", m["port"]), ("scene", m["scene_type"])):
            d = per[key]
            d["tp"] += list(tp)
            d["sc"] += [x["score"] for x in dets]
            d["n"] += len(gts)
        # 크기별 재현율 — 고정 임계값 기준
        conf = cfg["eval"]["conf_fixed"]
        matched = set(gidx[[i for i, d in enumerate(dets)
                            if tp[i] and d["score"] >= conf]]) if dets else set()
        for j, g in enumerate(gts):
            L = long_side_m(g["poly"], m["gw"], m["gh"])
            b = int(np.digitize(L, bins) - 1)
            b = min(max(b, 0), len(bins) - 2)
            tot_size[b] += 1
            if j in matched:
                hit_size[b] += 1

    ap50, pre, rec, sc = ap_from(all_tp, all_sc, n_gt)
    out = {"n_gt": n_gt, "n_det": len(all_sc), "AP50": ap50}

    # AP75, mAP50-95
    aps = {}
    for t in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]:
        tps, scs = [], []
        for img, gts in gt_by_img.items():
            dets = det_by_img.get(img, [])
            tp, _, _ = match(gts, dets, t)
            tps += list(tp)
            scs += [d["score"] for d in dets]
        aps[t] = ap_from(tps, scs, n_gt)[0]
    out["AP75"] = aps[0.75]
    out["mAP50_95"] = float(np.nanmean(list(aps.values())))

    # 1) PR curve  2) 최대 F1 3) 고정 임계값
    if len(sc):
        f1 = 2 * pre * rec / np.maximum(pre + rec, 1e-9)
        i = int(np.nanargmax(f1))
        out["bestF1"] = dict(conf=float(sc[i]), P=float(pre[i]),
                             R=float(rec[i]), F1=float(f1[i]))
        c = cfg["eval"]["conf_fixed"]
        k = int(np.searchsorted(-sc, -c, side="right")) - 1
        if k >= 0:
            tp_n = float(np.cumsum(np.asarray(all_tp)[np.argsort(-np.asarray(all_sc))])[k])
            fp_n = (k + 1) - tp_n
            P = tp_n / max(k + 1, 1)
            R = tp_n / max(n_gt, 1)
            out["fixed"] = dict(conf=c, P=P, R=R,
                                F1=2 * P * R / max(P + R, 1e-9),
                                TP=int(tp_n), FP=int(fp_n),
                                FN=int(n_gt - tp_n))
        else:
            out["fixed"] = dict(conf=c, P=0.0, R=0.0, F1=0.0, TP=0,
                                FP=0, FN=n_gt)
        out["pr_curve"] = dict(precision=[round(v, 5) for v in pre[::max(1, len(pre)//300)]],
                               recall=[round(v, 5) for v in rec[::max(1, len(rec)//300)]],
                               score=[round(float(v), 5) for v in sc[::max(1, len(sc)//300)]])
    out["by_size"] = {"%g-%g" % (bins[b], bins[b + 1]):
                      dict(gt=tot_size[b],
                           recall=(hit_size[b] / tot_size[b]) if tot_size[b] else None)
                      for b in range(len(bins) - 1)}
    out["by_group"] = {}
    for (kind, val), d in per.items():
        a, _, _, _ = ap_from(d["tp"], d["sc"], d["n"])
        out["by_group"]["%s:%s" % (kind, val)] = dict(AP50=a, n_gt=d["n"])
    return out


def load_gt(cfg, split_id, subset="test"):
    lab = os.path.join(rel(cfg, "data"), "labels_obb")
    man = {r["image_id"]: r for r in
           read_csv(os.path.join(rel(cfg, "manifests"), "images.csv"))}
    rows = read_csv(os.path.join(rel(cfg, "manifests"), "split.csv"))
    gt, meta = {}, {}
    cap = smoke_cap()
    for r in rows:
        if r[split_id] != subset:
            continue
        if cap and len(gt) >= cap[subset]:
            break
        stem = r["image_id"]
        p = os.path.join(lab, stem + ".txt")
        if not os.path.exists(p):
            continue
        polys = []
        for line in open(p, encoding="utf-8"):
            f = line.split()
            if len(f) == 9:
                polys.append(dict(poly=np.array([float(v) for v in f[1:]]).reshape(4, 2)))
        gt[stem] = polys
        m = man[stem]
        meta[stem] = dict(port=m["port"], scene_type=m["scene_type"] or "unknown",
                          gw=float(m["ground_w_m"] or 0),
                          gh=float(m["ground_h_m"] or 0))
    return gt, meta


def predict(weights, img_dir, stems, imgsz, nms_iou, device=None):
    from ultralytics import YOLO
    model = YOLO(weights)
    out, ms = {}, []
    for s in stems:
        p = os.path.join(img_dir, s + ".png")
        if not os.path.exists(p):
            p = os.path.join(img_dir, s + ".bmp")       # 원본은 BMP
        if not os.path.exists(p):
            out[s] = []
            continue
        t = time.time()
        r = model.predict(p, imgsz=imgsz, conf=0.001, iou=nms_iou,
                          verbose=False, device=device)[0]
        ms.append(1000 * (time.time() - t))
        H, W = r.orig_shape
        dets = []
        if r.obb is not None and len(r.obb):
            xy = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
            cf = r.obb.conf.cpu().numpy()
            for q, c in zip(xy, cf):
                dets.append(dict(poly=q / np.array([W, H]), score=float(c)))
        out[s] = dets
    return out, (float(np.median(ms)) if ms else float("nan"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="official")
    ap.add_argument("--device", default=None)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if a.smoke:
        os.environ["HRSC_SMOKE"] = "1"
    cfg = load_cfg(a.config)
    state_p = os.path.join(results_dir(), "train_state_%s.json" % a.split)
    if not os.path.exists(state_p):
        raise SystemExit("학습 결과가 없습니다: %s" % state_p)
    state = json.load(open(state_p, encoding="utf-8"))
    gt, meta = load_gt(cfg, a.split, "test")
    stems = sorted(gt)
    print("test %d장 · 정답 %d척" % (len(stems), sum(len(v) for v in gt.values())))

    data = rel(cfg, "data")
    arms = [x["id"] for x in cfg.get("arms", [])]      # 저해상도 비교군 (4차 본 실험엔 없음)
    results = []
    for key, run in state["done"].items():
        src, seed = run["source"], run["seed"]
        targets = [src] if src != "hr" else ["hr"] + arms
        for tgt in targets:
            cond = "matched" if src != "hr" else \
                ("upper_bound" if tgt == "hr" else "hr_trained")
            img_dir = rel(cfg, "images") if tgt == "hr" else os.path.join(data, "images_" + tgt)
            det, ms = predict(run["weights"], img_dir, stems,
                              cfg["train"]["imgsz"], cfg["eval"]["nms_iou"],
                              a.device)
            m = evaluate(gt, det, cfg, meta)
            m.update(condition=cond, train_source=src, eval_arm=tgt, seed=seed,
                     split=a.split, ms_per_image=ms, epochs=run["epochs"],
                     model=run["model"])
            results.append(m)
            f = m.get("fixed", {})
            print("  %-10s %-14s seed%d  AP50 %.3f  AP75 %.3f  mAP %.3f  "
                  "F1@fix %.3f  %.0f ms"
                  % (cond, tgt, seed, m["AP50"], m["AP75"], m["mAP50_95"],
                     f.get("F1", float("nan")), ms))
    out = os.path.join(results_dir(), "eval_%s.json" % a.split)
    dump_json(out, results)
    print("\n저장: %s  (%d행)" % (out, len(results)))


if __name__ == "__main__":
    main()
