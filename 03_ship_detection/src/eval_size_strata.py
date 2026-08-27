"""선박 크기별 층화 평가 — 코멘토 3차 업무 / 정승원

전체 mAP 하나가 가리는 것을 크기 축으로 펼친다.

왜 크기인가
-----------
10 m 해상도에서 선박은 2~70 px 다. 중앙값이 5.2 px 이니 대부분이 손톱만 하다.
그런데 mAP 는 큰 배와 작은 배를 한데 섞어 평균 낸다.
크기가 곧 "이 해상도에서 볼 수 있느냐"를 가르는 축이므로 그 축으로 나눠 본다.

무엇을 재는가
-----------
  AP50      IoU 0.5 로 맞췄을 때 찾았는가
  Recall    놓치지 않았는가
  평균 IoU  찾은 것이 얼마나 딱 맞는가  ← AP 만으로는 안 보이는 축

학습 결과에서 mAP50(0.86)과 mAP50-95(0.46)의 격차가 매우 컸다.
IoU 문턱을 올리면 급락한다는 뜻이고, 박스가 헐겁게 맞고 있다는 신호다.
그 헐거움이 크기에 따라 어떻게 달라지는지가 이 평가의 초점이다.
"""
import os, sys, glob, math, json, argparse
import numpy as np
import cv2

# 픽셀 구간과 대략의 실제 길이 (10 m/px)
BINS = [(0, 3), (3, 5), (5, 8), (8, 15), (15, 30), (30, 200)]


def obb_long(p):
    e = [math.dist(p[i], p[(i + 1) % 4]) for i in range(4)]
    return max(e[0], e[1])


def poly_iou(a, b):
    ia, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    if ia <= 0:
        return 0.0
    ua = (cv2.contourArea(a.astype(np.float32))
          + cv2.contourArea(b.astype(np.float32)) - ia)
    return ia / ua if ua > 0 else 0.0


def ap_from(matched, scores, n_gt):
    if n_gt == 0:
        return float('nan'), float('nan')          # 그 구간에 정답이 없음
    if len(scores) == 0:
        return 0.0, 0.0                            # 정답은 있는데 하나도 못 찾음
    o = np.argsort(-np.asarray(scores))
    tp = np.asarray(matched, dtype=float)[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec = ctp / n_gt
    prec = ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.], rec, [1.]])
    mpre = np.concatenate([[1.], prec, [0.]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(trap(np.interp(x, mrec, mpre), x)), float(rec[-1])


def load_gt(root, split, tile=320):
    out = {}
    for f in sorted(glob.glob(f'{root}/labels/{split}/*.txt')):
        stem = os.path.basename(f)[:-4]
        items = []
        if os.path.getsize(f):
            for ln in open(f):
                t = ln.split()
                if len(t) < 9:
                    continue
                p = np.array([[float(t[i]) * tile, float(t[i + 1]) * tile]
                              for i in range(1, 9, 2)], np.float32)
                items.append(p)
        out[stem] = items
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--root', default='C:/Users/seung/datasets/S2Ships/yolo')
    ap.add_argument('--split', default='test')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='outputs/size_strata.json')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)

    gt = load_gt(a.root, a.split, a.imgsz)
    n_inst = sum(len(v) for v in gt.values())
    print(f'{a.split}: 타일 {len(gt)}장, 인스턴스 {n_inst}개')

    recs = []                    # (장축, 맞췄나, 신뢰도, IoU)
    n_gt_bin = {b: 0 for b in BINS}
    for stem, items in gt.items():
        for p in items:
            L = obb_long(p)
            for b in BINS:
                if b[0] <= L < b[1]:
                    n_gt_bin[b] += 1

        img = f'{a.root}/images/{a.split}/{stem}.png'
        r = model.predict(img, imgsz=a.imgsz, conf=a.conf, verbose=False, device=a.device)[0]
        preds = []
        if r.obb is not None and len(r.obb) > 0:
            for poly, cf in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy()):
                preds.append((poly.reshape(4, 2), float(cf)))

        used = set()
        for poly, cf in sorted(preds, key=lambda z: -z[1]):
            best, bi = 0.0, -1
            for i, g in enumerate(items):
                if i in used:
                    continue
                v = poly_iou(poly, g)
                if v > best:
                    best, bi = v, i
            if best >= a.iou and bi >= 0:
                used.add(bi)
                recs.append((obb_long(items[bi]), 1, cf, best))
            else:
                # 오탐은 정답이 없으니 GT 크기를 못 쓴다. 대신 '예측한 박스의 크기'로
                # 구간에 넣는다. 4px 짜리 오탐은 소형선 구간의 오탐이다.
                # 모든 구간에 공통으로 더하면, GT 가 적은 대형선 구간이 오탐 수천 건에
                # 짓눌려 AP 가 구간끼리 비교 불가능해진다.
                recs.append((obb_long(poly), 0, cf, np.nan))
        for i, g in enumerate(items):
            if i not in used:
                recs.append((obb_long(g), 0, -1.0, np.nan))  # 미탐

    n_fp = sum(1 for r in recs if r[1] == 0 and r[2] >= 0)
    print()
    print('%-12s %8s %6s %6s %8s %8s %9s'
          % ('장축 구간', '실제 길이', 'GT', '오탐', 'AP50', 'Recall', '평균 IoU'))
    table = []
    for b in BINS:
        sel = [r for r in recs if b[0] <= r[0] < b[1]]
        det = [r for r in sel if r[2] >= 0]          # 탐지된 것(정탐+오탐)
        A, R = ap_from([r[1] for r in det], [r[2] for r in det], n_gt_bin[b])
        ious = [r[3] for r in sel if r[1] == 1 and r[3] == r[3]]
        mIoU = float(np.mean(ious)) if ious else float('nan')
        fp = sum(1 for r in det if r[1] == 0)
        rng = f'{b[0]*10}-{b[1]*10} m' if b[1] < 200 else f'{b[0]*10} m+'
        print('%-12s %8s %6d %6d %8.3f %8.3f %9.3f'
              % (f'{b[0]}-{b[1]} px', rng, n_gt_bin[b], fp, A, R, mIoU))
        table.append(dict(bin=f'{b[0]}-{b[1]}', metres=rng, n_gt=n_gt_bin[b],
                          n_fp=fp, ap50=A, recall=R, mean_iou=mIoU))

    print(f'\n오탐 합계 {n_fp}건 (신뢰도 하한 {a.conf} — PR 곡선을 그리려면 낮게 둔다)')
    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(dict(split=a.split, iou=a.iou, weights=os.path.basename(a.weights),
                   n_instances=n_inst, n_fp=n_fp, bins=table),
              open(a.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'저장: {a.out}')


if __name__ == '__main__':
    main()
