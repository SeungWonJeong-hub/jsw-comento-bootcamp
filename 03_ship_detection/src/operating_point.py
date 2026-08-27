"""운용 임계값 결정 — 코멘토 3차 업무 / 정승원

왜 필요한가
-----------
층화 평가에서 오탐이 test 2,353건 / val 6,082건 나왔다. 인스턴스가 각각 366, 349개인데
오탐이 6~17배다. 그런데 이 값은 신뢰도 하한을 0.001 로 둔 것이다.
PR 곡선을 끝까지 그리려면 낮게 둬야 해서 그런 것이지, 운용 설정이 아니다.

손실함수를 고치거나 재학습하기 전에 먼저 확인할 것 — 그 오탐이 정말 문제인가,
아니면 신뢰도가 낮아서 임계값 하나로 사라지는가.

무엇을 내는가
-----------
  신뢰도별 Precision / Recall / F1 / 오탐 수
  F1 최대점과, "오탐을 GT 수 이하로 누르는" 임계값
  각각에서의 성능

재학습이 필요 없다. 이미 있는 가중치로 예측 한 번만 하면 된다.
"""
import os, sys, glob, math, json, argparse
import numpy as np
import cv2


def poly_iou(a, b):
    ia, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    if ia <= 0:
        return 0.0
    ua = (cv2.contourArea(a.astype(np.float32))
          + cv2.contourArea(b.astype(np.float32)) - ia)
    return ia / ua if ua > 0 else 0.0


def load_gt(root, split, tile):
    out = {}
    for f in sorted(glob.glob(f'{root}/labels/{split}/*.txt')):
        stem = os.path.basename(f)[:-4]
        items = []
        if os.path.getsize(f):
            for ln in open(f):
                t = ln.split()
                if len(t) < 9:
                    continue
                items.append(np.array([[float(t[i]) * tile, float(t[i + 1]) * tile]
                                       for i in range(1, 9, 2)], np.float32))
        out[stem] = items
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--root', default='C:/Users/seung/datasets/S2Ships/yolo')
    ap.add_argument('--split', default='test')
    ap.add_argument('--iou', type=float, default=0.5)
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='outputs/operating_point.json')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    gt = load_gt(a.root, a.split, a.imgsz)
    n_gt = sum(len(v) for v in gt.values())
    print(f'{a.split}: 타일 {len(gt)}장, GT {n_gt}개')

    # 신뢰도를 아주 낮게 두고 한 번만 예측한 뒤, 임계값은 나중에 걸러서 쓴다
    recs = []                       # (신뢰도, 정탐인가)
    for stem, items in gt.items():
        r = model.predict(f'{a.root}/images/{a.split}/{stem}.png', imgsz=a.imgsz,
                          conf=0.001, verbose=False, device=a.device)[0]
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
                recs.append((cf, 1))
            else:
                recs.append((cf, 0))

    recs.sort(key=lambda r: -r[0])
    conf = np.array([r[0] for r in recs])
    tp = np.array([r[1] for r in recs], float)

    print()
    print('%8s %8s %10s %10s %8s %8s' % ('신뢰도', '탐지수', 'Precision', 'Recall', 'F1', '오탐'))
    rows = []
    for t in [0.001, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]:
        m = conf >= t
        n_det = int(m.sum())
        n_tp = float(tp[m].sum())
        p = n_tp / n_det if n_det else 0.0
        rec = n_tp / n_gt if n_gt else 0.0
        f1 = 2 * p * rec / (p + rec) if (p + rec) else 0.0
        rows.append(dict(conf=t, n_det=n_det, precision=p, recall=rec, f1=f1,
                         n_fp=n_det - int(n_tp)))
        print('%8.3f %8d %10.3f %10.3f %8.3f %8d'
              % (t, n_det, p, rec, f1, n_det - int(n_tp)))

    best = max(rows, key=lambda r: r['f1'])
    print(f"\nF1 최대: 신뢰도 {best['conf']:.2f}  "
          f"P {best['precision']:.3f}  R {best['recall']:.3f}  F1 {best['f1']:.3f}  "
          f"오탐 {best['n_fp']}건")

    # 오탐을 GT 수 이하로 누르는 가장 낮은 임계값 (재현율을 최대한 살리는 지점)
    ok = [r for r in rows if r['n_fp'] <= n_gt]
    if ok:
        q = min(ok, key=lambda r: r['conf'])
        print(f"오탐 <= GT({n_gt}) 지점: 신뢰도 {q['conf']:.2f}  "
              f"P {q['precision']:.3f}  R {q['recall']:.3f}  오탐 {q['n_fp']}건")
    else:
        q = None
        print(f"오탐을 GT({n_gt}) 이하로 누르는 임계값이 표 안에 없음")

    base = rows[0]
    print(f"\n임계값 0.001 -> {best['conf']:.2f} 로 올리면 "
          f"오탐 {base['n_fp']} -> {best['n_fp']}건 "
          f"({100*(1-best['n_fp']/max(base['n_fp'],1)):.1f}% 감소), "
          f"재현율 {base['recall']:.3f} -> {best['recall']:.3f}")

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(dict(split=a.split, weights=os.path.basename(a.weights), n_gt=n_gt,
                   iou=a.iou, curve=rows, best_f1=best, fp_under_gt=q),
              open(a.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'저장: {a.out}')


if __name__ == '__main__':
    main()
