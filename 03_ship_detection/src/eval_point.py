"""점 기반 평가 — 코멘토 3차 업무 / 정승원

왜 IoU 대신 점인가
------------------
한국 근해 선박은 길이 중앙값 30 m = 10 m 해상도에서 3 픽셀이다.
3 픽셀짜리 물체에 IoU 0.75 를 요구하는 것은 부분 픽셀 정확도를 요구하는 것과 같다.

더 중요한 이유가 있다. GFW 정답은 애초에 박스가 아니다.
  점(lat, lon) + 길이(m) + 방향(도)
회전 박스는 내가 그걸로 합성한 것이다. 그러면 IoU 오차에 두 가지가 섞인다.

  IoU 오차 = 모델의 위치 오차 + 내가 만든 박스의 오차

3 픽셀 물체에서는 뒷항이 앞항만큼 크다. 즉 mAP50-95 는 모델이 아니라
내 박스 합성을 재고 있다. 점 기반 평가는 그 교란을 통째로 제거한다.

무엇을 재는가
------------
예측 박스의 중심과 정답 점 사이 거리가 허용 반경 안이면 맞은 것으로 본다.
허용 반경은 고정값(픽셀)과 선박 길이 비례 중 큰 쪽을 쓴다.
  - 고정값: 아주 작은 배까지 최소한의 여유를 준다
  - 길이 비례: 큰 배는 중심이 좀 더 떨어져도 같은 배다

정답 하나에 예측 하나만 붙인다(헝가리안 대신 신뢰도 순 탐욕 매칭).
탐지 과제에서는 순위가 곧 신뢰도이므로 이게 관례에 맞다.
"""
import os, csv, glob, math, json, argparse
import numpy as np


def greedy_match(preds, gts, radius_fn):
    """신뢰도 높은 예측부터 가장 가까운 미사용 정답에 붙인다."""
    used = set()
    out = []                     # (신뢰도, 맞았나, 거리, 정답 길이)
    for cx, cy, cf in sorted(preds, key=lambda p: -p[2]):
        best, bi = 1e9, -1
        for i, (gx, gy, gl) in enumerate(gts):
            if i in used:
                continue
            d = math.hypot(cx - gx, cy - gy)
            if d < best:
                best, bi = d, i
        if bi >= 0 and best <= radius_fn(gts[bi][2]):
            used.add(bi)
            out.append((cf, 1, best, gts[bi][2]))
        else:
            out.append((cf, 0, best if bi >= 0 else float('nan'), float('nan')))
    for i, g in enumerate(gts):
        if i not in used:
            out.append((-1.0, 0, float('nan'), g[2]))     # 미탐
    return out


def ap_from(matched, scores, n_gt):
    if n_gt == 0:
        return float('nan'), float('nan')
    if len(scores) == 0:
        return 0.0, 0.0
    o = np.argsort(-np.asarray(scores))
    tp = np.asarray(matched, float)[o]
    ctp, cfp = np.cumsum(tp), np.cumsum(1 - tp)
    rec, prec = ctp / n_gt, ctp / np.maximum(ctp + cfp, 1e-9)
    mrec = np.concatenate([[0.], rec, [1.]])
    mpre = np.concatenate([[1.], prec, [0.]])
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    x = np.linspace(0, 1, 101)
    trap = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    return float(trap(np.interp(x, mrec, mpre), x)), float(rec[-1])


def load_gt_points(root, split, tile):
    """라벨 파일의 회전 박스에서 중심점과 장축을 뽑는다."""
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
                e = [math.dist(p[i], p[(i + 1) % 4]) for i in range(4)]
                items.append((float(p[:, 0].mean()), float(p[:, 1].mean()),
                              float(max(e[0], e[1]))))
        out[stem] = items
    return out


BINS = [(0, 3), (3, 5), (5, 8), (8, 15), (15, 30), (30, 200)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--weights', required=True)
    ap.add_argument('--root', default='C:/Users/seung/datasets/S2Ships/yolo')
    ap.add_argument('--split', default='test')
    ap.add_argument('--imgsz', type=int, default=320)
    ap.add_argument('--conf', type=float, default=0.001)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--radius-px', type=float, default=4.0,
                    help='허용 반경 하한 (픽셀)')
    ap.add_argument('--radius-frac', type=float, default=0.5,
                    help='선박 장축의 이 비율도 허용 반경 후보')
    ap.add_argument('--out', default='outputs/point_eval.json')
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)
    gt = load_gt_points(a.root, a.split, a.imgsz)
    n_gt = sum(len(v) for v in gt.values())
    print(f'{a.split}: 타일 {len(gt)}장, GT {n_gt}개')
    print(f'허용 반경 = max({a.radius_px} px, 장축 x {a.radius_frac})')

    radius = lambda L: max(a.radius_px, a.radius_frac * L)
    recs = []
    for stem, items in gt.items():
        r = model.predict(f'{a.root}/images/{a.split}/{stem}.png', imgsz=a.imgsz,
                          conf=a.conf, verbose=False, device=a.device)[0]
        preds = []
        if r.obb is not None and len(r.obb) > 0:
            for poly, cf in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy()):
                p = poly.reshape(4, 2)
                preds.append((float(p[:, 0].mean()), float(p[:, 1].mean()), float(cf)))
        recs += greedy_match(preds, items, radius)

    det = [r for r in recs if r[0] >= 0]
    A, R = ap_from([r[1] for r in det], [r[0] for r in det], n_gt)
    hit = [r[2] for r in recs if r[1] == 1]
    print()
    print(f'점 기반 AP  {A:.3f}   Recall {R:.3f}')
    print(f'맞은 탐지의 중심 오차: 중앙 {np.median(hit):.2f} px  '
          f'p90 {np.percentile(hit, 90):.2f} px  (= {np.median(hit)*10:.0f} m)')

    print()
    print('%-12s %6s %8s %8s %11s' % ('장축 구간', 'GT', 'AP', 'Recall', '중심오차px'))
    table = []
    for b in BINS:
        sel = [r for r in recs if r[3] == r[3] and b[0] <= r[3] < b[1]]
        n = len(sel)
        if n == 0:
            continue
        d = [r for r in sel if r[0] >= 0]
        # 오탐은 정답 길이가 없으므로 예측 크기를 모른다 -> 전체 오탐을 공통 분모로 두지 않고
        # 구간 재현율과 중심오차만 본다 (AP 는 전체값을 대표로 쓴다)
        rec = sum(r[1] for r in sel) / n
        er = [r[2] for r in sel if r[1] == 1]
        print('%-12s %6d %8s %8.3f %11.2f'
              % (f'{b[0]}-{b[1]} px', n, '-', rec, np.median(er) if er else float('nan')))
        table.append(dict(bin=f'{b[0]}-{b[1]}', n_gt=n, recall=rec,
                          center_err_px=float(np.median(er)) if er else None))

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    json.dump(dict(split=a.split, weights=os.path.basename(a.weights), n_gt=n_gt,
                   radius_px=a.radius_px, radius_frac=a.radius_frac,
                   ap=A, recall=R,
                   center_err_median_px=float(np.median(hit)),
                   center_err_p90_px=float(np.percentile(hit, 90)),
                   bins=table),
              open(a.out, 'w', encoding='utf-8'), indent=2, ensure_ascii=False)
    print(f'\n저장: {a.out}')


if __name__ == '__main__':
    main()
