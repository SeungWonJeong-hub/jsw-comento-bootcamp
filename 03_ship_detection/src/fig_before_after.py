"""원본과 탐지 결과 대비 그림 — 코멘토 3차 업무 / 정승원

PPT 3장에 들어갑니다. 왼쪽은 받은 그대로의 위성사진, 오른쪽은 같은 사진에
모델이 찾은 선박을 얹은 것입니다. 수치 표만 있으면 "정말 배를 찾는가" 를
확인할 수 없으므로, 눈으로 볼 수 있는 증거를 함께 냅니다.

왜 잘라내나
-----------
320 px 타일을 통째로 넣으면 배가 3~5 px 이라 화면에서 보이지 않습니다. 처음에
그렇게 만들었더니 노란 박스만 보이고 그 안에 무엇이 있는지 알 수 없었습니다.
그래서 선박이 몰린 자리를 작게 잘라 크게 확대합니다. 발트해는 물과 숲이 모두
어두워 대비도 함께 늘립니다.

고르는 기준
-----------
평가셋(34WFT, 학습에 안 쓴 해역)에서 고릅니다. 같은 장면이 반복되지 않도록
촬영일이 겹치면 한 장만 씁니다.

사용법
------
  py fig_before_after.py --weights weights/yolo11s_dota.pt
"""
import os
import glob
import argparse

import numpy as np
import cv2


def load_gt(label_path, tile):
    if not os.path.exists(label_path):
        return []
    out = []
    for line in open(label_path):
        p = line.split()
        if len(p) == 9:
            out.append(np.array([float(v) for v in p[1:]],
                                np.float32).reshape(4, 2) * tile)
    return out


def stretch(img, lo=1.0, hi=99.5):
    """백분위 대비 늘리기. 발트해는 물도 숲도 어두워 그냥은 안 보입니다."""
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        a, b = np.percentile(ch, lo), np.percentile(ch, hi)
        if b <= a:
            a, b = ch.min(), max(ch.max(), ch.min() + 1)
        out[:, :, c] = np.clip((ch - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)
    return out


def best_crop(gt, tile, size):
    """선박이 가장 많이 들어오는 정사각 창의 좌상단. 없으면 None."""
    if not gt:
        return None, 0
    cs = np.array([[p[:, 0].mean(), p[:, 1].mean()] for p in gt])
    best, bn = None, 0
    for cx, cy in cs:                       # 각 선박을 중심으로 후보 창을 놓습니다
        x = int(np.clip(cx - size / 2, 0, tile - size))
        y = int(np.clip(cy - size / 2, 0, tile - size))
        n = int(((cs[:, 0] >= x) & (cs[:, 0] < x + size)
                 & (cs[:, 1] >= y) & (cs[:, 1] < y + size)).sum())
        if n > bn:
            best, bn = (x, y), n
    return best, bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo11s_dota.pt")
    ap.add_argument("--data", default="C:/Users/seung/datasets/S2Ships/yolo")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=0.25)
    # DOTA 는 OBB 실험에 NMS 0.1 을 씁니다. 기본값 0.7 은 같은 배를 겹쳐 남깁니다.
    ap.add_argument("--iou", type=float, default=0.2)
    ap.add_argument("--grid", type=int, default=2)
    ap.add_argument("--tile", type=int, default=320)
    ap.add_argument("--crop", type=int, default=104, help="잘라낼 한 변 (px)")
    ap.add_argument("--scale", type=int, default=6, help="확대 배율")
    ap.add_argument("--gap", type=int, default=10)
    ap.add_argument("--out", default="outputs/fig6_before_after")
    a = ap.parse_args()

    from ultralytics import YOLO
    model = YOLO(a.weights)

    imgs = sorted(glob.glob(os.path.join(a.data, "images", a.split, "*.png")))
    print(f"{a.split} 타일 {len(imgs)}장")

    cands = []
    for ip in imgs:
        lp = ip.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        gt = load_gt(lp, a.tile)
        xy, n = best_crop(gt, a.tile, a.crop)
        if n >= 2:
            cands.append((n, ip, xy))
    cands.sort(key=lambda t: -t[0])

    # 같은 촬영일이 반복되면 그림이 단조로워집니다. 날짜당 최대 두 장까지만.
    # 한 장으로 제한하면 평가셋에 날짜가 셋뿐이라 격자가 빕니다.
    need = a.grid * a.grid
    picked, cnt = [], {}
    for cap in (1, 2, 99):
        for n, ip, xy in cands:
            if len(picked) == need:
                break
            day = os.path.basename(ip).rsplit("_", 1)[0]
            if cnt.get(day, 0) >= cap or (n, ip, xy) in picked:
                continue
            cnt[day] = cnt.get(day, 0) + 1
            picked.append((n, ip, xy))
        if len(picked) == need:
            break
    print(f"후보 {len(cands)}장 -> 사용 {len(picked)}장 "
          f"(창당 선박 {[n for n, _, _ in picked]})")

    S = a.crop * a.scale
    cellw = S + a.gap
    side = cellw * a.grid - a.gap
    left = np.full((side, side, 3), 250, np.uint8)
    right = left.copy()

    tg = td = 0
    for i, (n, ip, (x0, y0)) in enumerate(picked):
        img = cv2.imread(ip)
        lp = ip.replace("images", "labels").rsplit(".", 1)[0] + ".txt"
        gt = load_gt(lp, a.tile)
        r = model.predict(img, imgsz=a.tile, conf=a.conf, iou=a.iou, verbose=False)[0]
        dets = ([] if r.obb is None or len(r.obb) == 0
                else [p.reshape(4, 2) for p in r.obb.xyxyxyxy.cpu().numpy()])

        sub = stretch(img[y0:y0 + a.crop, x0:x0 + a.crop])
        # 최근접으로 늘립니다. 위성사진은 화소 하나가 10 m 이므로, 보간해서
        # 매끄럽게 만들면 없는 정보를 지어내면서 오히려 흐려 보입니다.
        big = cv2.resize(sub, (S, S), interpolation=cv2.INTER_NEAREST)
        vis = big.copy()
        kept = 0
        for poly in dets:
            q = (poly - [x0, y0]) * a.scale
            cx, cy = q[:, 0].mean(), q[:, 1].mean()
            if not (0 <= cx < S and 0 <= cy < S):
                continue
            kept += 1
            cv2.polylines(vis, [q.astype(np.int32)], True, (0, 212, 255),
                          3, cv2.LINE_AA)
        tg += n
        td += kept

        y, x = (i // a.grid) * cellw, (i % a.grid) * cellw
        left[y:y + S, x:x + S] = big
        right[y:y + S, x:x + S] = vis

    print(f"잘라낸 창 안 — 정답 {tg}척 · 탐지 {td}척 (conf {a.conf})")
    print(f"창 한 변 {a.crop} px = {a.crop * 10 / 1000:.2f} km")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    cv2.imwrite(f"{a.out}_raw.png", left)
    cv2.imwrite(f"{a.out}_det.png", right)
    print(f"저장: {a.out}_raw.png / {a.out}_det.png  ({side}x{side})")


if __name__ == "__main__":
    main()
