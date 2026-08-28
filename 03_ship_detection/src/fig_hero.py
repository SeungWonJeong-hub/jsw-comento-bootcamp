"""탐지 결과 대표 그림 — 코멘토 3차 업무 / 정승원

PPT 3장의 주인공입니다. 평가 해역(34WFT)의 장면 원본을 다시 받아, 선박이 가장
많이 몰린 6.4 x 3.6 km 를 한 장으로 그립니다.

배율을 어디에 맞추는가
----------------------
배는 장변이 5 화소입니다. 화면에서 형체(길쭉한지, 어느 쪽을 보는지)가 읽히려면
30 화소쯤은 되어야 하니 배율은 6배가 필요하고, 그러면 10 m 화소 하나가 발표
화면에서 0.34 mm 가 됩니다. 이 둘은 5 대 1 로 묶여 있어 따로 정할 수 없습니다.
배의 형체를 보려면 그 배를 이루는 화소도 함께 보인다는 뜻입니다.

앞서 320 px 타일 한 장(3.2 km)을 화면 폭에 맞춰 20배 넘게 늘렸을 때는 화소가
2 mm 가 되어 모자이크로 보였습니다. 6배에서는 최근접 대신 Lanczos 로 늘려
화소 경계를 부드럽게 두므로, 사진처럼 읽히면서 배의 꼬리 물결까지 남습니다.

창을 고르는 기준
----------------
학습에 쓰지 않은 해역에서만 고릅니다. 주석 폴리곤의 중심을 장면 좌표로 옮긴
뒤, 창을 밀어 가며 배가 가장 많이 담기는 자리를 씁니다.

사용법
------
  py fig_hero.py --weights weights/yolo11s_dota.pt
"""
import os
import json
import argparse

import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

CYAN = (228, 228, 38)          # BGR
FONTDIR = os.path.expandvars(r"%LOCALAPPDATA%\Microsoft\Windows\Fonts")


def font(name, size):
    return ImageFont.truetype(os.path.join(FONTDIR, f"Pretendard-{name}.ttf"), size)


def stretch(img, lo=0.5, hi=98.4):
    """백분위 대비 늘리기입니다.

    창을 넓게 잡으면 물·뭍·마을이 한 화면에 다 들어오므로, 백분위 양끝만 잘라
    펴는 것으로 충분합니다. 배를 억지로 띄우는 곡선은 쓰지 않습니다 — 배가
    어디 있는지는 청록 박스가 알려 주고, 사진은 사진대로 두는 편이 발표
    화면에서 읽기 좋습니다.
    """
    out = np.empty_like(img)
    for c in range(img.shape[2]):
        ch = img[:, :, c].astype(np.float32)
        a, b = np.percentile(ch, lo), np.percentile(ch, hi)
        if b <= a:
            a, b = ch.min(), max(ch.max(), ch.min() + 1)
        out[:, :, c] = np.clip((ch - a) * 255.0 / (b - a), 0, 255).astype(np.uint8)
    return out


def tone(img, desat=0.86, gamma=0.82):
    """채도를 죽여 청록 박스가 화면에서 유일한 색으로 남게 합니다."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    f = img.astype(np.float32) * (1 - desat) + g[:, :, None] * desat
    f = 255.0 * np.power(np.clip(f, 0, 255) / 255.0, gamma)
    return f.astype(np.uint8)


def rrect(dr, box, r, fill=None, outline=None, width=1):
    dr.rounded_rectangle(box, radius=r, fill=fill, outline=outline, width=width)


def load_centers(gpkg, date, crs, affine):
    """주석 폴리곤의 중심을 장면 픽셀 좌표로 옮깁니다."""
    import geopandas as gpd
    g = gpd.read_file(gpkg, layer=date).to_crs(crs)
    inv = ~affine
    return np.array([inv * (p.x, p.y) for p in g.geometry.centroid], np.float64)


def ship_contrast(img, centers):
    """배가 주변 물보다 몇 DN 밝은지의 중앙값입니다.

    같은 해역이라도 날에 따라 해 높이와 물빛이 달라, 이 값이 작으면 배가
    화면에서 물과 붙어 보입니다. 어느 날 장면을 쓸지 이 값으로 고릅니다.
    """
    L = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = L.shape
    out = []
    for c, r in centers:
        c, r = int(c), int(r)
        if not (10 < c < w - 10 and 10 < r < h - 10):
            continue
        ring = L[r - 9:r + 10, c - 9:c + 10]
        out.append(L[r - 1:r + 2, c - 1:c + 2].max()
                   - np.median(ring[ring < np.percentile(ring, 60)]))
    return float(np.median(out)) if out else 0.0


def best_window(cs, ww, hh, lim):
    """배가 가장 많이 담기는 창의 좌상단입니다.

    척수가 같으면 무리의 무게중심이 창 한가운데 오는 자리를 고릅니다.
    """
    W, H = lim
    best, bn, bd = (0, 0), -1, 1e18
    for cx, cy in cs:
        x = float(np.clip(cx - ww / 2, 0, max(0, W - ww)))
        y = float(np.clip(cy - hh / 2, 0, max(0, H - hh)))
        m = ((cs[:, 0] >= x) & (cs[:, 0] < x + ww)
             & (cs[:, 1] >= y) & (cs[:, 1] < y + hh))
        n = int(m.sum())
        if n == 0:
            continue
        d = np.hypot(cs[m, 0].mean() - (x + ww / 2), cs[m, 1].mean() - (y + hh / 2))
        if n > bn or (n == bn and d < bd):
            best, bn, bd = (int(x), int(y)), n, d
    return best, bn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo11s_dota.pt")
    ap.add_argument("--root", default="C:/Users/seung/datasets/S2Ships")
    ap.add_argument("--tile", default="34WFT", help="평가 해역 MGRS 코드")
    ap.add_argument("--conf", type=float, default=0.5,
                    help="F1 최고점 (outputs/operating_point_test.json)")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--win", type=int, default=640, help="창의 가로 (화소)")
    ap.add_argument("--scale", type=int, default=6, help="화면 배율")
    ap.add_argument("--out", default="outputs/fig6_hero")
    a = ap.parse_args()

    import rasterio
    from build_dataset import fetch
    from korea_ports import detect, on_water
    from ultralytics import YOLO

    scenes = json.load(open(f"{a.root}/scenes.json", encoding="utf-8"))
    gpkg = f"{a.root}/{a.tile}.gpkg"
    ww, hh = a.win, a.win * 9 // 16

    # 날짜마다 배가 가장 많이 담기는 창을 재고, 척수가 많은 날을 씁니다.
    # 배 대비만 보고 고르면 구름이 낀 빈 바다가 뽑히므로, 대비는 척수가
    # 비슷할 때의 저울로만 씁니다.
    cand = []
    for key, sc in sorted(scenes.items()):
        if not key.startswith(a.tile + "_"):
            continue
        date = key.split("_")[1]
        href = sc["assets"]["visual"]
        with rasterio.open(href) as s:
            T, crs, lim = s.transform, s.crs, (s.width, s.height)
        cs = load_centers(gpkg, date, crs, T)
        (x0, y0), n = best_window(cs, ww, hh, lim)
        arr, _, _ = fetch(href, (*(T * (x0, y0 + hh)), *(T * (x0 + ww, y0))))
        im0 = np.ascontiguousarray(np.transpose(arr[:3], (1, 2, 0))[:, :, ::-1])
        con = ship_contrast(im0, cs[(cs[:, 0] >= x0) & (cs[:, 0] < x0 + ww)
                                    & (cs[:, 1] >= y0) & (cs[:, 1] < y0 + hh)] - [x0, y0])
        cand.append((con, n, key, href, T, x0, y0, cs, im0))
        print(f"  {key}  창 안 {n:>3}척 · 배 대비 중앙값 {con:4.0f} DN · "
              f"구름 {sc.get('cloud') or 0:.1f}%")
    con, n_gt, key, href, T, x0, y0, cs, img = max(cand, key=lambda t: (t[1], t[0]))
    print(f"장면 {key} · 창 ({x0},{y0}) {ww}x{hh} = "
          f"{ww * 10 / 1000:.1f} x {hh * 10 / 1000:.1f} km · 정답 {n_gt}척 · 대비 {con:.0f} DN")

    hh, ww = img.shape[:2]

    model = YOLO(a.weights)
    raw = detect(img, model, tile=320, overlap=64, conf=a.conf, iou=a.iou)
    # 물 위 필터입니다. 장면 전체를 훑으면 갯바위·부두 가장자리가 배처럼
    # 잡힙니다. 탐지 주변 고리가 어둡고 균질한지를 보고 거르는데, 한국 항만
    # 파이프라인에서 쓰는 것과 같은 검사입니다.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    polys = [p for p, _ in raw
             if on_water(gray, p[:, 0].mean(), p[:, 1].mean()) is not False]
    print(f"물 위 필터 {len(raw)} → {len(polys)}척")

    # 창 안 정답과 짝지어 봅니다. 배가 5 화소라 IoU 대신 중심 거리로 셉니다
    # (2장에서 지표를 바꿔 확인한 것과 같은 기준입니다).
    gw = cs[(cs[:, 0] >= x0) & (cs[:, 0] < x0 + ww)
            & (cs[:, 1] >= y0) & (cs[:, 1] < y0 + hh)] - [x0, y0]
    dc = np.array([[p[:, 0].mean(), p[:, 1].mean()] for p in polys])
    dist = np.hypot(gw[:, None, 0] - dc[None, :, 0], gw[:, None, 1] - dc[None, :, 1])
    hit = int((dist.min(axis=1) <= 3).sum())          # 찾아낸 정답
    tp = int((dist.min(axis=0) <= 3).sum())           # 정답과 짝이 있는 탐지
    print(f"탐지 {len(polys)}척 · 정답 {len(gw)}척 중 {hit}척 적중 · "
          f"짝 없는 탐지 {len(polys) - tp}척 (conf {a.conf}, NMS {a.iou})")

    # 겹치지 않은 양쪽의 성격을 밝기로 재 둡니다. 놓친 쪽이 어둡고 짝 없는
    # 쪽이 밝으면, 후자는 헛짚은 것이 아니라 주석에 빠진 배로 읽어야 합니다.
    dn = {"hit": ship_contrast(img, gw[dist.min(axis=1) <= 3]),
          "miss": ship_contrast(img, gw[dist.min(axis=1) > 3]),
          "unmatched": ship_contrast(img, dc[dist.min(axis=0) > 3])}
    print("  밝기 차 중앙값 — 겹친 배 {hit:.0f} · 놓친 배 {miss:.0f} · "
          "짝 없는 탐지 {unmatched:.0f} DN".format(**dn))

    S = a.scale
    W, H = ww * S, hh * S
    shown = tone(stretch(img))
    base = cv2.resize(shown, (W, H), interpolation=cv2.INTER_LANCZOS4)

    # 청록 박스입니다. 배가 5 화소라 실제 크기대로 그리면 화면에서 선 두 줄이
    # 됩니다. 최소 한 변을 두어 눈에 걸리게 하되, 중심과 방향은 탐지 결과
    # 그대로입니다.
    MIN = 30
    boxes = []
    for p in polys:
        (cx, cy), (bw, bh), ang = cv2.minAreaRect((p * S).astype(np.float32))
        r = cv2.boxPoints(((cx, cy), (max(bw, MIN), max(bh, MIN)), ang))
        boxes.append(r.astype(np.int32))
    over = base.copy()
    for q in boxes:
        cv2.fillPoly(over, [q], CYAN)
    base = cv2.addWeighted(over, 0.13, base, 0.87, 0)
    for q in boxes:
        cv2.polylines(base, [q], True, CYAN, 3, cv2.LINE_AA)

    # 글자는 Pretendard 로 얹습니다 — 슬라이드 본문과 같은 서체입니다
    im = Image.fromarray(cv2.cvtColor(base, cv2.COLOR_BGR2RGB))
    dr = ImageDraw.Draw(im, "RGBA")
    cy = (38, 228, 228, 255)

    f_big, f_lab = font("Bold", 104), font("SemiBold", 27)
    num = str(len(polys))
    nw = dr.textlength(num, font=f_big)
    lw = dr.textlength("탐지된 선박", font=f_lab)
    bw = int(max(nw, lw) + 72)
    bx, by = W - 64 - bw, 64
    rrect(dr, [bx, by, bx + bw, by + 158], 16, fill=(8, 12, 14, 195),
          outline=(38, 228, 228, 95), width=2)
    dr.text((bx + bw / 2 - lw / 2, by + 22), "탐지된 선박", font=f_lab,
            fill=(150, 235, 235, 255))
    dr.text((bx + bw / 2 - nw / 2, by + 46), num, font=f_big, fill=cy)

    f_leg = font("Medium", 27)
    tw = dr.textlength("모델이 찾은 선박", font=f_leg)
    rrect(dr, [64, H - 64 - 62, 64 + tw + 92, H - 64], 12, fill=(8, 12, 14, 195))
    rrect(dr, [86, H - 64 - 41, 108, H - 64 - 19], 3,
          fill=(38, 228, 228, 55), outline=cy, width=3)
    dr.text((126, H - 64 - 47), "모델이 찾은 선박", font=f_leg, fill=(232, 232, 232, 255))

    # 축척 막대 — 화소 하나가 10 m 입니다
    f_sc = font("Medium", 25)
    mbar, sw = 200, 200 * S
    lab = f"{mbar * 10 // 1000} km"
    cwid = sw + 44 + dr.textlength(lab, font=f_sc) + 24
    cx0, cy0 = W - 64 - cwid, H - 64 - 62
    rrect(dr, [cx0, cy0, cx0 + cwid, H - 64], 12, fill=(8, 12, 14, 195))
    sx, sy = cx0 + 22, cy0 + 20
    for seg in ([sx, sy + 11, sx + sw, sy + 11], [sx, sy + 2, sx, sy + 20],
                [sx + sw, sy + 2, sx + sw, sy + 20]):
        dr.line(seg, fill=(232, 232, 232, 235), width=4)
    dr.text((sx + sw + 22, cy0 + 16), lab, font=f_sc, fill=(232, 232, 232, 235))

    # 아래쪽 글자는 밝은 뭍 위에서 안 보이므로 범례와 같은 어두운 칩에 얹습니다
    d = key.split("_")[1]
    at = f"Copernicus Sentinel-2 L2A · {a.tile} {d[:4]}-{d[4:6]}-{d[6:]}"
    f_at = font("Regular", 22)
    aw = dr.textlength(at, font=f_at)
    ax = 64 + tw + 92 + 16
    rrect(dr, [ax, H - 64 - 62, ax + aw + 44, H - 64], 12, fill=(8, 12, 14, 195))
    dr.text((ax + 22, H - 64 - 44), at, font=f_at, fill=(176, 176, 176, 235))

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    im.save(f"{a.out}_det.png")
    cv2.imwrite(f"{a.out}_raw.png",
                cv2.resize(shown, (W, H), interpolation=cv2.INTER_LANCZOS4))
    json.dump({"scene": key, "window": [x0, y0, ww, hh], "km": [ww / 100, hh / 100],
               "gt": int(len(gw)), "det_raw": len(raw), "det": len(polys), "hit": hit,
               "unmatched": len(polys) - tp, "dn": {k: round(v) for k, v in dn.items()},
               "conf": a.conf, "nms": a.iou},
              open(f"{a.out}.json", "w", encoding="utf-8"), indent=2)
    print(f"저장: {a.out}_det.png / {a.out}_raw.png  ({W}x{H})")


if __name__ == "__main__":
    main()
