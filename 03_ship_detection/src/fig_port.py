"""한국 항만 탐지 결과 대표 그림 — 코멘토 3차 업무 / 정승원

PPT 3장의 주인공입니다. `korea_ports.py` 와 같은 경로로 항만 한 곳을 훑고,
`fig_hero.py` 와 같은 방식으로 그려 두 그림을 나란히 놓을 수 있게 합니다.

정답은 어디서 오는가
--------------------
Global Fishing Watch 가 Sentinel-2 선박 탐지를 전 지구로 공개합니다. 같은 센서,
같은 10 m 격자이고 장면 번호까지 붙어 있어, 그 장면을 그대로 받아 좌표로 맞대면
됩니다. 부산 외항 정박지에서는 하루 55척까지 잡히고, 그중 34척은 AIS 와
대조돼 있습니다.

GFW 도 사람이 그린 정답은 아닙니다. 딥러닝 탐지 결과이고 자체 선박확률이
붙어 있으므로, 확률 하한을 두고 인프라는 뺍니다. '정답'보다는 '독립적인 참조'로
읽는 것이 맞고, 이 한계는 결과에 적어 둡니다.

배 크기도 핀란드와 다릅니다. 핀란드 평가셋은 장변 중앙값이 5.2 화소였지만, 부산 외항
정박지는 정박 중인 대형선이라 10~20 화소입니다. 그래서 확대를 덜 해도 형체가
읽히고, 한 화면에 수십 척이 들어옵니다.

장면은 고정합니다
-----------------
`find_scene` 은 오늘 기준으로 가장 최근의 무운 장면을 찾으므로, 다시 돌릴
때마다 날짜가 바뀝니다. 발표 자료의 수치와 그림이 어긋나지 않도록 장면 번호를
인자로 못 박고, 그 번호로 COG 주소를 직접 만듭니다.

사용법
------
  py fig_port.py --weights weights/yolo11s_dota.pt
"""
import os
import json
import argparse

import numpy as np
import cv2

from fig_hero import render, tone, stretch

COG = ("https://sentinel-cogs.s3.us-west-2.amazonaws.com/sentinel-s2-l2a-cogs/"
       "{utm}/{band}/{sq}/{y}/{m}/{sid}/TCI.tif")


def cog_href(scene_id):
    """장면 번호에서 AWS 공개 COG 주소를 만듭니다 (S2C_52SED_20260212_0_L2A)."""
    _, grid, date, *_ = scene_id.split("_")
    return COG.format(utm=grid[:2].lstrip("0"), band=grid[2], sq=grid[3:],
                      y=date[:4], m=str(int(date[4:6])), sid=scene_id)


def best_window(cs, ww, hh, lim):
    """탐지가 가장 많이 담기는 창의 좌상단입니다.

    한국 해역에는 정답이 없으므로 탐지 위치로 창을 잡습니다. 척수가 같으면
    무리의 무게중심이 가운데 오는 자리를 씁니다.
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


def load_gfw(path, bbox, date, min_presence):
    """그 장면 날짜의 GFW 탐지를 정답으로 모읍니다 (경위도)."""
    import csv
    lon0, lat0, lon1, lat1 = bbox
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if (r.get("detect_timestamp") or "")[:10].replace("-", "") != date:
                continue
            try:
                lon, lat = float(r["lon"]), float(r["lat"])
                if float(r["presence_score"]) < min_presence:
                    continue
            except (KeyError, ValueError, TypeError):
                continue
            if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
                continue
            if str(r.get("likely_infrastructure", "")).lower() == "true":
                continue
            out.append((lon, lat, r.get("length_m_inferred"), r.get("matching_score")))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo11s_dota.pt")
    ap.add_argument("--port", default="busan_anchorage")
    ap.add_argument("--scene", default="S2A_52SED_20221011_0_L2A",
                    help="장면 번호. GFW 정답이 있는 날로 못 박습니다")
    ap.add_argument("--gfw", default="C:/Users/seung/datasets/GFW/korea.csv")
    ap.add_argument("--min-presence", type=float, default=0.8)
    # korea_ports.py 는 낮은 임계값에 물 게이트를 짝으로 겁니다. 발표 그림은
    # 평가셋 F1 최고점(0.5)으로 한 번 더 걸러, 박스 하나하나가 확신 있는
    # 탐지만 남게 합니다.
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--show-conf", type=float, default=0.5, help="그림에 남길 신뢰도")
    ap.add_argument("--nms-iou", type=float, default=0.1)
    ap.add_argument("--win", type=int, default=1440, help="창의 가로 (화소)")
    ap.add_argument("--scale", type=int, default=3, help="화면 배율")
    ap.add_argument("--summary", default="outputs/korea/summary.json")
    ap.add_argument("--out", default="outputs/fig7_port")
    a = ap.parse_args()

    from korea_ports import PORTS, fetch_window, detect, on_water
    from ultralytics import YOLO

    label, *bbox = PORTS[a.port]
    scene = a.scene
    if not scene and os.path.exists(a.summary):
        scene = json.load(open(a.summary, encoding="utf-8"))[a.port]["scene"]
    href = cog_href(scene)
    print(f"{label} · 장면 {scene}")

    img, affine, crs = fetch_window(href, bbox)
    img = np.ascontiguousarray(img[:, :, ::-1])          # RGB -> BGR
    print(f"수신 {img.shape[1]}x{img.shape[0]} "
          f"({img.shape[1] / 100:.1f} x {img.shape[0] / 100:.1f} km)")

    model = YOLO(a.weights)
    raw = detect(img, model, conf=a.conf, iou=a.nms_iou)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gated = [(p, c) for p, c in raw
             if on_water(gray, p[:, 0].mean(), p[:, 1].mean()) is not False]
    keep = [p for p, c in gated if c >= a.show_conf]
    print(f"탐지 {len(raw)} → 물 위 {len(gated)} → 신뢰도 {a.show_conf} 이상 {len(keep)}척")

    # GFW 정답을 이 장면의 화소 좌표로 옮깁니다
    from rasterio.warp import transform as warp
    gfw = load_gfw(a.gfw, bbox, scene.split("_")[2], a.min_presence)
    xs, ys = warp("EPSG:4326", crs, [g[0] for g in gfw], [g[1] for g in gfw])
    inv = ~affine
    gt = np.array([inv * (x, y) for x, y in zip(xs, ys)]) if gfw else np.zeros((0, 2))
    ais = sum(1 for g in gfw if (g[3] or "") not in ("", "0") and float(g[3]) > 0.5)
    print(f"GFW 정답 {len(gt)}척 (AIS 대조 {ais}척)")

    ww, hh = a.win, a.win * 9 // 16
    cs = np.array([[p[:, 0].mean(), p[:, 1].mean()] for p in keep])
    (x0, y0), n = best_window(cs, ww, hh, (img.shape[1], img.shape[0]))
    sub = img[y0:y0 + hh, x0:x0 + ww]
    polys = [p - [x0, y0] for p in keep
             if x0 <= p[:, 0].mean() < x0 + ww and y0 <= p[:, 1].mean() < y0 + hh]
    lens = sorted(max(np.linalg.norm(p[i] - p[(i + 1) % 4]) for i in range(4)) * 10
                  for p in polys)
    print(f"창 ({x0},{y0}) {ww}x{hh} = {ww / 100:.1f} x {hh / 100:.1f} km · "
          f"{len(polys)}척 · 길이 중앙값 {lens[len(lens) // 2]:.0f} m")

    # 창 안에서 정답과 맞대 봅니다. 3 화소 물체에 IoU 를 요구하는 것은 지표의
    # 함정이고, GFW 정답이 애초에 점이라 점 기반이 정답의 성격에 맞습니다.
    gw = gt[(gt[:, 0] >= x0) & (gt[:, 0] < x0 + ww)
            & (gt[:, 1] >= y0) & (gt[:, 1] < y0 + hh)] - [x0, y0] if len(gt) else gt
    dc = np.array([[p[:, 0].mean(), p[:, 1].mean()] for p in polys])
    D = (np.hypot(gw[:, None, 0] - dc[None, :, 0], gw[:, None, 1] - dc[None, :, 1])
         if len(gw) else np.zeros((0, len(dc))))
    hit = int((D.min(axis=1) <= 3).sum()) if len(gw) else 0
    tp = int((D.min(axis=0) <= 3).sum()) if len(gw) else 0
    err = float(np.median(D.min(axis=1)[D.min(axis=1) <= 3])) if hit else 0.0
    print(f"창 안 정답 {len(gw)}척 중 {hit}척 적중 · 짝 없는 탐지 {len(polys) - tp}척 · "
          f"중심 오차 중앙값 {err:.2f} px")

    d = scene.split("_")[2]
    im, shown = render(sub, polys, a.scale,
                       f"Copernicus Sentinel-2 L2A · {scene.split('_')[1]} "
                       f"{d[:4]}-{d[4:6]}-{d[6:]}")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    im.save(f"{a.out}_det.png")
    cv2.imwrite(f"{a.out}_raw.png",
                cv2.resize(shown, im.size, interpolation=cv2.INTER_LANCZOS4))
    json.dump({"port": a.port, "label": label, "scene": scene,
               "window": [x0, y0, ww, hh], "km": [ww / 100, hh / 100],
               "gt": int(len(gw)), "det": len(polys), "hit": hit,
               "unmatched": len(polys) - tp, "err_px": round(err, 2),
               "gt_scene": len(gt), "ais": ais, "det_scene": len(keep),
               "median_m": round(lens[len(lens) // 2]),
               "conf": a.conf, "show_conf": a.show_conf, "nms": a.nms_iou},
              open(f"{a.out}.json", "w", encoding="utf-8"), indent=2)
    print(f"저장: {a.out}_det.png / {a.out}_raw.png  ({im.size[0]}x{im.size[1]})")


if __name__ == "__main__":
    main()
