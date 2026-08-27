"""한국 항만 선박 탐지 — 코멘토 3차 업무 / 정승원

항만 이름 하나만 주면 최신 구름 없는 Sentinel-2 장면을 찾아 받아서 탐지까지 돌린다.
4차 웹앱의 "항구를 고르면 탐지가 실행된다"가 이 함수 하나로 성립하도록 설계했다.

    python src/korea_ports.py --port busan_anchorage --weights weights/best.pt

항만 목록은 아래 PORTS 에 있고, 웹앱에서는 이 딕셔너리를 그대로 드롭다운으로 쓴다.
"""
import os, json, argparse, math
import numpy as np
import cv2

# 이름: (설명, lon0, lat0, lon1, lat1)
# 범위는 부두와 앞바다를 함께 덮도록 잡았다. 정박지는 항만 밖 대기 선박이 목적이다.
PORTS = {
    # 대형 상선
    "busan_north":      ("부산 북항",          128.98, 35.05, 129.12, 35.15),
    "busan_new":        ("부산 신항",          128.74, 35.02, 128.88, 35.12),
    "busan_anchorage":  ("부산 외항 정박지",    128.98, 34.94, 129.22, 35.08),
    "gwangyang":        ("광양항",            127.62, 34.84, 127.80, 34.96),
    "incheon":          ("인천항",            126.50, 37.38, 126.68, 37.52),
    "pyeongtaek":       ("평택·당진항",        126.74, 36.92, 126.90, 37.04),
    "daesan":           ("대산항",            126.28, 36.94, 126.44, 37.06),
    "gunsan":           ("군산항",            126.54, 35.92, 126.70, 36.04),
    "pohang":           ("포항 영일만항",      129.36, 36.00, 129.50, 36.12),
    # 액체화물·조선
    "ulsan":            ("울산항·앞바다",      129.32, 35.44, 129.50, 35.56),
    "yeosu":            ("여수·광양만",        127.68, 34.68, 127.84, 34.82),
    "geoje_okpo":       ("거제 옥포",          128.64, 34.82, 128.76, 34.94),
    "geoje_gohyeon":    ("거제 고현",          128.56, 34.84, 128.68, 34.96),
    "yeongam":          ("영암 현대삼호",      126.44, 34.73, 126.58, 34.85),
    # 어항·연안
    "busan_gamcheon":   ("부산 감천항",        128.96, 35.02, 129.06, 35.10),
    "mokpo":            ("목포항",            126.30, 34.72, 126.46, 34.84),
    "wando":            ("완도·다도해",        126.68, 34.24, 126.86, 34.40),
    "tongyeong":        ("통영항",            128.36, 34.78, 128.50, 34.90),
    "masan":            ("마산항",            128.52, 35.13, 128.64, 35.25),
    "jeju":             ("제주항",            126.46, 33.48, 126.60, 33.58),
    "sokcho":           ("속초항",            128.54, 38.15, 128.68, 38.27),
    "donghae":          ("동해·묵호항",        129.06, 37.46, 129.20, 37.58),
}

STAC = "https://earth-search.aws.element84.com/v1/search"


def find_scene(bbox, max_cloud=10, start="2024-01-01", end=None):
    """관심 항만을 덮는 가장 최근의 구름 적은 장면."""
    import requests
    from datetime import date
    end = end or date.today().isoformat()
    q = {"collections": ["sentinel-2-l2a"], "bbox": list(bbox),
         "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
         "query": {"eo:cloud_cover": {"lt": max_cloud}},
         "limit": 20, "sortby": [{"field": "properties.datetime", "direction": "desc"}]}
    r = requests.post(STAC, json=q, timeout=90).json()
    feats = r.get("features", [])
    if not feats:
        return None
    # STAC 는 bbox 와 '겹치기만' 해도 돌려준다. 항만이 타일 경계에 걸치면
    # 겨우 몇 픽셀만 덮는 장면이 뽑혀 아무것도 못 읽는다. 실제로 덮는 넓이로 고른다.
    def covered(f):
        b = f["bbox"]
        ov = (max(0.0, min(b[2], bbox[2]) - max(b[0], bbox[0]))
              * max(0.0, min(b[3], bbox[3]) - max(b[1], bbox[1])))
        need = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
        return ov / need if need > 0 else 0.0
    feats = [f for f in feats if covered(f) > 0.6] or feats
    feats.sort(key=lambda f: (-covered(f), f["properties"]["datetime"]), reverse=False)
    feats.sort(key=lambda f: -covered(f))
    f = feats[0]
    return {"id": f["id"], "datetime": f["properties"]["datetime"],
            "cloud": f["properties"].get("eo:cloud_cover"),
            "tile": f["properties"].get("grid:code"),
            "visual": f["assets"]["visual"]["href"],
            "nir": f["assets"]["nir"]["href"] if "nir" in f["assets"] else None}


def fetch_window(href, bbox):
    """항만 범위를 픽셀로 잘라 온다."""
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import from_bounds
    for k, v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                     GDAL_HTTP_MAX_RETRY="3").items():
        os.environ.setdefault(k, v)
    lon0, lat0, lon1, lat1 = bbox
    with rasterio.open(href) as s:
        xs, ys = transform("EPSG:4326", s.crs, [lon0, lon1], [lat0, lat1])
        w = from_bounds(xs[0], ys[0], xs[1], ys[1], s.transform).round_offsets().round_lengths()
        # 창이 타일 밖으로 나가면 read() 는 말없이 잘라 준다. 그런데 window_transform()
        # 은 '요청한' 창 기준이라 좌표가 어긋난다. 먼저 잘라서 맞춘다.
        # (항만이 타일 경계에 걸치면 창 폭이 0 이 되어 아무것도 못 읽는 일도 생긴다)
        w = w.crop(s.height, s.width)
        arr = s.read(window=w)
        return np.transpose(arr, (1, 2, 0)), s.window_transform(w), s.crs


def detect(img, model, tile=320, overlap=64, conf=0.25, iou=0.5):
    """큰 사진을 타일로 잘라 탐지하고 원래 좌표로 되돌린다.

    항만 하나가 1,000~2,000 px 이라 모델 입력(320)보다 크다. 그냥 리사이즈하면
    이미 몇 픽셀인 배가 더 작아져 사라진다. 그래서 겹치게 잘라 훑는다.
    """
    H, W = img.shape[:2]
    step = tile - overlap
    out = []
    for y0 in range(0, max(1, H - overlap), step):
        for x0 in range(0, max(1, W - overlap), step):
            y1, x1 = min(y0 + tile, H), min(x0 + tile, W)
            patch = img[y0:y1, x0:x1]
            if patch.shape[0] < 32 or patch.shape[1] < 32 or patch.max() == 0:
                continue
            if patch.shape[:2] != (tile, tile):        # 가장자리는 채워서 크기를 맞춘다
                pad = np.zeros((tile, tile, patch.shape[2]), patch.dtype)
                pad[:patch.shape[0], :patch.shape[1]] = patch
                patch = pad
            r = model.predict(patch[:, :, ::-1], imgsz=tile, conf=conf, verbose=False)[0]
            if r.obb is None or len(r.obb) == 0:
                continue
            for poly, cf in zip(r.obb.xyxyxyxy.cpu().numpy(), r.obb.conf.cpu().numpy()):
                out.append((poly.reshape(4, 2) + [x0, y0], float(cf)))
    return nms_obb(out, iou)


def poly_iou(a, b):
    ia, _ = cv2.intersectConvexConvex(a.astype(np.float32), b.astype(np.float32))
    if ia <= 0:
        return 0.0
    ua = cv2.contourArea(a.astype(np.float32)) + cv2.contourArea(b.astype(np.float32)) - ia
    return ia / ua if ua > 0 else 0.0


def nms_obb(dets, thr):
    """타일이 겹치므로 같은 배가 여러 번 잡힌다. 회전 박스 기준으로 합친다."""
    dets = sorted(dets, key=lambda d: -d[1])
    keep = []
    for poly, cf in dets:
        if all(poly_iou(poly, k[0]) < thr for k in keep):
            keep.append((poly, cf))
    return keep


def on_water(gray, cx, cy, r=22):
    """탐지 주변이 '물처럼' 보이는가 — 정답이 없을 때 쓰는 자동 품질 검사.

    물은 어둡고 균질하다. 육지·도심·부두는 밝고 얼룩덜룩하다.
    배 자체는 밝으므로 중심을 빼고 주변 고리만 본다.
    이 비율이 낮으면 그 항만의 탐지를 믿기 어렵다는 신호다.
    """
    h, w = gray.shape[:2]
    y0, y1 = max(0, int(cy) - r), min(h, int(cy) + r)
    x0, x1 = max(0, int(cx) - r), min(w, int(cx) + r)
    patch = gray[y0:y1, x0:x1]
    if patch.size < 200:
        return None
    m = np.ones(patch.shape, bool)
    a, b = max(0, int(cy) - y0 - 6), max(0, int(cx) - x0 - 6)
    m[a:a + 12, b:b + 12] = False          # 배 본체 제외
    ring = patch[m]
    if ring.size < 100:
        return None
    return bool(np.median(ring) < 90 and ring.std() < 45)


def to_lonlat(affine, crs, x, y):
    from rasterio.warp import transform as _t
    X, Y = affine * (x, y)
    lon, lat = _t(crs, "EPSG:4326", [X], [Y])
    return float(lon[0]), float(lat[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="busan_anchorage", choices=list(PORTS) + ["all"])
    ap.add_argument("--weights", required=True)
    ap.add_argument("--repo", default=None, help="커스텀 모듈이 있는 ultralytics 포크 경로")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--max-cloud", type=float, default=10)
    ap.add_argument("--outdir", default="outputs/korea")
    a = ap.parse_args()

    import sys
    if a.repo:
        sys.path.insert(0, a.repo)
    from ultralytics import YOLO
    model = YOLO(a.weights)

    os.makedirs(a.outdir, exist_ok=True)
    targets = list(PORTS) if a.port == "all" else [a.port]
    summary = {}

    for name in targets:
        label, *bbox = PORTS[name]
        sc = find_scene(bbox, a.max_cloud)
        if not sc:
            print(f"{label:<16} 구름 {a.max_cloud}% 이하 장면 없음")
            continue
        img, affine, crs = fetch_window(sc["visual"], bbox)
        dets = detect(img, model, conf=a.conf)

        gray = img.mean(2)
        vis = img[:, :, ::-1].copy()
        rows = []
        for poly, cf in dets:
            cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 212, 255), 2)
            cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
            lon, lat = to_lonlat(affine, crs, cx, cy)
            e = [math.dist(poly[i], poly[(i + 1) % 4]) for i in range(4)]
            rows.append({"lon": lon, "lat": lat, "conf": cf,
                         "length_px": max(e[0], e[1]), "length_m": max(e[0], e[1]) * 10,
                         "on_water": on_water(gray, cx, cy)})
        cv2.imwrite(f"{a.outdir}/{name}.jpg", vis)
        summary[name] = {"label": label, "scene": sc["id"], "datetime": sc["datetime"],
                         "cloud": sc["cloud"], "size_px": list(img.shape[:2]),
                         "n_detections": len(dets), "detections": rows}
        med = np.median([r["length_m"] for r in rows]) if rows else 0
        ws = [r["on_water"] for r in rows if r["on_water"] is not None]
        wr = 100 * sum(ws) / len(ws) if ws else 0
        summary[name]["water_ratio"] = wr
        print(f"{label:<16} {sc['datetime'][:10]}  구름 {sc['cloud']:>4.1f}%  "
              f"{img.shape[1]}x{img.shape[0]}px  탐지 {len(dets):>4}척  "
              f"길이중앙 {med:>5.0f} m  물 위 {wr:>3.0f}%")

    # 항만을 하나씩 돌리는 경우가 많아, 덮어쓰지 말고 이어 붙인다
    sp = f"{a.outdir}/summary.json"
    if os.path.exists(sp):
        try:
            old = json.load(open(sp, encoding="utf-8"))
            old.update(summary)
            summary = old
        except Exception:
            pass
    json.dump(summary, open(sp, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {a.outdir}")


if __name__ == "__main__":
    main()
