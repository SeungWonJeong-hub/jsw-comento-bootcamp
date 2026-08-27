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

        vis = img[:, :, ::-1].copy()
        rows = []
        for poly, cf in dets:
            cv2.polylines(vis, [poly.astype(np.int32)], True, (0, 212, 255), 2)
            cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
            lon, lat = to_lonlat(affine, crs, cx, cy)
            e = [math.dist(poly[i], poly[(i + 1) % 4]) for i in range(4)]
            rows.append({"lon": lon, "lat": lat, "conf": cf,
                         "length_px": max(e[0], e[1]), "length_m": max(e[0], e[1]) * 10})
        cv2.imwrite(f"{a.outdir}/{name}.jpg", vis)
        summary[name] = {"label": label, "scene": sc["id"], "datetime": sc["datetime"],
                         "cloud": sc["cloud"], "size_px": list(img.shape[:2]),
                         "n_detections": len(dets), "detections": rows}
        med = np.median([r["length_m"] for r in rows]) if rows else 0
        print(f"{label:<16} {sc['datetime'][:10]}  구름 {sc['cloud']:>4.1f}%  "
              f"{img.shape[1]}x{img.shape[0]}px  탐지 {len(dets):>4}척  길이중앙 {med:>5.0f} m")

    json.dump(summary, open(f"{a.outdir}/summary.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)
    print(f"\n저장: {a.outdir}")


if __name__ == "__main__":
    main()
