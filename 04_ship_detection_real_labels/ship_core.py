# -*- coding: utf-8 -*-
"""웹앱이 쓰는 순수 함수 — 화면과 분리해 시험할 수 있게 뺐습니다.

app_ship.py 는 streamlit 스크립트라 import 하는 것만으로 앱이 돌아갑니다.
그래서 시험 대상이 되는 계산은 전부 여기에 둡니다.
"""
import io
import os

import numpy as np
import cv2

# 한 화소의 지상 크기(m). HRSC2016 은 항만 위도에 따라 0.40~0.50 이라 app_ship.py 가
# 항만마다 계산합니다. 여기 값은 시험용 기본값입니다.
GSD = 0.45


def stretch(v, min_range=40.0):
    """장면마다 p1~p99.5 로 늘립니다.

    최소 폭을 둡니다. 물뿐인 자리에서 그대로 늘리면 잡음이 대비 가득
    증폭되어 모델이 물결을 물체로 읽습니다.
    """
    lo, hi = np.percentile(v, 1.0), np.percentile(v, 99.5)
    if hi - lo < min_range:
        hi = lo + min_range
    return (np.clip((v - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)


def read_upload(buf, name):
    """올린 파일 -> (RGB 0~255, 지리정보 또는 None, 8비트인가)."""
    ext = os.path.splitext(name)[1].lower()
    if ext == ".npy":
        a = np.load(io.BytesIO(buf)).astype(np.float32)
        if a.ndim != 3 or a.shape[2] < 3:
            raise ValueError("3 채널 이상이어야 합니다 (%s)" % (a.shape,))
        v = a[:, :, [2, 1, 0]] if a.shape[2] >= 4 else a[:, :, :3]
        return stretch(v), None, False
    if ext in (".tif", ".tiff"):
        try:
            import rasterio
        except Exception:
            raise ValueError("GeoTIFF 를 읽으려면 rasterio 가 필요합니다")
        with rasterio.MemoryFile(buf) as mf, mf.open() as ds:
            a = np.transpose(ds.read().astype(np.float32), (1, 2, 0))
            geo = None
            if ds.crs is not None:
                t = ds.transform
                geo = {"transform": [t.a, t.b, t.c, t.d, t.e, t.f],
                       "crs": str(ds.crs)}
        v = a[:, :, [2, 1, 0]] if a.shape[2] >= 3 else a
        return stretch(v), geo, False
    v = cv2.imdecode(np.frombuffer(buf, np.uint8), cv2.IMREAD_COLOR)
    if v is None:
        raise ValueError("그림을 읽지 못했습니다")
    return v[:, :, ::-1].copy(), None, True


def detect_tiled(model, rgb, conf, tile=640, stride=480):
    """겹쳐 훑고 합칩니다.

    큰 장면을 한 번에 넣으면 배가 뭉개지고, 겹치지 않고 자르면 타일
    경계에 걸린 배가 잘립니다. 실제로 재보니 경계 손실이 재현율을
    크게 갉아먹었습니다.
    """
    H, W = rgb.shape[:2]
    ys = list(range(0, max(H - tile, 0) + 1, stride)) or [0]
    xs = list(range(0, max(W - tile, 0) + 1, stride)) or [0]
    if ys[-1] + tile < H:
        ys.append(max(H - tile, 0))
    if xs[-1] + tile < W:
        xs.append(max(W - tile, 0))
    out = []
    for y in ys:
        for x in xs:
            crop = rgb[y:y + tile, x:x + tile]
            if crop.shape[0] < 32 or crop.shape[1] < 32:
                continue
            r = model.predict(crop[:, :, ::-1], imgsz=640, conf=conf,
                              verbose=False)[0]
            for b, c in zip(r.boxes.xyxy.cpu().numpy(),
                            r.boxes.conf.cpu().numpy()):
                out.append([b[0] + x, b[1] + y, b[2] + x, b[3] + y, float(c)])
    if not out:
        return []
    d = np.array(out)
    keep = []
    for i in np.argsort(-d[:, 4]):
        b = d[i]
        ok = True
        for k in keep:
            x0 = max(b[0], k[0]); y0 = max(b[1], k[1])
            x1 = min(b[2], k[2]); y1 = min(b[3], k[3])
            inter = max(x1 - x0, 0) * max(y1 - y0, 0)
            a1 = (b[2] - b[0]) * (b[3] - b[1])
            a2 = (k[2] - k[0]) * (k[3] - k[1])
            if inter / max(a1 + a2 - inter, 1e-9) > 0.3:
                ok = False
                break
        if ok:
            keep.append(b)
    return keep


def to_lonlat(x, y, geo):
    a, b, c, d, e, f = geo["transform"]
    X = a * (x + 0.5) + b * (y + 0.5) + c
    Y = d * (x + 0.5) + e * (y + 0.5) + f
    from pyproj import Transformer
    tr = Transformer.from_crs(geo["crs"], "EPSG:4326", always_xy=True)
    lon, lat = tr.transform(X, Y)
    return float(lon), float(lat)
