"""핀란드 연안 선박 데이터셋 구축 — 코멘토 3차 업무 / 정승원

Zenodo 에는 주석(GeoPackage)만 있고 위성 사진은 없다.
사진은 AWS 공개 COG(Sentinel-2 L1C)에서 인증 없이 받아 맞춘다.

절차
----
1. 타일·날짜별 주석 폴리곤을 읽는다 (EPSG:32634 / 32635)
2. 같은 타일·날짜의 Sentinel-2 L1C 장면을 AWS 에서 찾는다
3. 주석이 몰려 있는 영역만 창으로 잘라 온다 (전체 타일은 10,980² 이라 과하다)
4. 폴리곤을 픽셀 좌표로 옮기고 최소면적 회전 사각형(OBB)으로 바꾼다
5. 320x320 타일로 자르고 YOLO-OBB 라벨을 쓴다
6. 선박이 없는 타일도 일부 남긴다 (오탐을 재려면 음성 표본이 필요하다)

주의
----
- 논문은 L1C 의 TCI(트루컬러 8비트)로 학습했다. 여기서도 같은 자산을 쓴다.
- 타일 경계에 걸친 선박은 잘린 조각이 아니라 통째로 들어간 것만 남긴다.
  잘린 조각을 라벨로 두면 "절반짜리 배"를 학습하게 된다.
"""
import os, json, glob, argparse, math
import numpy as np
import cv2

TILE = 320
MIN_INSIDE = 0.75   # 타일 안에 이 비율 이상 들어와야 라벨로 인정


def load_annotations(gpkg_dir):
    """타일·날짜별 폴리곤을 {(tile, date): GeoDataFrame} 으로."""
    import geopandas as gpd
    import pyogrio
    out = {}
    for p in sorted(glob.glob(os.path.join(gpkg_dir, "*.gpkg"))):
        tile = os.path.basename(p)[:-5]
        for layer in pyogrio.list_layers(p):
            name = layer[0]
            g = gpd.read_file(p, layer=name)
            if len(g):
                out[(tile, name)] = g
    return out


def resolve_scenes(ann, out_path):
    """타일·날짜 → AWS 공개 COG 주소. 없으면 STAC 에 물어 만든다.

    주석은 L1C 기준으로 그려졌지만 L1C 자산은 요청자 부담 S3 라 무료로 못 받는다.
    L2A COG 는 같은 촬영·같은 격자라 기하가 동일하므로 그대로 쓸 수 있다.
    (대기 보정으로 화소값은 달라지지만 선박 위치는 안 바뀐다)
    """
    import requests
    if os.path.exists(out_path):
        return json.load(open(out_path, encoding="utf-8"))
    STAC = "https://earth-search.aws.element84.com/v1/search"
    found = {}
    for (tile, date) in sorted(ann):
        d = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        q = {"collections": ["sentinel-2-l2a"],
             "datetime": f"{d}T00:00:00Z/{d}T23:59:59Z",
             "query": {"grid:code": {"eq": "MGRS-" + tile}}, "limit": 5}
        try:
            feats = requests.post(STAC, json=q, timeout=90).json().get("features", [])
        except Exception as e:
            print(f"  {tile}_{date} STAC 실패: {e}")
            continue
        feats = [f for f in feats
                 if "visual" in f["assets"] and f["assets"]["visual"]["href"].startswith("https")]
        if not feats:
            print(f"  {tile}_{date} 장면 없음")
            continue
        f = feats[0]
        found[f"{tile}_{date}"] = {
            "id": f["id"], "cloud": f["properties"].get("eo:cloud_cover"),
            "assets": {"visual": f["assets"]["visual"]["href"]}}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    json.dump(found, open(out_path, "w"), indent=2)
    print(f"장면 {len(found)}/{len(ann)} 확보 → {out_path}")
    return found


def scene_window(gdf, pad=2000):
    """주석 전체를 감싸는 UTM 사각형 (미터 여유 포함)."""
    x0, y0, x1, y1 = gdf.total_bounds
    return x0 - pad, y0 - pad, x1 + pad, y1 + pad


def fetch(href, bounds, out_crs_check=None):
    """COG 에서 창 하나를 읽어 (배열, affine, crs) 로 돌려준다."""
    import rasterio
    from rasterio.windows import from_bounds
    for k, v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
                     AWS_NO_SIGN_REQUEST="YES",
                     GDAL_HTTP_MAX_RETRY="3", GDAL_HTTP_RETRY_DELAY="1").items():
        os.environ.setdefault(k, v)
    with rasterio.open(href) as s:
        w = from_bounds(*bounds, s.transform).round_offsets().round_lengths()
        # 창이 래스터 밖으로 나가면 read() 는 알아서 잘라 더 작은 배열을 준다.
        # 그런데 window_transform() 은 '요청한' 창 기준이라, 자른 만큼 라벨이 통째로 밀린다.
        # 주석이 타일 가장자리까지 닿아 있어 실제로 매번 발생한다. 먼저 잘라서 맞춘다.
        w = w.crop(s.height, s.width)
        arr = s.read(window=w)
        return arr, s.window_transform(w), s.crs


def polys_to_obb(gdf, affine):
    """폴리곤 → 픽셀 좌표 최소면적 회전 사각형 4점."""
    inv = ~affine
    out = []
    for geom in gdf.geometry:
        if geom is None or geom.is_empty:
            continue
        xs, ys = geom.exterior.coords.xy
        pts = np.array([inv * (x, y) for x, y in zip(xs, ys)], np.float32)
        rect = cv2.minAreaRect(pts)
        out.append(cv2.boxPoints(rect).astype(np.float32))
    return out


def inside_ratio(box, x0, y0, size):
    """박스가 타일 안에 얼마나 들어와 있나 (면적 비)."""
    poly = box - [x0, y0]
    clip = np.array([[0, 0], [size, 0], [size, size], [0, size]], np.float32)
    a, _ = cv2.intersectConvexConvex(poly.astype(np.float32), clip)
    full = cv2.contourArea(poly.astype(np.float32))
    return (a / full) if full > 0 else 0.0


def cut_tiles(img, boxes, stride, keep_empty_ratio, rng):
    """320x320 타일로 자르고 (이미지, 라벨) 목록을 만든다."""
    H, Wd = img.shape[:2]
    pos, neg = [], []
    for y0 in range(0, max(1, H - TILE + 1), stride):
        for x0 in range(0, max(1, Wd - TILE + 1), stride):
            patch = img[y0:y0 + TILE, x0:x0 + TILE]
            if patch.shape[:2] != (TILE, TILE):
                continue
            if patch.max() == 0:          # 타일 밖 검은 영역
                continue
            lab = []
            for b in boxes:
                cx, cy = b[:, 0].mean(), b[:, 1].mean()
                if not (x0 - TILE < cx < x0 + 2 * TILE and y0 - TILE < cy < y0 + 2 * TILE):
                    continue
                if inside_ratio(b, x0, y0, TILE) >= MIN_INSIDE:
                    lab.append(b - [x0, y0])
            (pos if lab else neg).append((patch, lab))
    if keep_empty_ratio > 0 and neg:
        k = min(len(neg), int(len(pos) * keep_empty_ratio))
        idx = rng.choice(len(neg), size=k, replace=False) if k else []
        neg = [neg[i] for i in idx]
    else:
        neg = []
    return pos + neg


def write_split(items, root, split, prefix, counter):
    os.makedirs(f"{root}/images/{split}", exist_ok=True)
    os.makedirs(f"{root}/labels/{split}", exist_ok=True)
    n_obj = 0
    for patch, lab in items:
        stem = f"{prefix}_{counter[0]:06d}"
        counter[0] += 1
        cv2.imwrite(f"{root}/images/{split}/{stem}.png", patch)
        with open(f"{root}/labels/{split}/{stem}.txt", "w") as f:
            for b in lab:
                c = np.clip(b, 0, TILE - 1) / TILE      # YOLO-OBB: 정규화 4점
                f.write("0 " + " ".join(f"{v:.6f}" for v in c.reshape(-1)) + "\n")
                n_obj += 1
    return n_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpkg", default="C:/Users/seung/datasets/S2Ships")
    ap.add_argument("--scenes", default="C:/Users/seung/datasets/S2Ships/scenes.json")
    ap.add_argument("--out", default="C:/Users/seung/datasets/S2Ships/yolo")
    ap.add_argument("--stride", type=int, default=256)
    ap.add_argument("--empty-ratio", type=float, default=0.65,
                    help="선박 있는 타일 대비 남길 빈 타일 비율")
    ap.add_argument("--val-tile", default="34VER", help="이 타일은 val 로 뺀다")
    ap.add_argument("--test-tile", default="34WFT", help="이 타일은 test 로 뺀다")
    a = ap.parse_args()

    rng = np.random.default_rng(0)
    ann = load_annotations(a.gpkg)
    scenes = resolve_scenes(ann, a.scenes)
    print(f"주석 장면 {len(ann)}개 · 총 폴리곤 {sum(len(g) for g in ann.values()):,}개\n")

    counter = [0]
    stats = {}
    for (tile, date), gdf in sorted(ann.items()):
        key = f"{tile}_{date}"
        if key not in scenes:
            print(f"  {key}  장면 없음 — 건너뜀")
            continue
        href = scenes[key]["assets"].get("visual")
        if not href:
            print(f"  {key}  visual(TCI) 자산 없음 — 건너뜀")
            continue

        split = "val" if tile == a.val_tile else ("test" if tile == a.test_tile else "train")
        # 학습은 겹쳐 잘라 표본을 늘리지만, 평가는 겹치면 같은 배를 여러 번 세게 된다.
        # 겹침 잘라내기로 만든 val/test 는 인스턴스가 1.5배로 부풀어 성능이 왜곡된다.
        stride = a.stride if split == "train" else TILE
        bounds = scene_window(gdf)
        arr, affine, crs = fetch(href, bounds)
        img = np.transpose(arr, (1, 2, 0))[:, :, ::-1]      # RGB -> BGR (cv2 저장용)
        boxes = polys_to_obb(gdf, affine)
        items = cut_tiles(img, boxes, stride, a.empty_ratio, rng)
        n_obj = write_split(items, a.out, split, key, counter)
        n_pos = sum(1 for _, l in items if l)
        stats.setdefault(split, [0, 0, 0])
        stats[split][0] += len(items); stats[split][1] += n_pos; stats[split][2] += n_obj
        print(f"  {key:<16} {split:<5} 사진 {img.shape[1]}x{img.shape[0]}  "
              f"폴리곤 {len(gdf):>4}  타일 {len(items):>4} (선박 {n_pos:>4})  라벨 {n_obj:>4}")

    print("\n%-6s %8s %8s %8s" % ("split", "타일", "선박있음", "인스턴스"))
    for k, v in stats.items():
        print("%-6s %8d %8d %8d" % (k, v[0], v[1], v[2]))

    os.makedirs(a.out, exist_ok=True)
    with open(f"{a.out}/s2ships.yaml", "w") as f:
        f.write(f"path: {a.out}\ntrain: images/train\nval: images/val\n"
                f"test: images/test\nnames:\n  0: vessel\n")
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
