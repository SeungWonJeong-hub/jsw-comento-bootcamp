"""GFW 선박 탐지로 학습셋 만들기 — 코멘토 3차 업무 / 정승원

핀란드 데이터의 한계
--------------------
맑은 발트해에서 사람이 손으로 그린 박스였다. 한국(특히 서해)에 가져가니
탁도 때문에 눈에 보이는 배를 놓쳤고, 박스도 뭉툭해서(길이 53m 인데 폭 43m)
회전 정보가 사실상 없었다.

GFW 가 주는 것
--------------
  점(lat, lon) + 길이(m) + 방향(도) + 선박확률 + 인프라 여부 + AIS 매칭
  황해 103만건 / 동중국해 67만건 / 한국 44만건, 전부 Sentinel-2 10 m

길이와 방향이 있으니 회전 박스를 계산으로 만든다. 추정할 필요가 없다.
그리고 AIS 매칭이 57~77% — 사람 판단이 아니라 선박 자신의 송신으로 검증된 정답이다.

어떻게 쓰는가
-------------
  train  황해 + 동중국해   탁수·소형선·양식장·항만 구조물을 배운다
  val    한국 (일부 장면)
  test   한국 (다른 장면)  ← 학습에 안 쓴 한국 장면으로만 평가한다

주의할 점
---------
  - 인프라(likely_infrastructure)는 배가 아니다. 라벨에서 빼되, 그 타일은
    음성 표본으로 남긴다. "여기 이렇게 생긴 건 배가 아니다"를 배워야 한다.
  - 선박확률이 낮은 탐지는 GFW 자신도 확신하지 않는다. 하한을 둔다.
  - 얼음(potential_ice)도 뺀다.
  - 한 장면에 최대 2,000 탐지가 있다. 전부 쓰면 몇 장면이 데이터를 지배하므로
    장면당 상한을 둔다.
"""
import os, csv, json, math, random, argparse, collections
import numpy as np
import cv2

TILE = 320
STAC = "https://earth-search.aws.element84.com/v1/search"


def load_detections(path, min_presence, max_per_scene, rng):
    """CSV → {scene_id: [(lon, lat, length_m, heading_deg, is_infra)]}

    주의 — max_per_scene 은 기본적으로 쓰지 않는다 (0 = 무제한).

    처음에는 "몇 장면이 데이터를 지배하지 않게" 장면당 60개로 무작위 추출했다.
    그런데 버려진 배는 사진에서 사라지지 않는다. 타일 안에 그대로 있는데
    라벨만 없어져, 학습에서 "저건 배가 아니다"로 배운다.
    실제로 탐지의 80%를 버렸고 재현율이 0.13~0.24 로 무너졌다.
    라벨 정렬은 97% 로 멀쩡했기 때문에 정렬 검사로는 안 잡히는 종류의 오류였다.

    장면 수로 규모를 조절하고, 고른 장면 안의 탐지는 전부 쓴다.
    """
    by_scene = collections.defaultdict(list)
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                p = float(r["presence_score"])
            except Exception:
                continue
            if str(r.get("potential_ice", "")).lower() == "true":
                continue
            infra = str(r.get("likely_infrastructure", "")).lower() == "true"
            if not infra and p < min_presence:
                continue
            try:
                lon, lat = float(r["lon"]), float(r["lat"])
                L = float(r["length_m_inferred"]) if r.get("length_m_inferred") else 30.0
                H = float(r["heading_deg_inferred"]) if r.get("heading_deg_inferred") else 0.0
            except Exception:
                continue
            by_scene[r["scene_id"]].append((lon, lat, L, H, infra))
    if not max_per_scene:                      # 0 = 무제한 (기본)
        return dict(by_scene)
    out = {}
    for k, v in by_scene.items():
        if len(v) > max_per_scene:
            v = [v[i] for i in rng.choice(len(v), max_per_scene, replace=False)]
        out[k] = v
    return out


def parse_scene(scene_id):
    """S2B_MSIL1C_20220215T021339_N0400_R060_T51PXN_... -> (타일, 날짜)"""
    parts = scene_id.split("_")
    tile, date = None, None
    for p in parts:
        if p.startswith("T") and len(p) == 6 and p[1:3].isdigit():
            tile = p[1:]
        if len(p) >= 8 and p[:8].isdigit() and "T" in p:
            date = p[:8]
    return tile, date


def resolve_cog(scene_ids, cache_path):
    """장면 ID → AWS 공개 COG (L2A TCI). L1C 자산은 요청자 부담 S3 라 무료로 못 받는다."""
    import requests
    cache = json.load(open(cache_path, encoding="utf-8")) if os.path.exists(cache_path) else {}
    todo = [s for s in scene_ids if s not in cache]
    print(f"COG 주소 확인 필요 {len(todo)} / 전체 {len(scene_ids)}")
    for i, sid in enumerate(todo, 1):
        tile, date = parse_scene(sid)
        if not (tile and date):
            cache[sid] = None
            continue
        d = f"{date[:4]}-{date[4:6]}-{date[6:]}"
        q = {"collections": ["sentinel-2-l2a"],
             "datetime": f"{d}T00:00:00Z/{d}T23:59:59Z",
             "query": {"grid:code": {"eq": "MGRS-" + tile}}, "limit": 3}
        try:
            fs = requests.post(STAC, json=q, timeout=60).json().get("features", [])
            fs = [f for f in fs if "visual" in f["assets"]
                  and f["assets"]["visual"]["href"].startswith("https")]
            cache[sid] = fs[0]["assets"]["visual"]["href"] if fs else None
        except Exception:
            cache[sid] = None
        if i % 25 == 0:
            print(f"  {i}/{len(todo)}", flush=True)
            json.dump(cache, open(cache_path, "w"), indent=0)
    json.dump(cache, open(cache_path, "w"), indent=0)
    return cache


def obb_from(cx, cy, length_px, heading_deg, width_ratio=0.28, min_w=2.0):
    """중심 + 길이 + 방향 → 회전 박스 4점.

    폭은 주지 않으므로 길이에 비례해 잡는다 (선박 길이:폭 = 대략 1:0.2~0.3).
    아주 작은 배는 최소 폭을 준다 — 10 m 해상도에서 0 폭 박스는 의미가 없다.
    방향은 북쪽 기준 시계방향(항해 관례)이므로 영상 좌표계로 바꾼다.
    """
    w = max(min_w, length_px * width_ratio)
    th = math.radians(90.0 - heading_deg)      # 북 기준 -> x축 기준
    dx, dy = math.cos(th), -math.sin(th)       # 영상은 y 가 아래로 증가
    px, py = -dy, dx
    hl, hw = length_px / 2, w / 2
    return np.array([
        [cx + dx * hl + px * hw, cy + dy * hl + py * hw],
        [cx + dx * hl - px * hw, cy + dy * hl - py * hw],
        [cx - dx * hl - px * hw, cy - dy * hl - py * hw],
        [cx - dx * hl + px * hw, cy - dy * hl + py * hw],
    ], np.float32)


def cut_scene(href, dets, out_root, split, prefix, counter,
              neg_ratio, rng, pad=400):
    """한 장면에서 탐지 주변 타일을 잘라 라벨과 함께 저장."""
    import rasterio
    from rasterio.warp import transform
    from rasterio.windows import from_bounds, Window
    for k, v in dict(GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR", AWS_NO_SIGN_REQUEST="YES",
                     GDAL_HTTP_MAX_RETRY="2").items():
        os.environ.setdefault(k, v)

    lons = [d[0] for d in dets]; lats = [d[1] for d in dets]
    n_pos = n_obj = 0
    with rasterio.open(href) as s:
        xs, ys = transform("EPSG:4326", s.crs, lons, lats)
        px = [(~s.transform) * (x, y) for x, y in zip(xs, ys)]
        # 탐지를 타일 격자로 묶는다 (한 탐지마다 자르면 겹침이 심해진다)
        buckets = collections.defaultdict(list)
        for (cx, cy), d in zip(px, dets):
            buckets[(int(cy) // TILE, int(cx) // TILE)].append((cx, cy, d))
        for (gy, gx), items in buckets.items():
            y0, x0 = gy * TILE, gx * TILE
            if not (0 <= x0 < s.width - TILE and 0 <= y0 < s.height - TILE):
                continue
            try:
                arr = s.read(window=Window(x0, y0, TILE, TILE))
            except Exception:
                continue
            if arr.shape[1:] != (TILE, TILE) or arr.max() == 0:
                continue
            img = np.transpose(arr, (1, 2, 0))[:, :, ::-1]
            labels = []
            for cx, cy, d in items:
                lon, lat, L, H, infra = d
                if infra:
                    continue                       # 인프라는 라벨에서 뺀다
                box = obb_from(cx - x0, cy - y0, L / 10.0, H)
                if box.min() < -8 or box.max() > TILE + 8:
                    continue
                labels.append(np.clip(box, 0, TILE - 1))
            stem = f"{prefix}_{counter[0]:07d}"
            counter[0] += 1
            os.makedirs(f"{out_root}/images/{split}", exist_ok=True)
            os.makedirs(f"{out_root}/labels/{split}", exist_ok=True)
            cv2.imwrite(f"{out_root}/images/{split}/{stem}.png", img)
            with open(f"{out_root}/labels/{split}/{stem}.txt", "w") as f:
                for b in labels:
                    f.write("0 " + " ".join(f"{v:.6f}" for v in (b / TILE).reshape(-1)) + "\n")
            if labels:
                n_pos += 1
                n_obj += len(labels)
    return n_pos, n_obj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gfw", default="C:/Users/seung/datasets/GFW")
    ap.add_argument("--out", default="C:/Users/seung/datasets/GFWShips/yolo")
    ap.add_argument("--min-presence", type=float, default=0.8)
    ap.add_argument("--max-per-scene", type=int, default=0,
                help="장면당 탐지 상한. 0=무제한. 상한을 걸면 버려진 배가 "
                     "라벨 없는 배로 남아 재현율이 무너진다")
    ap.add_argument("--train-scenes", type=int, default=110)
    ap.add_argument("--korea-scenes", type=int, default=45)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rng = np.random.default_rng(a.seed)
    os.makedirs(a.out, exist_ok=True)
    counter = [0]
    stats = {}

    plan = [("train", ["yellow_sea", "echina"], a.train_scenes),
            ("val",   ["korea"], a.korea_scenes // 3),
            ("test",  ["korea"], a.korea_scenes - a.korea_scenes // 3)]

    used_scenes = set()
    for split, regions, n_scenes in plan:
        pool = {}
        for reg in regions:
            d = load_detections(f"{a.gfw}/{reg}.csv", a.min_presence, a.max_per_scene, rng)
            pool.update({k: v for k, v in d.items() if k not in used_scenes})
        # 탐지가 많은 장면 위주로 고르되 다양성을 위해 상위에서 무작위 추출
        cand = sorted(pool, key=lambda k: -len(pool[k]))[:n_scenes * 4]
        pick = [cand[i] for i in rng.choice(len(cand), min(n_scenes, len(cand)), replace=False)]
        used_scenes.update(pick)
        cog = resolve_cog(pick, f"{a.gfw}/cog_cache.json")
        ok = [s for s in pick if cog.get(s)]
        print(f"\n[{split}] 장면 {len(ok)}/{len(pick)} 사용 가능")
        tp = to = 0
        for i, sid in enumerate(ok, 1):
            try:
                p, o = cut_scene(cog[sid], pool[sid], a.out, split, sid[:24], counter,
                                 0.0, rng)
                tp += p; to += o
            except Exception as e:
                print(f"  {sid[:30]} 실패: {type(e).__name__}")
            if i % 20 == 0:
                print(f"  {i}/{len(ok)}  타일 {tp}  라벨 {to}", flush=True)
        stats[split] = dict(scenes=len(ok), tiles_pos=tp, objects=to)
        print(f"[{split}] 완료 — 선박 있는 타일 {tp}, 인스턴스 {to}")

    open(f"{a.out}/gfw.yaml", "w").write(
        f"path: {a.out}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: vessel\n")
    json.dump(stats, open(f"{a.out}/build_stats.json", "w"), indent=2, ensure_ascii=False)
    print("\n", json.dumps(stats, indent=2, ensure_ascii=False))
    print("저장:", a.out)


if __name__ == "__main__":
    main()
