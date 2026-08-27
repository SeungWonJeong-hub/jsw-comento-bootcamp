"""Global Fishing Watch Sentinel-2 선박 탐지 — 관심 해역만 뽑아내기
코멘토 3차 업무 / 정승원

왜 이 데이터인가
----------------
핀란드 데이터로 학습한 모델을 한국에 가져오니 인천에서 눈에 보이는 배를 놓쳤다.
발트해는 맑고 서해는 탁해서, "어두운 물 위 밝은 배"라는 전제가 깨지기 때문이다.
그러면 탁한 물에서 라벨된 데이터가 필요한데, GFW 가 그걸 전 지구로 갖고 있다.

  Sentinel-2 10 m · 2019년~현재 · 대부분의 EEZ 와 해양보호구역
  탐지마다 위치 · 길이 · 방향 · 속도 · 선박일 확률
  Zenodo 16363632 (2022년 연간)

길이와 방향이 있으니 회전 박스를 직접 만들 수 있다. 폴리곤에서 추정할 필요가 없다.

문제와 해법
----------
탐지 파일이 6 GB 다. 그런데 우리가 쓸 건 한국 근해뿐이다.
전부 받아 저장하지 말고, 스트리밍으로 읽으면서 관심 사각형 안의 행만 남긴다.
디스크에는 필요한 것만 떨어진다.
"""
import os, csv, io, json, argparse, time

URL = ("https://zenodo.org/api/records/16363632/files/"
       "sentinel2_vessel_detections_pipev3_2022.csv/content")

# 관심 해역 (lon0, lat0, lon1, lat1)
REGIONS = {
    "korea":      (124.0, 32.5, 132.0, 39.0),   # 한반도 전 해역
    "yellow_sea": (119.0, 30.0, 127.0, 41.0),   # 황해 — 한중 공유, 탁도 높음
    "echina":     (117.0, 22.0, 128.0, 33.0),   # 동중국해
    "sea_asia":   (95.0, -11.0, 128.0, 24.0),   # 동남아 (베트남·인니·필리핀)
}


def stream_filter(url, boxes, out_dir, chunk=1 << 20, progress_every=200):
    """CSV 를 흘려 읽으며 관심 사각형 안의 행만 파일로 뺀다."""
    import requests
    os.makedirs(out_dir, exist_ok=True)
    r = requests.get(url, stream=True, timeout=300)
    r.raise_for_status()
    total = int(r.headers.get("content-length", 0))

    writers, files, counts = {}, {}, {k: 0 for k in boxes}
    header = None
    buf = ""
    seen = 0
    read = 0
    t0 = time.time()

    for raw in r.iter_content(chunk_size=chunk):
        read += len(raw)
        buf += raw.decode("utf-8", errors="replace")
        lines = buf.split("\n")
        buf = lines.pop()                      # 마지막 조각은 다음 덩어리와 이어 붙인다
        for ln in lines:
            if header is None:
                header = next(csv.reader([ln]))
                ilat, ilon = header.index("lat"), header.index("lon")
                for k in boxes:
                    f = open(f"{out_dir}/{k}.csv", "w", newline="", encoding="utf-8")
                    files[k] = f
                    writers[k] = csv.writer(f)
                    writers[k].writerow(header)
                continue
            if not ln.strip():
                continue
            try:
                row = next(csv.reader([ln]))
                lat, lon = float(row[ilat]), float(row[ilon])
            except Exception:
                continue
            seen += 1
            for k, (x0, y0, x1, y1) in boxes.items():
                if x0 <= lon <= x1 and y0 <= lat <= y1:
                    writers[k].writerow(row)
                    counts[k] += 1
        if total and read // (progress_every << 20) != (read - len(raw)) // (progress_every << 20):
            print(f"  {read/1e9:.1f} / {total/1e9:.1f} GB  "
                  f"({100*read/total:.0f}%)  읽은 행 {seen:,}  "
                  f"{dict(counts)}  {time.time()-t0:.0f}s", flush=True)

    for f in files.values():
        f.close()
    return header, seen, counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", nargs="+", default=["korea", "yellow_sea", "echina"],
                    choices=list(REGIONS))
    ap.add_argument("--out", default="C:/Users/seung/datasets/GFW")
    ap.add_argument("--url", default=URL)
    a = ap.parse_args()

    boxes = {k: REGIONS[k] for k in a.regions}
    print("관심 해역:")
    for k, b in boxes.items():
        print(f"  {k:<12} lon {b[0]}~{b[2]}  lat {b[1]}~{b[3]}")
    print("\n스트리밍 시작 (6 GB, 저장은 걸러낸 행만)")
    header, seen, counts = stream_filter(a.url, boxes, a.out)

    print(f"\n전체 탐지 {seen:,}건")
    for k, n in counts.items():
        print(f"  {k:<12} {n:>9,}건  ->  {a.out}/{k}.csv")
    json.dump({"regions": boxes, "total_rows": seen, "counts": counts,
               "header": header},
              open(f"{a.out}/summary.json", "w", encoding="utf-8"),
              indent=2, ensure_ascii=False)


if __name__ == "__main__":
    main()
