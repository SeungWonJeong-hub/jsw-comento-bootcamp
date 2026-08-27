"""한국 항만 정량 평가 — 코멘토 3차 업무 / 정승원

지금까지 한국 결과는 정답이 없어 '탐지가 물 위에 있는 비율'로 간접 추정했다.
GFW 가 한국 근해 443,344건을 주므로 이제 진짜 수치를 잰다.

절차
----
1. 항만 사각형 안에서, 그 장면과 같은 날짜의 GFW 탐지를 정답으로 모은다
2. 같은 장면 사진을 받아 모델을 돌린다
3. 점 기반으로 맞춘다 — 3 픽셀 물체에 IoU 를 요구하는 것은 지표의 함정이고,
   GFW 정답이 애초에 점이라 점 기반이 정답의 성격에 맞다

주의
----
GFW 도 완벽한 정답이 아니다. 딥러닝 탐지 결과이고, 자체 선박확률이 붙어 있다.
그래서 확률 하한을 두고, 인프라·얼음은 뺀다. '정답'이라기보다 '독립적인 참조'다.
이 한계는 결과에 명시한다.
"""
import os, csv, json, math, argparse, collections
import numpy as np
import cv2


def load_gfw(path, bbox, date=None, min_presence=0.8):
    """항만 사각형(+날짜) 안의 GFW 탐지."""
    lon0, lat0, lon1, lat1 = bbox
    out = []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            try:
                lon, lat = float(r["lon"]), float(r["lat"])
            except Exception:
                continue
            if not (lon0 <= lon <= lon1 and lat0 <= lat <= lat1):
                continue
            if str(r.get("likely_infrastructure", "")).lower() == "true":
                continue
            if str(r.get("potential_ice", "")).lower() == "true":
                continue
            try:
                if float(r["presence_score"]) < min_presence:
                    continue
            except Exception:
                continue
            ts = (r.get("detect_timestamp") or "")[:10].replace("-", "")
            if date and ts != date:
                continue
            L = float(r["length_m_inferred"]) if r.get("length_m_inferred") else 30.0
            out.append((lon, lat, L, ts, bool(r.get("mmsi"))))
    return out


def match(preds, gts, radius_px):
    """신뢰도 순 탐욕 매칭. (맞았나, 신뢰도, 거리, 정답 길이)"""
    used, recs = set(), []
    for cx, cy, cf in sorted(preds, key=lambda p: -p[2]):
        best, bi = 1e9, -1
        for i, (gx, gy, gl) in enumerate(gts):
            if i in used:
                continue
            d = math.hypot(cx - gx, cy - gy)
            if d < best:
                best, bi = d, i
        if bi >= 0 and best <= max(radius_px, 0.5 * gts[bi][2] / 10.0):
            used.add(bi)
            recs.append((1, cf, best, gts[bi][2]))
        else:
            recs.append((0, cf, float('nan'), float('nan')))
    for i, g in enumerate(gts):
        if i not in used:
            recs.append((0, -1.0, float('nan'), g[2]))
    return recs


def main():
    import sys
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--gfw", default="C:/Users/seung/datasets/GFW/korea.csv")
    ap.add_argument("--ports", nargs="+",
                    default=["busan_anchorage", "tongyeong", "yeosu", "gwangyang"])
    ap.add_argument("--conf", type=float, default=0.3)
    ap.add_argument("--radius-px", type=float, default=5.0)
    ap.add_argument("--max-cloud", type=float, default=10)
    ap.add_argument("--start", default="2022-01-01")
    ap.add_argument("--end", default="2022-12-31")
    ap.add_argument("--out", default="outputs/korea_quant.json")
    a = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from korea_ports import PORTS, find_scene, fetch_window, detect, to_lonlat
    from ultralytics import YOLO
    from rasterio.warp import transform as _t

    model = YOLO(a.weights)
    summary = {}
    for name in a.ports:
        label, *bbox = PORTS[name]
        # GFW 는 2022년 자료이므로 그 해 장면을 쓴다
        sc = find_scene(bbox, a.max_cloud, start=a.start, end=a.end)
        if not sc:
            print(f"{label:<16} 장면 없음")
            continue
        date = sc["datetime"][:10].replace("-", "")
        gts_ll = load_gfw(a.gfw, bbox, date)
        if not gts_ll:
            print(f"{label:<16} {sc['datetime'][:10]}  그 날짜 GFW 정답 없음 — 건너뜀")
            continue

        img, affine, crs = fetch_window(sc["visual"], bbox)
        inv = ~affine
        xs, ys = _t("EPSG:4326", crs, [g[0] for g in gts_ll], [g[1] for g in gts_ll])
        gts = []
        for (X, Y), g in zip(zip(xs, ys), gts_ll):
            cx, cy = inv * (X, Y)
            if 0 <= cx < img.shape[1] and 0 <= cy < img.shape[0]:
                gts.append((cx, cy, g[2]))
        if not gts:
            print(f"{label:<16} 창 안에 정답 없음")
            continue

        dets = detect(img, model, conf=a.conf)
        preds = [(p[:, 0].mean(), p[:, 1].mean(), c) for p, c in dets]
        recs = match(preds, gts, a.radius_px)

        det = [r for r in recs if r[1] >= 0]
        tp = sum(r[0] for r in det)
        prec = tp / len(det) if det else 0.0
        rec = tp / len(gts) if gts else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        err = [r[2] for r in recs if r[0] == 1]
        ais = sum(1 for g in gts_ll if g[4]) / len(gts_ll)
        print(f"{label:<16} {sc['datetime'][:10]}  GFW정답 {len(gts):>4}  탐지 {len(det):>4}  "
              f"P {prec:.3f}  R {rec:.3f}  F1 {f1:.3f}  "
              f"중심오차 {np.median(err) if err else float('nan'):.2f}px  AIS {100*ais:.0f}%")
        summary[name] = dict(label=label, scene=sc["id"], datetime=sc["datetime"],
                             n_gt=len(gts), n_det=len(det), precision=prec,
                             recall=rec, f1=f1,
                             center_err_px=float(np.median(err)) if err else None,
                             ais_ratio=ais, conf=a.conf)

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(summary, open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {a.out}")
    print("주의: GFW 도 딥러닝 탐지 결과이므로 완전한 정답이 아니다. 독립적인 참조로 본다.")


if __name__ == "__main__":
    main()
