# -*- coding: utf-8 -*-
"""HRSC2016 XML 메타데이터 전수 조사.

    py scripts/inspect_metadata.py

추측하지 않습니다. XML 이 실제로 담고 있는 것만 집계하고, 좌표를 근접
병합으로 묶어 군집별 표본 수·선박 수를 냅니다. 군집에 항구 이름을 붙이는
것은 다음 단계(build_port_manifest.py)에서 근거와 함께 합니다.

실측 결과 (2026-09-04)
----------------------
    XML 1,681개 · 파싱 실패 0
    좌표 보유 100% · Img_Resolution 보유 99.9%
    선박 2,977척 · Annotated=1 인 영상 1,070장
    좌표 군집 7개 (무르만스크 2개를 합치면 6개 -> 공식문서의 'six famous harbors')
"""
import os
import glob
import argparse
import collections
import xml.etree.ElementTree as ET

from common import (ROOT, load_cfg, rel, haversine_km, true_gsd, parse_zoom,
                    write_csv, dump_json)


def _t(node, tag):
    e = node.find(tag)
    return "" if e is None or e.text is None else e.text.strip()


def parse_one(p):
    r = ET.parse(p).getroot()
    objs = r.findall("./HRSC_Objects/HRSC_Object")
    loc = _t(r, "Img_Location")
    lat = lon = None
    if "," in loc:
        try:
            lat, lon = [float(v) for v in loc.split(",")[:2]]
        except ValueError:
            lat = lon = None
    try:
        field_gsd = float(_t(r, "Img_Resolution"))
    except ValueError:
        field_gsd = None
    zoom = parse_zoom(_t(r, "Img_Resolution_Layer"))
    return dict(
        img_id=_t(r, "Img_ID"), file_name=_t(r, "Img_FileName"),
        place_id=_t(r, "Place_ID"), source_id=_t(r, "Source_ID"),
        date=_t(r, "Img_Date"), cus_type=_t(r, "Img_CusType"),
        raw_location=loc, lat=lat, lon_abs=lon,
        field_gsd=field_gsd, zoom=zoom,
        width=int(_t(r, "Img_SizeWidth") or 0),
        height=int(_t(r, "Img_SizeHeight") or 0),
        annotated=_t(r, "Annotated"), n_ships=len(objs),
        gsd_formula=(true_gsd(lat, zoom) if lat is not None else None),
    )


def merge_clusters(rows, eps_km=25.0):
    cents = []
    for r in rows:
        if r["lat"] is None:
            r["cluster"] = -1
            continue
        best, bd = None, 1e9
        for i, c in enumerate(cents):
            d = haversine_km(r["lat"], r["lon_abs"], c["lat"], c["lon"])
            if d < bd:
                best, bd = i, d
        if best is not None and bd <= eps_km:
            c = cents[best]
            n = c["n"]
            c["lat"] = (c["lat"] * n + r["lat"]) / (n + 1)
            c["lon"] = (c["lon"] * n + r["lon_abs"]) / (n + 1)
            c["n"] = n + 1
            r["cluster"] = best
        else:
            cents.append(dict(lat=r["lat"], lon=r["lon_abs"], n=1))
            r["cluster"] = len(cents) - 1
    return cents


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "metadata"))
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    ann_dir = rel(cfg, "annotations")

    files = sorted(glob.glob(os.path.join(ann_dir, "*.xml")))
    if not files:
        raise SystemExit("XML 이 없습니다: %s\n먼저 download_hrsc2016.py 를 돌리세요."
                         % ann_dir)
    rows, bad = [], []
    for p in files:
        try:
            rows.append(parse_one(p))
        except Exception as e:
            bad.append((p, str(e)))

    n = len(rows)
    print("XML %d개 · 파싱 실패 %d개" % (n, len(bad)))
    print("좌표 보유 %d (%.1f%%) · Img_Resolution 보유 %d (%.1f%%)" % (
        sum(1 for r in rows if r["lat"] is not None),
        100.0 * sum(1 for r in rows if r["lat"] is not None) / n,
        sum(1 for r in rows if r["field_gsd"]),
        100.0 * sum(1 for r in rows if r["field_gsd"]) / n))
    print("선박 %d척 · Annotated=1 영상 %d장" % (
        sum(r["n_ships"] for r in rows),
        sum(1 for r in rows if r["annotated"] == "1")))

    fg = [r["field_gsd"] for r in rows if r["field_gsd"]]
    tg = [r["gsd_formula"] for r in rows if r["gsd_formula"]]
    if fg and tg:
        print("\nGSD 비교 — XML 필드 vs 위도보정 계산식")
        print("  필드값   %.2f ~ %.2f  (중앙 %.2f)" % (min(fg), max(fg),
                                                      sorted(fg)[len(fg) // 2]))
        print("  계산식   %.2f ~ %.2f  (중앙 %.2f)" % (min(tg), max(tg),
                                                      sorted(tg)[len(tg) // 2]))
        print("  -> 필드값은 위도 보정이 없는 명목값입니다. 계산식을 씁니다.")

    cents = merge_clusters(rows)
    per = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        d = per[r["cluster"]]
        d["img"] += 1
        d["ships"] += r["n_ships"]
        d["ann"] += 1 if r["annotated"] == "1" else 0

    print("\n좌표 군집 %d개 (25 km 기준)" % len(cents))
    print("| 군집 | 위도 | 경도(부호없음) | 영상 | 선박 | Annotated |")
    print("|---|---:|---:|---:|---:|---:|")
    for c in sorted((k for k in per if k >= 0), key=lambda k: -per[k]["img"]):
        print("| %d | %.4f | %.4f | %d | %d | %d |" % (
            c, cents[c]["lat"], cents[c]["lon"],
            per[c]["img"], per[c]["ships"], per[c]["ann"]))

    write_csv(os.path.join(a.out, "images_raw.csv"), rows)
    dump_json(os.path.join(a.out, "clusters.json"),
              [dict(cluster=c, **cents[c], images=per[c]["img"],
                    ships=per[c]["ships"]) for c in range(len(cents))])
    print("\n저장:", a.out)


if __name__ == "__main__":
    main()
