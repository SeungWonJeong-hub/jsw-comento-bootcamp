# -*- coding: utf-8 -*-
"""좌표 -> 항구 매핑과 manifests/images.csv · split.csv 생성.

    py pipeline/build_port_manifest.py
    py pipeline/build_port_manifest.py --verify-gsd

무엇을 근거로 항구를 정하는가
-----------------------------
XML 의 `Img_Location` 은 데이터셋이 제공한 사실입니다. 다만 **경도 부호가
없습니다**(전부 양수). 그대로 동경으로 읽으면 샌디에이고 좌표가 중국 내륙이
되어 항구가 있을 수 없습니다. 그래서 동경/서경 두 해석을 모두 시험하고,
알려진 항구 좌표와 25 km 안에 드는 것만 인정합니다.

  coord_match       25 km 안에 알려진 항구가 있음
  coord_unmatched   좌표는 있으나 안 맞음 -> port = unknown_us_port
  no_coord          좌표 없음

**지형만 보고 항구를 정하지 않습니다.** 좌표가 1차 근거이고, 지형은 사람이
눈으로 확인하는 보조 수단입니다(results/label_check, port_spotcheck.png).
"""
import os
import glob
import argparse
import collections
import xml.etree.ElementTree as ET

from common import (ROOT, PORTS, load_cfg, rel, true_gsd, parse_zoom,
                    identify_port, write_csv, dump_json)

# GSD 검증용.
#
# 종횡비만으로 함급을 자동 판별하는 방식은 **쓰지 않습니다.** 실제로 돌려보니
# 에버렛의 4:1 상자를 Nimitz 항모로 잡는 등 엉뚱한 짝이 대량으로 나옵니다.
# 종횡비는 식별이 아닙니다. 대신 사람이 영상을 직접 보고 확인한 몇 척만
# 하드코딩해 씁니다 — 표본은 적지만 각 항이 근거를 갖습니다.
#
#   (image_id, 함급, 실제 길이 m, 확인 방법)
CONFIRMED_HULLS = [
    ("100001155", "Nimitz class CVN", 332.8, "샌디에이고, 갑판 번호까지 보이는 항모"),
    ("100000873", "Arleigh Burke class DDG", 155.3, "노퍽, 부두에 나란한 구축함 3척"),
    ("100000637", "Wasp class LHD", 257.0, "에버렛, 5.2:1 비율의 강습상륙함"),
]
LONGEST_US_WARSHIP_M = 342.0        # USS Enterprise (CVN-65)
NIMITZ_LEN_M = 332.8                # 현역 최장 — p99 가 여기 맞아야 합니다

FIELDS = ["image_id", "source_image_id", "port", "country", "original_gsd",
          "target_gsd", "modality", "degradation", "upscale_method",
          "label_type", "split", "verification_status",
          "lat", "lon", "zoom", "width", "height", "n_ships",
          "scene_type", "ground_w_m", "ground_h_m", "field_gsd", "date"]


def _t(node, tag):
    e = node.find(tag)
    return "" if e is None or e.text is None else e.text.strip()


def read_imagesets(d):
    out = {}
    for s in ("train", "val", "test"):
        p = os.path.join(d, "%s.txt" % s)
        if not os.path.exists(p):
            continue
        for line in open(p, encoding="utf-8"):
            line = line.strip()
            if line:
                out[line] = s
    return out


def verify_gsd(rows, ann_dir):
    """GSD 계산식을 물리적으로 검증합니다.

    (1) 길이 상한 — 가정이 거의 없는 1차 검사
        미 해군 최장 함정은 Enterprise 342 m 입니다. 계산식 GSD 로 잰
        선박이 이보다 크게 길면 GSD 가 과대평가된 것입니다.

    (2) 함급 역산 — 배율 추정
        종횡비가 특정 함급과 뚜렷하게 갈리는 단독선박 영상만 골라, 그 함급의
        실제 길이로 나눠 m/px 를 얻습니다. **미국 항구만** 씁니다. 무르만스크
        함정을 미 함급 목록에 맞추면 아무 뜻 없는 수가 나옵니다.
    """
    us = [r for r in rows if r["country"] == "US" and r["gsd_formula"]]
    print("")
    print("[GSD 검증] 미국 항구 %d장" % len(us))

    lens = []
    for r in us:
        p = os.path.join(ann_dir, "%s.xml" % r["file_name"])
        for o in ET.parse(p).getroot().findall("./HRSC_Objects/HRSC_Object"):
            L = max(float(_t(o, "mbox_w")), float(_t(o, "mbox_h")))
            if L > 0:
                lens.append(L * r["gsd_formula"])
    if not lens:
        print("  선박이 없습니다.")
        return
    lens.sort()

    def q(x):
        return lens[min(int(x * len(lens)), len(lens) - 1)]

    print("")
    print("(1) 계산식 GSD 로 잰 선박 장변(m)")
    print("    p50 %.0f · p90 %.0f · p99 %.0f · 최대 %.0f"
          % (q(.5), q(.9), q(.99), lens[-1]))
    over = sum(1 for v in lens if v > LONGEST_US_WARSHIP_M)
    print("    %.0f m(Enterprise) 초과: %d척 (%.1f%%)"
          % (LONGEST_US_WARSHIP_M, over, 100.0 * over / len(lens)))
    # p99 는 항모여야 합니다. 그 가정에서 배율을 추정합니다 — 표본 2,964척이라
    # 아래 (2)의 육안 3척보다 훨씬 안정적입니다.
    k = q(.99) / NIMITZ_LEN_M
    print("    p99(%.0f m)가 Nimitz(%.0f m)라고 보면 계산식은 %+.0f%% 입니다."
          % (q(.99), NIMITZ_LEN_M, 100 * (k - 1)))
    print("    -> 목표 10 m 는 실제로 약 %.1f m 입니다." % (10.0 / k))

    print("")
    print("(2) 눈으로 확인한 함정으로 배율 점검")
    print("| 영상 | 함급 | 최장 상자(px) | 역산 m/px | 계산식 m/px | 배율 |")
    print("|---|---|---:|---:|---:|---:|")
    by_id = {r["file_name"]: r for r in us}
    ratios = []
    for img_id, hull, hull_len, how in CONFIRMED_HULLS:
        r = by_id.get(img_id)
        if not r:
            continue
        p = os.path.join(ann_dir, "%s.xml" % img_id)
        best = 0.0
        for o in ET.parse(p).getroot().findall("./HRSC_Objects/HRSC_Object"):
            best = max(best, max(float(_t(o, "mbox_w")), float(_t(o, "mbox_h"))))
        if best <= 0:
            continue
        est = hull_len / best
        f = r["gsd_formula"]
        ratios.append(f / est)
        print("| %s | %s | %.0f | %.3f | %.3f | %.2f배 |"
              % (img_id, hull, best, est, f, f / est))
        print("|  |  |  |  |  | %s |" % how)

    if not ratios:
        print("  확인된 영상을 찾지 못했습니다.")
        return
    ratios.sort()
    m = ratios[len(ratios) // 2]
    print("")
    print("확인 표본 %d척 · 계산식/역산 중앙 %.2f배 (편차 큼 — 참고용)"
          % (len(ratios), m))
    print("표본이 3척뿐이라 배율 추정은 위 (1)의 p99 기준을 씁니다.")
    print("")
    print("이 오차는 모든 비교군(A/B/C)에 똑같이 걸리므로 비교를 왜곡하지 않습니다.")
    print("영향은 '목표 10 m' 라는 이름표의 정확도뿐입니다.")
    dump_json(os.path.join(ROOT, "results", "gsd_verification.json"),
              dict(n_ships=len(lens), p50_m=q(.5), p99_m=q(.99),
                   max_m=lens[-1], over_longest=over,
                   hull_samples=len(ratios), formula_over_hull=m,
                   effective_target_m=10.0 / m))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--verify-gsd", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    ann_dir = rel(cfg, "annotations")
    target = float(cfg["gsd"]["target_m"])
    use_formula = cfg["gsd"]["source"] == "formula"
    sets = read_imagesets(rel(cfg, "imagesets"))

    rows = []
    for p in sorted(glob.glob(os.path.join(ann_dir, "*.xml"))):
        r = ET.parse(p).getroot()
        loc = _t(r, "Img_Location")
        lat = lon = None
        if "," in loc:
            try:
                lat, lon = [float(v) for v in loc.split(",")[:2]]
            except ValueError:
                pass
        zoom = parse_zoom(_t(r, "Img_Resolution_Layer"))
        try:
            field_gsd = float(_t(r, "Img_Resolution"))
        except ValueError:
            field_gsd = None
        if lat is None:
            port, status, slon, gsd_f = "unknown_us_port", "no_coord", None, None
        else:
            port, status, slon = identify_port(lat, lon)
            gsd_f = true_gsd(lat, zoom)
        stem = _t(r, "Img_FileName")
        w = int(_t(r, "Img_SizeWidth") or 0)
        h = int(_t(r, "Img_SizeHeight") or 0)
        gsd = gsd_f if (use_formula and gsd_f) else field_gsd
        rows.append(dict(
            image_id=stem, source_image_id=stem,
            port=port, country=PORTS.get(port, {}).get("country", ""),
            original_gsd=("%.4f" % gsd) if gsd else "",
            target_gsd="%.2f" % target,
            modality="optical_rgb", degradation="none", upscale_method="none",
            label_type="obb+hbb", split=sets.get(stem, "unassigned"),
            verification_status=status,
            lat=lat, lon=slon, zoom=zoom, width=w, height=h,
            n_ships=len(r.findall("./HRSC_Objects/HRSC_Object")),
            scene_type=_t(r, "Img_CusType"),
            ground_w_m=("%.1f" % (w * gsd)) if gsd else "",
            ground_h_m=("%.1f" % (h * gsd)) if gsd else "",
            field_gsd=field_gsd, annotated=_t(r, "Annotated"),
            date=_t(r, "Img_Date"),          # XML 의 촬영일. 비어 있으면 비워 둡니다
            gsd_formula=gsd_f, file_name=stem))

    per = collections.defaultdict(lambda: collections.Counter())
    for r in rows:
        if r.get("annotated") != "1":
            continue
        d = per[r["port"]]
        d["img"] += 1
        d["ships"] += r["n_ships"]
        d[r["split"]] += 1
    print("[항구별 표본]  (Annotated=1 만)")
    print("| 항구 | 영상 | 선박 | train | val | test | 비고 |")
    print("|---|---:|---:|---:|---:|---:|---|")
    tot = collections.Counter()
    for k in sorted(per, key=lambda k: -per[k]["img"]):
        d = per[k]
        note = "표본 부족" if d["img"] < 40 else ""
        print("| %s | %d | %d | %d | %d | %d | %s |"
              % (PORTS.get(k, {}).get("name", k), d["img"], d["ships"],
                 d["train"], d["val"], d["test"], note))
        for f in ("img", "ships", "train", "val", "test"):
            tot[f] += d[f]
    print("| **합계** | %d | %d | %d | %d | %d | |"
          % (tot["img"], tot["ships"], tot["train"], tot["val"], tot["test"]))
    print("")
    print("검증 상태: %s" % dict(collections.Counter(
        r["verification_status"] for r in rows)))

    pw = next(s for s in cfg["splits"] if s["id"] == "portwise")
    split_rows = []
    for r in rows:
        if r.get("annotated") != "1":
            continue
        if r["port"] in pw["test_ports"]:
            sp = "test"
        elif r["port"] in pw["val_ports"]:
            sp = "val"
        elif r["port"] in pw["train_ports"]:
            sp = "train"
        else:
            sp = "excluded"
        split_rows.append(dict(image_id=r["image_id"], port=r["port"],
                               official=r["split"], portwise=sp))
    mdir = rel(cfg, "manifests")
    write_csv(os.path.join(mdir, "images.csv"), rows, FIELDS)
    write_csv(os.path.join(mdir, "split.csv"), split_rows,
              ["image_id", "port", "official", "portwise"])
    dump_json(os.path.join(ROOT, "results", "port_summary.json"),
              {k: dict(per[k]) for k in per})
    print("저장: %s/images.csv · split.csv" % mdir)

    if a.verify_gsd:
        verify_gsd([r for r in rows if r.get("annotated") == "1"], ann_dir)


if __name__ == "__main__":
    main()
