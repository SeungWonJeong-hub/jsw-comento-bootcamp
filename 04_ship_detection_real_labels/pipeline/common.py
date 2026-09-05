# -*- coding: utf-8 -*-
"""스크립트 공통 유틸 — 설정 로드, 경로, 진짜 GSD 계산.

GSD 를 여기 한 곳에서만 계산합니다. XML 의 Img_Resolution 을 쓰지 않는
이유는 configs/experiment.yaml 에 적어 두었습니다.
"""
import os
import io
import sys
import csv
import json
import math

# 윈도 콘솔 기본 인코딩(cp949)은 em-dash 같은 글자에서 죽습니다. 모든
# 스크립트가 이 모듈을 import 하므로 여기서 한 번만 손봅니다.
for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# 항구 이름은 좌표 군집으로 식별한 것입니다. 데이터셋이 제공한 값이
# 아니므로 verification_status 로 근거를 남깁니다.
# 경도 부호가 XML 에 없어(전부 양수) 서반구는 부호를 뒤집었습니다.
PORTS = {
    "san_diego": dict(lat=32.6623, lon=-117.1211, country="US",
                      name="Naval Base San Diego, CA"),
    "norfolk":   dict(lat=36.9608, lon=-76.3286, country="US",
                      name="Norfolk Naval Base, VA"),
    "mayport":   dict(lat=30.3948, lon=-81.4097, country="US",
                      name="Mayport Naval Base, FL"),
    "everett":   dict(lat=47.9815, lon=-122.2277, country="US",
                      name="Naval Station Everett, WA"),
    "newport":   dict(lat=41.5283, lon=-71.3047, country="US",
                      name="Naval Station Newport, RI"),
    "murmansk":  dict(lat=69.10, lon=33.28, country="RU",
                      name="Murmansk / Severomorsk"),
}
PORT_MATCH_KM = 25.0


def results_dir():
    """결과 폴더. 스모크 테스트는 results/smoke/ 로 격리해 실제 결과를 안 건드립니다."""
    d = os.path.join(ROOT, "results", "smoke") if os.environ.get("HRSC_SMOKE") \
        else os.path.join(ROOT, "results")
    os.makedirs(d, exist_ok=True)
    return d


def smoke_cap():
    """스모크 모드면 분할별 영상 상한, 아니면 None."""
    return {"train": 24, "val": 8, "test": 8} if os.environ.get("HRSC_SMOKE") else None


def load_cfg(path=None):
    import yaml
    p = path or os.path.join(HERE, "config.yaml")
    with io.open(p, encoding="utf-8") as f:
        return yaml.safe_load(f)


def rel(cfg, key):
    """설정의 상대경로를 프로젝트 기준 절대경로로."""
    return os.path.normpath(os.path.join(ROOT, cfg["paths"][key]))


def haversine_km(a1, o1, a2, o2):
    R = 6371.0
    p1, p2 = math.radians(a1), math.radians(a2)
    dp, dl = p2 - p1, math.radians(o2 - o1)
    x = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * R * math.asin(math.sqrt(x))


def true_gsd(lat, zoom):
    """Google Earth 타일 레벨과 위도에서 실제 지상해상도(m/px).

        gsd = 156543.03392 * cos(lat) / 2^zoom

    XML 의 Img_Resolution 은 위도 보정이 없는 명목값이라 쓰지 않습니다.
    이 식은 함급을 아는 선박 치수로 교차검증했습니다
    (build_port_manifest.py --verify-gsd).
    """
    return 156543.03392 * math.cos(math.radians(lat)) / (2.0 ** zoom)


def parse_zoom(text, default=18):
    """'18' 또는 '17(eg:17th layer in google earth)' 같은 값을 정수로."""
    d = ""
    for ch in str(text):
        if ch.isdigit():
            d += ch
        elif d:
            break
    return int(d) if d else default


def identify_port(lat, lon):
    """좌표로 항구를 고릅니다. XML 경도가 부호 없이 저장돼 있으므로
    동경/서경 두 해석을 모두 시험하고, 25 km 안에 드는 것만 인정합니다.

    반환: (port_key, verification_status, 보정된 lon)
      coord_match       25 km 안에 알려진 항구가 있음
      unknown_us_port   좌표는 있으나 아는 항구와 안 맞음
    """
    best = (None, 1e9, lon)
    for signed in (lon, -lon):
        for k, p in PORTS.items():
            d = haversine_km(lat, signed, p["lat"], p["lon"])
            if d < best[1]:
                best = (k, d, signed)
    if best[1] <= PORT_MATCH_KM:
        return best[0], "coord_match", best[2]
    return "unknown_us_port", "coord_unmatched", lon


def write_csv(path, rows, fields=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    fields = fields or list(rows[0].keys())
    with io.open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def read_csv(path):
    with io.open(path, encoding="utf-8") as f:
        return list(csv.DictReader(f))


def dump_json(path, obj):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with io.open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
