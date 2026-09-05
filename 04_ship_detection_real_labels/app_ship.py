"""위성 선박탐지 웹앱 — 코멘토 4차 업무 / 정승원

    py -m streamlit run app_ship.py --server.port 8502

무엇을 하는가
-------------
고해상도(약 0.45 m) 광학 위성영상을 올리면 선박을 찾아 위치·길이·폭·방향을
냅니다. 항만은 HRSC2016 의 미국 해군기지 5곳이고, 학습에 쓰지 않은 test 영상을
항만별로 돌아가며 볼 수 있습니다.

    샌디에이고   test 217장 · 683척          에버렛   39장 · 52척
    노퍽         117장 · 308척              뉴포트   12장 · 23척
    메이포트      66장 · 160척

왜 회전상자인가
---------------
군함은 부두에 비스듬히 댑니다. 축정렬 상자는 이웃 배와 부두를 크게 포함해
길이·폭을 못 재지만, 회전상자(OBB)는 선체만 감쌉니다. HRSC2016 정답이 사람이
그린 회전상자라 모델도 회전상자를 냅니다.

성능 수치는 **측정한 것만** 적습니다
------------------------------------
전부 학습에 쓰지 않은 공식 test 분할(451장)에서 잰 값이고, 항만별 AP50 은
그 항만의 test 영상만으로 따로 쟀습니다. 에버렛(52척)·뉴포트(23척)는 표본이
적어 참고값입니다.

이전 판(핀란드 난탈리 · 톈진 · 도쿄만, Sentinel-2 10 m)은 이 판으로 대체했습니다.
같은 모델을 10 m 로 열화한 영상에 쓰면 재현율이 0.03 으로 무너지고, 10 m 로
다시 학습하면 F1 0.68 입니다(hrsc-sr-project 결과).
"""
import io
import os
import csv
import json
import math

import numpy as np
import cv2
import streamlit as st

from ship_core import read_upload, to_lonlat

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "data", "hrsc")

st.set_page_config(page_title="위성 선박탐지", page_icon="🛰️", layout="wide")


def gsd_of(lat, zoom=18):
    """Google Earth 타일 레벨과 위도에서 실제 지상해상도(m/px).
    XML 의 Img_Resolution(1.07)은 위도 보정이 없는 명목값이라 쓰지 않습니다."""
    return 156543.03392 * math.cos(math.radians(lat)) / (2.0 ** zoom)


# ---------------------------------------------------------------- 항만 정보
#
# 수치는 전부 **못 본 자료에서 잰 값** 입니다. HRSC2016 공식 test 분할이고,
# 학습 시드 3개의 평균입니다. 전체: precision 0.916 · recall 0.957 · F1 0.936
# · AP50 0.970 · AP50-95 0.770 (conf 0.25 · IoU 0.5).
PORTS = {
    "san_diego": {
        "label": "🇺🇸  샌디에이고 해군기지, CA", "lat": 32.6623,
        "AP50": 0.972, "n_test": 683,
        "note": "태평양함대 모항. 항공모함·강습상륙함·구축함이 나란히 접안합니다.",
    },
    "norfolk": {
        "label": "🇺🇸  노퍽 해군기지, VA", "lat": 36.9608,
        "AP50": 0.981, "n_test": 308,
        "note": "세계 최대 해군기지. 잠수함부터 항모까지 크기 편차가 큽니다.",
    },
    "mayport": {
        "label": "🇺🇸  메이포트 해군기지, FL", "lat": 30.3948,
        "AP50": 0.952, "n_test": 160,
        "note": "좁은 수로형 항만. 배가 부두에 붙어 있어 경계가 어렵습니다.",
    },
    "everett": {
        "label": "🇺🇸  에버렛 해군기지, WA", "lat": 47.9815,
        "AP50": 0.974, "n_test": 52,
        "note": "test 52척 — 표본이 적어 참고값입니다. 구축함 위주.",
    },
    "newport": {
        "label": "🇺🇸  뉴포트 해군기지, RI", "lat": 41.5283,
        "AP50": 0.933, "n_test": 23,
        "note": "test 23척 — 표본 부족. 퇴역 항공모함이 계류돼 있습니다.",
    },
}
# 다섯 항만은 **같은 모델 하나**를 씁니다. 다섯 곳을 함께 학습했고 성능은
# 항만별로 따로 쟀습니다.
MODELS = {"seed 0": "weights/hrsc_hr045_seed0.pt",
          "seed 1": "weights/hrsc_hr045_seed1.pt",
          "seed 2": "weights/hrsc_hr045_seed2.pt"}
OVERALL = {"precision": 0.916, "recall": 0.957, "f1": 0.936,
           "AP50": 0.970, "AP50_95": 0.770}
THR = 0.25


@st.cache_resource(show_spinner=False)
def get_model(path):
    from ultralytics import YOLO
    return YOLO(os.path.join(HERE, path))


@st.cache_data(show_spinner=False)
def load_test_index():
    """항만별 test 영상 목록. 배가 많은 장면부터 — 첫 화면에서 바로 보이게."""
    man = {r["image_id"]: r for r in csv.DictReader(
        io.open(os.path.join(DATA, "manifests", "images.csv"), encoding="utf-8"))}
    idx = {k: [] for k in PORTS}
    for r in csv.DictReader(io.open(os.path.join(DATA, "manifests", "split.csv"),
                                    encoding="utf-8")):
        if r["official"] == "test" and r["port"] in idx and \
                os.path.exists(os.path.join(DATA, "images", r["image_id"] + ".png")):
            idx[r["port"]].append(r["image_id"])
    for k in idx:
        idx[k].sort(key=lambda i: -int(man[i]["n_ships"] or 0))
    return idx, man


def load_gt(image_id, W, H):
    p = os.path.join(DATA, "labels", image_id + ".txt")
    out = []
    if os.path.exists(p):
        for line in io.open(p, encoding="utf-8"):
            f = line.split()
            if len(f) == 9:
                out.append(np.array([float(v) for v in f[1:]]).reshape(4, 2)
                           * np.array([W, H]))
    return out


def poly_iou(a, b):
    ia, _ = cv2.intersectConvexConvex(np.float32(a), np.float32(b))
    if ia <= 0:
        return 0.0
    u = cv2.contourArea(np.float32(a)) + cv2.contourArea(np.float32(b)) - ia
    return float(ia / u) if u > 0 else 0.0


# ==================================================================== 사이드바
st.sidebar.markdown("## 📍 항만")
key = st.sidebar.radio("탐지할 해역", list(PORTS),
                       format_func=lambda k: PORTS[k]["label"])
P = PORTS[key]
GSD = gsd_of(P["lat"])
mkey = st.sidebar.selectbox("가중치 (학습 시드)", list(MODELS))
ready = os.path.exists(os.path.join(HERE, MODELS[mkey]))

st.sidebar.markdown("### 📊 모델 성능")
st.sidebar.table({
    "지표": ["precision", "recall", "F1", "AP50", "AP50-95", "AP50 · 이 항만"],
    "값": ["%.3f" % OVERALL["precision"], "%.3f" % OVERALL["recall"],
           "%.3f" % OVERALL["f1"], "%.3f" % OVERALL["AP50"],
           "%.3f" % OVERALL["AP50_95"],
           "%.3f  (%d척)" % (P["AP50"], P["n_test"])],
})
st.sidebar.caption("학습에 쓰지 않은 test 451장에서 잰 값입니다. "
                   "항만 AP50 은 그 항만의 test 영상만으로 쟀습니다. " + P["note"])
if P["n_test"] < 80:
    st.sidebar.warning("이 항만은 test 표본이 적어 항만별 수치는 **참고값**입니다.")

st.sidebar.markdown("### ⚙️ 점수 문턱")
conf = st.sidebar.slider("점수 문턱", 0.05, 0.95, THR, 0.05,
                         key="thr_" + key, label_visibility="collapsed")
st.sidebar.caption(
    "모델이 각 탐지에 매기는 확신도의 최저선입니다. "
    "**올리면** 확실한 것만 남아 오탐이 줄지만 흐린 배를 놓치고, "
    "**낮추면** 더 많이 찾지만 부두·크레인까지 섞입니다."
)
show_zoom = st.sidebar.checkbox("확대 보기", value=True)
show_gt = st.sidebar.checkbox("정답 상자(노랑) 겹쳐 보기", value=True)

# ==================================================================== 본문
st.markdown("# 🛰️ 위성 선박탐지")
st.caption("HRSC2016 약 0.45 m · 미국 해군기지 5곳 · 회전상자(OBB) · YOLO11m")

if not ready:
    st.warning("**%s** 가중치가 없습니다. `%s` 에 넣습니다." % (mkey, MODELS[mkey]))
    st.stop()

test_idx, man = load_test_index()
ids = test_idx[key]
if "idx" not in st.session_state:
    st.session_state.idx = {}
i = st.session_state.idx.get(key, 0) % max(len(ids), 1)

c_up, c_prev, c_next = st.columns([4, 1, 1])
up = c_up.file_uploader("🖼️ 위성영상 올리기",
                        type=["npy", "tif", "tiff", "png", "jpg", "jpeg"],
                        help="약 0.4~0.5 m 급 광학영상 (.tif · .png · .npy). "
                             "Sentinel-2 10 m 는 이 모델이 거의 못 잡습니다.")
c_prev.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
c_next.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
if c_prev.button("◀ 이전 test 영상", use_container_width=True) and ids:
    i = (i - 1) % len(ids)
if c_next.button("다음 test 영상 ▶", use_container_width=True) and ids:
    i = (i + 1) % len(ids)
st.session_state.idx[key] = i

if up is not None:
    try:
        rgb, geo, is8 = read_upload(up.getvalue(), up.name)
    except Exception as e:
        st.error("읽지 못했습니다 — %s" % e)
        st.stop()
    name, gt = up.name, []
elif ids:
    image_id = ids[i]
    bgr = cv2.imread(os.path.join(DATA, "images", image_id + ".png"))
    rgb, geo = bgr[:, :, ::-1].copy(), None
    name = "test %s (%d / %d)" % (image_id, i + 1, len(ids))
    gt = load_gt(image_id, rgb.shape[1], rgb.shape[0])
else:
    st.info("위성영상을 올리면 선박을 찾습니다.")
    st.stop()

H, W = rgb.shape[:2]
st.success("%s · %d x %d 화소 · %.2f x %.2f km  (GSD %.2f m)"
           % (name, W, H, W * GSD / 1000, H * GSD / 1000, GSD))

with st.spinner("추론 중..."):
    model = get_model(MODELS[mkey])
    r = model.predict(rgb[:, :, ::-1], imgsz=640, conf=conf, iou=0.7,
                      verbose=False, device="cpu")[0]
    dets = []
    if r.obb is not None and len(r.obb):
        polys = r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2)
        xywhr = r.obb.xywhr.cpu().numpy()
        for q, (cx, cy, w, h, ang), s_ in zip(polys, xywhr, r.obb.conf.cpu().numpy()):
            dets.append({"poly": q, "cx": float(cx), "cy": float(cy),
                         "len": float(max(w, h)), "wid": float(min(w, h)),
                         "ang": float(math.degrees(ang)) % 180.0, "score": float(s_)})

# 정답이 있으면 IoU 0.5 그리디 매칭으로 맞힘/오탐/놓침을 셉니다
hit, used = set(), set()
if gt:
    for k in np.argsort([-d["score"] for d in dets]):
        best, bj = 0.5, -1
        for j, g in enumerate(gt):
            if j in used:
                continue
            v = poly_iou(dets[k]["poly"], g)
            if v >= best:
                best, bj = v, j
        if bj >= 0:
            used.add(bj)
            hit.add(int(k))

st.markdown("## 🚢 탐지 결과")
c1, c2, c3 = st.columns(3)
c1.metric("찾은 선박", "%d 척" % len(dets))
lens = [d["len"] * GSD for d in dets]
c2.metric("길이 중앙", "%.0f m" % (np.median(lens) if lens else 0))
c3.metric("점수 문턱", "%.2f" % conf)
if gt:
    st.caption("정답 %d척 · 맞힘 %d · 오탐 %d · 놓침 %d  (IoU ≥ 0.5)"
               % (len(gt), len(hit), len(dets) - len(hit), len(gt) - len(used)))


def draw(im, ds, gts=(), ox=0, oy=0, thick=2):
    """회전상자를 그립니다. 초록 = 탐지(맞힘), 빨강 = 오탐, 노랑 = 정답."""
    for g in gts:
        cv2.polylines(im, [np.int32(g - [ox, oy])], True, (255, 230, 0),
                      max(thick - 1, 1), cv2.LINE_AA)
    for k, d in ds:
        col = (0, 255, 0) if (k in hit or not gt) else (255, 60, 60)
        cv2.polylines(im, [np.int32(d["poly"] - [ox, oy])], True, col, thick, cv2.LINE_AA)
    return im


vis = draw(rgb.copy(), list(enumerate(dets)), gt if show_gt else ())
scale = min(1400.0 / max(W, 1), 3.0)
if abs(scale - 1.0) > 1e-6:
    vis = cv2.resize(vis, (int(W * scale), int(H * scale)),
                     interpolation=cv2.INTER_LINEAR)
st.image(vis, caption="초록 회전상자 = 탐지된 선박 (%d 척)%s"
         % (len(dets), " · 빨강 = 오탐 · 노랑 = 정답" if gt and show_gt else ""),
         use_container_width=True)

if show_zoom and dets:
    d0 = max(dets, key=lambda d: d["score"])
    cx, cy, s_ = int(d0["cx"]), int(d0["cy"]), 256
    x0, y0 = max(cx - s_, 0), max(cy - s_, 0)
    x1, y1 = min(cx + s_, W), min(cy + s_, H)
    z = rgb[y0:y1, x0:x1].copy()
    if z.size:
        inside = [(k, d) for k, d in enumerate(dets)
                  if x0 <= d["cx"] < x1 and y0 <= d["cy"] < y1]
        draw(z, inside, gt if show_gt else (), x0, y0, thick=1)
        z = cv2.resize(z, (z.shape[1] * 2, z.shape[0] * 2), interpolation=cv2.INTER_LINEAR)
        st.image(z, caption="확대 — 가장 확실한 배 주변 %.0f m" % ((x1 - x0) * GSD))

if dets:
    rows = sorted(dets, key=lambda d: -d["score"])
    table = {"ID": ["#%d" % (k + 1) for k in range(len(rows))],
             "점수": [round(d["score"], 3) for d in rows],
             "길이(m)": [round(d["len"] * GSD) for d in rows],
             "폭(m)": [round(d["wid"] * GSD) for d in rows],
             "방향(°)": [round(d["ang"]) for d in rows]}
    ll = None
    if geo is not None:
        ll = [to_lonlat(d["cx"], d["cy"], geo) for d in rows]
        table["중심(위도, 경도)"] = ["%.6f, %.6f" % (a[1], a[0]) for a in ll]
    else:
        table["중심(화소 x, y)"] = ["%.0f, %.0f" % (d["cx"], d["cy"]) for d in rows]
    st.markdown("### 선박별 상세")
    st.dataframe(table, use_container_width=True, height=320)

    stem = os.path.splitext(os.path.basename(name))[0].split(" ")[0]
    if ll is not None:
        feats = []
        for k, d in enumerate(rows):
            ring = [to_lonlat(x, y, geo) for x, y in d["poly"]]
            ring.append(ring[0])
            feats.append({"type": "Feature",
                          "geometry": {"type": "Polygon",
                                       "coordinates": [[[round(a, 6), round(b, 6)]
                                                        for a, b in ring]]},
                          "properties": {"id": k + 1, "score": round(d["score"], 3),
                                         "length_m": round(d["len"] * GSD, 1),
                                         "width_m": round(d["wid"] * GSD, 1),
                                         "heading_deg": round(d["ang"], 1)}})
        geo_out = {"type": "FeatureCollection", "name": "%s_ships" % stem,
                   "crs": {"type": "name", "properties": {"name": "EPSG:4326"}},
                   "properties": {"port": P["label"], "gsd_m": round(GSD, 3),
                                  "score_thr": conf, "model": MODELS[mkey],
                                  "kind": "obb"},
                   "features": feats}
        st.download_button("⬇ GeoJSON",
                           json.dumps(geo_out, ensure_ascii=False, indent=1),
                           file_name="%s_ships.geojson" % stem,
                           mime="application/geo+json")
else:
    st.info("이 영상에서 선박을 찾지 못했습니다. 점수 문턱을 낮춰 보십시오.")

st.caption("길이·폭은 회전상자의 장변·단변에 이 항만의 GSD(%.2f m)를 곱한 값입니다. "
           "GSD 는 Google Earth 타일 레벨 18 과 위도로 계산했고, 알려진 함급 치수와 "
           "약 4%% 안에서 맞습니다. 올린 영상은 같은 GSD 라고 가정합니다." % GSD)
