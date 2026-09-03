"""위성 선박탐지 웹앱 — 코멘토 4차 업무 / 정승원

    py -m streamlit run app_ship.py --server.port 8502

무엇을 하는가
-------------
Sentinel-2 위성영상을 올리면 선박을 찾아 위치와 크기를 냅니다. 항만마다
그 해역의 **실측 라벨**로 학습한 모델을 따로 씁니다.

    난탈리·투르쿠   핀란드환경연구소(SYKE) 사람 판독 · 10,697 라벨
    톈진            Allen AI Skylight 전문가 판독 점 라벨 · 2,173
    도쿄만          같은 자료 · 1,411

왜 항만마다 다른 모델인가
-------------------------
자료마다 라벨 규약이 다릅니다. 핀란드는 상자, Allen AI 는 **점** 입니다.
10 m 에서 배는 몇 화소뿐이라 상자 IoU 가 성립하지 않아 Allen AI 가 점으로
만든 것이고, 억지로 상자로 바꾸면 없는 정보를 지어내는 셈입니다. 그래서
규약대로 각각 학습합니다.

성능 수치는 **측정한 것만** 적습니다
------------------------------------
학습이 안 끝난 항만은 빈칸으로 둡니다. 짐작한 숫자를 화면에 올리면 쓰는
사람이 그것을 실측으로 읽습니다.
"""
import io
import os
import json

import numpy as np
import cv2
import streamlit as st

from ship_core import GSD, stretch, read_upload, detect_tiled, to_lonlat

HERE = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="위성 선박탐지", page_icon="🛰️", layout="wide")

# ---------------------------------------------------------------- 항만 정보
#
# 수치는 전부 **못 본 자료에서 잰 값** 입니다. 목표 항만을 학습에서 완전히
# 빼고 쟀습니다 — 난탈리는 34VEN 타일을 통째로 시험셋으로 돌렸습니다.
PORTS = {
    "naantali": {
        "label": "🇫🇮  핀란드 · 난탈리",
        "kind": "box",                       # 정답이 사람이 그린 박스
        "weights": "weights/naantali.pt",
        "thr": 0.25,
        "metrics": {"precision": 0.8544, "recall": 0.8230,
                    "f1": 0.8384, "AP50": 0.8669, "AP50_95": 0.3982},
    },
    "tianjin": {
        "label": "🇨🇳  중국 · 톈진",
        "kind": "point",                     # 정답이 선박 중심점
        "weights": "weights/point_tj_tokyo.pt",
        "thr": 0.15,
        "metrics": {"30m": (0.498, 0.522, 0.510),
                    "50m": (0.644, 0.675, 0.659)},
    },
    "tokyo": {
        "label": "🇯🇵  일본 · 도쿄만",
        "kind": "point",
        "weights": "weights/point_tj_tokyo.pt",
        "thr": 0.15,
        "metrics": {"30m": (0.436, 0.440, 0.438),
                    "50m": (0.691, 0.697, 0.694)},
    },
}
# 톈진과 도쿄만은 **같은 모델 하나**를 씁니다. 두 해역을 함께 학습했고
# 성능은 항만별로 따로 쟀습니다.


@st.cache_resource(show_spinner=False)
def get_model(path, kind):
    full = os.path.join(HERE, path)
    if kind == "point":
        import point_net
        return point_net.load(full)[0]
    from ultralytics import YOLO
    return YOLO(full)


# ==================================================================== 사이드바
st.sidebar.markdown("## 📍 항만")
key = st.sidebar.radio("탐지할 해역", list(PORTS),
                       format_func=lambda k: PORTS[k]["label"])
P = PORTS[key]
ready = os.path.exists(os.path.join(HERE, P["weights"]))

st.sidebar.markdown("### 📊 모델 성능")
m = P["metrics"]
if P["kind"] == "box":
    st.sidebar.table({
        "지표": ["precision", "recall", "F1", "AP50", "AP50-95"],
        "값": ["%.3f" % m["precision"], "%.3f" % m["recall"],
               "%.3f" % m["f1"], "%.3f" % m["AP50"], "%.3f" % m["AP50_95"]],
    })
    st.sidebar.caption("학습에 쓰지 않은 자료에서 잰 값입니다.")
else:
    st.sidebar.table({
        "허용 반경": ["30 m", "50 m"],
        "precision": ["%.3f" % m["30m"][0], "%.3f" % m["50m"][0]],
        "recall": ["%.3f" % m["30m"][1], "%.3f" % m["50m"][1]],
        "F1": ["%.3f" % m["30m"][2], "%.3f" % m["50m"][2]],
    })
    st.sidebar.caption(
        "점 탐지라 **정답점에서 몇 m 안이면 맞은 것으로 볼지**를 정해야 "
        "합니다. 10 m 해상도에서 길이 100 m 배의 중심을 찍는 일이라, "
        "30 m 는 선체 길이의 1/3 안을 요구하는 엄격한 기준입니다."
    )
    st.sidebar.warning(
        "이 해역은 아직 **진단 단계**입니다. 난탈리 수준으로 올리기 위한 "
        "원인 분석(라벨 오차·자료량·사전학습)을 README 에 정리했습니다."
    )

st.sidebar.markdown("### ⚙️ 점수 문턱")
conf = st.sidebar.slider("점수 문턱", 0.05, 0.95, P["thr"], 0.05,
                         key="thr_" + key, label_visibility="collapsed")
st.sidebar.caption(
    "모델이 각 탐지에 매기는 확신도의 최저선입니다. "
    "**올리면** 확실한 것만 남아 오탐이 줄지만 흐린 배를 놓치고, "
    "**낮추면** 더 많이 찾지만 물결·부표까지 섞입니다."
)
show_zoom = st.sidebar.checkbox("확대 보기", value=True)

# ==================================================================== 본문
st.markdown("# 🛰️ 위성 선박탐지")
st.caption("Sentinel-2 10 m · 항만마다 그 해역 실측 라벨로 학습한 모델")

if not ready:
    st.warning("**%s** 모델이 아직 없습니다. 학습이 끝나면 `%s` 에 넣습니다."
               % (P["label"], P["weights"]))
    st.stop()

up = st.file_uploader("🖼️ 위성영상 올리기",
                      type=["npy", "tif", "tiff", "png", "jpg", "jpeg"],
                      help="Sentinel-2 10 m 영상 (.npy · .tif · .png)")

if up is None:
    st.info("위성영상을 올리면 선박을 찾습니다.")
    st.stop()

try:
    rgb, geo, is8 = read_upload(up.getvalue(), up.name)
except Exception as e:
    st.error("읽지 못했습니다 — %s" % e)
    st.stop()

H, W = rgb.shape[:2]
st.success("%s · %d x %d 화소 · %.2f x %.2f km"
           % (up.name, W, H, W * GSD / 1000, H * GSD / 1000))

with st.spinner("추론 중..."):
    model = get_model(P["weights"], P["kind"])
    if P["kind"] == "point":
        import point_net
        # (x, y, 점수) -> 상자 없는 탐지 기록
        dets = [{"cx": x, "cy": y, "score": s_, "box": None}
                for x, y, s_ in point_net.detect(model, rgb, conf)]
    else:
        dets = [{"cx": (b[0] + b[2]) / 2, "cy": (b[1] + b[3]) / 2,
                 "score": b[4], "box": b[:4]}
                for b in detect_tiled(model, rgb, conf)]

st.markdown("## 🚢 탐지 결과")
c1, c2, c3 = st.columns(3)
c1.metric("찾은 선박", "%d 척" % len(dets))
if P["kind"] == "box":
    lens = [max(d["box"][2] - d["box"][0], d["box"][3] - d["box"][1]) * GSD
            for d in dets]
    c2.metric("길이 중앙", "%.0f m" % (np.median(lens) if lens else 0))
else:
    # 점 탐지는 크기를 내놓지 않습니다. 없는 값을 지어내지 않습니다.
    c2.metric("탐지 방식", "중심점")
c3.metric("점수 문턱", "%.2f" % conf)


def draw(im, ds, ox=0, oy=0, thick=2):
    """상자면 사각형, 점이면 십자. 좌표는 (ox, oy) 만큼 옮겨 그립니다."""
    for d in ds:
        if d["box"] is not None:
            b = d["box"]
            cv2.rectangle(im, (int(b[0] - ox), int(b[1] - oy)),
                          (int(b[2] - ox), int(b[3] - oy)), (0, 255, 0), thick)
        else:
            x, y = int(round(d["cx"] - ox)), int(round(d["cy"] - oy))
            r = 3 * thick
            cv2.line(im, (x - r, y), (x + r, y), (0, 255, 0), thick)
            cv2.line(im, (x, y - r), (x, y + r), (0, 255, 0), thick)
    return im


vis = draw(rgb.copy(), dets)
scale = min(1400.0 / max(W, 1), 3.0)
if abs(scale - 1.0) > 1e-6:
    vis = cv2.resize(vis, (int(W * scale), int(H * scale)),
                     interpolation=cv2.INTER_NEAREST)
mark = "초록 상자" if P["kind"] == "box" else "초록 십자"
st.image(vis, caption="%s = 탐지된 선박 (%d 척)" % (mark, len(dets)),
         use_container_width=True)

if show_zoom and dets:
    d0 = max(dets, key=lambda d: d["score"])
    cx, cy, s_ = int(d0["cx"]), int(d0["cy"]), 128
    x0, y0 = max(cx - s_, 0), max(cy - s_, 0)
    x1, y1 = min(cx + s_, W), min(cy + s_, H)
    z = rgb[y0:y1, x0:x1].copy()
    if z.size:
        draw(z, [d for d in dets if x0 <= d["cx"] < x1 and y0 <= d["cy"] < y1],
             x0, y0, thick=1)
        z = cv2.resize(z, (z.shape[1] * 4, z.shape[0] * 4),
                       interpolation=cv2.INTER_NEAREST)
        st.image(z, caption="확대 — 가장 확실한 배 주변 %.1f km"
                 % ((x1 - x0) * GSD / 1000))

if dets:
    rows = sorted(dets, key=lambda d: -d["score"])
    table = {"ID": ["#%d" % (i + 1) for i in range(len(rows))],
             "점수": [round(d["score"], 3) for d in rows]}
    if P["kind"] == "box":
        table["길이(m)"] = [round(max(d["box"][2] - d["box"][0],
                                     d["box"][3] - d["box"][1]) * GSD)
                           for d in rows]
    ll = None
    if geo is not None:
        ll = [to_lonlat(d["cx"], d["cy"], geo) for d in rows]
        table["중심(위도, 경도)"] = ["%.6f, %.6f" % (a[1], a[0]) for a in ll]
    else:
        table["중심(화소 x, y)"] = ["%.0f, %.0f" % (d["cx"], d["cy"])
                                  for d in rows]
    st.markdown("### 선박별 상세")
    st.dataframe(table, use_container_width=True, height=320)

    stem = os.path.splitext(os.path.basename(up.name))[0]
    if ll is not None:
        feats = []
        for i, (d, a) in enumerate(zip(rows, ll)):
            pr = {"id": i + 1, "score": round(d["score"], 3)}
            if d["box"] is not None:
                pr["length_m"] = round(max(d["box"][2] - d["box"][0],
                                           d["box"][3] - d["box"][1]) * GSD, 1)
            feats.append({"type": "Feature",
                          "geometry": {"type": "Point",
                                       "coordinates": [round(a[0], 6),
                                                       round(a[1], 6)]},
                          "properties": pr})
        geo_out = {"type": "FeatureCollection", "name": "%s_ships" % stem,
                   "crs": {"type": "name",
                           "properties": {"name": "EPSG:4326"}},
                   "properties": {"port": P["label"], "gsd_m": GSD,
                                  "score_thr": conf, "model": P["weights"],
                                  "kind": P["kind"]},
                   "features": feats}
        st.download_button("⬇ GeoJSON",
                           json.dumps(geo_out, ensure_ascii=False, indent=1),
                           file_name="%s_ships.geojson" % stem,
                           mime="application/geo+json")
else:
    st.info("이 영상에서 선박을 찾지 못했습니다. 점수 문턱을 낮춰 보십시오.")

if P["kind"] == "box":
    st.caption("길이는 탐지 상자의 장변입니다. 10 m 해상도에서 상자는 실제 "
               "선체보다 짧게 잡히는 경향이 있어 참고값입니다.")
else:
    st.caption("이 해역의 정답 라벨이 선박 중심점이라 모델도 중심점을 "
               "내놓습니다. 크기는 학습한 적이 없어 표시하지 않습니다.")
