"""위성 선박탐지 웹앱 — 코멘토 4차 업무 / 정승원

    py -m streamlit run app_ship.py --server.port 8502

HRSC2016(약 0.45 m) 미국 해군기지 5곳. 항만을 고르고 ◀ ▶ 을 누르면 학습에
쓰지 않은 test 영상에 회전상자 탐지를 그립니다. 성능표는 그 항만의 test 영상만으로
잰 값입니다.

    weights/hrsc_hr045_seed0.pt      YOLO11m-OBB, 67 epoch
    data/hrsc/images, labels         test 영상 451장 · 정답 회전상자
    data/hrsc/manifests              항만 · 촬영일 · 크기
    outputs/port_metrics.json        항만별 실측 성능
"""
import io
import os
import csv
import json
import math
import threading

import numpy as np
import cv2
import streamlit as st

# 자료·가중치는 과제 폴더(04_ship_detection_real_labels) 바로 아래에 있습니다.
# 개발 중엔 이 파일이 webapp/ 안에 있으니 한 단계 위, exe 로 묶이면 풀린 자리(_MEIPASS).
import sys
HERE = getattr(sys, "_MEIPASS", None) or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(HERE, "data", "hrsc")
WEIGHTS = "weights/hrsc_hr045_seed0.pt"
CONF, NMS = 0.25, 0.7

PORTS = {
    "san_diego": ("🇺🇸  샌디에이고 해군기지, CA", 32.6623),
    "norfolk":   ("🇺🇸  노퍽 해군기지, VA", 36.9608),
    "mayport":   ("🇺🇸  메이포트 해군기지, FL", 30.3948),
    "everett":   ("🇺🇸  에버렛 해군기지, WA", 47.9815),
    "newport":   ("🇺🇸  뉴포트 해군기지, RI", 41.5283),
}

st.set_page_config(page_title="위성 선박탐지", page_icon="🛰️", layout="wide")


def gsd_of(lat, zoom=18):
    return 156543.03392 * math.cos(math.radians(lat)) / (2.0 ** zoom)


_PREDICT_LOCK = threading.Lock()


@st.cache_resource(show_spinner=False)
def get_model():
    """모델을 한 번만 올리고, 여기서 바로 융합(fuse)까지 끝내 둡니다.

    ultralytics 는 첫 predict 에서 층을 융합하는데, 세션 둘이 동시에 첫 추론을
    걸면 한쪽이 융합 중인 모델을 다른 쪽이 또 융합하려다
    'Conv' object has no attribute 'bn' 으로 죽습니다. 캐시 함수 안에서
    빈 영상으로 한 번 돌려 융합을 마치면 그 뒤로는 안전합니다.
    """
    from ultralytics import YOLO
    m = YOLO(os.path.join(HERE, WEIGHTS))
    m.predict(np.zeros((64, 64, 3), np.uint8), imgsz=64, verbose=False, device="cpu")
    return m


def predict(model, bgr):
    with _PREDICT_LOCK:
        return model.predict(bgr, imgsz=640, conf=CONF, iou=NMS, verbose=False, device="cpu")[0]


@st.cache_data(show_spinner=False)
def load_index():
    man = {r["image_id"]: r for r in csv.DictReader(
        io.open(os.path.join(DATA, "manifests", "images.csv"), encoding="utf-8"))}
    idx = {k: [] for k in PORTS}
    for r in csv.DictReader(io.open(os.path.join(DATA, "manifests", "split.csv"), encoding="utf-8")):
        if r["official"] == "test" and r["port"] in idx and \
                os.path.exists(os.path.join(DATA, "images", r["image_id"] + ".jpg")):
            idx[r["port"]].append(r["image_id"])
    for k in idx:
        idx[k].sort(key=lambda i: -int(man[i]["n_ships"] or 0))
    metrics = json.load(io.open(os.path.join(HERE, "outputs", "port_metrics.json"), encoding="utf-8"))
    return idx, man, metrics


def load_gt(image_id, W, H):
    p = os.path.join(DATA, "labels", image_id + ".txt")
    out = []
    if os.path.exists(p):
        for line in io.open(p, encoding="utf-8"):
            f = line.split()
            if len(f) == 9:
                out.append(np.array([float(v) for v in f[1:]]).reshape(4, 2) * [W, H])
    return out


def poly_iou(a, b):
    ia, _ = cv2.intersectConvexConvex(np.float32(a), np.float32(b))
    if ia <= 0:
        return 0.0
    u = cv2.contourArea(np.float32(a)) + cv2.contourArea(np.float32(b)) - ia
    return float(ia / u) if u > 0 else 0.0


# ==================================================================== 사이드바
st.sidebar.markdown("## 📍 항만")
key = st.sidebar.radio("항만", list(PORTS), format_func=lambda k: PORTS[k][0],
                       label_visibility="collapsed")
label, lat = PORTS[key]
GSD = gsd_of(lat)
idx, man, metrics = load_index()
M = metrics[key]

st.sidebar.markdown("### 📊 이 항만의 성능")
# st.table 은 pyarrow(81 MB)를 끌고 옵니다. 실행 파일 용량 때문에 마크다운 표로 씁니다.
st.sidebar.markdown(
    "| 지표 | 값 |\n|---|---:|\n"
    "| precision | %.3f |\n| recall | %.3f |\n| F1 | %.3f |\n| AP50 | %.3f |"
    % (M["precision"], M["recall"], M["f1"], M["AP50"]))
st.sidebar.caption("test %d장 · 선박 %d척 · conf %.2f · IoU 0.5" % (M["n_images"], M["n_ships"], CONF))
show_gt = st.sidebar.checkbox("정답(노랑) 표시", value=True)

# ==================================================================== 본문
st.markdown("# 🛰️ 위성 선박탐지")

ids = idx[key]
if "idx" not in st.session_state:
    st.session_state.idx = {}
i = st.session_state.idx.get(key, 0) % len(ids)
c1, c2, c3 = st.columns([1, 1, 6])
if c1.button("◀ 이전", use_container_width=True):
    i = (i - 1) % len(ids)
if c2.button("다음 ▶", use_container_width=True):
    i = (i + 1) % len(ids)
st.session_state.idx[key] = i
image_id = ids[i]
m = man[image_id]
# XML 촬영일. 1900-01-01 은 자리표시자라 미기재로 봅니다.
date = m["date"] if m["date"] and not m["date"].startswith("1900") else "미기재"
c3.markdown("**%s** · %d / %d · 촬영일 **%s** · %s × %s 화소 · GSD %.2f m"
            % (label, i + 1, len(ids), date, m["width"], m["height"], GSD))

# cv2.imread 는 윈도에서 한글 경로(예: dist/위성선박탐지/)를 못 읽습니다.
bgr = cv2.imdecode(np.fromfile(os.path.join(DATA, "images", image_id + ".jpg"), np.uint8),
                   cv2.IMREAD_COLOR)
H, W = bgr.shape[:2]
gt = load_gt(image_id, W, H)

r = predict(get_model(), bgr)
dets = []
if r.obb is not None and len(r.obb):
    for q, (cx, cy, w, h, ang), s_ in zip(r.obb.xyxyxyxy.cpu().numpy().reshape(-1, 4, 2),
                                          r.obb.xywhr.cpu().numpy(), r.obb.conf.cpu().numpy()):
        dets.append(dict(poly=q, cx=float(cx), cy=float(cy), len=float(max(w, h)),
                         wid=float(min(w, h)), ang=float(math.degrees(ang)) % 180.0, score=float(s_)))

hit, used = set(), set()
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

vis = bgr[:, :, ::-1].copy()
if show_gt:
    for g in gt:
        cv2.polylines(vis, [np.int32(g)], True, (255, 230, 0), 2, cv2.LINE_AA)
for k, d in enumerate(dets):
    col = (0, 255, 0) if k in hit else (255, 60, 60)
    cv2.polylines(vis, [np.int32(d["poly"])], True, col, 3, cv2.LINE_AA)
st.image(vis, use_container_width=True,
         caption="초록 = 맞힘 · 빨강 = 오탐 · 노랑 = 정답")

a, b, c, d_ = st.columns(4)
a.metric("정답", len(gt))
b.metric("맞힘", len(hit))
c.metric("오탐", len(dets) - len(hit))
d_.metric("놓침", len(gt) - len(used))
