# -*- coding: utf-8 -*-
"""웹앱 계산부 시험.

여기서 잡으려는 것은 '모델이 얼마나 잘 맞히는가' 가 아니라 **모델 앞뒤의
배관이 조용히 틀리는 경우** 입니다. 실제로 겪은 사고를 시험으로 굳혔습니다.

  · 물뿐인 타일에서 대비를 끝까지 늘려 물결이 배로 보이던 것
  · 타일 경계에 걸친 배가 두 번 세어지던 것
  · 채널 순서가 뒤집혀 파랑과 빨강이 바뀌던 것
"""
import io
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import ship_core                                                  # noqa: E402


# ------------------------------------------------------------------ 대비 늘리기
def test_stretch_물뿐인_타일은_증폭하지_않는다():
    """값 폭이 좁으면 최소 폭을 강제합니다.

    잔물결만 있는 바다는 DN 폭이 몇 밖에 안 됩니다. 그대로 p1~p99.5 로
    늘리면 잡음이 0~255 로 펴져 모델이 물결을 물체로 읽습니다.
    """
    water = np.random.RandomState(0).normal(1000, 2, (64, 64, 3))
    out = ship_core.stretch(water, min_range=40.0)
    assert out.dtype == np.uint8
    # 폭 40 을 강제했으므로 잡음(표준편차 2)은 좁은 구간에 머물러야 합니다
    assert out.std() < 40, "물뿐인 타일이 대비 가득 증폭됐습니다"


def test_stretch_실제_대비가_있으면_펴진다():
    v = np.zeros((32, 32, 3), np.float32)
    v[8:24, 8:24] = 3000.0                      # 밝은 물체
    out = ship_core.stretch(v)
    assert out.max() == 255 and out.min() == 0


def test_stretch_상수_영상에서_터지지_않는다():
    out = ship_core.stretch(np.full((16, 16, 3), 500.0))
    assert out.shape == (16, 16, 3) and np.isfinite(out).all()


# ------------------------------------------------------------------ 파일 읽기
def test_read_upload_npy_는_BGR_을_RGB_로_돌린다():
    """.npy 는 B,G,R,NIR 순서로 저장돼 있어 뒤집어 읽어야 합니다."""
    a = np.zeros((8, 8, 4), np.float32)
    a[..., 2] = 4000.0                          # 세 번째 채널 = 빨강
    buf = io.BytesIO()
    np.save(buf, a)
    rgb, geo, is8 = ship_core.read_upload(buf.getvalue(), "x.npy")
    assert rgb.shape == (8, 8, 3) and geo is None and is8 is False
    assert rgb[..., 0].mean() > rgb[..., 2].mean(), "빨강이 파랑 자리에 갔습니다"


def test_read_upload_채널이_모자라면_거부한다():
    buf = io.BytesIO()
    np.save(buf, np.zeros((8, 8, 2), np.float32))
    with pytest.raises(ValueError):
        ship_core.read_upload(buf.getvalue(), "x.npy")


def test_read_upload_png_은_그대로_8비트():
    import cv2
    im = np.zeros((8, 8, 3), np.uint8)
    im[..., 2] = 200                            # OpenCV 는 BGR 로 씁니다
    ok, enc = cv2.imencode(".png", im)
    assert ok
    rgb, geo, is8 = ship_core.read_upload(enc.tobytes(), "x.png")
    assert is8 is True and rgb[0, 0, 0] == 200  # 읽으면 RGB 의 빨강


def test_read_upload_모르는_그림은_거부한다():
    with pytest.raises(ValueError):
        ship_core.read_upload(b"not an image", "x.png")


# ------------------------------------------------------------------ 좌표 변환
def test_to_lonlat_화소_중심을_가리킨다():
    """좌표는 화소의 **중심** 입니다 — 모서리가 아닙니다.

    화소 (100, 200) 의 중심은 원점에서 오른쪽으로 100.5 화소입니다.
    이 반 화소를 빠뜨리면 모든 탐지가 남동쪽으로 5 m 씩 밀립니다.
    """
    geo = {"crs": "EPSG:32650",
           "transform": [10.0, 0.0, 400000.0, 0.0, -10.0, 4400000.0]}
    lon, lat = ship_core.to_lonlat(100.0, 200.0, geo)
    from pyproj import Transformer
    back = Transformer.from_crs("EPSG:4326", geo["crs"], always_xy=True)
    X, Y = back.transform(lon, lat)
    assert abs(X - (400000.0 + 100.5 * 10.0)) < 0.1
    assert abs(Y - (4400000.0 - 200.5 * 10.0)) < 0.1


def test_to_lonlat_동쪽으로_갈수록_경도가_는다():
    geo = {"crs": "EPSG:32650",
           "transform": [10.0, 0.0, 400000.0, 0.0, -10.0, 4400000.0]}
    west = ship_core.to_lonlat(0.0, 100.0, geo)
    east = ship_core.to_lonlat(500.0, 100.0, geo)
    south = ship_core.to_lonlat(100.0, 500.0, geo)
    north = ship_core.to_lonlat(100.0, 0.0, geo)
    assert east[0] > west[0], "x 가 커지면 경도가 늘어야 합니다"
    assert north[1] > south[1], "y 가 커지면 위도가 줄어야 합니다"


# ------------------------------------------------------------------ 상자 합치기
class FakeYOLO:
    """지정한 자리에 상자를 내놓는 가짜 모델 — 합치는 논리만 시험합니다."""

    def __init__(self, boxes):
        self.boxes = boxes                      # 전역 좌표 [x0,y0,x1,y1]

    def predict(self, crop, imgsz=640, conf=0.0, verbose=False):
        import types
        h, w = crop.shape[:2]
        # 이 타일의 전역 원점을 모르므로, 시험에서는 한 타일만 씁니다
        out = [b for b in self.boxes]
        xy = np.array([[b[0], b[1], b[2], b[3]] for b in out], np.float32) \
            if out else np.zeros((0, 4), np.float32)
        cf = np.array([b[4] for b in out], np.float32) if out \
            else np.zeros((0,), np.float32)
        t = lambda a: types.SimpleNamespace(cpu=lambda: types.SimpleNamespace(
            numpy=lambda: a))
        return [types.SimpleNamespace(
            boxes=types.SimpleNamespace(xyxy=t(xy), conf=t(cf)))]


def test_detect_tiled_겹친_상자를_하나로_합친다():
    m = FakeYOLO([[10, 10, 30, 30, 0.9], [12, 12, 32, 32, 0.5]])
    keep = ship_core.detect_tiled(m, np.zeros((640, 640, 3), np.uint8), 0.1)
    assert len(keep) == 1, "IoU 0.3 을 넘는 중복이 남았습니다"
    assert keep[0][4] == pytest.approx(0.9), "점수가 낮은 쪽이 살아남았습니다"


def test_detect_tiled_떨어진_상자는_둘_다_남긴다():
    m = FakeYOLO([[10, 10, 30, 30, 0.9], [300, 300, 320, 320, 0.5]])
    keep = ship_core.detect_tiled(m, np.zeros((640, 640, 3), np.uint8), 0.1)
    assert len(keep) == 2


def test_detect_tiled_탐지가_없으면_빈_목록():
    keep = ship_core.detect_tiled(FakeYOLO([]),
                                  np.zeros((640, 640, 3), np.uint8), 0.1)
    assert keep == []
