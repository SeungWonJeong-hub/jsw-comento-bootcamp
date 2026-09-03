# -*- coding: utf-8 -*-
"""점 탐지기 시험.

점 탐지는 상자와 달리 **좌표가 곧 답** 입니다. 반 화소만 밀려도 30 m
기준에서 성능이 눈에 띄게 떨어집니다. 그래서 좌표를 만드는 길목을
전부 시험합니다.
"""
import os
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import point_net                                                  # noqa: E402

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CKPT = os.path.join(HERE, "weights", "point_tj_tokyo.pt")


def heat(shape, pts, sigma=1.0):
    """정답점 자리에 가우시안을 찍은 히트맵 — 학습 때와 같은 방식."""
    h = np.zeros(shape, np.float32)
    yy, xx = np.mgrid[0:shape[0], 0:shape[1]]
    for x, y in pts:
        h = np.maximum(h, np.exp(-((xx-x)**2 + (yy-y)**2) / (2*sigma**2)))
    return h


# ------------------------------------------------------------------ 봉우리 뽑기
def test_peaks_봉우리를_정확한_자리에서_찾는다():
    p = heat((64, 64), [(20.0, 33.0)])
    got = point_net.peaks(p, thr=0.3)
    assert len(got) == 1
    x, y, s = got[0]
    assert (x, y) == (20.0, 33.0)
    assert s == pytest.approx(1.0)


def test_peaks_문턱_아래는_버린다():
    p = heat((64, 64), [(20.0, 33.0)]) * 0.1
    assert point_net.peaks(p, thr=0.3) == []


def test_peaks_붙어있는_두_봉우리_중_하나만_남긴다():
    """3x3 안에 둘이 있으면 지역 최대는 하나뿐입니다."""
    p = heat((64, 64), [(30.0, 30.0)])
    p[30, 31] = 0.95                            # 바로 옆에 낮은 봉우리
    got = point_net.peaks(p, thr=0.3)
    assert len(got) == 1 and (got[0][0], got[0][1]) == (30.0, 30.0)


def test_peaks_떨어진_두_봉우리는_둘_다_남는다():
    p = heat((64, 64), [(10.0, 10.0), (40.0, 40.0)])
    assert len(point_net.peaks(p, thr=0.3)) == 2


def test_peaks_비어있는_히트맵():
    assert point_net.peaks(np.zeros((32, 32), np.float32), thr=0.3) == []


# ------------------------------------------------------------------ 중복 지우기
def test_dedup_가까운_점은_점수가_높은_것만():
    out = point_net.dedup([(10.0, 10.0, 0.4), (11.0, 10.0, 0.9)], radius=3.0)
    assert len(out) == 1 and out[0][2] == 0.9


def test_dedup_반경_밖은_둘_다_남는다():
    out = point_net.dedup([(10.0, 10.0, 0.4), (20.0, 10.0, 0.9)], radius=3.0)
    assert len(out) == 2


def test_dedup_순서에_상관없이_같은_답():
    pts = [(10.0, 10.0, 0.4), (11.0, 10.0, 0.9), (40.0, 40.0, 0.7)]
    a = sorted(point_net.dedup(pts))
    b = sorted(point_net.dedup(list(reversed(pts))))
    assert a == b


# ------------------------------------------------------------------ 구조
def test_UNet_출력이_입력과_같은_크기():
    """출력 보폭이 1 이어야 좌표가 화소 격자로 반올림되지 않습니다."""
    net = point_net.UNet(base=8).eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 3, 64, 64))
    assert out.shape == (1, 1, 64, 64)


def test_UNet_처음에는_배경이라고_답한다():
    """출력 편향이 -4 라 초기 확률이 2% 근처여야 합니다."""
    net = point_net.UNet(base=8).eval()
    with torch.no_grad():
        p = torch.sigmoid(net(torch.zeros(1, 3, 32, 32)))
    assert p.mean() < 0.10


# ------------------------------------------------------------------ 학습된 모델
가중치필요 = pytest.mark.skipif(not os.path.exists(CKPT), reason="가중치가 없습니다")


@가중치필요
def test_체크포인트에_잰_성능이_함께_들어있다():
    net, val = point_net.load(CKPT)
    assert 0.0 < val["f1"] <= 1.0
    assert val["thr"] == pytest.approx(0.15, abs=0.01)


@가중치필요
def test_잡음만_있는_영상에서는_거의_찾지_않는다():
    """물결을 배로 읽으면 여기서 걸립니다."""
    net, _ = point_net.load(CKPT)
    rng = np.random.RandomState(0)
    water = (rng.rand(256, 256, 3) * 30 + 20).astype(np.uint8)
    assert len(point_net.detect(net, water, thr=0.15)) <= 2


@가중치필요
def test_타일_경계에_걸친_배도_한_번만_센다():
    """겹쳐 훑으므로 경계의 배가 두 타일에서 잡힙니다. 합쳐야 합니다."""
    net, _ = point_net.load(CKPT)
    im = (np.random.RandomState(1).rand(700, 700, 3) * 20).astype(np.uint8)
    im[348:353, 300:360] = 255                  # 경계(448) 근처 밝은 선체
    got = point_net.detect(net, im, thr=0.15, tile=512, stride=448)
    xs = sorted(g[0] for g in got)
    for a, b in zip(xs, xs[1:]):
        assert b - a > 3.0, "중복이 남았습니다"


@가중치필요
def test_영상이_타일보다_작아도_돈다():
    net, _ = point_net.load(CKPT)
    small = (np.random.RandomState(2).rand(96, 96, 3) * 20).astype(np.uint8)
    point_net.detect(net, small, thr=0.15)      # 예외가 없으면 통과
