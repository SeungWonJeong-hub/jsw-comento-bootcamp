# -*- coding: utf-8 -*-
"""점 탐지기 — 톈진·도쿄만용.

이 두 항만의 정답 라벨(Allen AI Skylight)은 박스가 아니라 **선박 중심점**
입니다. 없는 박스를 지어내는 대신 중심점을 직접 맞히도록 학습했습니다.

구조는 CenterNet 계열입니다. 정답점마다 가우시안을 찍은 히트맵을 만들고,
UNet 이 그 히트맵을 예측한 뒤 지역 최대점을 뽑습니다. 출력 보폭이 1 이라
좌표가 화소 격자로 반올림되지 않습니다.

가우시안 폭 sigma 는 1.0 입니다. 2.0 으로 학습한 것과 같은 조건에서
비교했을 때 F1@3px 가 0.340 에서 0.491 로 올랐습니다.
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def block(cin, cout):
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True),
        nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout),
        nn.ReLU(inplace=True))


class UNet(nn.Module):
    """얕은 U-Net. 세 번만 줄입니다 — 더 줄이면 10 화소 배가 사라집니다."""

    def __init__(self, base=32):
        super().__init__()
        b = base
        self.e1 = block(3, b); self.e2 = block(b, b * 2)
        self.e3 = block(b * 2, b * 4); self.e4 = block(b * 4, b * 8)
        self.u3 = nn.ConvTranspose2d(b * 8, b * 4, 2, 2); self.d3 = block(b * 8, b * 4)
        self.u2 = nn.ConvTranspose2d(b * 4, b * 2, 2, 2); self.d2 = block(b * 4, b * 2)
        self.u1 = nn.ConvTranspose2d(b * 2, b, 2, 2); self.d1 = block(b * 2, b)
        self.out = nn.Conv2d(b, 1, 1)
        # 처음부터 '거의 배경' 이라고 답하게 편향을 줍니다. 안 그러면
        # 초기 몇 epoch 을 전부 배경이라고 배우는 데 씁니다.
        nn.init.constant_(self.out.bias, -4.0)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        e1 = self.e1(x); e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2)); e4 = self.e4(self.pool(e3))
        d3 = self.d3(torch.cat([self.u3(e4), e3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), e2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), e1], 1))
        return self.out(d1)


def peaks(prob, thr, nms=3):
    """국소 최대값 -> 점 목록. 학습·평가와 같은 식을 씁니다."""
    if isinstance(prob, np.ndarray):
        prob = torch.from_numpy(prob)
    p = prob.view(1, 1, *prob.shape[-2:])
    m = F.max_pool2d(p, nms, stride=1, padding=nms // 2)
    keep = (p == m) & (p >= thr)
    return [(float(x), float(y), float(p[0, 0, y, x]))
            for y, x in keep[0, 0].nonzero().tolist()]


def load(path, device="cpu"):
    """학습된 체크포인트를 읽어 모델과 저장된 성능을 돌려줍니다."""
    ck = torch.load(path, map_location=device)
    net = UNet(ck.get("base", 32))
    net.load_state_dict(ck["state"])
    net.eval().to(device)
    return net, ck.get("val", {})


@torch.no_grad()
def detect(net, rgb, thr=0.15, tile=512, stride=448, device="cpu"):
    """큰 영상을 겹치게 잘라 훑고 중복을 없앱니다.

    타일 경계에 걸친 배가 두 번 잡히므로, 이어붙인 뒤 nms 반경 안의
    점을 점수가 높은 것만 남깁니다.
    """
    H, W = rgb.shape[:2]
    found = []
    ys = list(range(0, max(H-tile, 0)+1, stride)) or [0]
    xs = list(range(0, max(W-tile, 0)+1, stride)) or [0]
    if ys[-1] + tile < H:
        ys.append(H-tile)
    if xs[-1] + tile < W:
        xs.append(W-tile)
    for y in ys:
        for x in xs:
            crop = rgb[y:y+tile, x:x+tile]
            if crop.shape[0] < 8 or crop.shape[1] < 8:
                continue
            t = torch.from_numpy(crop.astype(np.float32)/255.0)
            t = t.permute(2, 0, 1)[None].to(device)
            pr = torch.sigmoid(net(t))[0, 0]
            for px, py, s in peaks(pr, thr):
                found.append((px+x, py+y, s))
    return dedup(found)


def dedup(pts, radius=3.0):
    """가까운 점 중 점수가 높은 것만 남깁니다."""
    out = []
    for x, y, s in sorted(pts, key=lambda p: -p[2]):
        if all((x-a)**2 + (y-b)**2 > radius*radius for a, b, _ in out):
            out.append((x, y, s))
    return out
