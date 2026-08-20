"""과제 예시 코드 (업무.pdf 2차 업무 p.14~15) 를 원문 그대로 재현한 모듈.

이 모듈의 함수들은 예시 코드의 동작을 바꾸지 않는다. 개선안과 같은 조건에서
비교하려면 대조군이 원문 그대로여야 하기 때문이다. 문제점은 여기서 고치지 않고
실험 결과로 드러낸 뒤 report 에서 다룬다.

원문 (p.15 심화 코드)
    image = cv2.imread('sample.jpg')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    depth_map = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
    h, w = depth_map.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)
    points_3d = np.dstack((X, Y, Z))
"""

from __future__ import annotations

import cv2
import numpy as np


def generate_depth_map(image: np.ndarray) -> np.ndarray:
    """예시 코드 p.14 의 generate_depth_map 을 그대로 옮긴 것.

    실제로는 깊이를 계산하지 않는다. 그레이스케일에 JET 색상표를 입힌
    '깊이처럼 보이는 그림'이며, 반환값은 (H, W, 3) BGR 이다.
    """
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.applyColorMap(grayscale, cv2.COLORMAP_JET)


def image_to_points_3d(image: np.ndarray) -> np.ndarray:
    """예시 코드 p.15 의 3D 포인트 클라우드 변환을 그대로 옮긴 것.

    Returns
    -------
    (H, W, 3) 배열. 각 원소는 (X, Y, Z) = (열 인덱스, 행 인덱스, 밝기) 다.

    주의
        X, Y 는 픽셀 인덱스이고 Z 는 0~255 밝기값이라 세 축의 단위가 서로 다르다.
        카메라 내부 파라미터를 쓰지 않았으므로 원근 투영이 반영되지 않았고,
        결과는 미터 단위 3D 좌표가 아니다.
    """
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    Z = gray.astype(np.float32)
    return np.dstack((X, Y, Z))


def points_3d_to_cloud(points_3d: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """(H, W, 3) 격자를 (N, 3) 포인트 클라우드로 편다.

    예시 코드는 여기까지 가지 않지만, 정량 비교를 하려면 (N, 3) 형태가 필요하다.
    배경(우주)을 걸러내지 않으면 검은 하늘 전체가 Z=0 평면으로 들어가므로
    mask 를 받아 물체 화소만 남긴다.
    """
    points_3d = np.asarray(points_3d, dtype=np.float64)
    if points_3d.ndim != 3 or points_3d.shape[2] != 3:
        raise ValueError(f"(H, W, 3) 배열이 필요합니다: {points_3d.shape}")

    flat = points_3d.reshape(-1, 3)
    if mask is None:
        return flat
    mask = np.asarray(mask).astype(bool)
    if mask.shape != points_3d.shape[:2]:
        raise ValueError(f"마스크 크기 {mask.shape} 가 격자 {points_3d.shape[:2]} 와 다릅니다.")
    return flat[mask.reshape(-1)]


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    """미터 단위 깊이 맵을 JET 색상표로 시각화한다.

    예시 코드의 applyColorMap 사용법은 그대로 두되, 입력을 '진짜 깊이'로
    바꾼 버전이다. 결과 그림을 나란히 놓으면 차이가 눈으로 보인다.
    """
    depth = np.asarray(depth, dtype=np.float64)
    valid = np.isfinite(depth)
    out = np.zeros(depth.shape + (3,), dtype=np.uint8)
    if not np.any(valid):
        return out

    lo = float(np.min(depth[valid]))
    hi = float(np.max(depth[valid]))
    span = hi - lo if hi > lo else 1.0
    norm = np.zeros(depth.shape, dtype=np.uint8)
    norm[valid] = np.clip((depth[valid] - lo) / span * 255.0, 0, 255).astype(np.uint8)

    colored = cv2.applyColorMap(norm, cv2.COLORMAP_JET)
    colored[~valid] = 0
    return colored
