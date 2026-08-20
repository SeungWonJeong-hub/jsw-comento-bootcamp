"""깊이 추정 — 대조군과 스케일 정렬.

여기서 다루는 것은 과제 예시(업무.pdf p.15)의 방식이다.

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    Z = gray.astype(np.float32)          # 밝기를 그대로 깊이로 쓴다

밝기는 표면 반사율과 조명 입사각의 곱이라 거리 정보를 담고 있지 않다.
특히 우주 영상에서는 태양전지판처럼 가까우면서 어두운 면과, 정반사로
밝게 뜨는 먼 면이 함께 나타나 관계가 뒤집힌다.

이 모듈은 그 방식을 버리지 않고 대조군으로 남긴다. 최소자승으로 스케일과
오프셋을 정답에 맞춰 최대한 유리한 조건을 준 뒤에도 오차가 큰지를 확인해야
'개선했다'는 주장에 근거가 생기기 때문이다.

스테레오 삼각측량(Z = fx * B / d)은 이 데이터셋에 쓸 수 없다. SPE3R 은
카메라 병진이 항상 (0, 0, Z) 라 좌우 베이스라인이 0 이다. 자세한 측정값은
src/spe3r.py 의 문서를 참고할 것.
"""

from __future__ import annotations

import cv2
import numpy as np


def brightness_depth(image: np.ndarray, mask: np.ndarray = None,
                     invert: bool = False) -> np.ndarray:
    """밝기를 깊이로 간주한다 (과제 예시 방식, 대조군).

    Parameters
    ----------
    image : (H, W) 또는 (H, W, 3)
    mask : 물체 영역. 배경(우주)은 깊이가 정의되지 않으므로 NaN 으로 둔다.
    invert : True 면 '밝을수록 멀다'로 뒤집는다.

    Returns
    -------
    (H, W) float64. 단위 없는 상대값이며 미터가 아니다.
    """
    if image is None:
        raise ValueError("입력된 이미지가 없습니다.")
    image = np.asarray(image)
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"(H, W) 또는 (H, W, 3) 배열이 필요합니다: {image.shape}")

    relative = gray.astype(np.float64) / 255.0
    if invert:
        relative = 1.0 - relative

    if mask is not None:
        mask = np.asarray(mask).astype(bool)
        if mask.shape != relative.shape:
            raise ValueError(f"마스크 크기 {mask.shape} 가 영상 {relative.shape} 와 다릅니다.")
        relative = np.where(mask, relative, np.nan)
    return relative


def align_scale_shift(pred: np.ndarray, reference: np.ndarray,
                      mask: np.ndarray = None):
    """pred 에 a * pred + b 를 적용해 reference 에 최소자승으로 맞춘다.

    상대 깊이만 내놓는 추정기를 미터 단위 정답과 비교하려면 스케일과 오프셋을
    먼저 정렬해야 한다. 이 정렬은 추정기에게 최대한 유리한 조건을 주는 것이며,
    그럼에도 오차가 크다면 그 방식 자체에 거리 정보가 없다는 뜻이다.

    Returns
    -------
    (정렬된 pred, a, b)
    """
    pred = np.asarray(pred, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if pred.shape != reference.shape:
        raise ValueError(f"크기가 다릅니다: {pred.shape} vs {reference.shape}")

    valid = np.isfinite(pred) & np.isfinite(reference)
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    if valid.sum() < 2:
        raise ValueError("정렬에 쓸 유효 픽셀이 2개 미만입니다.")

    x = pred[valid]
    y = reference[valid]
    if np.allclose(x, x[0]):
        raise ValueError("pred 가 상수라서 스케일을 결정할 수 없습니다.")

    A = np.stack([x, np.ones_like(x)], axis=1)
    (a, b), *_ = np.linalg.lstsq(A, y, rcond=None)
    return a * pred + b, float(a), float(b)


def depth_correlation(pred: np.ndarray, reference: np.ndarray,
                      mask: np.ndarray = None) -> float:
    """추정 깊이와 정답 깊이의 피어슨 상관계수.

    1 에 가까우면 스케일만 맞추면 되는 상태, 0 근처면 거리 정보가 없는 상태,
    음수면 관계가 뒤집힌 상태다. RMSE 하나만 보는 것보다 원인 진단에 좋다.
    """
    pred = np.asarray(pred, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    valid = np.isfinite(pred) & np.isfinite(reference)
    if mask is not None:
        valid &= np.asarray(mask).astype(bool)
    if valid.sum() < 2:
        raise ValueError("상관계수 계산에 쓸 유효 픽셀이 2개 미만입니다.")

    x = pred[valid]
    y = reference[valid]
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])
