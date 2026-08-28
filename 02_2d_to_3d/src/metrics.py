"""3D 복원과 깊이 추정의 정량 평가.

Chamfer 거리와 F-score 는 3D 복원 평가에서 널리 쓰는 정의를 따릅니다.
정규화된 모델 좌표계에서 계산하므로 단위는 없습니다.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial import cKDTree


def chamfer_distance(pred: np.ndarray, target: np.ndarray, norm: int = 1) -> dict:
    """두 포인트 클라우드 사이의 Chamfer 거리.

        d(A -> B) = mean_{a in A} min_{b in B} ||a - b||

    양방향 평균을 대칭 Chamfer 로 씁니다. 단방향 값도 함께 돌려주는데,
    복원 결과가 정답보다 부풀었는지(pred -> target 이 작고 target -> pred 가 큼)
    깎였는지를 구분할 수 있기 때문입니다.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.ndim != 2 or pred.shape[1] != 3:
        raise ValueError(f"pred 는 (N, 3) 이어야 합니다: {pred.shape}")
    if target.ndim != 2 or target.shape[1] != 3:
        raise ValueError(f"target 은 (N, 3) 이어야 합니다: {target.shape}")
    if len(pred) == 0 or len(target) == 0:
        raise ValueError("빈 포인트 클라우드는 비교할 수 없습니다.")
    if norm not in (1, 2):
        raise ValueError(f"norm 은 1 또는 2 여야 합니다: {norm}")

    d_pt, _ = cKDTree(target).query(pred, k=1)
    d_tp, _ = cKDTree(pred).query(target, k=1)

    if norm == 2:
        d_pt = d_pt ** 2
        d_tp = d_tp ** 2

    forward = float(d_pt.mean())
    backward = float(d_tp.mean())
    return {
        "chamfer": (forward + backward) / 2.0,
        "pred_to_target": forward,
        "target_to_pred": backward,
    }


def f_score(pred: np.ndarray, target: np.ndarray, threshold: float) -> dict:
    """임계 거리 안에 대응점이 있는 비율로 정밀도/재현율을 계산합니다."""
    if threshold <= 0:
        raise ValueError(f"임계 거리는 양수여야 합니다: {threshold}")
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    d_pt, _ = cKDTree(target).query(pred, k=1)
    d_tp, _ = cKDTree(pred).query(target, k=1)

    precision = float((d_pt < threshold).mean())
    recall = float((d_tp < threshold).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f,
            "threshold": float(threshold)}


def depth_metrics(pred: np.ndarray, target: np.ndarray,
                  mask: np.ndarray = None) -> dict:
    """깊이 맵 오차 지표.

    두 맵 모두 유한한 픽셀에서만 계산하고, 그 비율(valid_ratio)도 함께 보고합니다.
    유효 픽셀이 적으면 오차가 작아도 좋은 결과가 아니기 때문입니다.
    """
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if pred.shape != target.shape:
        raise ValueError(f"크기가 다릅니다: {pred.shape} vs {target.shape}")

    domain = np.isfinite(target)
    if mask is not None:
        domain &= np.asarray(mask).astype(bool)
    valid = domain & np.isfinite(pred)

    n_domain = int(domain.sum())
    n_valid = int(valid.sum())
    if n_valid == 0:
        # 정상 경로와 키가 같아야 합니다. median_abs 를 빼 두면 호출부가
        # m["median_abs"] 에서 KeyError 로 죽습니다. 값이 없는 것은 NaN 으로 알립니다.
        return {"rmse": float("nan"), "mae": float("nan"),
                "median_abs": float("nan"),
                "valid_ratio": 0.0, "n_valid": 0, "n_domain": n_domain}

    err = pred[valid] - target[valid]
    return {
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "mae": float(np.mean(np.abs(err))),
        "median_abs": float(np.median(np.abs(err))),
        "valid_ratio": n_valid / n_domain if n_domain else 0.0,
        "n_valid": n_valid,
        "n_domain": n_domain,
    }


def mask_iou(pred: np.ndarray, target: np.ndarray) -> float:
    """두 이진 마스크의 IoU. 자세 규약이 맞는지 검증할 때 씁니다."""
    pred = np.asarray(pred).astype(bool)
    target = np.asarray(target).astype(bool)
    if pred.shape != target.shape:
        raise ValueError(f"크기가 다릅니다: {pred.shape} vs {target.shape}")
    union = np.logical_or(pred, target).sum()
    if union == 0:
        return 0.0
    return float(np.logical_and(pred, target).sum() / union)
