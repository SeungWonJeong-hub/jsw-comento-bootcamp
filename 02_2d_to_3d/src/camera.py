"""핀홀 카메라 모델과 자세 표현.

좌표계 규약 (OpenCV / SPEED+ 표준)
    x : 오른쪽,  y : 아래,  z : 광축 방향(카메라 앞이 양수)
    깊이(depth)는 광축 방향 성분 Z 를 뜻하며 광선 길이가 아닙니다.

자세 라벨
    q_vbs2tango_true : 카메라(vbs) 좌표계 벡터를 타겟 동체(tango) 좌표계로
                       옮기는 쿼터니언. 스칼라가 앞에 오는 [w, x, y, z] 순서.
    r_Vo2To_vbs_true : 카메라 좌표계에서 본 타겟 중심 위치 [m].
"""

from __future__ import annotations

import numpy as np


class PinholeCamera:
    """왜곡이 없는 핀홀 카메라."""

    def __init__(self, width, height, fx, fy=None, cx=None, cy=None):
        if width <= 0 or height <= 0:
            raise ValueError(f"영상 크기는 양수여야 합니다: ({width}, {height})")
        if fx <= 0:
            raise ValueError(f"초점거리는 양수여야 합니다: fx={fx}")
        fy = float(fx) if fy is None else float(fy)
        if fy <= 0:
            raise ValueError(f"초점거리는 양수여야 합니다: fy={fy}")

        self.width = int(width)
        self.height = int(height)
        self.fx = float(fx)
        self.fy = fy
        self.cx = (self.width - 1) / 2.0 if cx is None else float(cx)
        self.cy = (self.height - 1) / 2.0 if cy is None else float(cy)

    def __repr__(self):
        return (f"PinholeCamera({self.width}x{self.height}, "
                f"f=({self.fx:.2f}, {self.fy:.2f}), c=({self.cx:.1f}, {self.cy:.1f}))")

    @property
    def K(self) -> np.ndarray:
        """3x3 내부 파라미터 행렬."""
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    @property
    def shape(self) -> tuple:
        """(height, width) — numpy 배열 순서."""
        return self.height, self.width

    def pixel_rays(self) -> np.ndarray:
        """픽셀마다의 시선 벡터를 (H, W, 3) 로 반환합니다.

        z 성분을 1 로 고정했기 때문에 광선 매개변수 s 가 곧 깊이 Z 가 됩니다.
        즉  P_cam = origin + s * dir  이고  P_cam.z = s (origin.z = 0 일 때).
        """
        uu, vv = np.meshgrid(np.arange(self.width, dtype=np.float64),
                             np.arange(self.height, dtype=np.float64))
        x = (uu - self.cx) / self.fx
        y = (vv - self.cy) / self.fy
        return np.stack([x, y, np.ones_like(x)], axis=-1)

    def project(self, points: np.ndarray) -> np.ndarray:
        """카메라 좌표계 3D 점 (N, 3) -> 픽셀 좌표 (N, 2)."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"(N, 3) 배열이 필요합니다. 입력: {points.shape}")
        z = points[:, 2]
        if np.any(z <= 0):
            raise ValueError("카메라 뒤쪽(z <= 0)의 점은 투영할 수 없습니다.")
        u = self.fx * points[:, 0] / z + self.cx
        v = self.fy * points[:, 1] / z + self.cy
        return np.stack([u, v], axis=-1)

    def unproject(self, depth: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        """깊이 맵 -> 카메라 좌표계 3D 점 (N, 3)."""
        if depth is None:
            raise ValueError("입력된 깊이 맵이 없습니다.")
        depth = np.asarray(depth, dtype=np.float64)
        if depth.shape != self.shape:
            raise ValueError(f"깊이 맵 크기 {depth.shape} 가 카메라 {self.shape} 와 다릅니다.")

        valid = np.isfinite(depth) & (depth > 0)
        if mask is not None:
            mask = np.asarray(mask)
            if mask.shape != self.shape:
                raise ValueError(f"마스크 크기 {mask.shape} 가 카메라 {self.shape} 와 다릅니다.")
            valid &= mask.astype(bool)

        return self.pixel_rays()[valid] * depth[valid][:, None]

    def in_view(self, uv: np.ndarray) -> np.ndarray:
        """픽셀 좌표가 영상 안에 있는지 (N,) bool."""
        uv = np.asarray(uv)
        return ((uv[:, 0] >= 0) & (uv[:, 0] <= self.width - 1)
                & (uv[:, 1] >= 0) & (uv[:, 1] <= self.height - 1))


def quaternion_to_rotation(q, scalar_first: bool = True) -> np.ndarray:
    """단위 쿼터니언 -> 3x3 회전 행렬.

    반환되는 R 은 q 가 표현하는 좌표계 변환을 그대로 따릅니다.
    """
    q = np.asarray(q, dtype=np.float64).reshape(4)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"쿼터니언의 크기가 0 에 가깝습니다: {q}")
    q = q / n
    if scalar_first:
        w, x, y, z = q
    else:
        x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def rotation_angle_deg(R_a: np.ndarray, R_b: np.ndarray) -> float:
    """두 회전 사이의 각도 [deg]."""
    R = np.asarray(R_a) @ np.asarray(R_b).T
    cos = (np.trace(R) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


class Pose:
    """타겟 동체 좌표계 -> 카메라 좌표계 강체 변환.

        p_cam = R @ p_body + t
    """

    def __init__(self, R: np.ndarray, t):
        R = np.asarray(R, dtype=np.float64)
        if R.shape != (3, 3):
            raise ValueError(f"회전 행렬은 (3, 3) 이어야 합니다: {R.shape}")
        if not np.allclose(R @ R.T, np.eye(3), atol=1e-6):
            raise ValueError("회전 행렬이 직교하지 않습니다.")
        self.R = R
        self.t = np.asarray(t, dtype=np.float64).reshape(3)

    def __repr__(self):
        return f"Pose(t={np.round(self.t, 4).tolist()})"

    def apply(self, points_body: np.ndarray) -> np.ndarray:
        """동체 좌표계 점 (N, 3) -> 카메라 좌표계 점 (N, 3)."""
        points_body = np.asarray(points_body, dtype=np.float64)
        if points_body.ndim != 2 or points_body.shape[1] != 3:
            raise ValueError(f"(N, 3) 배열이 필요합니다: {points_body.shape}")
        return points_body @ self.R.T + self.t

    def inverse_apply(self, points_cam: np.ndarray) -> np.ndarray:
        """카메라 좌표계 점 -> 동체 좌표계 점."""
        points_cam = np.asarray(points_cam, dtype=np.float64)
        return (points_cam - self.t) @ self.R
