"""스테레오 삼각측량으로 깊이 맵을 만들고 3D 포인트 클라우드로 변환한다.

과제가 지정한 경로 그대로다.

    영상 2장  ->  시차 d  ->  깊이 맵 Z = f * B / d  ->  포인트 클라우드

SPE3R 에서 두 번째 시점을 어떻게 얻는가
    카메라 좌표계 병진은 항상 (0, 0, Z) 라 옆으로 움직이지 않는다. 그러나 타겟이
    매 프레임 무작위로 회전하므로, 두 뷰 사이의 상대 자세를 계산하면 타겟 기준
    으로는 카메라가 궤도를 돈 것과 같아진다.

        R_ij = R_j · R_i^T
        t_ij = t_j - R_ij · t_i

    거리 5~6 m 에서 회전각 4 도면 유효 베이스라인이 약 0.45 m 생긴다.
    이 (R_ij, t_ij) 를 cv2.stereoRectify 에 넣으면 평행 정렬된 쌍이 되어
    표준 StereoSGBM 을 그대로 쓸 수 있다.

    다만 조건이 까다롭다. 회전각이 크면 표면 대응이 끊기고, 병진이 광축 방향에
    치우치면(전방 이동) 에피폴이 화면 안으로 들어와 정합이 무너진다. 그래서
    회전각이 작으면서 횡방향 성분이 지배적인 쌍만 골라야 한다.
"""

from __future__ import annotations

import cv2
import numpy as np

from .camera import PinholeCamera


def relative_pose(pose_i, pose_j):
    """뷰 i 카메라에서 뷰 j 카메라로 가는 상대 강체 변환.

    두 뷰 모두 같은 강체(타겟)를 보고 있으므로, 타겟을 고정으로 두면
    카메라가 움직인 것과 같다.

        p_j = R_ij · p_i + t_ij
        R_ij = R_j · R_i^T,   t_ij = t_j - R_ij · t_i
    """
    R_ij = pose_j.R @ pose_i.R.T
    t_ij = pose_j.t - R_ij @ pose_i.t
    return R_ij, t_ij


def baseline_geometry(pose_i, pose_j) -> dict:
    """두 뷰 사이 베이스라인의 크기와 방향 성분을 정리한다."""
    R_ij, t_ij = relative_pose(pose_i, pose_j)
    lateral = float(np.linalg.norm(t_ij[:2]))
    forward = float(abs(t_ij[2]))
    cos = np.clip((np.trace(R_ij) - 1.0) / 2.0, -1.0, 1.0)
    return {
        "R_ij": R_ij,
        "t_ij": t_ij,
        "baseline": float(np.linalg.norm(t_ij)),
        "lateral": lateral,
        "forward": forward,
        # 이 비가 클수록 옆으로 나란한 배치에 가까워 정합이 안정적이다.
        "lateral_ratio": lateral / (forward + 1e-9),
        "rotation_deg": float(np.degrees(np.arccos(cos))),
    }


def find_pairs(model, max_rotation_deg: float = 8.0,
               min_lateral_ratio: float = 2.0, limit: int = None) -> list:
    """스테레오에 쓸 만한 뷰 쌍을 고른다.

    선별 기준
        1. 두 뷰의 자세 차이가 작을 것 — 표면 대응이 살아 있어야 한다
        2. 베이스라인의 횡방향 성분이 전방 성분보다 클 것 — 전방 이동은
           에피폴이 화면 안에 들어와 정합이 무너진다

    Returns
    -------
    baseline_geometry() 결과에 인덱스를 더한 dict 목록. 횡방향 비 내림차순.
    """
    if max_rotation_deg <= 0:
        raise ValueError(f"최대 회전각은 양수여야 합니다: {max_rotation_deg}")

    q = np.array([lb["q_vbs2tango_true"] for lb in model.labels], dtype=np.float64)
    q /= np.linalg.norm(q, axis=1, keepdims=True)
    gram = np.abs(q @ q.T)
    np.fill_diagonal(gram, 0.0)
    angles = np.degrees(2.0 * np.arccos(np.clip(gram, 0.0, 1.0)))

    iu = np.triu_indices(len(model), k=1)
    close = np.flatnonzero(angles[iu] < max_rotation_deg)

    out = []
    for c in close:
        i, j = int(iu[0][c]), int(iu[1][c])
        geo = baseline_geometry(model.pose(i), model.pose(j))
        if geo["lateral_ratio"] < min_lateral_ratio:
            continue
        geo.update({"i": i, "j": j, "distance": float(np.linalg.norm(model.pose(i).t))})
        out.append(geo)

    out.sort(key=lambda g: -g["lateral_ratio"])
    return out[:limit] if limit else out


class RectifiedPair:
    """평행 정렬된 스테레오 쌍과 그 기하."""

    def __init__(self, camera, R_ij, t_ij, alpha: float = -1.0, size=None):
        size = (camera.width, camera.height) if size is None else tuple(size)
        t_ij = np.ascontiguousarray(t_ij, dtype=np.float64).reshape(3)
        if np.linalg.norm(t_ij) < 1e-9:
            raise ValueError("두 시점의 베이스라인이 0 입니다. 스테레오가 성립하지 않습니다.")
        K = camera.K
        D = np.zeros(5)

        R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
            K, D, K, D, (camera.width, camera.height),
            np.ascontiguousarray(R_ij, dtype=np.float64), t_ij.reshape(3, 1),
            flags=cv2.CALIB_ZERO_DISPARITY, alpha=alpha, newImageSize=size)

        # 베이스라인이 세로 방향이면 OpenCV 는 오프셋을 P2[1, 3] 에 싣는다.
        self.horizontal = abs(P2[0, 3]) >= abs(P2[1, 3])
        if self.horizontal:
            baseline = abs(P2[0, 3] / P2[0, 0])
        else:
            baseline = abs(P2[1, 3] / P2[1, 1])
        if baseline < 1e-9:
            raise ValueError("정렬 후 베이스라인이 0 입니다. 스테레오가 성립하지 않습니다.")

        self.size = size
        self.R1, self.R2, self.P1, self.P2, self.Q = R1, R2, P1, P2, Q
        self.baseline = float(baseline)
        self.focal = float(P1[0, 0])
        if self.focal <= 0:
            raise ValueError(f"정렬 후 초점거리가 유효하지 않습니다: {self.focal}")

        self.camera = PinholeCamera(size[0], size[1], self.focal, P1[1, 1],
                                    P1[0, 2], P1[1, 2])
        self._map1 = cv2.initUndistortRectifyMap(K, D, R1, P1, size, cv2.CV_32FC1)
        self._map2 = cv2.initUndistortRectifyMap(K, D, R2, P2, size, cv2.CV_32FC1)

    def expected_disparity(self, distance: float) -> float:
        """주어진 거리에서 예상되는 시차 [pixel]."""
        return self.focal * self.baseline / float(distance)

    def remap(self, left, right, mask=None):
        """원본 영상을 정렬된 좌표로 옮긴다."""
        L = cv2.remap(left, *self._map1, cv2.INTER_LINEAR)
        R = cv2.remap(right, *self._map2, cv2.INTER_LINEAR)
        M = None
        if mask is not None:
            M = cv2.remap(np.asarray(mask).astype(np.uint8) * 255, *self._map1,
                          cv2.INTER_NEAREST) > 0
        if not self.horizontal:
            # 세로 스테레오는 90도 돌려 가로로 만든 뒤 표준 매처를 쓴다.
            L, R = np.rot90(L).copy(), np.rot90(R).copy()
            if M is not None:
                M = np.rot90(M).copy()
        return L, R, M

    def unrotate(self, image):
        """세로 스테레오에서 돌려 놓았던 영상을 원래 방향으로 되돌린다."""
        return np.rot90(image, k=-1).copy() if not self.horizontal else image

    def to_body(self, points_rect, pose_i):
        """정렬된 왼쪽 카메라 좌표계의 점을 타겟 동체 좌표계로 옮긴다.

        정렬은 왼쪽 카메라를 R1 만큼 돌린 것이므로 먼저 R1 을 되돌린 뒤,
        뷰 i 의 자세를 역으로 적용한다.
        """
        points_left = np.asarray(points_rect, dtype=np.float64) @ self.R1
        return pose_i.inverse_apply(points_left)


def compute_disparity(left, right, num_disparities: int, block_size: int = 3,
                      min_disparity: int = 0, uniqueness: int = 5) -> np.ndarray:
    """StereoSGBM 으로 시차 맵을 만든다. 매칭 실패는 NaN.

    block_size 기본값을 3 으로 둔 근거
        SGBM 은 블록 안에서 시차가 일정하다고 가정하므로, 블록이 클수록 기울어진
        면에서 깊이가 뭉개진다. 정답 깊이가 있는 합성 장면에서 잰 값:

            block  RMSE     중앙값   5cm이내  유효화소  복원 깊이 폭 (정답 0.898)
              3    0.0413   0.0105    95.4%    97.6%    0.915
              5    0.0439   0.0098    96.2%    98.2%    0.911
              7    0.0543   0.0093    95.5%    99.0%    0.908
              9    0.0552   0.0092    95.6%    99.3%    0.903

        블록이 커지면 유효 화소와 중앙값은 조금 좋아지지만 RMSE 는 나빠진다.
        RMSE 가 가장 낮은 3 을 기본값으로 두되 차이는 크지 않다.
        uniquenessRatio 는 1~10 사이에서 결과가 거의 바뀌지 않아 5 를 유지한다.

    주의 — 성능은 파라미터보다 표면 무늬에 더 크게 좌우된다
        같은 장면에서 조명 방향만 뒤집어 카메라를 향한 면을 그늘에 넣으면
        복원 깊이 폭이 0.915 m 에서 0.572 m 로 무너진다. 기하 조건은 그대로이고
        보이는 무늬만 사라진 것인데, 블록 크기를 3~9 어디로 바꿔도 회복되지 않는다.
        파라미터를 조정하기 전에 입력 영상에 매칭할 단서가 있는지부터 확인할 것.
    """
    if left is None or right is None:
        raise ValueError("좌/우 영상이 모두 필요합니다.")
    left = np.asarray(left)
    right = np.asarray(right)
    if left.shape != right.shape:
        raise ValueError(f"좌/우 영상 크기가 다릅니다: {left.shape} vs {right.shape}")
    if num_disparities <= 0 or num_disparities % 16 != 0:
        raise ValueError(f"num_disparities 는 16 의 양의 배수여야 합니다: {num_disparities}")

    if left.ndim == 3:
        left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
        right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    matcher = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * block_size ** 2,
        P2=32 * block_size ** 2,
        disp12MaxDiff=1,
        uniquenessRatio=uniqueness,
        speckleWindowSize=100,
        speckleRange=2,
        preFilterCap=31,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )
    raw = matcher.compute(left.astype(np.uint8), right.astype(np.uint8))
    disparity = raw.astype(np.float64) / 16.0
    return np.where(disparity > min_disparity, disparity, np.nan)


def filter_disparity(disparity, max_speckle_size: int = 400,
                     max_diff_px: float = 1.0, median_kernel: int = 3) -> np.ndarray:
    """시차 맵의 이상치를 제거한다 (OpenCV 표준 후처리).

    왜 필요한가
        SGBM 은 무텍스처 영역과 가림(occlusion) 경계에서 이웃과 동떨어진 시차를
        작은 덩어리로 만들어낸다. 화소 수는 적지만 시차가 조금만 틀려도 깊이는
        Z = f·B/d 를 따라 크게 튀므로 RMSE 를 지배해 버린다.

    측정 효과 (SPE3R aqua, img000638/img000784 기준)
        필터 없음                RMSE 0.1239  중앙값 0.0101  5cm이내 87.0%
        filterSpeckles(400, 1)   RMSE 0.0679  중앙값 0.0096  5cm이내 89.9%
        + median 3x3             RMSE 0.0677  중앙값 0.0093  5cm이내 90.3%

        중앙값은 거의 그대로인데 RMSE 가 절반으로 준다. 즉 이 필터는 정상 화소를
        건드리지 않고 소수의 심한 오정합만 걷어낸다.

    Parameters
    ----------
    max_speckle_size : 이보다 작은 연결 덩어리는 제거한다 [pixel]
    max_diff_px : 같은 덩어리로 볼 시차 차이 한계 [pixel]
    median_kernel : 중앙값 필터 크기. 0 이면 적용하지 않는다.
    """
    disparity = np.asarray(disparity, dtype=np.float64)
    finite = np.isfinite(disparity)
    if not np.any(finite):
        return disparity.copy()
    if max_speckle_size < 0 or max_diff_px < 0:
        raise ValueError("필터 파라미터는 0 이상이어야 합니다.")

    out = disparity.copy()
    if max_speckle_size > 0:
        # filterSpeckles 는 16 배 고정소수점 int16 을 요구한다.
        buf = np.where(finite, disparity * 16.0, -16.0).astype(np.int16)
        cv2.filterSpeckles(buf, -16, int(max_speckle_size),
                           int(round(max_diff_px * 16)))
        out = np.where(buf > 0, buf.astype(np.float64) / 16.0, np.nan)

    if median_kernel and median_kernel >= 3:
        keep = np.isfinite(out)
        filled = np.where(keep, out, 0.0).astype(np.float32)
        smoothed = cv2.medianBlur(filled, int(median_kernel))
        out = np.where(keep, smoothed.astype(np.float64), np.nan)

    return out


def disparity_to_depth(disparity, focal: float, baseline: float) -> np.ndarray:
    """시차 -> 깊이.  Z = f * B / d.  d <= 0 은 NaN.

    이 한 줄이 과제가 요구한 '깊이 맵 생성'의 본체다. 밝기와 달리 이 값은
    카메라 기하에서 유도되므로 미터 단위 물리량이다.
    """
    if focal <= 0:
        raise ValueError(f"초점거리는 양수여야 합니다: {focal}")
    if baseline <= 0:
        raise ValueError(f"베이스라인은 양수여야 합니다: {baseline}")

    disparity = np.asarray(disparity, dtype=np.float64)
    depth = np.full(disparity.shape, np.nan)
    good = np.isfinite(disparity) & (disparity > 0)
    depth[good] = focal * baseline / disparity[good]
    return depth


def depth_resolution(depth: float, focal: float, baseline: float) -> float:
    """시차 1 pixel 오차가 만드는 깊이 오차 [m].   dZ = Z^2 / (f * B)"""
    return float(depth) ** 2 / (focal * baseline)


def reconstruct(pair: "RectifiedPair", left, right, mask=None,
                distance: float = None, disparity_margin: float = 1.6,
                depth_range: tuple = None, postfilter: bool = True) -> dict:
    """정렬된 쌍에서 깊이 맵과 포인트 클라우드를 만든다.

    Parameters
    ----------
    depth_range : (lo, hi) 밖의 깊이를 버린다. 타겟 크기를 알면 물리적으로
        불가능한 값을 걸러낼 수 있다. 호출부(run_3d_experiment.py)는
        [d0 - R, d0 + R] 을 쓰는데, 창의 **폭** 2R 은 타겟 치수에서 오지만
        창의 **중심** d0 는 데이터셋의 정답 거리다. 화소 단위 정답 깊이를
        쓰는 것은 아니지만 정답 자세에 의존하는 것은 맞다. 실제 상대항법이면
        추정 거리를 넣어야 한다. README 7절 한계에 함께 적었다.
    postfilter : filter_disparity() 로 이상치를 제거할지 여부.

    Returns
    -------
    dict(left, right, mask, disparity, depth, points, valid, num_disparities)
        points 는 정렬된 왼쪽 카메라 좌표계 (N, 3) 이다.
    """
    L, R, M = pair.remap(left, right, mask)

    if distance is None:
        num_disp = 128
    else:
        expected = pair.expected_disparity(distance)
        num_disp = int(np.ceil(expected * disparity_margin / 16)) * 16
        num_disp = max(16, min(num_disp, 16 * ((pair.size[0] - 16) // 16)))

    disparity = compute_disparity(L, R, num_disparities=num_disp)
    if postfilter:
        disparity = filter_disparity(disparity)
    depth = disparity_to_depth(disparity, pair.focal, pair.baseline)

    if M is not None:
        depth = np.where(M, depth, np.nan)
    if depth_range is not None:
        lo, hi = depth_range
        depth = np.where((depth >= lo) & (depth <= hi), depth, np.nan)

    points = pair.camera.unproject(depth)
    valid = np.isfinite(depth)
    return {
        "left": L, "right": R, "mask": M,
        "disparity": disparity, "depth": depth,
        "points": points, "valid": valid,
        "num_disparities": num_disp,
    }
