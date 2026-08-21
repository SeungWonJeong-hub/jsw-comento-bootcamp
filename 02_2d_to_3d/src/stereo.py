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

# SGBM 정합 블록 크기의 기본값. 근거는 compute_disparity() 의 설명에 있다.
# 한 곳에서만 정하지 않으면 reconstruct() 의 기본값이 조용히 다른 값을 쓴다.
DEFAULT_BLOCK_SIZE = 11


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
    """평행 정렬된 스테레오 쌍과 그 기하.

    정렬 창을 원본과 같은 크기로 두면 안 된다
        정렬은 두 카메라를 평행하게 만드는 회전이라, 정렬된 평면에서 원본 화면은
        기울어진 사각형이 된다. 그 사각형의 경계 상자는 원본보다 크다. 창을 원본
        크기 그대로 두면 OpenCV 가 잡아 주는 위치에서 잘려 나가고, 타겟이 그
        바깥에 걸리면 조용히 사라진다. 후보 20 쌍 중 14 쌍에서 타겟이 창 테두리에
        닿아 있었고, 심한 쌍은 실루엣의 절반을 잃고 있었다.

        잘린 부분은 마스크에서도 함께 사라지므로 "커버리지가 낮다"로도 안 잡힌다.
        채점 영역 자체가 줄어들 뿐이다.

    fit_window=True (기본)
        원본 네 귀퉁이가 정렬 평면 어디에 떨어지는지 계산해 아무것도 잘리지 않는
        최소 창을 잡는다. 초점거리는 그대로 두고 주점만 옮기므로 시차와
        베이스라인은 변하지 않는다. 최적 쌍에서 채점 영역이 8,442 -> 9,485 화소로
        늘고 RMSE·중앙값·5cm·유효화소가 모두 함께 좋아진다.

        newImageSize 로 창을 키우는 방법과는 다르다. 그쪽은 OpenCV 가 화각을
        다시 잡아 초점거리와 주점이 함께 바뀌고, 2 배만 줘도 타겟이 창 밖으로
        나가 아무것도 복원되지 않는다.
    """

    @staticmethod
    def _fit_window(camera, R1, P1, P2, max_side):
        """원본이 잘리지 않는 최소 정렬 창과 그에 맞춘 투영 행렬을 만든다."""
        W, H = camera.width, camera.height
        corners = np.array([[0, 0], [W - 1, 0], [0, H - 1], [W - 1, H - 1]],
                           dtype=np.float64).reshape(-1, 1, 2)
        pts = cv2.undistortPoints(corners, camera.K, np.zeros(5),
                                  R=R1, P=P1).reshape(-1, 2)
        lo = np.floor(pts.min(axis=0))
        hi = np.ceil(pts.max(axis=0))
        new_w, new_h = int(hi[0] - lo[0]) + 1, int(hi[1] - lo[1]) + 1

        # 회전이 심한 쌍에서는 경계 상자가 수천 화소까지 커진다. 정합 비용이
        # 넓이에 비례하므로 상한을 두고, 넘으면 전체를 균일하게 축소한다.
        # 축소해도 잘리지는 않는다.
        s = min(1.0, max_side / max(new_w, new_h))
        new_w, new_h = max(16, int(new_w * s)), max(16, int(new_h * s))

        out = []
        for P in (P1, P2):
            Q = P.copy()
            Q[:2, :] *= s                    # 초점거리·주점·베이스라인 항을 함께
            Q[0, 2] -= lo[0] * s
            Q[1, 2] -= lo[1] * s
            out.append(Q)
        return (new_w, new_h), out[0], out[1]

    def __init__(self, camera, R_ij, t_ij, alpha: float = -1.0, size=None,
                 fit_window: bool = True, max_side: int = 1024):
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

        if fit_window:
            size, P1, P2 = self._fit_window(camera, R1, P1, P2, max_side)

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

    @property
    def match_width(self) -> int:
        """매처가 실제로 보는 영상의 가로 길이.

        세로 스테레오는 remap() 에서 90 도 돌려 매칭하므로 가로 길이가 창의
        세로 길이가 된다. 시차 탐색 폭의 상한은 이 값으로 잡아야 한다.
        정사각 영상에서는 둘이 같아 차이가 드러나지 않는다.
        """
        return self.size[0] if self.horizontal else self.size[1]

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


def reference_depth(pair: "RectifiedPair", pose, points_body) -> np.ndarray:
    """동체 좌표계 점들을 정렬된 왼쪽 카메라로 투영해 기준 깊이 맵을 만든다.

    reconstruct() 와 짝이다. 둘이 같은 좌표계를 돌려줘야 화소 단위로 비교할 수
    있는데, 좌표 규약의 양쪽이 서로 다른 파일에 흩어져 있으면 한쪽만 고치는
    사고가 난다. 실제로 그랬다. reconstruct 가 세로 쌍에서 돌아간 채로 반환하던
    것을 고치면서 이 함수의 unrotate 를 안 걷어내, 기준 깊이가 90 도 어긋난
    채로 채점되고 있었다. 마스크 안에서 기준 깊이가 정의된 비율이 27.9% 까지
    떨어졌다. 그래서 같은 모듈로 옮기고 테스트를 붙였다.

    pair.camera 는 돌리기 전 정렬 카메라이므로 결과는 이미 정렬된 왼쪽 카메라
    좌표계다. 방향을 더 손대면 안 된다.

    Parameters
    ----------
    points_body : (N, 3) 동체 좌표계 점. 메시를 조밀하게 샘플링한 것을 넣는다.
    """
    from . import pointcloud

    p_rect = pose.apply(points_body) @ pair.R1.T
    return pointcloud.zbuffer_depth(p_rect, pair.camera, splat=1, fill_holes=True)


def compute_disparity(left, right, num_disparities: int,
                      block_size: int = DEFAULT_BLOCK_SIZE,
                      min_disparity: int = 0, uniqueness: int = 5) -> np.ndarray:
    """StereoSGBM 으로 시차 맵을 만든다. 매칭 실패는 NaN.

    block_size 기본값을 11 로 둔 근거
        처음에는 3 이었다. **합성 장면**에서 RMSE 가 가장 낮았기 때문이다.

            block  RMSE     중앙값   5cm이내  유효화소  복원 깊이 폭 (정답 0.898)
              3    0.0413   0.0105    95.4%    97.6%    0.915
              5    0.0439   0.0098    96.2%    98.2%    0.911
              7    0.0543   0.0093    95.5%    99.0%    0.908
              9    0.0552   0.0092    95.6%    99.3%    0.903

        그런데 합성 장면은 무늬가 넉넉해서 어떤 블록이든 대응이 잡힌다. 그
        조건에서는 작은 블록이 기울어진 면을 덜 뭉개니 3 이 이긴다. **실제
        위성 영상은 정반대다.** 정합할 단서가 부족해서 작은 블록은 애초에
        대응을 못 찾는다. 같은 쌍·같은 채점 영역에서 블록만 바꿔 재면
        (run_3d_experiment.py 의 block_size_ablation, 최적 쌍 기준):

            block  중앙값    5cm이내  유효화소
              3    0.0089    91.6%    60.2%
              7    0.0079    93.9%    72.2%
             11    0.0074    95.5%    80.8%   <- 채택
             13    0.0074    95.3%    83.5%
             17    0.0077    94.3%    90.3%

        11 은 5cm 이내 비율이 가장 높고 중앙값도 사실상 최저다. 더 키우면
        유효화소만 오르고 정확도는 다시 나빠진다. 격자의 끝이 아니라 안쪽에서
        고른 값이라 우연히 걸린 값이 아니다.

        **교훈은 값 자체가 아니라 절차다.** 합성 데이터로 고른 하이퍼파라미터를
        실데이터에 검증 없이 옮기면 안 된다. 두 데이터의 병목이 다르기 때문이다.
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

    측정 효과 (SPE3R aqua, img000638/img000784 기준. 나머지 조건은 고정하고
    필터만 바꾼 값이며 run_3d_experiment.py 의 filter_ablation() 이 만든다)
        필터 없음                RMSE 0.1239  중앙값 0.0101  5cm이내 87.0%
        filterSpeckles(400, 1)   RMSE 0.0679  중앙값 0.0096  5cm이내 89.9%
        + median 3x3             RMSE 0.0675  중앙값 0.0093  5cm이내 90.4%

        효과는 거의 전부 filterSpeckles 에서 나온다. 중앙값 필터가 더하는 것은
        RMSE 0.0004 뿐이지만, 아래 주석대로 경계 편향을 없애 두어야 정상 화소를
        건드리지 않는다는 말이 실제로 성립한다.

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
        k = int(median_kernel)
        keep = np.isfinite(out)
        # 무효 화소를 0 으로 채우고 median 을 돌리면, 경계에서 창의 절반 가까이가
        # 0 이 되어 중앙값이 0 쪽으로 끌려간다. 시차 0 은 무한원점이므로 깊이가
        # 밀려난다. 실측하면 경계 화소(3x3 안에 무효가 있는 화소)만 평균
        # -10.9 px 편향됐고 내부 화소는 -0.019 px 로 영향이 없었다. 즉 이 편향은
        # 전적으로 0 채움에서 온다. 유효 이웃이 창의 과반일 때만 median 을
        # 적용하고, 그렇지 않으면 원래 값을 둔다.
        filled = np.where(keep, out, 0.0).astype(np.float32)
        smoothed = cv2.medianBlur(filled, k)
        enough = cv2.medianBlur((keep.astype(np.uint8) * 255), k) > 127
        out = np.where(keep,
                       np.where(enough, smoothed.astype(np.float64), out),
                       np.nan)

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
                depth_range: tuple = None, postfilter: bool = True,
                block_size: int = DEFAULT_BLOCK_SIZE) -> dict:
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
    block_size : SGBM 정합 블록 크기. 기본 3 은 합성 장면에서 고른 값이라
        실데이터에 그대로 맞는다는 보장이 없다. 호출부에서 실측으로
        비교할 수 있게 열어 둔다 (README 8절 개선점).

    Returns
    -------
    dict(left, right, mask, disparity, depth, points, valid, num_disparities)
        화소 맵은 모두 **정렬된 왼쪽 카메라 좌표계**로 돌려준다. 세로 스테레오
        에서 매칭을 위해 돌렸던 것은 반환 전에 되돌리므로, 호출부는 가로/세로를
        구분할 필요가 없다. points 는 그 좌표계의 (N, 3) 이다.
    """
    L, R, M = pair.remap(left, right, mask)

    if distance is None:
        num_disp = 128
    else:
        expected = pair.expected_disparity(distance)
        num_disp = int(np.ceil(expected * disparity_margin / 16)) * 16
        num_disp = max(16, min(num_disp, 16 * ((pair.match_width - 16) // 16)))

    disparity = compute_disparity(L, R, num_disparities=num_disp,
                                  block_size=block_size)
    if postfilter:
        disparity = filter_disparity(disparity)
    depth = disparity_to_depth(disparity, pair.focal, pair.baseline)

    if M is not None:
        depth = np.where(M, depth, np.nan)
    if depth_range is not None:
        lo, hi = depth_range
        depth = np.where((depth >= lo) & (depth <= hi), depth, np.nan)

    # 세로 스테레오는 remap() 에서 90도 돌려 매칭했다. 반환하기 전에 되돌려
    # 놓아야 한다. 그러지 않으면 호출부가 돌아간 깊이 맵을 똑바로 선 기준
    # 깊이와 비교하게 되고, pair.camera 는 돌리기 전 정렬 카메라이므로
    # unproject() 도 틀린 광선을 쓴다. 영상이 정사각이면 shape 이 같아서
    # 예외 없이 조용히 틀린다.
    if not pair.horizontal:
        L, R = pair.unrotate(L), pair.unrotate(R)
        disparity, depth = pair.unrotate(disparity), pair.unrotate(depth)
        if M is not None:
            M = pair.unrotate(M)

    points = pair.camera.unproject(depth)
    valid = np.isfinite(depth)
    return {
        "left": L, "right": R, "mask": M,
        "disparity": disparity, "depth": depth,
        "points": points, "valid": valid,
        "num_disparities": num_disp,
    }
