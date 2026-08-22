"""스테레오 삼각측량으로 깊이 맵을 만들고 3D 포인트 클라우드로 변환한다.

과제가 지정한 경로 그대로다.

    영상 2장  ->  시차 d  ->  깊이 맵 Z = f * B / d  ->  포인트 클라우드

두 번째 시점을 어떻게 얻는가
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

        SGBM 은 블록 안에서 시차가 일정하다고 가정한다. 블록이 크면 대응을
        찾기 쉬워지지만 기울어진 면에서 깊이가 뭉개지고, 작으면 반대가 된다.
        어느 쪽이 유리한지는 **장면에 무늬가 얼마나 있는지**로 갈린다.

        달 지형(티코 크레이터)에서 같은 쌍·같은 채점 영역으로 재면
        (run_3d_experiment.py 의 [4] 절):

            block  Z 오차 중앙값   시차 환산   값이 나온 화소
              3      65.1 m       0.14 px      71.8%
              5      66.1 m       0.15 px      72.0%
             11      70.3 m       0.16 px      73.1%
             15      73.4 m       0.16 px      73.1%

        크레이터와 그림자로 무늬가 넉넉해 **작은 블록이 정확하다.** 그래서 달
        실험은 호출부에서 5 를 쓴다. 여기 기본값 11 은 무늬가 적은 장면까지
        고려한 무난한 값이고, 실제로 쓸 때는 데이터에서 다시 골라야 한다.

        **교훈은 값이 아니라 절차다.** 한 데이터에서 고른 하이퍼파라미터를
        다른 데이터에 검증 없이 옮기면 안 된다. 병목이 다르기 때문이다.
        uniquenessRatio 는 1~10 사이에서 결과가 거의 바뀌지 않아 5 를 유지한다.

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

    측정 효과 (달 지형 티코 크레이터 기준. 나머지 조건은 고정하고
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


def local_contrast(image, ksize: int = 7) -> np.ndarray:
    """화소 주변의 밝기 표준편차. 정합할 단서가 얼마나 있는지를 나타낸다.

    블록 정합은 창 안의 밝기 패턴을 맞춘다. 그 패턴이 평평하면 어디에 갖다
    대도 비용이 비슷해서, 매처는 탐색 구간 안 어딘가를 고르기는 하지만 그것이
    옳다는 보장이 없다.

    표준편차는 적분 영상으로 한 번에 구한다.  Var = E[x^2] - E[x]^2
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"창 크기는 3 이상 홀수여야 합니다: {ksize}")
    f = np.asarray(image, dtype=np.float32)
    if f.ndim == 3:
        f = cv2.cvtColor(f.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
    mean = cv2.blur(f, (ksize, ksize))
    var = cv2.blur(f * f, (ksize, ksize)) - mean * mean
    return np.sqrt(np.maximum(var, 0.0))


def texture_mask(image, min_contrast: float, ksize: int = 7) -> np.ndarray:
    """정합할 단서가 있는 화소만 True.

    왜 필요한가 — 실측 (달 지형 티코 크레이터, 시차 1 px = 450 m)

        대비 하한   값이 나온 화소   Z 오차 중앙값   1 px 초과
             0          72.0%          66.1 m        0.99%
             2          69.4%          64.9 m        0.69%
             4          63.0%          63.3 m        0.75%
             8          47.2%          60.9 m        0.94%

        하한 2 는 화소를 2.6 %p 만 내주고 **크게 틀린 화소를 3분의 1 줄인다.**
        더 올리면 덮는 범위만 빠르게 잃는다. 크게 틀린 화소들은 실제로 어두운
        곳에 몰려 있었다(평균 밝기 62 대 전체 114) - 크레이터 안쪽 그늘이다.

        다만 이것으로 "대응이 없다" 를 가려낼 수는 없다. 좌우에 같은 영상을
        넣어도 화소의 21% 가 값을 내는데, 관문을 걸어도 20% 로 거의 줄지
        않는다. 매처는 탐색 구간 안에서 반드시 무언가를 고르기 때문이다.
        신뢰도를 함께 내보내는 것이 제대로 된 해법이고, README 9절에 적었다.
    """
    if min_contrast < 0:
        raise ValueError(f"대비 하한은 0 이상이어야 합니다: {min_contrast}")
    return local_contrast(image, ksize) >= min_contrast


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


# ---------------------------------------------------------------------------
# 정답 없이 쓰는 신뢰도
#
# 지금까지는 어느 화소가 맞았는지를 정답 고도와 비교해야만 알 수 있었다.
# 실제 운용에는 정답이 없으므로, 두 영상만으로 계산되는 단서가 필요하다.
# 아래 셋은 모두 정답을 쓰지 않는다.
# ---------------------------------------------------------------------------


def compute_disparity_both(left, right, num_disparities: int,
                           block_size: int = DEFAULT_BLOCK_SIZE,
                           min_disparity: int = 0, uniqueness: int = 5):
    """왼쪽 기준 시차와 오른쪽 기준 시차를 함께 구한다.

    오른쪽 기준 시차는 두 영상을 좌우로 뒤집어 역할을 맞바꾼 뒤 정합하고
    결과를 다시 뒤집으면 얻어진다. 뒤집으면 시차의 방향도 뒤집히므로 같은
    부호 규약으로 돌아온다. StereoSGBM 을 한 번 더 돌리는 것 말고는 새로
    필요한 것이 없다.
    """
    dl = compute_disparity(left, right, num_disparities, block_size,
                           min_disparity, uniqueness)
    dr = compute_disparity(right[:, ::-1], left[:, ::-1], num_disparities,
                           block_size, min_disparity, uniqueness)[:, ::-1]
    return dl, dr


def left_right_consistency(disparity_left, disparity_right,
                           max_diff_px: float = 1.0) -> np.ndarray:
    """좌우 일관성 검사 — 두 기준의 시차가 서로를 가리키는 화소만 True.

    왜 이것이 필요한가
        한쪽 영상에서만 보이는 곳(가림)에서 매처는 반드시 무언가를 고른다.
        고를 대상이 없는데도 값을 내므로, 그 값이 틀렸다는 사실을 매처 자신은
        알려 주지 않는다. 왼쪽 기준으로 x 가 x-d 에 대응한다면, 오른쪽 기준
        으로도 x-d 가 x 를 가리켜야 한다. 가림에서는 이 왕복이 깨진다.

        크레이터 안쪽 그늘처럼 한쪽에만 보이는 곳이 정확히 여기서 걸린다.
        정답 고도를 전혀 쓰지 않는다는 점이 핵심이다.

    Parameters
    ----------
    max_diff_px : 왕복했을 때 허용할 시차 차이 [pixel]
    """
    if disparity_left.shape != disparity_right.shape:
        raise ValueError("좌우 시차 맵의 크기가 다릅니다: "
                         f"{disparity_left.shape} vs {disparity_right.shape}")
    h, w = disparity_left.shape
    cols = np.arange(w, dtype=np.float64)[None, :] - disparity_left
    inside = np.isfinite(disparity_left) & (cols >= 0) & (cols <= w - 1)

    idx = np.clip(np.nan_to_num(np.round(cols)), 0, w - 1).astype(np.intp)
    mate = np.take_along_axis(disparity_right, idx, axis=1)
    agree = np.isfinite(mate) & (np.abs(disparity_left - mate) <= max_diff_px)
    return inside & agree


def photometric_residual(left, right, disparity, ksize: int = 7) -> np.ndarray:
    """시차대로 오른쪽 영상을 옮겨 놓고 남는 밝기 차이.

    잘 맞춘 화소는 두 영상의 같은 지점을 겹치므로 잔차가 잡음 수준에
    머문다. 틀린 화소는 다른 지점을 겹치므로 잔차가 크다. 정답이 아니라
    입력 영상 두 장만 쓴다.

    Returns
    -------
    ksize x ksize 창에서 평균한 절대 잔차. 시차가 없는 곳은 NaN.
    """
    if ksize < 1 or ksize % 2 == 0:
        raise ValueError(f"ksize 는 1 이상의 홀수여야 합니다: {ksize}")
    h, w = left.shape
    map_x = (np.arange(w, dtype=np.float32)[None, :]
             - np.nan_to_num(disparity).astype(np.float32))
    map_y = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    warped = cv2.remap(right.astype(np.float32), map_x, map_y,
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT,
                       borderValue=float("nan"))

    diff = np.abs(left.astype(np.float32) - warped)
    ok = np.isfinite(diff) & np.isfinite(disparity)
    # 상자 필터는 NaN 을 퍼뜨린다. 유효한 값만 더하고 그 개수로 나눈다.
    total = cv2.boxFilter(np.where(ok, diff, 0.0), -1, (ksize, ksize),
                          normalize=False)
    count = cv2.boxFilter(ok.astype(np.float32), -1, (ksize, ksize),
                          normalize=False)
    out = np.where(count > 0, total / np.maximum(count, 1e-9), np.nan)
    return np.where(ok, out, np.nan)


def sparsification(error, confidence, steps: int = 20):
    """신뢰도가 진짜 오차를 예측하는지 재는 자 (희소화 곡선 · AUSE).

    무엇을 하는가
        신뢰도가 낮은 화소부터 일정 비율씩 버리면서, 남은 화소의 오차
        중앙값을 기록한다. 신뢰도가 쓸모 있다면 버릴수록 오차가 빠르게
        떨어진다. 같은 일을 **실제 오차 순서로** 하면 이론상 가장 빠르게
        떨어지는 곡선(오라클)이 나온다. 두 곡선 사이의 넓이가 AUSE 다.

        0 에 가까울수록 그 신뢰도가 오차를 잘 예측한다는 뜻이다. 신뢰도와
        오차가 무관하면 곡선이 평평해져 넓이가 커진다.

    여기서 정답을 쓰는 것은 **채점할 때뿐**이다. 신뢰도 자체는 정답 없이
    계산된 값이어야 하며, 그것이 이 함수를 쓰는 이유다.

    Returns
    -------
    dict : fractions, curve(신뢰도 순), oracle(실제 오차 순), ause
    """
    if steps < 2:
        raise ValueError(f"steps 는 2 이상이어야 합니다: {steps}")
    err = np.asarray(error, dtype=np.float64).ravel()
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    ok = np.isfinite(err) & np.isfinite(conf)
    err, conf = err[ok], conf[ok]
    if len(err) < steps:
        raise ValueError(f"유효한 화소가 너무 적습니다: {len(err)}")

    by_conf = np.argsort(-conf, kind="stable")     # 신뢰도 높은 순
    by_err = np.argsort(err, kind="stable")        # 오차 작은 순
    fractions = np.linspace(0.0, 0.9, steps)

    def curve(order):
        # 평균 절대오차로 그린다. 중앙값으로 그리면 크게 틀린 소수를 버리는
        # 효과가 곡선에 거의 나타나지 않아, 신뢰도가 하는 일을 과소평가한다.
        out = []
        for f in fractions:
            keep = order[:max(1, int(round(len(order) * (1.0 - f))))]
            out.append(float(np.mean(err[keep])))
        return out

    got, oracle = curve(by_conf), curve(by_err)
    base = got[0] if got[0] > 0 else 1.0
    # numpy 2 에서 trapz 가 사라졌다. 사다리꼴 적분을 직접 쓴다.
    gap = np.array(got) - np.array(oracle)
    ause = float(np.sum((gap[:-1] + gap[1:]) / 2.0 * np.diff(fractions)) / base)
    return {"fractions": [float(f) for f in fractions], "curve": got,
            "oracle": oracle, "ause": ause,
            "spearman": rank_correlation(-conf, err)}


def rank_correlation(a, b) -> float:
    """스피어만 순위 상관. 두 값의 순서가 얼마나 같이 가는지만 본다.

    신뢰도와 오차의 관계는 직선이 아니므로 피어슨 상관은 뜻이 흐리다.
    순위로 바꾸면 "신뢰도가 낮은 화소가 실제로도 더 틀리는가" 라는 질문에
    바로 답한다. 1 에 가까울수록 그렇다는 뜻이다.
    """
    x = np.asarray(a, dtype=np.float64).ravel()
    y = np.asarray(b, dtype=np.float64).ravel()
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(x) < 3:
        return float("nan")

    def ranks(v):
        order = np.argsort(v, kind="stable")
        r = np.empty(len(v), dtype=np.float64)
        r[order] = np.arange(len(v), dtype=np.float64)
        return r

    rx, ry = ranks(x), ranks(y)
    rx -= rx.mean()
    ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 0 else float("nan")


# ---------------------------------------------------------------------------
# 촬영 조건에서 스스로 정하는 파라미터
#
# 상수로 박아 둔 값은 그 지형에서 훑어 고른 값이다. 지형이 바뀌면 다시
# 골라야 하는데, 그러면 자동으로 도는 파이프라인이 아니다. 영상 자체에서
# 유도하면 따라온다.
# ---------------------------------------------------------------------------


def estimate_noise_sigma(image) -> float:
    """영상의 잡음 표준편차를 추정한다 (Immerkaer 1996).

    라플라시안을 두 번 겹친 마스크는 완만한 밝기 변화와 일정한 기울기를
    모두 지운다. 남는 것은 잡음뿐이므로, 그 응답의 평균 절대값에서 잡음
    크기를 되돌릴 수 있다. 지형이 무엇이든 영상 한 장이면 된다.

        sigma = sqrt(pi/2) / (6 (W-2) (H-2)) * sum |I * N|
        N = [[1, -2, 1], [-2, 4, -2], [1, -2, 1]]
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"2차원 영상이어야 합니다: {img.shape}")
    h, w = img.shape
    if h < 3 or w < 3:
        raise ValueError(f"3x3 마스크를 걸 수 없습니다: {img.shape}")
    mask = np.array([[1.0, -2.0, 1.0], [-2.0, 4.0, -2.0], [1.0, -2.0, 1.0]])
    r = cv2.filter2D(img, -1, mask, borderType=cv2.BORDER_REPLICATE)
    return float(np.sqrt(np.pi / 2.0) / (6.0 * (w - 2) * (h - 2))
                 * np.abs(r[1:-1, 1:-1]).sum())


def autocorrelation_length(image, max_lag: int = 32) -> float:
    """가로 방향 자기상관이 절반으로 떨어지는 지연 [pixel].

    무늬가 잘면 자기상관이 금방 떨어지고, 밋밋하면 오래 간다. 정합 블록은
    "무늬 하나가 들어갈 만큼" 이어야 하므로 이 길이가 블록 크기의 기준이
    된다. 창 안에 무늬가 하나도 없으면 어디에 맞춰도 비슷해 보이고, 너무
    크면 경사면에서 깊이가 뭉개진다.
    """
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2:
        raise ValueError(f"2차원 영상이어야 합니다: {img.shape}")
    x = img - img.mean(axis=1, keepdims=True)
    n = img.shape[1]
    f = np.fft.rfft(x, n=2 * n, axis=1)
    ac = np.fft.irfft(f * np.conj(f), axis=1)[:, :min(max_lag + 1, n)]
    ac = ac.mean(axis=0)
    if ac[0] <= 0:
        return 1.0
    ac = ac / ac[0]
    below = np.flatnonzero(ac < 0.5)
    if not len(below):
        return float(len(ac) - 1)
    k = int(below[0])
    if k == 0:
        return 0.0
    # 0.5 를 지나는 지점을 두 표본 사이에서 선형으로 찾는다.
    a, b = ac[k - 1], ac[k]
    return float(k - 1 + (a - 0.5) / max(a - b, 1e-12))


def suggest_block_size(image, min_block: int = 3, max_block: int = 15) -> int:
    """무늬의 크기에서 정합 블록 크기를 정한다. 항상 홀수."""
    lag = autocorrelation_length(image)
    block = int(2.0 * lag + 1.0) | 1
    return int(np.clip(block, min_block, max_block)) | 1


def contrast_floor(image, k: float = 2.0) -> float:
    """대비 하한을 잡음 크기의 배수로 정한다.

    지금까지 쓰던 "회색조 2단계" 는 이 영상에서 훑어 고른 절대값이라,
    밝기 분포가 다른 영상에 그대로 쓰면 뜻이 달라진다. 잡음의 k 배로 두면
    "잡음보다 이만큼은 뚜렷한 무늬" 라는 같은 뜻이 어디서나 유지된다.
    """
    if k <= 0:
        raise ValueError(f"k 는 양수여야 합니다: {k}")
    return float(k * estimate_noise_sigma(image))


def _windowed_zncc(a, b, valid, ksize):
    """창 안에서 두 영상의 정규화 상관. 평균과 분산을 빼고 재므로 밝기
    차이에 흔들리지 않는다 — 시점에 따라 밝기가 달라지는 달 표면에서 중요하다."""
    a = np.where(valid, a, 0.0).astype(np.float32)
    b = np.where(valid, b, 0.0).astype(np.float32)
    w = valid.astype(np.float32)
    k = (ksize, ksize)

    n = cv2.boxFilter(w, -1, k, normalize=False)
    n = np.maximum(n, 1.0)
    ma = cv2.boxFilter(a, -1, k, normalize=False) / n
    mb = cv2.boxFilter(b, -1, k, normalize=False) / n
    saa = cv2.boxFilter(a * a, -1, k, normalize=False) / n - ma * ma
    sbb = cv2.boxFilter(b * b, -1, k, normalize=False) / n - mb * mb
    sab = cv2.boxFilter(a * b, -1, k, normalize=False) / n - ma * mb
    return sab / np.sqrt(np.maximum(saa * sbb, 1e-9))


def matching_margin(left, right, disparity, ksize: int = 9):
    """이긴 시차가 이웃보다 얼마나 뚜렷하게 이겼는지 잰다.

    왜 이것이 가장 강한 단서인가
        매처는 탐색 구간에서 비용이 가장 낮은 시차를 고른다. 그 최소가
        **뾰족하면** 다른 후보와 확실히 구별된 것이고, **평평하면** 아무
        데나 골라도 비슷했다는 뜻이다. 정합기 안에서 실제로 일어난 일에
        가장 가까운 단서이며, 정답을 쓰지 않는다.

        OpenCV 의 StereoSGBM 은 비용 부피를 밖으로 내주지 않는다. 그래서
        이긴 시차와 그 양옆에서 상관을 **다시 재서** 봉우리 모양을 복원한다.

    Returns
    -------
    dict : margin(1등과 2등의 차), curvature(봉우리의 뾰족함), score(1등 점수)
    """
    if ksize < 3 or ksize % 2 == 0:
        raise ValueError(f"창 크기는 3 이상 홀수여야 합니다: {ksize}")
    h, w = left.shape
    L = left.astype(np.float32)
    R = right.astype(np.float32)
    base = np.arange(w, dtype=np.float32)[None, :]
    rows = np.repeat(np.arange(h, dtype=np.float32)[:, None], w, axis=1)
    d = np.nan_to_num(disparity).astype(np.float32)
    have = np.isfinite(disparity)

    out = []
    for offset in (-1.0, 0.0, 1.0):
        x = base - d - offset
        inside = have & (x >= 0) & (x <= w - 1)
        warped = cv2.remap(R, np.clip(x, 0, w - 1), rows, cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_REPLICATE)
        out.append(np.where(inside, _windowed_zncc(L, warped, inside, ksize),
                            np.nan))

    lo, mid, hi = out
    second = np.fmax(lo, hi)
    return {"margin": mid - second,
            "curvature": 2.0 * mid - lo - hi,
            "score": mid}


def rank_fuse(*signals):
    """여러 신호를 순위로 바꿔 평균한다.

    단위도 분포도 다른 신호를 그냥 더할 수는 없다. 각각을 순위(0~1)로 바꾸면
    같은 자 위에 놓인다. 하나가 놓치는 것을 다른 하나가 잡을 때 합이 각각보다
    나아진다 — 실제로 그런지는 재 봐야 안다.
    """
    if not signals:
        raise ValueError("합칠 신호가 없습니다.")
    shape = np.asarray(signals[0]).shape
    total = np.zeros(shape, dtype=np.float64)
    count = np.zeros(shape, dtype=np.float64)
    for sig in signals:
        v = np.asarray(sig, dtype=np.float64)
        if v.shape != shape:
            raise ValueError(f"신호 크기가 다릅니다: {v.shape} vs {shape}")
        ok = np.isfinite(v)
        flat = v[ok]
        order = np.argsort(flat, kind="stable")
        r = np.empty(len(flat), dtype=np.float64)
        r[order] = np.arange(len(flat), dtype=np.float64)
        if len(flat) > 1:
            r /= len(flat) - 1
        norm = np.full(shape, np.nan)
        norm[ok] = r
        total = np.where(ok, total + np.nan_to_num(norm), total)
        count = count + ok
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def blunder_auc(error, confidence, threshold: float) -> float:
    """크게 틀린 화소를 신뢰도가 얼마나 잘 골라내는가 (ROC 아래 넓이).

    왜 채점 대상을 바꾸는가
        오차의 대부분은 부화소 잡음이고, 그것은 원리적으로 예측할 수 없다.
        실제로 필요한 판단은 "이 값을 버릴까 말까" 이므로, **크게 틀린 화소를
        골라내는가** 를 물어야 한다. 희소화 곡선이 잘 안 나오는 것과 이 질문에
        잘 답하는 것은 동시에 성립할 수 있다.

        0.5 가 무작위, 1.0 이 완벽이다. 순위 통계로 계산하므로 문턱값을
        훑을 필요가 없다.
    """
    err = np.asarray(error, dtype=np.float64).ravel()
    conf = np.asarray(confidence, dtype=np.float64).ravel()
    ok = np.isfinite(err) & np.isfinite(conf)
    err, conf = err[ok], conf[ok]

    bad = err > threshold
    n_bad, n_good = int(bad.sum()), int((~bad).sum())
    if n_bad == 0 or n_good == 0:
        return float("nan")

    # 신뢰도가 낮을수록 크게 틀렸다고 예측하는 것이므로 부호를 뒤집는다.
    pred = -conf
    order = np.argsort(pred, kind="stable")
    ranks = np.empty(len(pred), dtype=np.float64)
    ranks[order] = np.arange(1, len(pred) + 1, dtype=np.float64)
    return float((ranks[bad].sum() - n_bad * (n_bad + 1) / 2.0)
                 / (n_bad * n_good))

