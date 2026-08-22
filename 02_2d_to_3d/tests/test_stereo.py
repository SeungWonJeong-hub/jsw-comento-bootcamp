"""스테레오 삼각측량 Unit Test.

과제가 지정한 경로 —  영상 2장 -> 시차 -> 깊이 맵 -> 포인트 클라우드 —  의
각 단계를 손으로 풀 수 있는 조건에서 수치까지 검증한다.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import metrics, stereo  # noqa: E402
from src.camera import PinholeCamera, Pose, quaternion_to_rotation  # noqa: E402

@pytest.fixture
def cam():
    """256x256 · 초점거리 1277 px 짜리 카메라."""
    return PinholeCamera(width=256, height=256, fx=1277.37226, cx=128.0, cy=128.0)


@pytest.fixture
def wide_cam():
    """합성 스테레오 검증용 카메라.

256x256 에서 베이스라인 0.4 m 를 쓰면 시차가 약 102 px 인데, SGBM 은
    왼쪽 numDisparities 폭을 통째로 무효 처리한다. 물체 대부분이 그 무효
    영역에 들어가므로, 초점거리는 같게 두고 화면만 넓혀 여유를 확보한다.
    """
    return PinholeCamera(width=640, height=512, fx=1277.37226, cx=320.0, cy=256.0)



def test_disparity_to_depth_matches_formula():
    """Z = f * B / d 를 그대로 만족해야 한다."""
    f, B = 1277.37226, 0.45
    d = np.array([[10.0, 50.0, 96.0]])

    z = stereo.disparity_to_depth(d, f, B)
    np.testing.assert_allclose(z, f * B / d, rtol=0, atol=1e-12)


def test_depth_disparity_roundtrip():
    """깊이 -> 시차 -> 깊이 왕복에서 값이 보존되어야 한다."""
    f, B = 900.0, 0.3
    z_true = np.array([[4.0, 5.5, 7.25, 12.0]])

    d = f * B / z_true
    np.testing.assert_allclose(stereo.disparity_to_depth(d, f, B), z_true, atol=1e-12)


def test_baseline_scaling_invariance():
    """베이스라인과 시차를 함께 2배 하면 깊이는 그대로다."""
    f = 1000.0
    d = np.array([[40.0]])

    z1 = stereo.disparity_to_depth(d, f, 0.25)
    z2 = stereo.disparity_to_depth(2 * d, f, 0.50)
    np.testing.assert_allclose(z1, z2, atol=1e-12)


def test_nonpositive_disparity_becomes_nan():
    """시차 0 은 무한원점을 뜻한다. 0 나눗셈이 새어 나가면 안 된다."""
    z = stereo.disparity_to_depth(np.array([[0.0, -3.0, np.nan, 25.0]]), 1000.0, 0.4)

    assert np.isnan(z[0, 0]) and np.isnan(z[0, 1]) and np.isnan(z[0, 2])
    assert np.isfinite(z[0, 3])
    assert not np.any(np.isinf(z))


def test_depth_resolution_formula():
    """dZ = Z^2 / (f * B). 멀수록 제곱으로 나빠진다."""
    f, B = 1277.37226, 0.589
    assert stereo.depth_resolution(5.63, f, B) == pytest.approx(
        5.63 ** 2 / (f * B), rel=1e-12)
    # 거리가 2배면 분해능은 4배 나빠진다.
    assert (stereo.depth_resolution(10.0, f, B)
            == pytest.approx(4 * stereo.depth_resolution(5.0, f, B), rel=1e-12))


def test_disparity_to_depth_rejects_bad_geometry():
    with pytest.raises(ValueError, match="초점거리"):
        stereo.disparity_to_depth(np.ones((2, 2)), 0.0, 0.4)
    with pytest.raises(ValueError, match="베이스라인"):
        stereo.disparity_to_depth(np.ones((2, 2)), 1000.0, 0.0)


def test_compute_disparity_rejects_bad_disparity_count():
    a = np.zeros((32, 32), np.uint8)
    with pytest.raises(ValueError, match="16 의 양의 배수"):
        stereo.compute_disparity(a, a, num_disparities=50)



def test_relative_pose_maps_view_i_to_view_j():
    """p_j = R_ij p_i + t_ij 가 실제로 성립해야 한다."""
    pose_i = Pose(quaternion_to_rotation([0.9, 0.1, -0.2, 0.35]), (0.0, 0.0, 5.4))
    pose_j = Pose(quaternion_to_rotation([0.88, 0.2, -0.1, 0.4]), (0.0, 0.0, 5.1))

    rng = np.random.default_rng(0)
    p_body = rng.uniform(-0.4, 0.4, size=(200, 3))
    p_i = pose_i.apply(p_body)
    p_j = pose_j.apply(p_body)

    R_ij, t_ij = stereo.relative_pose(pose_i, pose_j)
    np.testing.assert_allclose(p_i @ R_ij.T + t_ij, p_j, atol=1e-12)


def test_identical_views_have_zero_baseline():
    pose = Pose(quaternion_to_rotation([0.5, 0.5, 0.5, 0.5]), (0.0, 0.0, 5.0))
    geo = stereo.baseline_geometry(pose, pose)

    assert geo["baseline"] == pytest.approx(0.0, abs=1e-12)
    assert geo["rotation_deg"] == pytest.approx(0.0, abs=1e-7)


def test_target_rotation_creates_lateral_baseline():
    """카메라 병진이 (0, 0, Z) 로 같아도 타겟이 돌면 횡방향 베이스라인이 생긴다.

    이것이 한 대의 카메라로도 스테레오가 성립하는 이유다. 거리 d 에서 각 t 만큼
    돌면 유효 베이스라인은 2·d·sin(t/2) 이다.
    """
    dist = 5.5
    angle = np.deg2rad(4.0)
    axis_y = np.array([[np.cos(angle), 0, np.sin(angle)],
                       [0, 1, 0],
                       [-np.sin(angle), 0, np.cos(angle)]])

    pose_i = Pose(np.eye(3), (0.0, 0.0, dist))
    pose_j = Pose(axis_y, (0.0, 0.0, dist))
    geo = stereo.baseline_geometry(pose_i, pose_j)

    expected = 2 * dist * np.sin(angle / 2)
    assert geo["baseline"] == pytest.approx(expected, rel=1e-9)
    assert geo["lateral"] > 10 * geo["forward"]   # 거의 순수 횡방향



def test_rectified_pair_recovers_baseline(cam):
    """순수 횡방향 이동이면 정렬 후 베이스라인이 원래 크기와 같아야 한다."""
    B = 0.4
    pair = stereo.RectifiedPair(cam, np.eye(3), np.array([-B, 0.0, 0.0]))

    assert pair.horizontal
    assert pair.baseline == pytest.approx(B, rel=1e-6)
    assert pair.focal == pytest.approx(cam.fx, rel=1e-6)


def test_rectified_pair_expected_disparity(cam):
    B, Z = 0.4, 5.0
    pair = stereo.RectifiedPair(cam, np.eye(3), np.array([-B, 0.0, 0.0]))
    assert pair.expected_disparity(Z) == pytest.approx(cam.fx * B / Z, rel=1e-6)


def test_rectified_pair_rejects_zero_baseline(cam):
    with pytest.raises(ValueError, match="베이스라인이 0"):
        stereo.RectifiedPair(cam, np.eye(3), np.array([0.0, 0.0, 0.0]))


def test_fit_window_keeps_geometry_and_contains_the_source(cam):
    """정렬 창을 원본이 잘리지 않는 크기로 잡되, 시차 기하는 그대로여야 한다.

    창을 넓히는 것은 주점을 옮기는 것뿐이므로 초점거리와 베이스라인이 변하면
    안 된다. 변하면 Z = f·B/d 가 통째로 어긋난다.
    """
    B = 0.4
    plain = stereo.RectifiedPair(cam, np.eye(3), np.array([-B, 0.0, 0.0]),
                                 fit_window=False)
    fitted = stereo.RectifiedPair(cam, np.eye(3), np.array([-B, 0.0, 0.0]))

    assert fitted.baseline == pytest.approx(plain.baseline, rel=1e-9)
    assert fitted.focal == pytest.approx(plain.focal, rel=1e-9)
    assert fitted.size[0] >= plain.size[0] and fitted.size[1] >= plain.size[1]

    # 원본 네 귀퉁이가 전부 창 안에 들어와야 한다.
    rng = np.random.default_rng(2)
    img = rng.integers(1, 255, size=cam.shape, dtype=np.uint8)
    L, _, _ = fitted.remap(img, img)
    assert (L > 0).sum() >= (img > 0).sum() * 0.99, "정렬 창이 원본을 자른다"


def test_fit_window_respects_the_size_cap(cam):
    """회전이 심하면 경계 상자가 수천 화소가 된다. 상한을 두고 균일 축소한다.

    실제 데이터에서는 창 최대변이 423 이라 이 분기가 한 번도 실행되지 않는다.
    안 타는 코드는 틀려도 모르므로 여기서 강제로 태운다. 축소는 균일해야
    하므로 Z = f·B/d 의 f·B 비율이 보존돼야 한다.
    """
    B = 0.4
    t = np.array([-B, 0.0, 0.0])
    big = stereo.RectifiedPair(cam, np.eye(3), t, max_side=4096)
    small = stereo.RectifiedPair(cam, np.eye(3), t, max_side=64)

    assert max(small.size) <= 64, f"상한을 넘었다: {small.size}"
    assert max(big.size) > 64, "이 테스트는 상한이 실제로 걸리는 조건을 전제로 한다"

    # 균일 축소이므로 초점거리는 줄고 베이스라인은 그대로다.
    assert small.focal < big.focal
    assert small.baseline == pytest.approx(big.baseline, rel=1e-9)
    # 같은 거리에서 예상 시차는 초점거리에 비례해 함께 줄어야 한다.
    ratio = small.focal / big.focal
    assert small.expected_disparity(5.0) == pytest.approx(
        big.expected_disparity(5.0) * ratio, rel=1e-9)


# --------------------------------------------------------------------------
# 3-1. 세로 스테레오 — 돌려서 매칭하고 되돌리는 경로
#
# 후보 20 쌍 중 절반이 세로다. 이 경로가 조용히 틀리면 후보의 절반을 통째로
# 잃는데, 영상이 정사각(256x256)이면 shape 이 같아 예외가 나지 않는다.
# --------------------------------------------------------------------------

def test_vertical_baseline_is_detected(cam):
    """세로 이동이면 horizontal 이 False 여야 하고 베이스라인은 그대로다."""
    B = 0.4
    pair = stereo.RectifiedPair(cam, np.eye(3), np.array([0.0, -B, 0.0]))

    assert not pair.horizontal
    assert pair.baseline == pytest.approx(B, rel=1e-6)


def test_vertical_remap_rotates_and_unrotate_restores(wide_cam):
    """remap 은 90도 돌려 가로로 만들고, unrotate 는 정확히 되돌려야 한다."""
    pair = stereo.RectifiedPair(wide_cam, np.eye(3), np.array([0.0, -0.4, 0.0]))
    rng = np.random.default_rng(11)
    img = rng.integers(0, 255, size=wide_cam.shape, dtype=np.uint8)

    L, _, _ = pair.remap(img, img)
    # 정렬 창은 원본과 크기가 다를 수 있다(fit_window). 기준은 pair.camera 다.
    assert L.shape == pair.camera.shape[::-1]
    assert pair.unrotate(L).shape == pair.camera.shape
    # 매처가 보는 가로 길이는 창의 세로 길이다.
    assert pair.match_width == pair.size[1]
    # 왕복이 항등이어야 한다.
    np.testing.assert_array_equal(np.rot90(pair.unrotate(L)), L)
