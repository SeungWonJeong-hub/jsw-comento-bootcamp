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

from src import metrics, scene, stereo  # noqa: E402
from src.camera import PinholeCamera, Pose, quaternion_to_rotation  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "spe3r")
HAS_SPE3R = os.path.exists(os.path.join(DATA_DIR, "camera.json"))
needs_spe3r = pytest.mark.skipif(
    not HAS_SPE3R, reason="SPE3R 데이터 없음. 'py -3 tools/get_spe3r_aqua.py' 실행 필요")


@pytest.fixture
def cam():
    """SPE3R 과 같은 규격."""
    return PinholeCamera(width=256, height=256, fx=1277.37226, cx=128.0, cy=128.0)


@pytest.fixture
def wide_cam():
    """합성 스테레오 검증용 카메라.

    SPE3R 규격(256x256)에서 베이스라인 0.4 m 를 쓰면 시차가 약 102 px 인데,
    SGBM 은 왼쪽 numDisparities 폭을 통째로 무효 처리한다. 256 px 영상에서는
    물체 대부분이 그 무효 영역에 들어가므로, 초점거리는 같게 두고 화면만
    넓혀 시차 탐색 여유를 확보한다.
    """
    return PinholeCamera(width=640, height=512, fx=1277.37226, cx=320.0, cy=256.0)


# --------------------------------------------------------------------------
# 1. 삼각측량 공식 — 해석해 검증
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# 2. 상대 자세 — 두 번째 시점을 만드는 근거
# --------------------------------------------------------------------------

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

    이것이 SPE3R 에서 스테레오가 성립하는 이유다. 거리 d 에서 각 t 만큼
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


# --------------------------------------------------------------------------
# 3. 정렬(rectification)
# --------------------------------------------------------------------------

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


def test_to_body_roundtrip_for_identity_rectification(cam):
    """정렬 회전이 항등이면 동체 변환은 자세의 역적용과 같아야 한다."""
    pair = stereo.RectifiedPair(cam, np.eye(3), np.array([-0.4, 0.0, 0.0]))
    pose = Pose(quaternion_to_rotation([0.7, 0.1, 0.2, -0.68]), (0.0, 0.0, 5.2))

    rng = np.random.default_rng(4)
    p_body = rng.uniform(-0.3, 0.3, size=(100, 3))
    p_rect = pose.apply(p_body) @ pair.R1.T      # 정렬 좌표계로

    np.testing.assert_allclose(pair.to_body(p_rect, pose), p_body, atol=1e-9)


# --------------------------------------------------------------------------
# 4. 전체 파이프라인 — 정답 깊이가 있는 합성 장면
# --------------------------------------------------------------------------

def _synthetic_pair(camera, baseline):
    """정답 깊이가 있는 합성 스테레오 쌍을 만든다.

    오른쪽 카메라를 +B 옮기는 것은 타겟을 -B 옮기는 것과 같다.
    """
    prims = scene.default_satellite()
    pose_l = Pose(quaternion_to_rotation([0.94, 0.0, 0.342, 0.0]), (0.0, 0.0, 5.0))
    pose_r = Pose(pose_l.R, pose_l.t - np.array([baseline, 0.0, 0.0]))
    left = scene.render(camera, pose_l, prims, texture_strength=0.35)
    right = scene.render(camera, pose_r, prims, texture_strength=0.35)
    return left, right


def test_full_pipeline_recovers_analytic_depth(wide_cam):
    """영상 2장 -> 시차 -> 깊이 맵 -> 포인트 클라우드 전 과정을 정답과 대조한다.

    합성 장면은 광선-도형 교차를 해석적으로 풀기 때문에 정답 깊이에
    렌더링 오차가 없다. 따라서 오차는 전부 정합 알고리즘에서 온 것이다.
    """
    B = 0.40
    left, right = _synthetic_pair(wide_cam, B)

    disparity = stereo.compute_disparity(left["image"], right["image"],
                                         num_disparities=144)
    depth = stereo.disparity_to_depth(disparity, wide_cam.fx, B)
    depth = np.where(left["mask"], depth, np.nan)

    m = metrics.depth_metrics(depth, left["depth"], mask=left["mask"])
    assert m["valid_ratio"] > 0.5, f"물체 화소의 절반 이상에서 시차를 찾아야 한다: {m}"
    assert m["median_abs"] < 0.05, f"깊이 오차 중앙값이 5 cm 미만이어야 한다: {m}"

    points = wide_cam.unproject(depth, mask=left["mask"])
    assert len(points) > 1000
    assert np.isfinite(points).all()
    # 복원된 점은 타겟 주변(약 5 m)에 모여 있어야 한다.
    assert 4.0 < np.median(points[:, 2]) < 6.0


def test_pipeline_beats_brightness_baseline_on_same_scene(wide_cam):
    """같은 장면에서 스테레오가 과제 예시(밝기->깊이)보다 정확해야 한다.

    대조군에는 정답에 대한 최적 아핀 정렬까지 해 주므로 최대한 유리한 조건이다.
    그럼에도 화소 단위 정확도에서 크게 뒤진다.
    """
    from src import depth as depth_mod

    B = 0.40
    left, right = _synthetic_pair(wide_cam, B)
    mask, gt = left["mask"], left["depth"]

    d = stereo.compute_disparity(left["image"], right["image"], num_disparities=144)
    z_stereo = np.where(mask, stereo.disparity_to_depth(d, wide_cam.fx, B), np.nan)

    z_bright = depth_mod.brightness_depth(left["image"], mask=mask)
    z_bright, _, _ = depth_mod.align_scale_shift(z_bright, gt, mask=mask)

    m_stereo = metrics.depth_metrics(z_stereo, gt, mask=mask)
    m_bright = metrics.depth_metrics(z_bright, gt, mask=mask)

    def within(z, tol=0.05):
        v = np.isfinite(z) & np.isfinite(gt) & mask
        return float((np.abs(z[v] - gt[v]) < tol).mean())

    # 중앙값 오차는 4배 이상 좋아야 한다 (측정값 0.0137 vs 0.0672).
    assert m_stereo["median_abs"] < 0.35 * m_bright["median_abs"]
    # 5 cm 이내로 맞춘 화소 비율이 2배 이상 (측정값 85.5% vs 35.2%).
    assert within(z_stereo) > 2.0 * within(z_bright)
    # RMSE 도 앞선다. 다만 격차는 작다 — 아래 테스트가 그 이유를 설명한다.
    assert m_stereo["rmse"] < m_bright["rmse"]


def test_stereo_recovers_full_depth_range(wide_cam):
    """복원된 깊이 폭이 실제 깊이 폭과 맞아야 한다.

    밝기 기반 대조군은 아무리 정렬해도 깊이 폭이 거의 0 으로 눌린다(측정값
    0.017 m). 스테레오는 태양전지판 양 끝의 0.9 m 차이를 그대로 되살린다.
    """
    B = 0.40
    left, right = _synthetic_pair(wide_cam, B)
    mask, gt = left["mask"], left["depth"]

    d = stereo.compute_disparity(left["image"], right["image"], num_disparities=144)
    d = stereo.filter_disparity(d)
    z = np.where(mask, stereo.disparity_to_depth(d, wide_cam.fx, B), np.nan)

    v = np.isfinite(z) & mask
    span_pred = float(z[v].max() - z[v].min())
    span_true = float(np.nanmax(gt[mask]) - np.nanmin(gt[mask]))

    assert span_pred == pytest.approx(span_true, rel=0.15)


def test_unlit_surfaces_break_matching(wide_cam):
    """표면 무늬가 없으면 어떤 정합 설정도 소용없다.

    조명 방향만 뒤집어 카메라를 향한 면을 그늘에 넣는 통제 실험이다. 베이스라인,
    자세, 카메라는 그대로이고 보이는 무늬만 사라진다. 정합 성능이 파라미터보다
    입력 영상의 무늬에 더 크게 좌우된다는 것을 고정해 둔다.
    """
    B = 0.40
    prims = scene.default_satellite()
    pose_l = Pose(quaternion_to_rotation([0.94, 0.0, 0.342, 0.0]), (0.0, 0.0, 5.0))
    pose_r = Pose(pose_l.R, pose_l.t - np.array([B, 0.0, 0.0]))

    def run(sun):
        left = scene.render(wide_cam, pose_l, prims, sun_direction=sun,
                            texture_strength=0.35)
        right = scene.render(wide_cam, pose_r, prims, sun_direction=sun,
                             texture_strength=0.35)
        d = stereo.filter_disparity(
            stereo.compute_disparity(left["image"], right["image"],
                                     num_disparities=144))
        z = np.where(left["mask"], stereo.disparity_to_depth(d, wide_cam.fx, B), np.nan)
        v = np.isfinite(z) & left["mask"]
        return left["image"][left["mask"]].mean(), (
            float(z[v].max() - z[v].min()) if v.sum() else 0.0)

    lit_brightness, lit_span = run((0.42, 0.30, 0.86))
    dark_brightness, dark_span = run((-0.42, -0.30, -0.86))

    # 측정값: 깊이 폭 0.915 m (정상 조명) -> 0.572 m (앞면 그늘), 약 62% 수준
    assert dark_brightness < 0.5 * lit_brightness   # 앞면이 그늘에 들어간다
    assert dark_span < 0.75 * lit_span              # 깊이 구조가 뭉개진다


# --------------------------------------------------------------------------
# 5. 실제 데이터 연동
# --------------------------------------------------------------------------

@needs_spe3r
def test_find_pairs_returns_lateral_dominant_pairs():
    from src.spe3r import SPE3RModel

    model = SPE3RModel(DATA_DIR, "aqua")
    pairs = stereo.find_pairs(model, max_rotation_deg=8.0, min_lateral_ratio=2.0)

    assert len(pairs) > 0
    for g in pairs[:10]:
        assert g["rotation_deg"] < 8.0
        assert g["lateral"] > 2.0 * g["forward"]
        assert g["baseline"] > 0.05


@needs_spe3r
def test_spe3r_pair_relative_pose_is_consistent_with_mesh():
    """상대 자세로 메시를 옮기면 뷰 j 의 자세로 옮긴 것과 같아야 한다."""
    from src.spe3r import SPE3RModel

    model = SPE3RModel(DATA_DIR, "aqua")
    vertices, _ = model.load_mesh()
    sub = vertices[::200]

    pose_i, pose_j = model.pose(0), model.pose(137)
    R_ij, t_ij = stereo.relative_pose(pose_i, pose_j)

    np.testing.assert_allclose(pose_i.apply(sub) @ R_ij.T + t_ij,
                               pose_j.apply(sub), atol=1e-9)
