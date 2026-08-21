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


def test_vertical_reconstruct_returns_unrotated_maps(wide_cam):
    """reconstruct 는 세로 쌍에서도 정렬된 왼쪽 카메라 좌표계로 돌려줘야 한다.

    돌아간 채로 반환하면 호출부가 똑바로 선 기준 깊이와 비교하게 되고,
    pair.camera 는 돌리기 전 카메라라 unproject 도 틀린 광선을 쓴다.
    정사각 영상에서는 shape 이 같아 예외 없이 조용히 틀린다.
    """
    B = 0.4
    prims = scene.default_satellite()
    pose_l = Pose(quaternion_to_rotation([0.94, 0.0, 0.342, 0.0]), (0.0, 0.0, 5.0))
    # 타겟을 세로로 옮기면 카메라가 세로로 움직인 것과 같다.
    pose_r = Pose(pose_l.R, pose_l.t - np.array([0.0, B, 0.0]))
    left = scene.render(wide_cam, pose_l, prims, texture_strength=0.35)
    right = scene.render(wide_cam, pose_r, prims, texture_strength=0.35)

    pair = stereo.RectifiedPair(wide_cam, np.eye(3), np.array([0.0, -B, 0.0]))
    assert not pair.horizontal
    out = stereo.reconstruct(pair, left["image"], right["image"],
                             mask=left["mask"], distance=5.0)

    # 반환된 화소 맵은 모두 정렬된 왼쪽 카메라 좌표계여야 한다.
    assert out["depth"].shape == pair.camera.shape
    assert out["mask"].shape == pair.camera.shape

    ref = stereo.reference_depth(
        pair, pose_l, pose_l.inverse_apply(unproject_scene(wide_cam, left)))
    v = np.isfinite(out["depth"]) & np.isfinite(ref)
    assert v.sum() > 500, f"겹치는 화소가 너무 적다: {int(v.sum())}"
    assert float(np.median(np.abs(out["depth"][v] - ref[v]))) < 0.05


def unproject_scene(camera, rendered):
    """렌더 결과의 정답 깊이를 카메라 좌표계 점구름으로 편다."""
    depth, mask = rendered["depth"], rendered["mask"]
    v = np.isfinite(depth) & mask
    return camera.pixel_rays()[v] * depth[v][:, None]


@pytest.mark.parametrize("direction, shift", [
    ("가로", np.array([0.4, 0.0, 0.0])),
    ("세로", np.array([0.0, 0.4, 0.0])),
])
def test_reference_depth_lands_in_the_same_frame_as_reconstruct(
        wide_cam, direction, shift):
    """reference_depth 와 reconstruct 가 같은 좌표계를 돌려줘야 한다.

    이 둘은 좌표 규약을 공유하는 짝이다. 한쪽만 고치면 채점이 조용히 어긋난다.
    실제로 reconstruct 의 세로 경로를 고치면서 reference_depth 의 unrotate 를
    안 걷어내, 마스크 안에서 기준 깊이가 정의된 비율이 27.9% 까지 떨어진 적이
    있다. 가로/세로가 서로 독립인 경우이므로 parametrize 로 나눈다. 한 방향이
    깨져도 다른 방향은 끝까지 실행되고, 어느 쪽이 깨졌는지 테스트 이름에 남는다.
    """
    prims = scene.default_satellite()
    pose_l = Pose(quaternion_to_rotation([0.94, 0.0, 0.342, 0.0]), (0.0, 0.0, 5.0))
    pose_r = Pose(pose_l.R, pose_l.t - shift)
    left = scene.render(wide_cam, pose_l, prims, texture_strength=0.35)
    right = scene.render(wide_cam, pose_r, prims, texture_strength=0.35)
    pair = stereo.RectifiedPair(wide_cam, np.eye(3), -shift)

    out = stereo.reconstruct(pair, left["image"], right["image"],
                             mask=left["mask"], distance=5.0)
    ref = stereo.reference_depth(
        pair, pose_l, pose_l.inverse_apply(unproject_scene(wide_cam, left)))

    assert ref.shape == out["depth"].shape
    # 마스크 안에서 기준 깊이가 거의 전부 정의되어야 한다. 방향이 어긋나면
    # 이 비율이 무너진다.
    covered = (np.isfinite(ref) & out["mask"]).sum() / max(1, out["mask"].sum())
    assert covered > 0.9, f"{direction}: 마스크 안 기준 깊이 정의 비율 {covered:.1%}"

    v = np.isfinite(out["depth"]) & np.isfinite(ref) & out["mask"]
    assert v.sum() > 500, f"{direction}: 겹치는 화소 {int(v.sum())}"
    assert float(np.median(np.abs(out["depth"][v] - ref[v]))) < 0.05


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
