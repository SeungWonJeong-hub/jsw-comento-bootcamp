"""달 지형 스테레오의 Unit Test.

출력 크기와 자료형만 보는 테스트는 수식이
틀려도 통과하므로, **손으로 답을 낼 수 있는 조건**을 만들어 숫자까지 대조합니다.

  해석해      평면을 내려다보면 깊이가 정확히 고도와 같아야 합니다
  불변식      베이스라인 = 2·H·tan(수렴각/2), 카메라 두 대가 같은 점을 봅니다
  왕복        깊이 맵을 3D 로 펴서 되돌리면 원래 고도가 나와야 합니다
  회귀        렌더링한 두 장의 시차가 f·B/Z 와 맞아야 한다  <- 실제 버그를 잡은 것
  경계        잘못된 입력은 조용히 넘어가지 말고 예외를 냅니다
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import stereo, terrain  # noqa: E402
from src.camera import PinholeCamera, Pose  # noqa: E402

DTM = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "data", "moon", "tycho_118m.tif")
needs_dtm = pytest.mark.skipif(not os.path.exists(DTM),
                               reason="달 고도 모델이 없습니다 (README 1절 참고)")


# --------------------------------------------------------------------------
# 1. 지형 격자 - 해석해
# --------------------------------------------------------------------------

def test_surface_points_places_grid_centre_at_origin():
    """격자의 가운데가 원점에 오고, 화소 간격이 지상 거리와 같아야 합니다."""
    elev = np.zeros((5, 5))
    pts = terrain.surface_points(elev, gsd=10.0)

    centre = pts[len(pts) // 2]
    np.testing.assert_allclose(centre, [0.0, 0.0, 0.0], atol=1e-12)
    # 가로로 이웃한 두 점은 정확히 gsd 만큼 떨어져 있습니다.
    assert np.isclose(pts[1, 0] - pts[0, 0], 10.0)
    # 세로는 위쪽 행이 북쪽(+Y)입니다.
    assert pts[0, 1] > pts[-1, 1]


def test_surface_points_keeps_elevation_as_z():
    rng = np.random.default_rng(0)
    elev = rng.normal(size=(7, 7)) * 100
    pts = terrain.surface_points(elev, gsd=5.0)
    np.testing.assert_allclose(pts[:, 2], elev.ravel(), atol=1e-12)


def test_normals_of_a_flat_plane_point_straight_up():
    n = terrain.surface_normals(np.zeros((16, 16)), gsd=4.0)
    np.testing.assert_allclose(n[..., 0], 0.0, atol=1e-12)
    np.testing.assert_allclose(n[..., 1], 0.0, atol=1e-12)
    np.testing.assert_allclose(n[..., 2], 1.0, atol=1e-12)


def test_normal_of_a_constant_slope_matches_the_angle():
    """기울기 tan(t) 인 사면의 법선은 시선과 정확히 t 만큼 기울어야 합니다."""
    gsd, slope = 5.0, 0.25            # 동쪽으로 갈수록 높아집니다
    x = np.arange(32) * gsd
    elev = np.tile(slope * x, (32, 1))
    n = terrain.surface_normals(elev, gsd)[16, 16]

    expected = np.array([-slope, 0.0, 1.0]) / np.sqrt(1 + slope ** 2)
    np.testing.assert_allclose(n, expected, atol=1e-9)


@pytest.mark.parametrize("gsd", [0.0, -1.0])
def test_normals_reject_bad_pixel_size(gsd):
    with pytest.raises(ValueError):
        terrain.surface_normals(np.zeros((4, 4)), gsd)


# --------------------------------------------------------------------------
# 2. 밝기 - 해석해
# --------------------------------------------------------------------------

def test_overhead_sun_on_flat_ground_gives_albedo_plus_ambient():
    """태양이 바로 위면 램버트 반사는 albedo 그대로입니다."""
    img = terrain.shade(np.zeros((8, 8)), gsd=5.0, sun_elevation_deg=90.0,
                        albedo=0.2, ambient=0.05)
    np.testing.assert_allclose(img, 0.25, atol=1e-9)


def test_sun_at_the_horizon_leaves_only_ambient():
    img = terrain.shade(np.zeros((8, 8)), gsd=5.0, sun_elevation_deg=0.0,
                        albedo=0.2, ambient=0.05)
    np.testing.assert_allclose(img, 0.05, atol=1e-9)


def test_slope_facing_the_sun_is_brighter_than_the_one_away():
    """해를 마주 보는 사면이 밝아야 합니다.

    부호를 헷갈리기 쉽습니다. 고도가 동쪽으로 갈수록 높아지는 지형은 **서쪽 비탈**
    입니다. 그 면의 법선이 서쪽을 향하므로 서쪽에 뜬 해에 밝아집니다. 처음에는
    동쪽 해로 적었다가 이 테스트에서 걸렸습니다.
    """
    gsd = 5.0
    x = np.arange(32) * gsd
    up_east = np.tile(0.3 * x, (32, 1))          # 서쪽 비탈
    west, east = 270.0, 90.0
    lit = terrain.shade(up_east, gsd, sun_elevation_deg=20.0,
                        sun_azimuth_deg=west)
    dark = terrain.shade(up_east, gsd, sun_elevation_deg=20.0,
                         sun_azimuth_deg=east)
    assert lit[16, 16] > dark[16, 16]


# --------------------------------------------------------------------------
# 3. 카메라 배치 - 불변식
# --------------------------------------------------------------------------

def test_look_at_returns_an_orthonormal_pose_aimed_at_the_target():
    pose = terrain.look_at((100.0, -50.0, 2000.0), (0.0, 0.0, 0.0))
    np.testing.assert_allclose(pose.R @ pose.R.T, np.eye(3), atol=1e-12)
    # 바라보는 점은 카메라 앞(+z)에 있고 광축 위에 있습니다.
    target_cam = pose.apply(np.zeros((1, 3)))[0]
    assert target_cam[2] > 0
    np.testing.assert_allclose(target_cam[:2], 0.0, atol=1e-9)


def test_look_at_rejects_a_degenerate_up_hint():
    with pytest.raises(ValueError):
        terrain.look_at((0.0, 0.0, 100.0), (0.0, 0.0, 0.0),
                        up_hint=(0.0, 0.0, 1.0))


def test_nadir_cameras_do_not_flip_their_right_direction():
    """수직에 가까운 두 시점이 같은 쪽을 영상 오른쪽으로 삼아야 합니다.

    up 기준을 수직(0,0,1) 로 두면 외적이 0 에 가까워져 방향이 홱 뒤집힙니다.
    실제로 그렇게 짰다가 두 카메라의 상대 회전이 20도가 아니라 180도로 나왔습니다.
    """
    left, right, _ = terrain.stereo_cameras(2000.0, convergence_deg=20.0)
    R_ij, _ = stereo.relative_pose(left, right)
    angle = np.degrees(np.arccos(np.clip((np.trace(R_ij) - 1) / 2, -1, 1)))
    assert angle == pytest.approx(20.0, abs=1e-6)


def test_stereo_baseline_matches_the_convergence_formula():
    """B = 2 * H * tan(수렴각 / 2)"""
    H, conv = 5000.0, 24.0
    left, right, baseline = terrain.stereo_cameras(H, conv)
    assert baseline == pytest.approx(2 * H * np.tan(np.radians(conv) / 2))

    _, t_ij = stereo.relative_pose(left, right)
    assert np.linalg.norm(t_ij) == pytest.approx(baseline, rel=1e-12)


@pytest.mark.parametrize("bad", [0.0, -5.0, 90.0, 120.0])
def test_stereo_cameras_reject_impossible_convergence(bad):
    with pytest.raises(ValueError):
        terrain.stereo_cameras(1000.0, bad)


# --------------------------------------------------------------------------
# 4. 렌더링 - 해석해와 왕복
# --------------------------------------------------------------------------

def test_nadir_view_of_flat_ground_has_constant_depth_equal_to_altitude():
    """수직으로 평지를 보면 모든 화소의 깊이가 고도와 같습니다.

    광축에 수직인 평면은 어느 화소에서 보든 z 성분이 같기 때문입니다. 값이
    조금이라도 다르면 투영식이나 광선 방향이 틀린 것입니다.
    """
    H = 1500.0
    cam = PinholeCamera(64, 64, 400.0)
    pose = terrain.look_at((0.0, 0.0, H), (0.0, 0.0, 0.0))
    elev = np.zeros((64, 64))
    _, depth = terrain.render_heightfield(elev, 5.0, np.full_like(elev, 0.5),
                                          cam, pose)
    inside = np.isfinite(depth)
    assert inside.mean() > 0.9
    np.testing.assert_allclose(depth[inside], H, atol=1e-6)


def test_rendered_depth_recovers_the_original_elevation():
    """깊이 맵을 3D 로 펴서 지형 좌표계로 되돌리면 원래 고도가 나와야 합니다."""
    elev, gsd = terrain.synthetic_dtm(size=128, gsd=5.0, relief=200.0, seed=1)
    H = 400.0 * gsd
    cam = PinholeCamera(128, 128, 400.0)
    pose = terrain.look_at((0.0, 0.0, H), (0.0, 0.0, 0.0))
    _, depth = terrain.render_heightfield(elev, gsd, terrain.shade(elev, gsd),
                                          cam, pose)

    world = pose.inverse_apply(cam.unproject(depth))
    ok = np.isfinite(world).all(axis=1)
    assert world[ok, 2].min() == pytest.approx(elev.min(), abs=2.0)
    assert world[ok, 2].max() == pytest.approx(elev.max(), abs=2.0)


@pytest.mark.parametrize("gsd", [0.0, -5.0])
def test_render_rejects_bad_pixel_size(gsd):
    """화소 크기가 0 이면 조용히 0 으로 나누지 말고 예외를 내야 합니다.

    법선 계산은 검사하는데 렌더링은 빠져 있었습니다. 0 으로 나눈 좌표가 그대로
    보간에 들어가 경고만 남기고 엉뚱한 그림이 나옵니다.
    """
    cam = PinholeCamera(32, 32, 100.0)
    pose = terrain.look_at((0.0, 0.0, 500.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        terrain.render_heightfield(np.zeros((32, 32)), gsd, np.zeros((32, 32)),
                                   cam, pose)
    with pytest.raises(ValueError):
        terrain.surface_points(np.zeros((32, 32)), gsd)


def test_render_rejects_mismatched_elevation_and_intensity():
    cam = PinholeCamera(32, 32, 100.0)
    pose = terrain.look_at((0.0, 0.0, 500.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError):
        terrain.render_heightfield(np.zeros((32, 32)), 5.0, np.zeros((16, 16)),
                                   cam, pose)


# --------------------------------------------------------------------------
# 5. 회귀 - 실제로 났던 버그
# --------------------------------------------------------------------------

def test_rendered_pair_aligns_at_the_disparity_the_geometry_predicts():
    """렌더링한 두 장이 f·B/Z 에서 가장 잘 겹쳐야 합니다.

    실제로 났던 버그를 고정합니다. 점을 화소에 흩뿌려 그릴 때 위치를 반올림하면
    그 오차가 표면 기울기 방향으로 쏠립니다. 수렴 촬영의 두 카메라는 서로 반대로
    기울어 있어 좌우가 반대로 쏠리고, 결과가 **시차 치우침**입니다. 티코 지형에서
    +2.6 px, 깊이로 1.2 km 였습니다. 화소마다 광선을 쏘는 방식으로 바꿔 −0.1 px 가
    됐습니다.

    기하식만 대조하면 P1, P2 의 정의상 항상 맞아 떨어져 아무것도 검증하지
    못합니다. 그래서 **렌더링한 영상 자체**를 시차만큼 밀어 겹쳐 봅니다.
    """
    import cv2

    elev, gsd = terrain.synthetic_dtm(size=192, gsd=5.0, relief=250.0, seed=2)
    H = 500.0 * gsd
    cam = PinholeCamera(192, 192, 500.0)
    left, right, _ = terrain.stereo_cameras(H, 15.0)
    pair = stereo.RectifiedPair(cam, *stereo.relative_pose(left, right),
                                alpha=-1.0)
    shading = terrain.shade(elev, gsd)
    shading = (shading - shading.min()) / np.ptp(shading)

    L, _ = terrain.render_heightfield(elev, gsd, shading, cam, left)
    R, _ = terrain.render_heightfield(elev, gsd, shading, cam, right)
    Lr, Rr, _ = pair.remap(L, R)

    rect_pose = Pose(pair.R1 @ left.R, pair.R1 @ left.t)
    _, ref = terrain.render_heightfield(elev, gsd, shading, pair.camera,
                                        rect_pose)
    disparity = np.nan_to_num(pair.focal * pair.baseline / ref,
                              nan=0.0).astype(np.float32)

    h, w = Lr.shape
    xx, yy = np.meshgrid(np.arange(w, dtype=np.float32),
                         np.arange(h, dtype=np.float32))
    scores = {}
    for offset in (-2.0, -1.0, 0.0, 1.0, 2.0):
        moved = (xx - disparity - np.float32(offset)).astype(np.float32)
        warped = cv2.remap(Rr.astype(np.float32), moved, yy,
                           cv2.INTER_LINEAR, borderValue=0)
        seen = (Lr > 0) & (warped > 0) & (disparity > 0)
        scores[offset] = float(np.abs(Lr[seen].astype(float) - warped[seen]).mean())

    assert min(scores, key=scores.get) == 0.0, scores


def test_scatter_render_would_bias_the_disparity():
    """흩뿌리기 방식이 기울어진 시점에서 실제로 치우침을 만드는지 고정합니다.

    고친 방식(광선)과 옛 방식(흩뿌리기)을 같은 장면에 돌려 비교합니다. 옛 방식이
    더 크게 치우쳐야 합니다. 이 관계가 깨지면 렌더러를 되돌려도 테스트가 통과해
    버리므로, 회귀를 막으려면 둘을 함께 재야 합니다.
    """
    elev, gsd = terrain.synthetic_dtm(size=160, gsd=5.0, relief=250.0, seed=3)
    H = 400.0 * gsd
    cam = PinholeCamera(160, 160, 400.0)
    pose = terrain.look_at((H * 0.15, 0.0, H), (0.0, 0.0, 0.0))
    shading = terrain.shade(elev, gsd)

    _, ray = terrain.render_heightfield(elev, gsd, shading, cam, pose)
    pts = terrain.surface_points(elev, gsd)
    _, scat = terrain.render(pts, shading, cam, pose, splat=1)

    both = np.isfinite(ray) & np.isfinite(scat)
    assert both.mean() > 0.5
    # 흩뿌리기는 깊이를 카메라 쪽으로 당긴다 (3x3 최소값 필터와 같습니다).
    assert np.median(scat[both] - ray[both]) < 0.0


# --------------------------------------------------------------------------
# 6. 합성 지형과 실제 고도 모델
# --------------------------------------------------------------------------

def test_synthetic_terrain_is_reproducible_and_scaled():
    a, gsd = terrain.synthetic_dtm(size=64, gsd=5.0, relief=300.0, seed=7)
    b, _ = terrain.synthetic_dtm(size=64, gsd=5.0, relief=300.0, seed=7)
    c, _ = terrain.synthetic_dtm(size=64, gsd=5.0, relief=300.0, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    assert gsd == 5.0
    # 크레이터가 파이므로 기복은 relief 보다 커집니다.
    assert a.max() - a.min() > 300.0


def test_synthetic_terrain_rejects_a_tiny_grid():
    with pytest.raises(ValueError):
        terrain.synthetic_dtm(size=8)


@needs_dtm
def test_real_dtm_loads_with_metric_pixel_size():
    elev, gsd = terrain.load_dtm(DTM)
    assert elev.ndim == 2 and elev.shape[0] > 100
    assert 1.0 < gsd < 10000.0
    assert np.isfinite(elev).all()
    # 티코 크레이터는 바닥과 테두리의 고도 차가 수 km 다.
    assert elev.max() - elev.min() > 1000.0


@needs_dtm
def test_real_dtm_crop_keeps_the_centre():
    full, _ = terrain.load_dtm(DTM)
    small, _ = terrain.load_dtm(DTM, crop=128)
    assert small.shape == (128, 128)
    h, w = full.shape
    np.testing.assert_allclose(
        small, full[(h - 128) // 2:(h - 128) // 2 + 128,
                    (w - 128) // 2:(w - 128) // 2 + 128])


@needs_dtm
def test_real_dtm_rejects_an_oversized_crop():
    with pytest.raises(ValueError):
        terrain.load_dtm(DTM, crop=100000)


# --------------------------------------------------------------------------
# 7. 전체 파이프라인 - 두 장에서 고도까지
# --------------------------------------------------------------------------

def test_full_pipeline_recovers_the_terrain_within_one_pixel():
    """렌더링 → 정렬 → 정합 → 깊이 까지 한 번에 돌려 정확도를 확인합니다.

    조각마다 맞아도 이어 붙이면 틀릴 수 있습니다. 좌표계를 한 군데서 잘못 돌리면
    각 단계는 통과하면서 최종 깊이만 어긋납니다. 그래서 끝에서 끝까지 한 번
    돌려 **시차 1픽셀에 해당하는 깊이** 안에 드는지 봅니다.

    1픽셀을 기준으로 두는 이유는, 그것이 이 촬영 기하가 원리적으로 구분할 수
    있는 한계이기 때문입니다. 그보다 작은 오차를 요구하면 정합 알고리즘이 아니라
    운을 시험하게 됩니다.
    """
    from src import metrics

    elev, gsd = terrain.synthetic_dtm(size=256, gsd=5.0, relief=250.0, seed=5)
    focal, conv = 500.0, 15.0
    altitude = focal * gsd
    cam = PinholeCamera(256, 256, focal)
    left, right, _ = terrain.stereo_cameras(altitude, conv)
    pair = stereo.RectifiedPair(cam, *stereo.relative_pose(left, right),
                                alpha=-1.0)

    shading = terrain.shade(elev, gsd)
    shading = (shading - shading.min()) / np.ptp(shading)
    L, _ = terrain.render_heightfield(elev, gsd, shading, cam, left)
    R, _ = terrain.render_heightfield(elev, gsd, shading, cam, right)
    Lr, Rr, _ = pair.remap(L, R)

    rect_pose = Pose(pair.R1 @ left.R, pair.R1 @ left.t)
    _, reference = terrain.render_heightfield(elev, gsd, shading, pair.camera,
                                              rect_pose)

    lo, hi = np.nanmin(reference) - 100.0, np.nanmax(reference) + 100.0
    d_lo = pair.focal * pair.baseline / hi
    d_hi = pair.focal * pair.baseline / lo
    min_disp = max(0, int(np.floor(d_lo / 16)) * 16)
    n_disp = max(16, int(np.ceil((d_hi - min_disp) / 16)) * 16)

    disparity = stereo.filter_disparity(stereo.compute_disparity(
        Lr, Rr, num_disparities=n_disp, min_disparity=min_disp, block_size=5))
    depth = stereo.disparity_to_depth(disparity, pair.focal, pair.baseline)
    depth = np.where((depth >= lo) & (depth <= hi), depth, np.nan)

    m = metrics.depth_metrics(depth, reference, mask=np.isfinite(reference))
    one_pixel = stereo.depth_resolution(altitude, pair.focal, pair.baseline)

    # 정렬 창은 원본 화각 밖까지 포함하므로 값이 나오는 화소는 3분의 1 정도입니다.
    # 기준 깊이가 정의된 영역 자체가 88% 이고, 거기서 그늘과 가장자리가 빠집니다.
    assert m["valid_ratio"] > 0.25
    # 실측 0.30 px. 절반을 상한으로 두면 정합이 망가졌을 때만 걸립니다.
    assert m["median_abs"] < 0.5 * one_pixel


def test_texture_gate_keeps_patterned_areas_and_drops_flat_ones():
    """무늬가 있는 곳만 남기고 평평한 곳은 버려야 합니다.

    블록 정합은 창 안의 밝기 패턴을 맞춥니다. 패턴이 없으면 어디에 갖다 대도
    비용이 비슷해서, 매처가 고른 값이 옳다는 보장이 없습니다.
    """
    flat = np.full((64, 64), 120, dtype=np.uint8)
    assert not stereo.texture_mask(flat, 2.0).any()

    rng = np.random.default_rng(0)
    speckled = np.clip(120 + rng.normal(0, 25, (64, 64)), 0, 255).astype(np.uint8)
    assert stereo.texture_mask(speckled, 2.0).mean() > 0.95

    # 절반만 무늬가 있으면 그 절반만 남습니다.
    half = flat.copy()
    half[:, 32:] = speckled[:, 32:]
    kept = stereo.texture_mask(half, 2.0)
    assert kept[:, :28].mean() < 0.05 and kept[:, 36:].mean() > 0.95


@pytest.mark.parametrize("bad", [2, 4, 1])
def test_local_contrast_rejects_bad_window(bad):
    with pytest.raises(ValueError):
        stereo.local_contrast(np.zeros((16, 16), np.uint8), ksize=bad)


def test_render_rejects_terrain_above_the_camera():
    """지형이 카메라보다 높으면 멈춰야 합니다.

    고정점 반복은 광선이 아래로 내려가며 지면을 한 번 만난다고 가정합니다.
    지형이 카메라보다 높으면 그 전제가 깨지는데, 예외 없이 두면 값이 거의
    안 나오는 결과가 조용히 돌아옵니다. 실제로 기복 4 km 지형을 고도 2.5 km
    에서 찍는 설정이 유효화소 3% 를 내놓고도 아무 말이 없었습니다.
    """
    elev, gsd = terrain.synthetic_dtm(size=64, gsd=5.0, relief=4000.0, seed=1)
    cam = PinholeCamera(64, 64, 500.0)
    pose = terrain.look_at((0.0, 0.0, 2500.0), (0.0, 0.0, 0.0))
    with pytest.raises(ValueError, match="카메라 고도"):
        terrain.render_heightfield(elev, gsd, terrain.shade(elev, gsd), cam, pose)


# --------------------------------------------------------------------------
# 8. 여러 각도 융합
# --------------------------------------------------------------------------

def _run_module():
    import importlib
    return importlib.import_module("run_3d_experiment")


def test_grid_binning_puts_points_in_the_right_cell():
    """지형 격자에 얹을 때 셀 위치가 어긋나면 안 됩니다."""
    M = _run_module()
    elev = np.zeros((5, 5))
    gsd = 10.0
    # 격자 가운데(0,0)와 오른쪽 위 모서리에 점을 하나씩
    pts = np.array([[0.0, 0.0, 7.0], [20.0, 20.0, 3.0]])
    grid = M.to_grid(pts, elev, gsd)
    assert grid[2, 2] == pytest.approx(7.0)
    assert grid[0, 4] == pytest.approx(3.0)
    assert np.isnan(grid).sum() == 23


def test_fusion_weights_the_precise_layer_more():
    """정밀한 층(분해능이 작은 층)이 더 무겁게 반영돼야 합니다."""
    M = _run_module()
    coarse = np.full((4, 4), 100.0)      # 분해능 400 m
    fine = np.full((4, 4), 200.0)        # 분해능 100 m -> 16배 무겁습니다
    fused, n, spread = M.fuse([coarse, fine], [400.0, 100.0], max_spread=1e9)
    expected = (100.0 / 400.0 ** 2 + 200.0 / 100.0 ** 2) /                (1 / 400.0 ** 2 + 1 / 100.0 ** 2)
    assert fused[0, 0] == pytest.approx(expected)
    assert (n == 2).all()
    assert spread[0, 0] == pytest.approx(100.0)


def test_fusion_drops_cells_where_the_layers_disagree():
    """각도끼리 어긋나는 셀은 버려야 합니다.

    정답을 보지 않고 "믿을 수 없는 곳" 을 가려낼 수 있는 단서입니다. 서로 다른
    기하에서 본 값이 다르면 둘 중 하나 이상이 틀린 것입니다.
    """
    M = _run_module()
    a = np.array([[10.0, 10.0]])
    b = np.array([[10.5, 90.0]])         # 오른쪽 셀에서 크게 어긋납니다
    fused, _, _ = M.fuse([a, b], [100.0, 100.0], max_spread=5.0)
    assert np.isfinite(fused[0, 0])
    assert np.isnan(fused[0, 1])


def test_fusion_keeps_a_cell_only_one_layer_saw():
    """한 층만 본 셀은 견줄 상대가 없으므로 그대로 둡니다."""
    M = _run_module()
    a = np.array([[np.nan, 5.0]])
    b = np.array([[7.0, np.nan]])
    fused, n, spread = M.fuse([a, b], [100.0, 100.0], max_spread=0.001)
    assert fused[0, 0] == pytest.approx(7.0)
    assert fused[0, 1] == pytest.approx(5.0)
    assert (n == 1).all() and np.isnan(spread).all()
