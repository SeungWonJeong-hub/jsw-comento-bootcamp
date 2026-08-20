"""2D -> 3D 변환 파이프라인 Unit Test.

테스트 설계 원칙
    과제 예시(p.14)의 테스트는 출력 크기와 자료형만 확인한다. 그것만으로는
    수식이 틀려도 통과한다. 여기서는 답을 손으로 풀 수 있는 조건을 만들어
    수치까지 검증하고, 원리적으로 성립해야 하는 불변식을 함께 확인한다.

    - 해석해 검증  : 구/직육면체처럼 깊이를 손으로 계산할 수 있는 도형을 쓴다
    - 불변식 검증  : 강체 변환에서 거리가 보존되는지 등 반드시 참인 성질
    - 실패 특성화  : 대조군이 '어떻게' 실패하는지를 테스트로 못박는다
    - 경계 조건    : 빈 입력, 크기 불일치, 0 나눗셈
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import baseline, carving, depth, metrics, pointcloud, scene  # noqa: E402
from src.camera import PinholeCamera, Pose, quaternion_to_rotation, rotation_angle_deg  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                        "data", "spe3r")
HAS_SPE3R = os.path.exists(os.path.join(DATA_DIR, "camera.json"))
needs_spe3r = pytest.mark.skipif(
    not HAS_SPE3R, reason="SPE3R 데이터 없음. 'py -3 tools/get_spe3r_aqua.py' 실행 필요")


@pytest.fixture
def cam():
    """SPE3R 과 같은 규격의 카메라."""
    return PinholeCamera(width=256, height=256, fx=1277.37226, cx=128.0, cy=128.0)


@pytest.fixture
def front_pose():
    """타겟이 광축 위 5 m 앞에 정면으로 놓인 자세."""
    return Pose(R=np.eye(3), t=(0.0, 0.0, 5.0))


# --------------------------------------------------------------------------
# 1. 카메라 기하 — 해석해와 왕복 검증
# --------------------------------------------------------------------------

def test_pixel_rays_have_unit_z(cam):
    """광선의 z 성분이 1 이어야 매개변수가 곧 깊이 Z 가 된다."""
    rays = cam.pixel_rays()
    assert rays.shape == (cam.height, cam.width, 3)
    np.testing.assert_allclose(rays[..., 2], 1.0)


def test_projection_unprojection_roundtrip(cam):
    """3D 점 -> 투영 -> 역투영으로 원래 점이 복원되어야 한다."""
    rng = np.random.default_rng(0)
    points = np.stack([rng.uniform(-1, 1, 500),
                       rng.uniform(-1, 1, 500),
                       rng.uniform(3, 8, 500)], axis=1)

    uv = cam.project(points)
    z = points[:, 2]
    x = (uv[:, 0] - cam.cx) * z / cam.fx
    y = (uv[:, 1] - cam.cy) * z / cam.fy
    recovered = np.stack([x, y, z], axis=1)

    np.testing.assert_allclose(recovered, points, atol=1e-9)


def test_projection_of_optical_axis_hits_principal_point(cam):
    """광축 위의 점은 항상 주점에 맺혀야 한다."""
    uv = cam.project(np.array([[0.0, 0.0, 4.2], [0.0, 0.0, 9.7]]))
    np.testing.assert_allclose(uv[:, 0], cam.cx)
    np.testing.assert_allclose(uv[:, 1], cam.cy)


def test_analytic_sphere_depth_at_center(cam, front_pose):
    """정면에서 본 구의 중심 화소 깊이는 (거리 - 반지름) 이다."""
    sphere = scene.Sphere(center=(0.0, 0.0, 0.0), radius=0.35)
    out = scene.render(cam, front_pose, [sphere])

    center_depth = out["depth"][int(cam.cy), int(cam.cx)]
    assert center_depth == pytest.approx(5.0 - 0.35, abs=1e-9)


def test_analytic_sphere_silhouette_radius(cam, front_pose):
    """구의 실루엣 반지름은 r_px = f * R / sqrt(d^2 - R^2) 로 결정된다."""
    R, d = 0.35, 5.0
    out = scene.render(cam, front_pose, [scene.Sphere((0, 0, 0), R)])

    expected_px = cam.fx * R / np.sqrt(d ** 2 - R ** 2)
    measured_px = np.sqrt(out["mask"].sum() / np.pi)
    assert measured_px == pytest.approx(expected_px, rel=0.01)


def test_unproject_recovers_analytic_depth(cam, front_pose):
    """역투영한 점이 실제로 구 표면 위에 놓여야 한다."""
    R = 0.35
    out = scene.render(cam, front_pose, [scene.Sphere((0, 0, 0), R)])
    pts_cam = cam.unproject(out["depth"], mask=out["mask"])

    # 카메라 좌표계 구 중심은 (0, 0, 5)
    radii = np.linalg.norm(pts_cam - np.array([0.0, 0.0, 5.0]), axis=1)
    np.testing.assert_allclose(radii, R, atol=1e-9)


# --------------------------------------------------------------------------
# 2. 자세 표현 — 쿼터니언과 강체 변환
# --------------------------------------------------------------------------

def test_quaternion_gives_proper_rotation():
    """쿼터니언에서 만든 행렬은 직교하고 행렬식이 +1 이어야 한다."""
    rng = np.random.default_rng(3)
    for _ in range(20):
        q = rng.normal(size=4)
        R = quaternion_to_rotation(q)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


def test_identity_quaternion_is_identity_matrix():
    np.testing.assert_allclose(quaternion_to_rotation([1, 0, 0, 0]), np.eye(3), atol=1e-15)


def test_pose_apply_inverse_roundtrip():
    """동체 -> 카메라 -> 동체 왕복에서 점이 보존되어야 한다."""
    R = quaternion_to_rotation([0.5, 0.5, -0.5, 0.5])
    pose = Pose(R=R, t=(0.1, -0.2, 5.3))

    rng = np.random.default_rng(7)
    pts = rng.uniform(-0.5, 0.5, size=(300, 3))
    np.testing.assert_allclose(pose.inverse_apply(pose.apply(pts)), pts, atol=1e-12)


def test_rigid_transform_preserves_pairwise_distances():
    """강체 변환은 점 사이 거리를 바꾸지 않는다 (SE(3) 불변식)."""
    rng = np.random.default_rng(11)
    pts = rng.uniform(-1, 1, size=(200, 3))
    R = quaternion_to_rotation(rng.normal(size=4))

    moved = pointcloud.transform_points(pts, R, t=(3.0, -1.0, 7.0))
    before = np.linalg.norm(pts[:50, None, :] - pts[None, :50, :], axis=-1)
    after = np.linalg.norm(moved[:50, None, :] - moved[None, :50, :], axis=-1)
    np.testing.assert_allclose(after, before, atol=1e-12)


def test_rotation_angle_between_identical_rotations_is_zero():
    R = quaternion_to_rotation([0.2, 0.5, 0.1, -0.3])
    assert rotation_angle_deg(R, R) == pytest.approx(0.0, abs=1e-7)


# --------------------------------------------------------------------------
# 3. 실루엣 기반 복원 (voxel space carving)
# --------------------------------------------------------------------------

def test_carving_recovers_sphere_radius(cam):
    """구의 visual hull 은 구 자신이다. 반지름이 참값으로 수렴해야 한다."""
    R_true = 0.35
    sphere = [scene.Sphere((0, 0, 0), R_true)]
    poses = scene.orbit_poses(30, distance=5.0, seed=1)
    masks = [scene.render_mask(cam, p, sphere) for p in poses]

    # 합성 장면의 실루엣은 해석적으로 정확하므로 여유를 주지 않는다.
    result = carving.carve(cam, masks, poses, bounds=0.5, resolution=64,
                           mask_margin=0)
    pts = carving.surface_points(result["occupancy"], result["centers"])

    assert len(pts) > 0
    measured = float(np.linalg.norm(pts, axis=1).max())
    # 유한 개의 시점으로 만든 hull 은 구를 살짝 감싸므로 위쪽으로 치우친다.
    assert R_true - 2 * result["spacing"] <= measured <= R_true * 1.15


def test_carving_shrinks_monotonically(cam):
    """시점을 추가할수록 남는 복셀은 줄기만 해야 한다."""
    prims = scene.default_satellite()
    poses = scene.orbit_poses(12, distance=5.0, seed=2)
    masks = [scene.render_mask(cam, p, prims) for p in poses]

    result = carving.carve(cam, masks, poses, bounds=0.6, resolution=48)
    history = result["history"]
    assert len(history) == 12
    assert all(b <= a for a, b in zip(history, history[1:]))


def test_visual_hull_contains_true_shape(cam):
    """visual hull 은 실제 형상을 포함한다 (원리적으로 항상 참).

    복원 결과가 실제보다 작아지면 알고리즘이 틀린 것이다.
    """
    R_true = 0.3
    sphere = [scene.Sphere((0, 0, 0), R_true)]
    poses = scene.orbit_poses(24, distance=5.0, seed=4)
    masks = [scene.render_mask(cam, p, sphere) for p in poses]

    result = carving.carve(cam, masks, poses, bounds=0.5, resolution=48)
    occ = result["occupancy"]
    centers = result["centers"]

    # 참 표면보다 확실히 안쪽에 있는 점들은 모두 채워져 있어야 한다.
    inner = np.linalg.norm(centers, axis=-1) < R_true - 2 * result["spacing"]
    assert occ[inner].all()


def test_undersized_silhouette_erodes_shape_as_views_grow(cam):
    """실루엣이 참 투영보다 작으면 뷰를 늘릴수록 복원이 나빠진다.

    visual hull 은 실루엣이 참 투영을 '포함'해야 성립한다. 부족분은 시점마다
    다른 방향을 향하므로, 시점을 더할수록 물체가 사방에서 깎여 나간다.
    실제 SPE3R 마스크가 메시 투영보다 약 1 픽셀 작아 이 현상이 관측되었고,
    carve() 의 mask_margin 기본값을 1 로 둔 근거가 된다.
    """
    R_true = 0.35
    sphere = [scene.Sphere((0, 0, 0), R_true)]

    def radius_after(n_views, margin):
        poses = scene.orbit_poses(n_views, distance=5.0, seed=6)
        masks = [scene.render_mask(cam, p, sphere) for p in poses]
        res = carving.carve(cam, masks, poses, bounds=0.5, resolution=64,
                            mask_margin=margin)
        pts = carving.surface_points(res["occupancy"], res["centers"])
        return float(np.linalg.norm(pts, axis=1).max()) if len(pts) else 0.0

    # 실루엣을 1 픽셀 줄이면 뷰가 늘수록 반지름이 작아진다.
    shrunk_few = radius_after(4, margin=-1)
    shrunk_many = radius_after(32, margin=-1)
    assert shrunk_many < shrunk_few

    # 정확한 실루엣이면 뷰를 늘려도 참값 아래로 무너지지 않는다.
    exact_many = radius_after(32, margin=0)
    assert exact_many > R_true - 2 * (1.0 / 63)
    assert exact_many > shrunk_many


def test_surface_points_excludes_interior():
    """표면 추출은 속이 빈 껍질만 남겨야 한다."""
    occ = np.zeros((9, 9, 9), dtype=bool)
    occ[2:7, 2:7, 2:7] = True
    centers = carving.make_voxel_grid(1.0, 9)[0]

    surface = carving.surface_points(occ, centers)
    assert len(surface) == 5 ** 3 - 3 ** 3   # 껍질만 = 125 - 27


def test_carving_rejects_mismatched_view_counts(cam, front_pose):
    masks = [np.ones(cam.shape, dtype=bool)]
    with pytest.raises(ValueError, match="수가 다릅니다"):
        carving.carve(cam, masks, [front_pose, front_pose], bounds=0.5, resolution=8)


def test_carving_rejects_empty_views(cam):
    with pytest.raises(ValueError, match="시점이 하나도 없습니다"):
        carving.carve(cam, [], [], bounds=0.5, resolution=8)


def test_render_depth_matches_carved_geometry(cam, front_pose):
    """복원한 점을 되쏘아 만든 깊이 맵이 물체 앞면 거리와 맞아야 한다."""
    R_true = 0.35
    pts_body = np.array([[0.0, 0.0, -R_true]])   # 카메라를 향한 최근접점
    d = carving.render_depth(pts_body, front_pose, cam, fill_holes=False)

    finite = np.isfinite(d)
    assert finite.sum() == 1
    assert d[finite][0] == pytest.approx(5.0 - R_true, abs=1e-9)


# --------------------------------------------------------------------------
# 4. 포인트 클라우드 입출력과 배경 처리
# --------------------------------------------------------------------------

def test_background_excluded_from_pointcloud(cam, front_pose):
    """우주 배경(마스크 밖)은 포인트 클라우드에 들어가면 안 된다.

    배경을 거르지 않으면 검은 하늘 전체가 하나의 평면으로 섞여 들어간다.
    """
    out = scene.render(cam, front_pose, [scene.Sphere((0, 0, 0), 0.35)])
    pts = pointcloud.depth_to_pointcloud(cam, out["depth"], mask=out["mask"])

    assert len(pts) == int(out["mask"].sum())
    assert np.isfinite(pts).all()
    assert (pts[:, 2] > 0).all()


def test_ply_write_read_roundtrip(tmp_path):
    rng = np.random.default_rng(5)
    pts = rng.uniform(-1, 1, size=(1000, 3))

    path = pointcloud.write_ply(str(tmp_path / "cloud.ply"), pts)
    loaded = pointcloud.read_ply(path)

    assert loaded.shape == pts.shape
    np.testing.assert_allclose(loaded, pts, atol=1e-6)   # float32 저장 오차


def test_mesh_surface_sampling_is_area_weighted():
    """면적이 큰 삼각형에서 비례해 더 많이 뽑혀야 한다."""
    vertices = np.array([[0, 0, 0], [10, 0, 0], [0, 10, 0],      # 큰 삼각형 (면적 50)
                         [0, 0, 5], [1, 0, 5], [0, 1, 5]])       # 작은 삼각형 (면적 0.5)
    faces = np.array([[0, 1, 2], [3, 4, 5]])

    pts = pointcloud.sample_mesh_surface(vertices, faces, count=10000, seed=0)
    on_large = (pts[:, 2] < 2.5).mean()
    assert on_large == pytest.approx(50 / 50.5, abs=0.02)


def test_normalize_scale_maps_to_unit_ball():
    rng = np.random.default_rng(9)
    pts = rng.uniform(-3, 3, size=(500, 3)) + np.array([10.0, -4.0, 7.0])

    normalized, center, sc = pointcloud.normalize_scale(pts)
    assert np.linalg.norm(normalized, axis=1).max() == pytest.approx(1.0)
    np.testing.assert_allclose(normalized * sc + center, pts, atol=1e-9)


# --------------------------------------------------------------------------
# 5. 평가 지표
# --------------------------------------------------------------------------

def test_chamfer_is_zero_for_identical_clouds():
    rng = np.random.default_rng(13)
    pts = rng.uniform(-1, 1, size=(500, 3))
    assert metrics.chamfer_distance(pts, pts)["chamfer"] == pytest.approx(0.0, abs=1e-12)


def test_chamfer_grows_with_scale_error():
    """복원이 부풀수록 Chamfer 가 커져야 한다 (지표의 단조성)."""
    rng = np.random.default_rng(17)
    v = rng.normal(size=(2000, 3))
    sphere = v / np.linalg.norm(v, axis=1, keepdims=True)

    d = [metrics.chamfer_distance(sphere * s, sphere)["chamfer"]
         for s in (1.02, 1.10, 1.25)]
    assert d[0] < d[1] < d[2]


def test_chamfer_direction_reveals_visual_hull_inflation():
    """visual hull 처럼 참 형상을 감싸며 부푼 복원은 두 방향이 비대칭이다.

    hull 은 참 표면을 포함하므로 target -> pred 는 0 에 가깝고,
    참 표면 바깥의 여분 때문에 pred -> target 은 커진다. 이 비대칭이
    '덜 깎였다'와 '잘못 깎였다'를 구분해 준다.
    """
    rng = np.random.default_rng(19)
    v = rng.normal(size=(3000, 3))
    sphere = v / np.linalg.norm(v, axis=1, keepdims=True)

    hull = np.vstack([sphere, sphere * 1.3])   # 참 표면 + 바깥 여분
    out = metrics.chamfer_distance(hull, sphere)

    assert out["target_to_pred"] < 0.05
    assert out["pred_to_target"] > 5 * out["target_to_pred"]


def test_depth_metrics_reports_valid_ratio():
    gt = np.full((10, 10), 5.0)
    pred = gt.copy()
    pred[:, :5] = np.nan   # 절반은 추정 실패

    m = metrics.depth_metrics(pred, gt)
    assert m["rmse"] == pytest.approx(0.0)
    assert m["valid_ratio"] == pytest.approx(0.5)


def test_mask_iou_bounds():
    a = np.zeros((8, 8), dtype=bool)
    a[:4] = True
    assert metrics.mask_iou(a, a) == pytest.approx(1.0)
    assert metrics.mask_iou(a, ~a) == pytest.approx(0.0)


# --------------------------------------------------------------------------
# 6. 대조군(과제 예시 코드)의 실패 특성화
# --------------------------------------------------------------------------

def test_example_code_runs_and_keeps_shape():
    """과제 예시 코드가 원문 그대로 동작하는지 확인한다 (요구사항 충족 증빙)."""
    image = np.zeros((100, 120, 3), dtype=np.uint8)
    image[:, 60:] = 200

    dmap = baseline.generate_depth_map(image)
    assert dmap.shape == image.shape
    assert isinstance(dmap, np.ndarray)

    pts = baseline.image_to_points_3d(image)
    assert pts.shape == (100, 120, 3)


def test_example_code_rejects_none_input():
    with pytest.raises(ValueError, match="입력된 이미지가 없습니다"):
        baseline.generate_depth_map(None)


def test_brightness_depth_confuses_albedo_with_distance():
    """같은 거리에 있어도 반사율이 다르면 다른 깊이로 읽힌다.

    이것이 대조군의 근본 결함이다. 밝기는 (반사율 x 조명각)이라
    거리와 독립적으로 변한다.
    """
    image = np.zeros((40, 40, 3), dtype=np.uint8)
    image[:, :20] = 40     # 어두운 면 (예: 태양전지판)
    image[:, 20:] = 220    # 밝은 면 (예: 단열재)
    mask = np.ones((40, 40), dtype=bool)

    est = depth.brightness_depth(image, mask=mask)
    dark = est[:, :20].mean()
    bright = est[:, 20:].mean()

    # 실제로는 두 면이 같은 거리에 있는데 추정값은 4배 이상 벌어진다.
    assert bright / dark > 4.0


def test_brightness_depth_no_better_than_predicting_the_mean(cam):
    """대조군이 '평균값을 찍는 것'보다 나은지 여러 자세에서 확인한다.

    비교 기준을 상수 예측으로 잡은 이유
        임의의 임계값(예: RMSE < 0.1 m)은 물체 크기에 따라 의미가 달라진다.
        반면 '깊이의 표준편차'는 아무 정보 없이 평균만 찍었을 때의 RMSE 이므로,
        rmse / std 가 1 에 가깝다는 것은 추정기가 정보를 전혀 더하지 못했다는 뜻이다.
        (rmse / std)^2 = 1 - R^2 관계로 결정계수와 직접 연결된다.

    자세 하나만 보면 우연히 잘 맞을 수 있어 12 개 자세의 중앙값으로 판정한다.
    """
    ratios, r2s, corrs = [], [], []
    for pose in scene.orbit_poses(12, distance=5.0, seed=42):
        out = scene.render(cam, pose)
        mask = out["mask"]
        if mask.sum() < 500:
            continue

        est = depth.brightness_depth(out["image"], mask=mask)
        aligned, _, _ = depth.align_scale_shift(est, out["depth"], mask=mask)
        m = metrics.depth_metrics(aligned, out["depth"], mask=mask)

        std = float(out["depth"][mask].std())
        ratios.append(m["rmse"] / std)
        r2s.append(1.0 - (m["rmse"] / std) ** 2)
        corrs.append(abs(depth.depth_correlation(est, out["depth"], mask=mask)))

    assert len(ratios) >= 10
    # 상수 예측 대비 개선이 거의 없다 (측정값 중앙값 0.98).
    assert np.median(ratios) > 0.6
    # 깊이 분산을 거의 설명하지 못한다 (측정값 중앙값 0.05).
    assert np.median(r2s) < 0.4
    # 밝기와 거리 사이에 신뢰할 만한 상관도 없다 (측정값 중앙값 0.05).
    assert np.median(corrs) < 0.5


def test_align_scale_shift_is_exact_for_affine_input():
    """정답에 아핀 변환만 가한 입력은 완전히 복원되어야 한다 (정렬기 자체 검증)."""
    rng = np.random.default_rng(23)
    gt = rng.uniform(4.0, 6.0, size=(50, 50))
    pred = 0.37 * gt - 11.2

    aligned, a, b = depth.align_scale_shift(pred, gt)
    np.testing.assert_allclose(aligned, gt, atol=1e-9)
    assert a == pytest.approx(1 / 0.37, rel=1e-9)


# --------------------------------------------------------------------------
# 7. 경계 조건과 예외 처리
# --------------------------------------------------------------------------

def test_unproject_rejects_none():
    cam = PinholeCamera(16, 16, 100.0)
    with pytest.raises(ValueError, match="깊이 맵이 없습니다"):
        cam.unproject(None)


def test_unproject_rejects_wrong_shape():
    cam = PinholeCamera(16, 16, 100.0)
    with pytest.raises(ValueError, match="크기"):
        cam.unproject(np.ones((8, 8)))


def test_camera_rejects_nonpositive_focal_length():
    with pytest.raises(ValueError, match="초점거리"):
        PinholeCamera(16, 16, 0.0)


def test_project_rejects_points_behind_camera():
    cam = PinholeCamera(16, 16, 100.0)
    with pytest.raises(ValueError, match="카메라 뒤쪽"):
        cam.project(np.array([[0.0, 0.0, -1.0]]))


def test_pose_rejects_non_orthogonal_rotation():
    with pytest.raises(ValueError, match="직교"):
        Pose(R=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 2]], dtype=float), t=(0, 0, 5))


def test_quaternion_rejects_zero_norm():
    with pytest.raises(ValueError, match="크기가 0"):
        quaternion_to_rotation([0, 0, 0, 0])


def test_chamfer_rejects_empty_cloud():
    with pytest.raises(ValueError, match="빈 포인트 클라우드"):
        metrics.chamfer_distance(np.zeros((0, 3)), np.ones((5, 3)))


def test_infinite_depth_never_reaches_pointcloud(cam):
    """무한대/NaN 깊이가 포인트 클라우드로 새어 나가면 안 된다."""
    d = np.full(cam.shape, np.nan)
    d[10, 10] = np.inf
    d[20, 20] = 5.0

    pts = cam.unproject(d)
    assert len(pts) == 1
    assert np.isfinite(pts).all()


# --------------------------------------------------------------------------
# 8. 실제 데이터 연동 (SPE3R)
# --------------------------------------------------------------------------

@needs_spe3r
def test_spe3r_loads_with_expected_geometry():
    from src.spe3r import SPE3RModel

    model = SPE3RModel(DATA_DIR, "aqua")
    assert len(model) == 1000
    assert model.camera.width == 256 and model.camera.height == 256
    assert model.camera.fx == pytest.approx(1277.37226)


@needs_spe3r
def test_spe3r_translation_has_no_stereo_baseline():
    """데이터셋에 좌우 베이스라인이 없다는 사실을 테스트로 못박는다.

    이 성질 때문에 스테레오 삼각측량 대신 실루엣 기반 복원을 택했다.
    """
    from src.spe3r import SPE3RModel

    model = SPE3RModel(DATA_DIR, "aqua")
    r = np.array([lb["r_Vo2To_vbs_true"] for lb in model.labels], dtype=float)

    np.testing.assert_allclose(r[:, 0], 0.0)
    np.testing.assert_allclose(r[:, 1], 0.0)
    assert r[:, 2].min() == pytest.approx(5.0)
    assert r[:, 2].max() == pytest.approx(6.0)


@needs_spe3r
def test_spe3r_pose_convention_matches_ground_truth_mask():
    """확정한 자세 규약으로 메시를 투영하면 정답 마스크와 겹쳐야 한다."""
    from src.spe3r import SPE3RModel

    model = SPE3RModel(DATA_DIR, "aqua")
    vertices, _ = model.load_mesh()

    ious = []
    for i in (0, 250, 500, 750):
        pose = model.pose(i)
        p_cam = pose.apply(vertices)
        uv = model.camera.project(p_cam)
        inside = model.camera.in_view(uv)

        pred = np.zeros(model.camera.shape, dtype=bool)
        ui = np.round(uv[inside, 0]).astype(int)
        vi = np.round(uv[inside, 1]).astype(int)
        pred[vi, ui] = True

        gt = model.load_mask(i)
        # 정점만 찍은 실루엣은 성기므로 정답 안에 들어가는 비율로 본다.
        ious.append(float((pred & gt).sum() / max(1, pred.sum())))

    assert np.mean(ious) > 0.9


@needs_spe3r
def test_spe3r_view_selection_spreads_orientations():
    """뷰 선택이 자세를 고르게 흩어 놓는지 확인한다."""
    from src.spe3r import SPE3RModel

    model = SPE3RModel(DATA_DIR, "aqua")
    chosen = model.select_views(12, seed=0)

    assert len(chosen) == len(set(chosen)) == 12
    angles = [rotation_angle_deg(model.pose(a).R, model.pose(b).R)
              for i, a in enumerate(chosen) for b in chosen[i + 1:]]
    assert min(angles) > 30.0
