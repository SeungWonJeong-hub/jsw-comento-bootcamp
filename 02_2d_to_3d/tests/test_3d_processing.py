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

from src import baseline, depth, metrics, pointcloud, scene  # noqa: E402
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

@pytest.mark.parametrize("seed", range(8))
def test_quaternion_gives_proper_rotation(seed):
    """쿼터니언에서 만든 행렬은 직교하고 행렬식이 +1 이어야 한다.

    무작위 쿼터니언마다 독립적인 경우이므로 seed 로 나눈다. 한 번에 루프를
    돌면 첫 실패에서 멈춰 나머지가 실행되지 않고, 어느 값에서 깨졌는지도
    테스트 이름에 남지 않는다.
    """
    q = np.random.default_rng(seed).normal(size=4)
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
# 3. 포인트 클라우드 — 변환과 입출력
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
# 4. 평가 지표
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
# 5. 대조군(과제 예시 코드)의 실패 특성화
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
    같은 이유로 parametrize 로 나누지 않는다. 자세별 판정이 아니라 분포로
    판정하는 것이 이 테스트의 의도다.
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
# 6. 경계 조건과 예외 처리
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
# 7. 실제 데이터 연동 (SPE3R)
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

    이 성질이 본 실험의 출발점이다. 카메라 좌표계에는 베이스라인이 없으므로
    상대 자세 (R_ij, t_ij) 에서 유효 베이스라인을 끌어내야 한다. 그 경로가
    실제로 성립하는지는 test_stereo.py 의 test_target_rotation_creates_
    lateral_baseline 과 test_spe3r_pair_relative_pose_is_consistent_with_mesh
    가 확인한다.
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

    # IoU 가 아니라 정밀도(예측 실루엣 안에 정답이 든 비율)를 잰다. 정점만
    # 투영하면 실루엣이 성겨서 합집합 기준 IoU 는 낮게 나온다. 규약이 틀리면
    # 이 값이 0.3 아래로 떨어지므로 판별에는 충분하다.
    #
    # 여기는 parametrize 로 나누지 않는다. 뷰마다 값이 0.87~0.94 로 흔들려서
    # 뷰 하나씩 0.9 를 넘으라고 걸면 뷰 750(0.8749) 에서 깨진다. 임계값을
    # 낮추면 규약이 틀렸을 때를 못 잡는다. 개별 판정이 아니라 평균으로 보는
    # 것이 이 테스트의 의도다.
    precisions = []
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
        precisions.append(float((pred & gt).sum() / max(1, pred.sum())))

    assert np.mean(precisions) > 0.9


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
