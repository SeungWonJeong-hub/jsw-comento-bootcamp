"""해석적 합성 장면 — Unit Test 픽스처.

이 모듈이 필요한 이유
    SPE3R 은 실제 위성 데이터지만 픽셀 단위 정답 깊이가 없다. 그래서 카메라
    투영, 역투영, 삼각측량 같은 기하 연산이 '수식대로' 맞는지 검증할 근거가
    없다. 여기서는 구와 직육면체처럼 답을 손으로 풀 수 있는 도형만 써서
    실루엣과 깊이를 해석적으로 만든다. 정답에 렌더링 오차가 섞이지 않으므로
    1e-9 수준의 엄격한 단위 테스트가 가능하다.

구가 특히 유용한 이유
    구는 중심 화소의 깊이가 (거리 - 반지름), 실루엣 반지름이
    f·R/sqrt(d^2 - R^2) 로 닫힌 형태다. 정합 결과를 이 값과 직접 대조할 수
    있어서 '그럴듯한 그림' 과 '수식대로 맞는 값' 을 구분할 수 있다.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_EPS = 1e-12


@dataclass
class Sphere:
    """동체 좌표계의 구."""

    center: tuple = (0.0, 0.0, 0.0)
    radius: float = 0.35
    albedo: float = 0.6

    def __post_init__(self):
        if self.radius <= 0:
            raise ValueError(f"반지름은 양수여야 합니다: {self.radius}")


@dataclass
class Box:
    """동체 좌표계에 축 정렬된 직육면체."""

    center: tuple = (0.0, 0.0, 0.0)
    half_extents: tuple = (0.2, 0.2, 0.3)
    albedo: float = 0.5

    def __post_init__(self):
        if min(self.half_extents) <= 0:
            raise ValueError(f"half_extents 는 모두 양수여야 합니다: {self.half_extents}")


def default_satellite() -> list:
    """본체 + 태양전지판 2 + 안테나로 구성한 위성 목업 (정규화 좌표).

    태양전지판은 어둡고(albedo 0.10) 본체는 밝다(0.55). 밝기를 깊이로 쓰는
    대조군이 어떻게 무너지는지 보이기 위한 의도적인 배치다.
    """
    return [
        Box((0.0, 0.0, 0.0), (0.14, 0.14, 0.20), albedo=0.55),
        Box((-0.42, 0.0, 0.0), (0.28, 0.01, 0.16), albedo=0.10),
        Box((0.42, 0.0, 0.0), (0.28, 0.01, 0.16), albedo=0.10),
        Sphere((0.0, -0.20, 0.02), 0.08, albedo=0.80),
    ]


def _ray_sphere(origin_b, dirs_b, center, radius):
    """광선-구 교차. 반환은 (t_hit, normal)."""
    oc = origin_b - np.asarray(center, dtype=np.float64)
    a = np.einsum("...k,...k->...", dirs_b, dirs_b)
    b = 2.0 * np.einsum("...k,k->...", dirs_b, oc)
    c = float(oc @ oc) - float(radius) ** 2

    disc = b * b - 4.0 * a * c
    hit = disc >= 0.0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0 = (-b - sq) / (2.0 * a)
    t1 = (-b + sq) / (2.0 * a)
    t = np.where(t0 > 0.0, t0, t1)
    t = np.where(hit & (t > 0.0), t, np.inf)

    point = origin_b + dirs_b * np.where(np.isfinite(t), t, 0.0)[..., None]
    normal = point - np.asarray(center, dtype=np.float64)
    n = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal = normal / np.where(n < _EPS, 1.0, n)
    return t, normal


def _ray_box(origin_b, dirs_b, lo, hi):
    """광선-직육면체 교차 (slab method). 반환은 (t_hit, normal)."""
    safe = np.where(np.abs(dirs_b) < _EPS, _EPS, dirs_b)
    t1 = (lo - origin_b) / safe
    t2 = (hi - origin_b) / safe

    t_near = np.minimum(t1, t2)
    t_far = np.maximum(t1, t2)

    axis = np.argmax(t_near, axis=-1)
    t_min = np.take_along_axis(t_near, axis[..., None], axis=-1)[..., 0]
    t_max = np.min(t_far, axis=-1)

    hit = (t_max >= t_min) & (t_max > 0.0) & (t_min > 0.0)
    t = np.where(hit, t_min, np.inf)

    dir_on_axis = np.take_along_axis(dirs_b, axis[..., None], axis=-1)[..., 0]
    sign = np.where(dir_on_axis < 0.0, 1.0, -1.0)
    normal = np.zeros(dirs_b.shape)
    np.put_along_axis(normal, axis[..., None], sign[..., None], axis=-1)
    return t, normal


def _hash_noise(points_body: np.ndarray, freq: float, seed: int) -> np.ndarray:
    """동체 좌표계 위치에 붙는 결정론적 격자 잡음 (0~1).

    스테레오 정합은 표면에 식별 가능한 무늬가 있어야 동작한다. 무늬를 화면이
    아니라 '동체 좌표계'에서 계산하는 것이 핵심이다. 그래야 좌/우 영상에서
    같은 무늬가 같은 표면 지점에 붙어 대응 관계가 성립한다.
    실제 위성 표면도 MLI 주름, 셀 경계 등으로 미세 무늬를 갖는다.
    """
    q = np.floor(points_body * freq).astype(np.int64)
    h = (q[..., 0] * np.int64(73856093)) ^ (q[..., 1] * np.int64(19349663)) \
        ^ (q[..., 2] * np.int64(83492791)) ^ np.int64(seed)
    h = (h ^ (h >> np.int64(13))) * np.int64(1274126177)
    h = h ^ (h >> np.int64(16))
    return (h & np.int64(0xFFFF)).astype(np.float64) / 65535.0


def render(camera, pose, primitives=None,
           sun_direction=(0.42, 0.30, 0.86), ambient: float = 0.10,
           texture_strength: float = 0.0, seed: int = 20260816) -> dict:
    """장면을 한 시점에서 해석적으로 렌더링한다.

    Parameters
    ----------
    camera : PinholeCamera
    pose : Pose (동체 -> 카메라)
    primitives : Sphere / Box 목록. 생략하면 default_satellite()
    sun_direction : 광선이 '진행하는' 방향(태양에서 물체를 향하는 방향).
        카메라를 향한 면(법선 -z)이 빛을 받으려면 z 성분이 양수여야 한다.
        부호를 뒤집으면 앞면이 전부 그늘에 들어가 정합할 무늬가 사라진다.
    texture_strength : 표면 무늬 대비. 0 이면 무늬 없음(순수 음영).
        스테레오 실험에서는 0.3 안팎을 준다. 무늬가 없으면 정합할 단서가 없다.

    Returns
    -------
    dict(image uint8, depth float64 [배경 NaN], mask bool, part_id int)
    """
    primitives = default_satellite() if primitives is None else primitives
    if not primitives:
        raise ValueError("렌더링할 도형이 하나도 없습니다.")

    dirs = camera.pixel_rays()                    # (H, W, 3), z 성분 = 1
    # 카메라 좌표계 -> 동체 좌표계. 광선 매개변수는 두 좌표계에서 같다.
    origin_b = pose.inverse_apply(np.zeros((1, 3)))[0]
    dirs_b = dirs @ pose.R                        # (R^T dir)^T 와 같다

    H, W = camera.shape
    best_t = np.full((H, W), np.inf)
    best_normal = np.zeros((H, W, 3))
    best_part = np.full((H, W), -1, dtype=np.int64)
    albedo = np.zeros((H, W))

    for idx, prim in enumerate(primitives):
        if isinstance(prim, Sphere):
            t, normal = _ray_sphere(origin_b, dirs_b, prim.center, prim.radius)
        elif isinstance(prim, Box):
            c = np.asarray(prim.center, dtype=np.float64)
            h = np.asarray(prim.half_extents, dtype=np.float64)
            t, normal = _ray_box(origin_b, dirs_b, c - h, c + h)
        else:
            raise TypeError(f"지원하지 않는 도형입니다: {type(prim)}")

        closer = t < best_t
        best_t = np.where(closer, t, best_t)
        best_normal = np.where(closer[..., None], normal, best_normal)
        best_part = np.where(closer, idx, best_part)
        albedo = np.where(closer, prim.albedo, albedo)

    mask = np.isfinite(best_t)
    # 광선 방향의 z 성분이 1 이므로 매개변수가 곧 카메라 좌표계 깊이 Z 다.
    depth = np.where(mask, best_t, np.nan)

    normals_cam = best_normal @ pose.R.T
    sun = np.asarray(sun_direction, dtype=np.float64)
    sun = sun / np.linalg.norm(sun)
    lambert = np.clip(np.einsum("hwk,k->hw", normals_cam, -sun), 0.0, 1.0)

    intensity = albedo * (ambient + (1.0 - ambient) * lambert)

    if texture_strength > 0.0:
        hit_b = origin_b + dirs_b * np.where(mask, best_t, 0.0)[..., None]
        noise = (0.7 * _hash_noise(hit_b, 55.0, seed)
                 + 0.3 * _hash_noise(hit_b, 180.0, seed + 977))
        intensity = intensity * (1.0 + texture_strength * (noise - 0.5) * 2.0)

    image = np.clip(np.where(mask, intensity, 0.0) * 255.0, 0, 255).astype(np.uint8)

    return {"image": image, "depth": depth, "mask": mask, "part_id": best_part}


def render_mask(camera, pose, primitives=None) -> np.ndarray:
    """실루엣 마스크만 필요할 때 쓰는 단축 함수."""
    return render(camera, pose, primitives)["mask"]


def orbit_poses(count: int, distance: float = 5.0, seed: int = 0) -> list:
    """타겟을 중심으로 시선 방향이 고르게 흩어진 자세들을 만든다.

    SPE3R 과 같은 구성(카메라는 (0, 0, distance) 에 고정, 타겟만 회전)을 따른다.
    황금각 나선으로 회전축을 흩어 놓아 적은 개수로도 SO(3) 를 고르게 덮는다.
    """
    from .camera import Pose

    if count <= 0:
        raise ValueError(f"뷰 개수는 1 이상이어야 합니다: {count}")

    rng = np.random.default_rng(seed)
    golden = np.pi * (3.0 - np.sqrt(5.0))
    poses = []
    for i in range(count):
        z = 1.0 - 2.0 * (i + 0.5) / count
        r = np.sqrt(max(0.0, 1.0 - z * z))
        phi = golden * i
        axis = np.array([r * np.cos(phi), r * np.sin(phi), z])
        angle = rng.uniform(0.0, 2.0 * np.pi)

        K = np.array([[0.0, -axis[2], axis[1]],
                      [axis[2], 0.0, -axis[0]],
                      [-axis[1], axis[0], 0.0]])
        R = np.eye(3) + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        poses.append(Pose(R=R, t=(0.0, 0.0, float(distance))))
    return poses
