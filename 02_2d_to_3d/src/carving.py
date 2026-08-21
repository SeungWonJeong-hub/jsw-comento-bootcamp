"""실루엣 기반 3D 복원 (voxel space carving).

원리
    3D 공간을 복셀 격자로 자르고, 각 복셀 중심을 모든 시점의 영상에 투영한다.
    한 시점에서라도 실루엣 바깥으로 떨어지면 그 복셀은 물체가 아니므로 깎아낸다.
    남은 복셀의 합집합이 visual hull 이며, 뷰가 많아질수록 실제 형상에 가까워진다.

왜 SPE3R 에 이 방법도 함께 두는가
    본 실험의 주 방법은 스테레오 삼각측량이다. 카메라 좌표계에는 베이스라인이
    없지만 두 뷰의 상대 자세에서 유효 베이스라인이 나온다 (src/stereo.py).
    다만 조건이 까다로워 1,000 뷰에서 쓸 만한 쌍이 5 개뿐이다.

    무작위 자세는 시선 방향이 SO(3) 를 고르게 덮으므로 실루엣 기반 복원에는
    오히려 이상적이다. 특징점 매칭이 필요 없어 조명 변화와 무텍스처 표면에도
    강건하고 뒷면까지 복원된다. 스테레오와 약점이 상보적이라 대안으로 함께 둔다.

한계 (반드시 함께 보고해야 하는 것)
    visual hull 은 실루엣의 교집합이므로 오목한(concave) 부분을 복원하지 못한다.
    안테나 접시 안쪽, 본체 사이의 홈 등은 원리적으로 메워진 채로 남는다.
"""

from __future__ import annotations

import numpy as np


def make_voxel_grid(bounds, resolution: int = 128):
    """정육면체 영역을 균일 복셀로 나눈다.

    Parameters
    ----------
    bounds : (min_xyz, max_xyz) 또는 스칼라 반폭
    resolution : 한 축당 복셀 개수

    Returns
    -------
    centers : (R, R, R, 3) 복셀 중심
    spacing : float 복셀 한 변 길이
    """
    if resolution < 2:
        raise ValueError(f"해상도는 2 이상이어야 합니다: {resolution}")

    if np.isscalar(bounds):
        lo = np.full(3, -float(bounds))
        hi = np.full(3, float(bounds))
    else:
        lo = np.asarray(bounds[0], dtype=np.float64).reshape(3)
        hi = np.asarray(bounds[1], dtype=np.float64).reshape(3)
    if np.any(hi <= lo):
        raise ValueError(f"bounds 의 최대값이 최소값보다 커야 합니다: {lo} ~ {hi}")

    axes = [np.linspace(lo[k], hi[k], resolution) for k in range(3)]
    gx, gy, gz = np.meshgrid(*axes, indexing="ij")
    centers = np.stack([gx, gy, gz], axis=-1)
    spacing = float((hi - lo).max() / (resolution - 1))
    return centers, spacing


def bounds_from_mesh(vertices: np.ndarray, margin: float = 0.05):
    """메시의 경계 상자에 여유를 준 정육면체 영역을 만든다."""
    v = np.asarray(vertices, dtype=np.float64)
    lo, hi = v.min(axis=0), v.max(axis=0)
    center = (lo + hi) / 2.0
    half = float((hi - lo).max()) / 2.0 * (1.0 + margin)
    return center - half, center + half


def carve(camera, masks, poses, bounds, resolution: int = 128,
          mask_margin: int = 1, verbose: bool = False):
    """여러 시점의 실루엣으로 복셀 격자를 깎는다.

    Parameters
    ----------
    camera : PinholeCamera
    masks : 길이 N 의 (H, W) bool 배열 목록
    poses : 길이 N 의 Pose 목록 (동체 -> 카메라)
    bounds : make_voxel_grid 에 넘길 영역
    resolution : 복셀 해상도
    mask_margin : 마스크 경계를 이 픽셀만큼 넓힌다 (음수면 줄인다).

    화면 밖으로 나가는 복셀
        관측되지 않은 것으로 보고 깎지 않는다. 자세한 근거는 아래 구현의 주석.

    mask_margin 이 왜 필요한가
        visual hull 은 실루엣이 물체의 참 투영을 '포함'해야 성립한다. 실루엣이
        조금이라도 작으면 그 부족분이 시점마다 다른 방향에서 물체를 깎아내고,
        뷰를 추가할수록 복원이 나빠진다.

        SPE3R 의 정답 마스크를 실측한 결과 메시 투영 실루엣보다 약 1 픽셀 작다.
        (마스크 대비 실루엣 면적비 1.111, 1 픽셀 팽창 시 0.982, IoU 0.900 -> 0.953)
        따라서 기본값을 1 로 두어 실루엣을 과대 근사 쪽으로 맞춘다.

    Returns
    -------
    dict(occupancy, centers, spacing, kept_ratio, history)
    """
    if len(masks) != len(poses):
        raise ValueError(f"마스크 {len(masks)} 개와 자세 {len(poses)} 개의 수가 다릅니다.")
    if len(masks) == 0:
        raise ValueError("시점이 하나도 없습니다.")

    centers, spacing = make_voxel_grid(bounds, resolution)
    flat = centers.reshape(-1, 3)
    alive = np.ones(len(flat), dtype=bool)
    history = []

    for i, (mask, pose) in enumerate(zip(masks, poses)):
        mask = np.asarray(mask).astype(bool)
        if mask.shape != camera.shape:
            raise ValueError(f"마스크 크기 {mask.shape} 가 카메라 {camera.shape} 와 다릅니다.")
        if mask_margin != 0:
            import cv2
            k = np.ones((3, 3), np.uint8)
            op = cv2.dilate if mask_margin > 0 else cv2.erode
            mask = op(mask.astype(np.uint8), k,
                      iterations=abs(mask_margin)).astype(bool)

        idx = np.flatnonzero(alive)
        if idx.size == 0:
            break

        p_cam = flat[idx] @ pose.R.T + pose.t
        z = p_cam[:, 2]
        ok = z > 1e-9

        u = np.full(idx.size, -1.0)
        v = np.full(idx.size, -1.0)
        u[ok] = camera.fx * p_cam[ok, 0] / z[ok] + camera.cx
        v[ok] = camera.fy * p_cam[ok, 1] / z[ok] + camera.cy

        ui = np.round(u).astype(np.int64)
        vi = np.round(v).astype(np.int64)
        inside_image = ok & (ui >= 0) & (ui < camera.width) & (vi >= 0) & (vi < camera.height)

        # 화면 밖으로 투영된 복셀은 '관측되지 않음'이지 '물체 아님'이 아니다.
        # 깎아 버리면 visual hull 이 참 형상을 포함한다는 성질이 깨진다. 실제로
        # SPE3R 은 1,000 뷰 중 552 뷰에서 실루엣이 화면 테두리에 닿고, 카빙에
        # 쓰는 20 뷰 중에도 15 뷰가 그렇다. 합성 장면에서 타겟이 모든 뷰에서
        # 잘리게 만들면 복원 반지름이 0.300 -> 0.252 로 깎여 나갔다.
        #
        # 대신 유지하면 그 방향으로는 아무 제약이 걸리지 않으므로, 뷰가 전부
        # 잘려 있으면 hull 이 격자 경계까지 부푼다(같은 실험에서 0.866).
        # 과하게 깎는 것은 틀린 답이고 과하게 남기는 것은 느슨한 답이라,
        # 포함 성질을 지키는 쪽을 택한다.
        keep = np.ones(idx.size, dtype=bool)
        sel = np.flatnonzero(inside_image)
        keep[sel] = mask[vi[sel], ui[sel]]

        alive[idx] = keep
        history.append(int(alive.sum()))
        if verbose:
            print(f"  view {i + 1:3d}/{len(masks)}  남은 복셀 {alive.sum():,}")

    occupancy = alive.reshape(centers.shape[:3])
    return {
        "occupancy": occupancy,
        "centers": centers,
        "spacing": spacing,
        "kept_ratio": float(occupancy.mean()),
        "history": history,
    }


def surface_points(occupancy: np.ndarray, centers: np.ndarray) -> np.ndarray:
    """채워진 복셀 중 표면에 있는 것들의 중심 좌표를 뽑는다.

    내부 복셀까지 모두 내보내면 표면 비교(Chamfer)에서 정답과 어긋난다.
    6-이웃이 하나라도 비어 있으면 표면으로 본다.
    """
    occ = np.asarray(occupancy).astype(bool)
    if occ.ndim != 3:
        raise ValueError(f"(R, R, R) 배열이 필요합니다: {occ.shape}")

    interior = np.ones_like(occ)
    interior[1:, :, :] &= occ[:-1, :, :]
    interior[:-1, :, :] &= occ[1:, :, :]
    interior[:, 1:, :] &= occ[:, :-1, :]
    interior[:, :-1, :] &= occ[:, 1:, :]
    interior[:, :, 1:] &= occ[:, :, :-1]
    interior[:, :, :-1] &= occ[:, :, 1:]
    # 격자 경계에 닿은 복셀은 내부로 볼 수 없다.
    interior[0, :, :] = interior[-1, :, :] = False
    interior[:, 0, :] = interior[:, -1, :] = False
    interior[:, :, 0] = interior[:, :, -1] = False

    surface = occ & ~interior
    return centers[surface]


def render_depth(points_body: np.ndarray, pose, camera,
                 fill_holes: bool = True, splat: int = 0) -> np.ndarray:
    """복원한 3D 점을 한 시점으로 되쏘아 깊이 맵을 만든다 (3D -> 2D 역변환).

    과제의 '깊이 맵 생성'을 실루엣 복원 결과로부터 얻는 경로다.
    같은 픽셀에 여러 점이 겹치면 가장 가까운 것(z 최소)을 남긴다 (z-buffer).

    Returns
    -------
    (H, W) float64. 물체가 없는 픽셀은 NaN.
    """
    points_body = np.asarray(points_body, dtype=np.float64)
    if points_body.ndim != 2 or points_body.shape[1] != 3:
        raise ValueError(f"(N, 3) 배열이 필요합니다: {points_body.shape}")

    p_cam = points_body @ pose.R.T + pose.t
    z = p_cam[:, 2]
    ok = z > 1e-9
    if not np.any(ok):
        return np.full(camera.shape, np.nan)

    u = camera.fx * p_cam[ok, 0] / z[ok] + camera.cx
    v = camera.fy * p_cam[ok, 1] / z[ok] + camera.cy
    zz = z[ok]

    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inside = (ui >= 0) & (ui < camera.width) & (vi >= 0) & (vi < camera.height)
    ui, vi, zz = ui[inside], vi[inside], zz[inside]

    depth = np.full(camera.shape, np.inf)
    # 같은 픽셀에 여러 점이 오므로 최소값 누적을 쓴다.
    # splat > 0 이면 점 하나를 주변 화소까지 번지게 한다. 복셀 중심을 투영하면
    # 표면이 성기게 찍혀 화면에 구멍이 생기는데, 이를 메우기 위한 것이다.
    offsets = [(dy, dx)
               for dy in range(-splat, splat + 1)
               for dx in range(-splat, splat + 1)]
    for dy, dx in offsets:
        yy = vi + dy
        xx = ui + dx
        good = (yy >= 0) & (yy < camera.height) & (xx >= 0) & (xx < camera.width)
        np.minimum.at(depth, (yy[good], xx[good]), zz[good])
    depth[~np.isfinite(depth)] = np.nan

    if fill_holes:
        depth = _close_pinholes(depth)
    return depth


def _close_pinholes(depth: np.ndarray) -> np.ndarray:
    """복셀 투영이 성기어 생긴 한두 픽셀짜리 구멍을 이웃 중앙값으로 메운다."""
    import cv2

    valid = np.isfinite(depth).astype(np.uint8)
    if valid.sum() == 0:
        return depth

    kernel = np.ones((3, 3), np.uint8)
    closed = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, kernel)
    holes = (closed > 0) & (valid == 0)
    if not np.any(holes):
        return depth

    filled = depth.copy()
    substitute = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    weight = valid.astype(np.float32)
    num = cv2.blur(substitute, (3, 3))
    den = cv2.blur(weight, (3, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = num / den
    filled[holes] = avg[holes]
    return filled
