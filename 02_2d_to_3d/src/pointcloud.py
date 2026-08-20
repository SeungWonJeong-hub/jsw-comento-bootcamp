"""포인트 클라우드 생성, 변환, 저장.

Open3D 를 쓰지 않고 numpy 와 PLY 파일 규격만으로 처리한다. PLY 는
MeshLab, CloudCompare, Blender 등 대부분의 3D 뷰어에서 바로 열린다.
"""

from __future__ import annotations

import os

import numpy as np


def depth_to_pointcloud(camera, depth: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    """깊이 맵 -> 카메라 좌표계 포인트 클라우드 (N, 3)."""
    return camera.unproject(depth, mask=mask)


def zbuffer_depth(points_cam: np.ndarray, camera, splat: int = 0,
                  fill_holes: bool = False) -> np.ndarray:
    """카메라 좌표계 점들을 깊이 맵으로 투영한다 (z-buffer).

    같은 화소에 여러 점이 오면 가장 가까운 것을 남긴다. 메시를 조밀하게
    샘플링해 넣으면 기준 깊이 맵(reference depth)을 만들 수 있고, 복원한
    포인트 클라우드를 넣으면 3D -> 2D 역변환이 된다.

    Parameters
    ----------
    splat : 점 하나를 주변 몇 화소까지 번지게 할지. 점이 성길 때 구멍을 막는다.

    Returns
    -------
    (H, W) float64. 점이 닿지 않은 화소는 NaN.
    """
    points_cam = np.asarray(points_cam, dtype=np.float64)
    if points_cam.ndim != 2 or points_cam.shape[1] != 3:
        raise ValueError(f"(N, 3) 배열이 필요합니다: {points_cam.shape}")
    if splat < 0:
        raise ValueError(f"splat 은 0 이상이어야 합니다: {splat}")

    z = points_cam[:, 2]
    ok = z > 1e-9
    if not np.any(ok):
        return np.full(camera.shape, np.nan)

    u = camera.fx * points_cam[ok, 0] / z[ok] + camera.cx
    v = camera.fy * points_cam[ok, 1] / z[ok] + camera.cy
    zz = z[ok]

    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)

    depth = np.full(camera.shape, np.inf)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            yy, xx = vi + dy, ui + dx
            good = ((yy >= 0) & (yy < camera.height)
                    & (xx >= 0) & (xx < camera.width))
            np.minimum.at(depth, (yy[good], xx[good]), zz[good])

    depth[~np.isfinite(depth)] = np.nan
    if fill_holes:
        depth = _close_pinholes(depth)
    return depth


def _close_pinholes(depth: np.ndarray) -> np.ndarray:
    """한두 화소짜리 구멍을 이웃 평균으로 메운다."""
    import cv2

    valid = np.isfinite(depth).astype(np.uint8)
    if valid.sum() == 0:
        return depth

    closed = cv2.morphologyEx(valid, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    holes = (closed > 0) & (valid == 0)
    if not np.any(holes):
        return depth

    substitute = np.where(np.isfinite(depth), depth, 0.0).astype(np.float32)
    num = cv2.blur(substitute, (3, 3))
    den = cv2.blur(valid.astype(np.float32), (3, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        avg = num / den

    filled = depth.copy()
    filled[holes] = avg[holes]
    return filled


def transform_points(points: np.ndarray, R: np.ndarray, t=None) -> np.ndarray:
    """강체 변환 p' = R p + t."""
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"(N, 3) 배열이 필요합니다: {points.shape}")
    R = np.asarray(R, dtype=np.float64)
    if R.shape != (3, 3):
        raise ValueError(f"회전 행렬은 (3, 3) 이어야 합니다: {R.shape}")
    out = points @ R.T
    if t is not None:
        out = out + np.asarray(t, dtype=np.float64).reshape(3)
    return out


def sample_mesh_surface(vertices: np.ndarray, faces: np.ndarray,
                        count: int = 30000, seed: int = 0) -> np.ndarray:
    """메시 표면에서 면적 비례로 점을 샘플링한다 (정답 포인트 클라우드 생성용).

    면적 비례로 뽑아야 큰 면과 작은 면이 고르게 대표된다. 정점만 쓰면
    메시의 삼각형 밀도 편향이 그대로 정답에 실린다.
    """
    if count <= 0:
        raise ValueError(f"샘플 개수는 1 이상이어야 합니다: {count}")
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.size == 0:
        raise ValueError("삼각형 면이 없습니다.")

    a = vertices[faces[:, 0]]
    b = vertices[faces[:, 1]]
    c = vertices[faces[:, 2]]
    area = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    total = area.sum()
    if total <= 0:
        raise ValueError("메시 전체 면적이 0 입니다.")

    rng = np.random.default_rng(seed)
    pick = rng.choice(len(faces), size=count, p=area / total)

    # 삼각형 내부 균등 샘플링
    u = rng.random(count)
    v = rng.random(count)
    flip = u + v > 1.0
    u[flip] = 1.0 - u[flip]
    v[flip] = 1.0 - v[flip]

    return a[pick] + u[:, None] * (b[pick] - a[pick]) + v[:, None] * (c[pick] - a[pick])


def normalize_scale(points: np.ndarray):
    """중심을 원점으로 옮기고 최대 반지름을 1 로 맞춘다.

    Returns
    -------
    (정규화된 점, center, scale)
    """
    points = np.asarray(points, dtype=np.float64)
    if len(points) == 0:
        raise ValueError("빈 포인트 클라우드입니다.")
    center = points.mean(axis=0)
    centered = points - center
    scale = float(np.linalg.norm(centered, axis=1).max())
    if scale <= 0:
        raise ValueError("모든 점이 한 자리에 겹쳐 있습니다.")
    return centered / scale, center, scale


def write_ply(path: str, points: np.ndarray, colors: np.ndarray = None) -> str:
    """포인트 클라우드를 바이너리 PLY 로 저장한다."""
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"(N, 3) 배열이 필요합니다: {points.shape}")

    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    n = len(points)

    header = ["ply", "format binary_little_endian 1.0", f"element vertex {n}",
              "property float x", "property float y", "property float z"]
    if colors is not None:
        colors = np.asarray(colors)
        if colors.shape != points.shape:
            raise ValueError(f"색상 크기 {colors.shape} 가 점 {points.shape} 와 다릅니다.")
        colors = np.clip(colors, 0, 255).astype(np.uint8)
        header += ["property uchar red", "property uchar green", "property uchar blue"]
    header.append("end_header")

    with open(path, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        if colors is None:
            f.write(points.tobytes())
        else:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec = np.empty(n, dtype=dtype)
            rec["x"], rec["y"], rec["z"] = points[:, 0], points[:, 1], points[:, 2]
            rec["r"], rec["g"], rec["b"] = colors[:, 0], colors[:, 1], colors[:, 2]
            f.write(rec.tobytes())
    return path


def read_ply(path: str) -> np.ndarray:
    """write_ply 로 저장한 바이너리 PLY 를 다시 읽는다 (왕복 검증용)."""
    with open(path, "rb") as f:
        count = None
        has_color = False
        while True:
            line = f.readline().decode("ascii").strip()
            if line.startswith("element vertex"):
                count = int(line.split()[-1])
            elif line.startswith("property uchar red"):
                has_color = True
            elif line == "end_header":
                break
            elif not line:
                raise ValueError(f"PLY 헤더가 올바르지 않습니다: {path}")
        if count is None:
            raise ValueError(f"정점 개수를 찾지 못했습니다: {path}")

        if has_color:
            dtype = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                              ("r", "u1"), ("g", "u1"), ("b", "u1")])
            rec = np.frombuffer(f.read(count * dtype.itemsize), dtype=dtype)
            return np.stack([rec["x"], rec["y"], rec["z"]], axis=1).astype(np.float64)
        raw = np.frombuffer(f.read(count * 12), dtype="<f4")
        return raw.reshape(count, 3).astype(np.float64)
