"""달 지형(DTM)에서 스테레오 영상 쌍을 만든다.

왜 이 모듈이 필요한가
    궤도에서 찍은 원본 영상으로 곧장 스테레오를 돌리려면 카메라 자세를 담은
    커널을 읽고 지도 투영을 맞추는 전용 도구 체계가 필요하다. 그 준비만으로
    며칠이 간다.

    대신 **이미 측정된 고도 모델(DTM)** 에서 출발한다. 고도를 3D 점으로 펴고,
    자세를 아는 카메라 두 대로 다시 찍는다. 그러면
      - 정답 고도가 처음부터 손에 있고 (레이저 고도계로 잰 값)
      - 카메라 자세도 우리가 정한 값이라 정확하며
      - 복원 결과를 원래 고도와 바로 견줄 수 있다.

    착륙 위험 탐지 연구에서 실제로 쓰는 방식이다. 다만 **기하는 실제 달이지만
    밝기는 렌더링**이라는 점은 분명히 해야 한다. 그림자와 표면 반사 특성은
    실제 영상과 다르다.

좌표 규약
    지형 좌표계는 X 동쪽, Y 북쪽, Z 위(고도)다. 카메라는 이 좌표계 안의
    위치와 자세로 놓는다. 카메라 좌표계는 저장소의 다른 모듈과 같다
    (z 가 시선 방향, y 가 영상 아래쪽).
"""

from __future__ import annotations

import cv2
import numpy as np

from .camera import Pose

__all__ = ["load_dtm", "synthetic_dtm", "surface_normals", "surface_points",
           "shade", "render", "render_heightfield", "look_at",
           "stereo_cameras", "camera_center", "sensor_image",
           "load_image", "estimate_illumination", "derive_albedo",
           "LUNAR_LAMBERT_COEFFS"]

#: 위상각에 따른 Lunar-Lambert 계수 L(g) 의 다항식 계수 (McEwen 1991).
#: g 는 도 단위이며, L(0)=1 에서 시작해 위상각이 커질수록 0 쪽으로 간다.
#: NASA Ames Stereo Pipeline 과 ISIS 가 달에 쓰는 값과 같다.
LUNAR_LAMBERT_COEFFS = (-1.9e-2, 2.42e-4, -1.46e-6)


def load_dtm(path: str, crop: int = None):
    """GeoTIFF 고도 모델을 읽는다.

    Returns
    -------
    (elev, gsd) : 고도 [m] 배열과 화소 한 변의 지상 거리 [m/px].

    화소 크기를 함께 돌려주는 이유는, 이것이 이후 모든 계산의 기준이 되기
    때문이다. 카메라 초점거리도 "고도 / 화소 크기" 로 정한다.
    """
    import rasterio

    with rasterio.open(path) as src:
        elev = src.read(1).astype(np.float64)
        gsd = float(abs(src.transform.a))
        nodata = src.nodata

    if nodata is not None:
        elev = np.where(elev == nodata, np.nan, elev)
    if not np.isfinite(elev).any():
        raise ValueError(f"유효한 고도 값이 없습니다: {path}")

    # 빈 값은 주변 평균으로 메운다. 스테레오는 구멍을 만나면 그 자리에서
    # 엉뚱한 대응을 만들어 내므로, 없는 곳을 남겨 두면 안 된다.
    if np.isnan(elev).any():
        elev = np.where(np.isnan(elev), np.nanmean(elev), elev)

    if crop:
        h, w = elev.shape
        if crop > min(h, w):
            raise ValueError(f"crop {crop} 이 지형 크기 {elev.shape} 보다 큽니다.")
        y0, x0 = (h - crop) // 2, (w - crop) // 2
        elev = elev[y0:y0 + crop, x0:x0 + crop]
    return elev, gsd


def synthetic_dtm(size: int = 512, gsd: float = 5.0, relief: float = 300.0,
                  n_craters: int = 25, seed: int = 0):
    """실제 DTM 이 없을 때 쓰는 가짜 달 지형.

    데이터를 받기 전에도 파이프라인 전체를 돌려 볼 수 있어야 하고, 단위
    테스트도 데이터 없이 통과해야 한다. 큰 굴곡 위에 크레이터를 파 넣는다.

    크레이터를 넣는 이유는 무늬 때문이다. 매끈한 경사면만 있으면 정합할
    단서가 없어 시차를 못 찾는다. 실제 달 표면이 정합이 잘 되는 것도
    크레이터와 바위가 만드는 그림자 덕분이다.
    """
    if size < 32:
        raise ValueError(f"size 는 32 이상이어야 합니다: {size}")
    rng = np.random.default_rng(seed)

    # 저주파 굴곡: 무작위 잡음을 여러 배율로 겹친다.
    elev = np.zeros((size, size))
    for k, amp in ((4, 1.0), (8, 0.5), (16, 0.25), (32, 0.12)):
        small = rng.normal(size=(k, k))
        big = np.kron(small, np.ones((size // k + 1, size // k + 1)))[:size, :size]
        elev += amp * big
    elev = elev / np.abs(elev).max() * relief

    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    for _ in range(n_craters):
        r = rng.uniform(size * 0.02, size * 0.10)
        cx, cy = rng.uniform(r, size - r, size=2)
        d = np.hypot(xx - cx, yy - cy) / r
        inside = d < 1.0
        # 사발 모양 바닥 + 테두리 둔덕. 깊이는 지름에 비례한다.
        depth = r * gsd * 0.2
        elev -= np.where(inside, depth * (1.0 - d ** 2), 0.0)
        rim = (d >= 1.0) & (d < 1.25)
        elev += np.where(rim, depth * 0.25 * (1.25 - d) / 0.25, 0.0)
    return elev, gsd


def surface_normals(elev: np.ndarray, gsd: float) -> np.ndarray:
    """고도 격자에서 표면 법선을 구한다. (H, W, 3)"""
    if gsd <= 0:
        raise ValueError(f"화소 크기는 양수여야 합니다: {gsd}")
    dzdy, dzdx = np.gradient(np.asarray(elev, dtype=np.float64), gsd)
    n = np.dstack([-dzdx, -dzdy, np.ones_like(elev)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def surface_points(elev: np.ndarray, gsd: float) -> np.ndarray:
    """고도 격자를 (N, 3) 지형 좌표 점으로 편다. X 동, Y 북, Z 고도."""
    if gsd <= 0:
        raise ValueError(f"화소 크기는 양수여야 합니다: {gsd}")
    h, w = np.asarray(elev).shape
    y, x = np.mgrid[0:h, 0:w]
    # 격자의 가운데를 원점으로 둔다. 카메라를 원점 위에 놓기 편하다.
    X = (x - (w - 1) / 2.0) * gsd
    Y = ((h - 1) / 2.0 - y) * gsd
    return np.column_stack([X.ravel(), Y.ravel(), np.asarray(elev).ravel()])


def shade(elev: np.ndarray, gsd: float, sun_elevation_deg: float = 25.0,
          sun_azimuth_deg: float = 135.0, albedo=0.12, ambient: float = 0.06,
          viewer=None):
    """지형의 밝기를 만든다.

    태양 고도를 낮게 두는 것이 기본값인 이유는, 그림자가 길어야 지형의
    무늬가 살아나기 때문이다. 실제 달 영상도 태양이 낮을 때 찍은 것이
    지형 판독에 쓰인다. 태양이 머리 위면 표면이 밋밋해져 정합이 어려워진다.

    viewer 를 주면 달 광도함수 (Lunar-Lambert) 를 쓴다
        램버트 반사는 밝기가 태양 방향에만 의존한다. 그러면 같은 지면
        조각이 두 사진에서 **정확히 같은 밝기**로 찍힌다. 수렴각을 아무리
        벌려도 밝기는 변하지 않으므로, 정합기 입장에서는 실제보다 훨씬 쉬운
        문제가 된다.

        실제 달 표면은 후방산란이 강해서 보는 방향에 따라 밝기가 달라진다.
        그래서 수렴 촬영의 두 사진은 같은 조각을 다른 밝기로 담는다. 이
        차이가 정합의 진짜 난이도이며, viewer 를 주지 않으면 그 난이도가
        실험에서 통째로 빠진다.

            I = A [ 2 L(g) mu0 / (mu0 + mu) + (1 - L(g)) mu0 ]

            mu0 = cos(입사각)   법선과 태양 방향의 사이각
            mu  = cos(방출각)   법선과 카메라 방향의 사이각
            g   = 위상각        태양 방향과 카메라 방향의 사이각
            L(g) = 1 + A1 g + A2 g^2 + A3 g^3   (McEwen 1991, 도 단위)

        L=0 이면 램버트, L=1 이면 Lommel-Seeliger 가 되는 보간식이다.
        위상각이 커질수록 L 이 작아지며 램버트 쪽으로 간다.

    Parameters
    ----------
    viewer : 카메라 위치 (3,) [m]. 지형 좌표계. None 이면 램버트.
    """
    az = np.radians(sun_azimuth_deg)
    el = np.radians(sun_elevation_deg)
    sun = np.array([np.sin(az) * np.cos(el), np.cos(az) * np.cos(el),
                    np.sin(el)])
    n = surface_normals(elev, gsd)
    mu0 = np.clip(n @ sun, 0.0, None)
    if viewer is None:
        return np.clip(np.asarray(albedo) * mu0 + ambient, 0.0, 1.0)

    viewer = np.asarray(viewer, dtype=np.float64).reshape(3)
    pts = surface_points(elev, gsd).reshape(elev.shape + (3,))
    v = viewer - pts
    norm = np.linalg.norm(v, axis=2, keepdims=True)
    if np.any(norm <= 0):
        raise ValueError("카메라가 지형면 위에 놓여 있습니다.")
    v = v / norm
    mu = np.clip((n * v).sum(axis=2), 0.0, None)

    g = np.degrees(np.arccos(np.clip(v @ sun, -1.0, 1.0)))
    a1, a2, a3 = LUNAR_LAMBERT_COEFFS
    L = np.clip(1.0 + a1 * g + a2 * g ** 2 + a3 * g ** 3, 0.0, 1.0)

    denom = mu0 + mu
    # 입사와 방출이 모두 스치는 각이면 두 항이 함께 0 이라 값이 없다.
    lommel = np.where(denom > 1e-9, 2.0 * mu0 / np.maximum(denom, 1e-9), 0.0)
    refl = L * lommel + (1.0 - L) * mu0
    return np.clip(np.asarray(albedo) * refl + ambient, 0.0, 1.0)


def camera_center(pose) -> np.ndarray:
    """카메라 중심을 지형 좌표계에서 구한다.

    자세는 p_cam = R p + t 이므로 p_cam = 0 인 점이 중심이고, 곧 -R^T t 다.
    """
    return -pose.R.T @ pose.t


def sensor_image(intensity, snr: float = 100.0, blur_px: float = 0.7,
                 bits: int = 8, read_noise_e: float = 20.0, seed: int = 0):
    """렌더링한 밝기를 실제 카메라가 찍은 것처럼 망가뜨린다.

    왜 필요한가
        렌더링 영상은 잡음이 0 이고 초점이 완벽하다. 그런 영상에서 정합이
        잘 되는 것은 당연하며, 그 성적을 실제 영상에 옮겨 말할 수 없다.
        실제 궤도 카메라에는 광학 흐림(PSF)이 있고, 빛은 광자 단위로 오므로
        밝기에 따라 커지는 산탄 잡음이 있으며, 읽어 낼 때 잡음이 더해지고,
        마지막에 정해진 비트 수로 잘린다. 넷을 모두 넣는다.

    Parameters
    ----------
    snr : 최대 밝기에서의 신호 대 잡음비. 광자 수 N 의 잡음이 sqrt(N) 이므로
          가득 찬 우물의 광자 수를 snr^2 으로 잡는 것과 같다. LROC NAC 이
          대략 100 수준이다.
    blur_px : 광학 흐림의 표준편차 [pixel]
    bits : 양자화 비트 수
    read_noise_e : 읽기 잡음 [전자]
    """
    if snr <= 0:
        raise ValueError(f"snr 은 양수여야 합니다: {snr}")
    if not 1 <= bits <= 16:
        raise ValueError(f"bits 는 1~16 이어야 합니다: {bits}")
    if blur_px < 0:
        raise ValueError(f"blur_px 는 0 이상이어야 합니다: {blur_px}")

    img = np.asarray(intensity, dtype=np.float64) / 255.0
    if blur_px > 0:
        img = cv2.GaussianBlur(img, (0, 0), blur_px,
                               borderType=cv2.BORDER_REPLICATE)

    rng = np.random.default_rng(seed)
    full_well = float(snr) ** 2
    electrons = rng.poisson(np.clip(img, 0.0, 1.0) * full_well)
    electrons = electrons + rng.normal(0.0, read_noise_e, electrons.shape)

    levels = 2 ** bits - 1
    dn = np.clip(electrons / full_well, 0.0, 1.0)
    return (np.round(dn * levels) / levels * 255.0).astype(np.uint8)


def look_at(position, target, up_hint=(0.0, 1.0, 0.0)) -> Pose:
    """지형 좌표계의 한 점을 바라보는 카메라 자세를 만든다.

    카메라 좌표계는 z 가 시선, x 가 영상 오른쪽, y 가 영상 아래쪽이다.

    up_hint 를 **수평 방향(기본 북쪽)** 으로 두는 이유
        아래를 내려다보는 카메라는 시선이 거의 -Z 라, 흔히 쓰는 up=(0,0,1)
        과 거의 나란해진다. 그러면 외적이 0 에 가까워져 영상의 오른쪽이
        시선의 미세한 수평 성분으로 정해지고, 카메라 위치가 조금만 달라져도
        방향이 홱 뒤집힌다. 실제로 수렴 촬영 두 대의 상대 회전이 20도가
        아니라 180도로 나왔다. 수평 기준을 주면 두 대가 같은 쪽을 오른쪽으로
        삼는다.
    """
    position = np.asarray(position, dtype=np.float64)
    forward = np.asarray(target, dtype=np.float64) - position
    n = np.linalg.norm(forward)
    if n < 1e-9:
        raise ValueError("카메라와 바라보는 점이 같은 위치입니다.")
    forward = forward / n

    up_hint = np.asarray(up_hint, dtype=np.float64)
    right = np.cross(forward, up_hint)
    if np.linalg.norm(right) < 1e-9:
        raise ValueError("시선이 up_hint 와 나란해 자세를 정할 수 없습니다.")
    right /= np.linalg.norm(right)
    down = np.cross(forward, right)

    R_wc = np.vstack([right, down, forward])      # 지형 -> 카메라
    return Pose(R_wc, -R_wc @ position)


def stereo_cameras(altitude: float, convergence_deg: float,
                   target=(0.0, 0.0, 0.0)):
    """같은 지점을 바라보는 카메라 두 대를 만든다.

    실제 궤도 스테레오는 두 대를 나란히 놓는 것이 아니라, 지나가며 앞뒤로
    기울여 같은 지역을 두 번 찍는다(수렴 촬영). 그래서 여기서도 두 카메라가
    같은 점을 바라보게 둔다.

    convergence_deg 가 무엇을 정하는가
        두 시선이 벌어진 각이다. 이 각이 깊이 정확도를 지배한다. 베이스라인은
        B = 2 * H * tan(각/2) 이고, 깊이 분해능은 dZ = Z^2 / (f * B) 이므로,
        각이 크면 고도를 정밀하게 얻지만 두 영상이 서로 달라 보여 정합이
        어려워진다. 실제 임무도 15~30도 사이에서 타협한다.
    """
    if altitude <= 0:
        raise ValueError(f"고도는 양수여야 합니다: {altitude}")
    if not 0 < convergence_deg < 90:
        raise ValueError(f"수렴각은 0~90도 사이여야 합니다: {convergence_deg}")

    target = np.asarray(target, dtype=np.float64)
    half = np.radians(convergence_deg) / 2.0
    dx = altitude * np.tan(half)
    left = look_at(target + np.array([-dx, 0.0, altitude]), target)
    right = look_at(target + np.array([+dx, 0.0, altitude]), target)
    return left, right, 2.0 * dx


def _scatter(points_cam, intensity, camera, splat):
    """한 번의 z-buffer 통과. 가장 가까운 점의 깊이와 그 점의 밝기를 남긴다."""
    z = points_cam[:, 2]
    ok = z > 1e-9
    if not np.any(ok):
        raise ValueError("카메라 앞에 있는 점이 없습니다. 자세를 확인하세요.")
    u = camera.fx * points_cam[ok, 0] / z[ok] + camera.cx
    v = camera.fy * points_cam[ok, 1] / z[ok] + camera.cy
    zz, ii = z[ok], intensity[ok]
    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)

    depth = np.full(camera.shape, np.inf)
    img = np.zeros(camera.shape)
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            yy, xx = vi + dy, ui + dx
            good = ((yy >= 0) & (yy < camera.height)
                    & (xx >= 0) & (xx < camera.width))
            np.minimum.at(depth, (yy[good], xx[good]), zz[good])
    for dy in range(-splat, splat + 1):
        for dx in range(-splat, splat + 1):
            yy, xx = vi + dy, ui + dx
            good = ((yy >= 0) & (yy < camera.height)
                    & (xx >= 0) & (xx < camera.width))
            yy, xx, zk, ik = yy[good], xx[good], zz[good], ii[good]
            win = np.abs(zk - depth[yy, xx]) < 1e-6
            img[yy[win], xx[win]] = ik[win]
    depth[~np.isfinite(depth)] = np.nan
    return img, depth


def render(points: np.ndarray, intensity: np.ndarray, camera, pose: Pose,
           splat: int = 1):
    """지형 점들을 카메라에 투영해 영상과 깊이 맵을 함께 만든다.

    splat 을 **빈 화소를 메우는 데만** 쓰는 이유
        점을 주변 화소까지 번지게 하면서 깊이도 함께 쓰면, 깊이 버퍼에 3x3
        최소값 필터를 건 것과 같아진다. 표면이 카메라 쪽으로 당겨지고, 그
        당김은 표면 기울기 방향을 따른다. 수렴 촬영의 두 카메라는 서로 반대로
        기울어 있으므로 당김 방향도 반대가 되어, **시차가 한쪽으로 치우친다.**

        실측하면 티코 지형에서 splat=1 이 시차를 +2.6 px 밀어 깊이가 1.2 km
        치우쳤다. splat=0 으로 두면 +0.6 px 로 줄지만 이번에는 구멍이 남는다.

        그래서 두 번 훑는다. 먼저 번지지 않고(splat=0) 정직한 깊이를 얻고,
        그때 비어 있는 화소만 번진 통과로 메운다. 채워 넣은 자리는 원래
        정보가 없던 곳이라 치우침을 만들지 않는다.

    Returns
    -------
    (image, depth) : uint8 영상과 float64 깊이 맵. 빈 화소는 깊이가 NaN.
    """
    points = np.asarray(points, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64).ravel()
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"(N, 3) 배열이 필요합니다: {points.shape}")
    if len(intensity) != len(points):
        raise ValueError(f"점 {len(points)}개와 밝기 {len(intensity)}개가 다릅니다.")
    if splat < 0:
        raise ValueError(f"splat 은 0 이상이어야 합니다: {splat}")

    cam = pose.apply(points)
    img, depth = _scatter(cam, intensity, camera, 0)
    if splat > 0:
        img_f, depth_f = _scatter(cam, intensity, camera, splat)
        hole = ~np.isfinite(depth)
        depth = np.where(hole, depth_f, depth)
        img = np.where(hole, img_f, img)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8), depth


def render_heightfield(elev, gsd, intensity, camera, pose: Pose, iters: int = 12):
    """화소마다 광선을 쏘아 지형과 만나는 점을 찾는다 (역방향 렌더링).

    왜 흩뿌리기(scatter)를 버렸는가
        점을 화소에 던지는 방식은 위치를 반올림한다. 그 반올림 오차가 표면
        기울기 방향으로 쏠리는데, 수렴 촬영의 두 카메라는 반대로 기울어
        있으므로 좌우가 서로 다른 쪽으로 쏠린다. 결과가 시차 치우침이다.
        티코에서 splat 을 끄고도 +0.6 px 가 남았다.

        광선을 쏘면 화소 중심이 정확히 정의되고 지형은 이중선형으로 뽑으므로
        반올림이 없다. 구멍도 생기지 않는다.

    푸는 방법
        고도장 z(X, Y) 와 광선 P = C + t·d 의 교점은 닫힌 형태가 없다.
        z_s 를 현재 추정 고도로 두고 t = (z_s - C_z)/d_z 로 t 를 갱신하는
        고정점 반복으로 푼다. 기복이 촬영 거리보다 훨씬 작으면 몇 번에
        수렴한다.

    수렴하지 않는 광선
        지형 격자 밖으로 나간 광선은 고도 표본이 가장자리 값으로 고정되어
        값이 진동한다. 실측하면 그런 광선이 약 10% 있었고, 수렴하지 않은 채
        깊이를 내놓아 최악 300 m 까지 틀렸다. 마지막 갱신 폭이 화소 크기의
        1/100 을 넘으면 값을 내지 않는다 - "모른다" 를 틀린 값으로 채우는
        것보다 비워 두는 편이 낫다.

    한계
        가림(occlusion)을 다루지 않는다. 광선이 앞쪽 봉우리를 지나쳐 뒤쪽
        지면에 닿아도 그대로 받는다. 시선이 지면에 거의 수직이고 경사가
        완만하면 문제가 없지만, 수렴각을 크게 키우면 확인이 필요하다.
    """
    from scipy.ndimage import map_coordinates

    elev = np.asarray(elev, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    if elev.shape != intensity.shape:
        raise ValueError(f"고도 {elev.shape} 와 밝기 {intensity.shape} 가 다릅니다.")
    if gsd <= 0:
        raise ValueError(f"화소 크기는 양수여야 합니다: {gsd}")
    if iters < 1:
        raise ValueError(f"반복 횟수는 1 이상이어야 합니다: {iters}")
    h, w = elev.shape

    centre = -pose.R.T @ pose.t                     # 지형 좌표계의 카메라 위치
    # 지형이 카메라보다 높으면 고정점 반복의 전제(광선이 아래로 내려가며 한 번
    # 만난다)가 깨진다. 값이 거의 안 나오는 결과가 조용히 돌아오는 대신
    # 여기서 멈춘다. 실제로 기복 4 km 지형을 고도 2.5 km 에서 찍는 설정이
    # 유효화소 3% 를 내놓고도 아무 말이 없었다.
    if np.nanmax(elev) >= centre[2]:
        raise ValueError(
            f"지형 최고점 {np.nanmax(elev):.1f} m 가 카메라 고도 "
            f"{centre[2]:.1f} m 보다 높습니다.")
    dirs = camera.pixel_rays().reshape(-1, 3) @ pose.R   # 화소별 시선 (지형 좌표계)
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    usable = dirs[:, 2] < -1e-6                     # 아래를 향하는 광선만
    d = dirs[usable]
    z_guess = np.full(usable.sum(), float(np.nanmean(elev)))
    step = np.full(usable.sum(), np.inf)
    for _ in range(iters):
        tt = (z_guess - centre[2]) / d[:, 2]
        P = centre + tt[:, None] * d
        col = P[:, 0] / gsd + (w - 1) / 2.0
        row = (h - 1) / 2.0 - P[:, 1] / gsd
        z_new = map_coordinates(elev, [row, col], order=1, mode="nearest")
        step = np.abs(z_new - z_guess)
        z_guess = z_new
    # 반복이 끝난 뒤 한 번 더 갱신한다. 그러지 않으면 t 가 마지막 고도 표본보다
    # 한 걸음 뒤처져, 수렴했는데도 잔차가 남는다.
    tt = (z_guess - centre[2]) / d[:, 2]

    t = np.full(len(dirs), np.nan)
    settled = step < gsd / 100.0
    t[np.flatnonzero(usable)[settled]] = tt[settled]

    P = centre + t[:, None] * dirs
    col = P[:, 0] / gsd + (w - 1) / 2.0
    row = (h - 1) / 2.0 - P[:, 1] / gsd
    ok = np.isfinite(t) & (col >= 0) & (col <= w - 1) & (row >= 0) & (row <= h - 1)

    img = np.zeros(len(dirs))
    img[ok] = map_coordinates(intensity, [row[ok], col[ok]], order=1,
                              mode="nearest")
    depth = np.full(len(dirs), np.nan)
    depth[ok] = pose.apply(P[ok])[:, 2]
    depth[ok] = np.where(depth[ok] > 0, depth[ok], np.nan)

    return ((np.clip(img, 0, 1) * 255).astype(np.uint8).reshape(camera.shape),
            depth.reshape(camera.shape))


def load_image(path: str, crop: int = None):
    """실제로 찍은 영상을 읽는다 (고도 모델과 같은 격자여야 한다)."""
    import rasterio

    with rasterio.open(path) as src:
        if crop:
            w = min(crop, src.width, src.height)
            r0 = (src.height - w) // 2
            c0 = (src.width - w) // 2
            from rasterio.windows import Window
            img = src.read(1, window=Window(c0, r0, w, w))
        else:
            img = src.read(1)
    return np.asarray(img, dtype=np.float64)


def estimate_illumination(image, elev, gsd, step_deg: float = 5.0):
    """찍힌 영상이 어느 방향에서 비춰진 것인지 지형으로부터 되찾는다.

    왜 필요한가
        모자이크 영상에는 촬영 당시의 음영이 이미 들어 있다. 그것을 그대로
        알베도로 쓰면 **음영을 두 번 입히게 된다** — 우리가 다시 조명을 걸기
        때문이다. 빼내려면 원래 조명 방향을 알아야 하는데, 모자이크에는
        그 값이 붙어 있지 않다.

        지형의 법선은 알고 있으므로, 여러 태양 방향을 넣어 보고 영상과 가장
        잘 맞는 방향을 고르면 된다. 음영이 지형에서 온 것이라면 반드시 어떤
        방향에서 상관이 뚜렷하게 높아진다.

    Returns
    -------
    (방위각, 고도, 상관계수) — 상관이 낮으면 음영보다 무늬가 지배적이라는 뜻이다.
    """
    n = surface_normals(elev, gsd)
    img = np.asarray(image, dtype=np.float64)
    if img.shape != elev.shape:
        raise ValueError(f"영상과 고도 모델의 격자가 다릅니다: "
                         f"{img.shape} vs {elev.shape}")
    y = img.ravel() - img.mean()
    denom_y = np.sqrt((y ** 2).sum())
    if denom_y <= 0:
        raise ValueError("영상이 완전히 균일합니다.")

    best = (0.0, 45.0, -1.0)
    for az in np.arange(0.0, 360.0, step_deg):
        for el in np.arange(10.0, 85.0, step_deg):
            a, e = np.radians(az), np.radians(el)
            sun = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e),
                            np.sin(e)])
            lit = np.clip(n @ sun, 0.0, None).ravel()
            x = lit - lit.mean()
            den = np.sqrt((x ** 2).sum()) * denom_y
            if den <= 0:
                continue
            r = float((x * y).sum() / den)
            if r > best[2]:
                best = (float(az), float(el), r)
    return best


def derive_albedo(image, elev, gsd, mean_albedo: float = 0.12,
                  floor: float = 0.25):
    """찍힌 영상에서 음영을 빼내고 표면 무늬만 남긴다.

    영상 = 알베도 x 음영 이므로, 추정한 조명으로 만든 음영으로 나누면 알베도가
    남는다. 스치는 각에서는 음영이 0 에 가까워 나눗셈이 폭발하므로 바닥을 둔다.

    남는 것은 티코의 광조(ray) 무늬, 어둡고 밝은 물질, 고도 모델에는 없는
    잔크레이터다. **고도 모델이 매끈해서 잔무늬가 적다** 는 한계를 이것이 메운다.

    Returns
    -------
    (albedo, info) : 평균이 mean_albedo 인 알베도 지도와 추정한 조명 정보
    """
    if not 0.0 < mean_albedo < 1.0:
        raise ValueError(f"평균 알베도는 0~1 이어야 합니다: {mean_albedo}")
    if not 0.0 < floor <= 1.0:
        raise ValueError(f"바닥은 0~1 이어야 합니다: {floor}")

    az, el, corr = estimate_illumination(image, elev, gsd)
    a, e = np.radians(az), np.radians(el)
    sun = np.array([np.sin(a) * np.cos(e), np.cos(a) * np.cos(e), np.sin(e)])
    lit = np.clip(surface_normals(elev, gsd) @ sun, floor, None)

    img = np.asarray(image, dtype=np.float64)
    albedo = img / lit
    albedo = np.clip(albedo, *np.percentile(albedo, [1, 99]))
    albedo = albedo / albedo.mean() * mean_albedo
    return albedo, {"azimuth_deg": az, "elevation_deg": el,
                    "correlation": corr}
