"""실제로 찍은 달 스테레오 쌍 한 구역을 내려받는다 (Kaguya TC).

무엇을 받는가
    Kaguya(SELENE) 지형 카메라는 앞뒤로 기울어진 두 대(TC1, TC2)를 함께 실은
    **스테레오 전용 카메라**다. 같은 궤도에서 같은 지역을 두 각도로 찍으므로,
    이 실험이 가정한 수렴 촬영과 기하가 같다. 수렴각은 약 30도다.

    USGS 가 그 관측을 지도 투영해 Cloud Optimized GeoTIFF 로 공개해 두었다.
    두 장이 **같은 좌표계의 같은 격자**에 놓이므로 카메라 자세 커널 없이도
    시차를 잴 수 있다 — 지형의 높낮이가 두 장 사이의 어긋남으로 남기 때문이다.

    구면에 투영한 것이라 고도에 비례하는 어긋남이 그대로 남는다.

        어긋남 = 고도 x (tan(e1) + tan(e2))

    e 는 각 카메라의 방출각이다. 공칭 ±15도이면 계수가 0.536 이므로, 5 m 화소
    한 칸이 고도 약 9 m 에 해당한다.

사용법
    py -3 tools/get_kaguya_pair.py [--size 2048] [--out data/moon]
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

BUCKET = ("https://astrogeo-ard.s3-us-west-2.amazonaws.com/moon/kaguya/"
          "terrain_camera/stereoscopic/uncontrolled/")

#: 티코 부근을 지나는 같은 궤도(07450)의 TC1 / TC2 관측.
#: STAC 목록에서 겹치는 쌍을 골랐다.
#: https://stac.astrogeology.usgs.gov/api/collections/
#:     kaguya_terrain_camera_stereoscopic_uncontrolled_observations
PAIR = ("TC1W2B0_01_07450S430E3481", "TC2W2B0_01_07450S435E3480")

GSD = 5.0          # 공통 격자의 화소 크기 [m]. 원본이 4.79~4.96 이라 그 사이로 잡는다.


def url(name: str) -> str:
    return f"/vsicurl/{BUCKET}{name}/{name}.tif"


def overlap_bounds(paths):
    """두 영상이 함께 덮는 지도 좌표 범위."""
    import rasterio

    boxes = []
    for p in paths:
        with rasterio.open(p) as src:
            if src.crs is None:
                raise RuntimeError(f"좌표계가 없습니다: {p}")
            boxes.append(src.bounds)
    left = max(b.left for b in boxes)
    right = min(b.right for b in boxes)
    bottom = max(b.bottom for b in boxes)
    top = min(b.top for b in boxes)
    if right <= left or top <= bottom:
        raise RuntimeError("두 관측이 겹치지 않습니다.")
    return left, bottom, right, top


def fetch(name, transform, width, height, crs, out_path):
    """한 관측을 공통 격자로 다시 샘플링해 받는다."""
    import rasterio
    from rasterio.warp import Resampling, reproject

    dst = np.zeros((height, width), dtype=np.float32)
    with rasterio.open(url(name)) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  resampling=Resampling.bilinear)

    # 원본은 int16 이지만 실제 값은 12비트 근처다. 0~255 로 펴서 저장한다.
    valid = dst > 0
    if not valid.any():
        raise RuntimeError(f"내려받은 구역이 비어 있습니다: {name}")
    lo, hi = np.percentile(dst[valid], [0.5, 99.5])
    scaled = np.clip((dst - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * 255.0
    scaled = np.where(valid, scaled, 0.0).astype(np.uint8)

    with rasterio.open(out_path, "w", driver="GTiff", height=height,
                       width=width, count=1, dtype="uint8", crs=crs,
                       transform=transform, compress="deflate") as dstf:
        dstf.write(scaled, 1)
    return float(valid.mean())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--size", type=int, default=2048,
                    help="한 변의 화소 수 (기본 2048 = 약 10 km)")
    ap.add_argument("--out", default=os.path.join("data", "moon"))
    args = ap.parse_args()

    try:
        import rasterio
        from affine import Affine
    except ImportError:
        print("rasterio 가 필요합니다: pip install rasterio", file=sys.stderr)
        return 1

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    os.makedirs(args.out, exist_ok=True)

    paths = [url(n) for n in PAIR]
    left, bottom, right, top = overlap_bounds(paths)
    print(f"겹치는 범위  {(right-left)/1000:.1f} x {(top-bottom)/1000:.1f} km")

    # 겹치는 범위의 한가운데에서 정사각형을 오려 낸다.
    cx, cy = (left + right) / 2.0, (bottom + top) / 2.0
    half = args.size * GSD / 2.0
    if half * 2 > min(right - left, top - bottom):
        half = min(right - left, top - bottom) / 2.0
        size = int(2 * half / GSD)
        print(f"  겹치는 범위가 좁아 {size} 화소로 줄인다")
    else:
        size = args.size
    transform = Affine(GSD, 0.0, cx - half, 0.0, -GSD, cy + half)

    with rasterio.open(paths[0]) as src:
        crs = src.crs

    for name, tag in zip(PAIR, ("tc1", "tc2")):
        out = os.path.join(args.out, f"tycho_kaguya_{tag}.tif")
        share = fetch(name, transform, size, size, crs, out)
        print(f"  {tag}  {out}  ({size}x{size} · 값이 있는 화소 {share*100:.1f}%)")

    print(f"\n중심  x {cx:.0f}  y {cy:.0f}  [m, IAU_2015:30115]")
    print(f"화소  {GSD:.1f} m")
    return 0


if __name__ == "__main__":
    sys.exit(main())
