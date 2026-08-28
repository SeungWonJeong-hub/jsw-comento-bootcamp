"""실제 스테레오 쌍과 같은 격자에 LOLA 정답 고도를 얹습니다.

왜 따로 받는가
    Kaguya 영상은 IAU_2015:30115 로, LOLA 전역 모델은 단순원통도법으로
    저장돼 있습니다. 좌표계가 다르면 화소끼리 견줄 수 없습니다. LOLA 쪽을 영상과
    **같은 격자로 다시 샘플링**해 두어야 오차를 화소 단위로 잴 수 있습니다.

    LOLA 는 118 m/화소이고 영상은 5 m/화소입니다. 늘려서 얹는 것이므로 정답의
    실제 해상도가 올라가지는 않습니다. 큰 지형의 높낮이를 맞추는지 보는 용도입니다.

사용법
    py -3 tools/get_kaguya_truth.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "moon")
SOURCE = ("https://planetarymaps.usgs.gov/mosaic/"
          "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif")
REFERENCE = os.path.join(DEST, "tycho_kaguya_tc1.tif")
OUT = os.path.join(DEST, "tycho_kaguya_lola.tif")


def main() -> int:
    import rasterio
    from rasterio.warp import Resampling, reproject

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    if not os.path.exists(REFERENCE):
        print("먼저 tools/get_kaguya_pair.py 를 실행하세요.", file=sys.stderr)
        return 1

    with rasterio.open(REFERENCE) as ref:
        transform, crs = ref.transform, ref.crs
        height, width = ref.height, ref.width

    dst = np.full((height, width), np.nan, dtype=np.float32)
    with rasterio.open("/vsicurl/" + SOURCE) as src:
        reproject(source=rasterio.band(src, 1), destination=dst,
                  src_transform=src.transform, src_crs=src.crs,
                  dst_transform=transform, dst_crs=crs,
                  src_nodata=src.nodata, dst_nodata=float("nan"),
                  resampling=Resampling.bilinear)
        scale = src.scales[0]

    elev = dst * scale
    ok = np.isfinite(elev)
    if not ok.any():
        print("유효한 고도가 없습니다.", file=sys.stderr)
        return 1

    with rasterio.open(OUT, "w", driver="GTiff", height=height, width=width,
                       count=1, dtype="float32", crs=crs, transform=transform,
                       nodata=float("nan"), tiled=True, blockxsize=256,
                       blockysize=256) as f:
        f.write(elev.astype("float32"), 1)

    print(f"저장 -> {os.path.relpath(OUT, ROOT)}")
    print(f"  고도 {np.nanmin(elev):.0f} ~ {np.nanmax(elev):.0f} m "
          f"(기복 {np.nanmax(elev)-np.nanmin(elev):.0f} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
