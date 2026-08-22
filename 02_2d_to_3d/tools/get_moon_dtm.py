"""달 고도 모델에서 관심 구역 한 장만 잘라 받는다.

왜 잘라 받는가
    LRO LOLA 전역 고도 모델은 8.5 GB 다. 실험에 쓰는 것은 크레이터 하나
    주변뿐이므로, 원격 파일에서 필요한 창만 읽어 온다. GeoTIFF 는 부분 읽기를
    지원해서 전체를 받지 않아도 된다.

기본값이 티코 크레이터인 이유
    지름 85 km, 바닥에서 테두리까지 5 km 가 넘는다. 기복이 크면 시차 폭이
    넓어 스테레오가 실제로 무엇을 하는지 눈에 보인다. 평평한 바다(mare)를
    고르면 시차가 몇 픽셀뿐이라 결과가 잡음에 묻힌다.

사용법
    py -3 tools/get_moon_dtm.py
    py -3 tools/get_moon_dtm.py --lat -43.31 --lon -11.36 --size 640
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "moon")

# 달 반지름 [m]. 이 고도 모델의 좌표는 이 구면을 편 단순원통도법이다.
MOON_RADIUS = 1737400.0
SOURCE = ("https://planetarymaps.usgs.gov/mosaic/"
          "Lunar_LRO_LOLA_Global_LDEM_118m_Mar2014.tif")


def fetch(lat: float, lon: float, size: int, name: str) -> str:
    import rasterio
    from rasterio.windows import Window

    os.environ.setdefault("GDAL_DISABLE_READDIR_ON_OPEN", "EMPTY_DIR")
    out = os.path.join(DEST, name)
    os.makedirs(DEST, exist_ok=True)

    with rasterio.open("/vsicurl/" + SOURCE) as src:
        row, col = src.index(MOON_RADIUS * np.radians(lon),
                             MOON_RADIUS * np.radians(lat))
        window = Window(col - size // 2, row - size // 2, size, size)
        raw = src.read(1, window=window).astype(np.float64)
        transform = src.window_transform(window)
        crs, scale, nodata = src.crs, src.scales[0], src.nodata

    # 값은 정수로 저장돼 있고 배율이 따로 붙어 있다. 미터로 되돌린다.
    elev = np.where(raw == nodata, np.nan, raw * scale).astype("float32")
    if not np.isfinite(elev).any():
        raise ValueError(f"유효한 고도 값이 없습니다: 위도 {lat}, 경도 {lon}")

    profile = dict(driver="GTiff", width=size, height=size, count=1,
                   dtype="float32", crs=crs, transform=transform,
                   nodata=float("nan"), tiled=True, blockxsize=256,
                   blockysize=256)
    with rasterio.open(out, "w", **profile) as dst:
        dst.write(elev, 1)
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="달 고도 모델 구역 내려받기")
    p.add_argument("--lat", type=float, default=-43.31, help="위도 [도]")
    p.add_argument("--lon", type=float, default=-11.36, help="경도 [도]")
    p.add_argument("--size", type=int, default=640, help="한 변의 화소 수")
    p.add_argument("--name", default="tycho_118m.tif", help="저장할 파일 이름")
    a = p.parse_args()

    if a.size < 64:
        print("size 는 64 이상이어야 합니다.")
        return 1

    print(f"내려받는 중 — 위도 {a.lat}, 경도 {a.lon}, {a.size}x{a.size} 화소")
    try:
        path = fetch(a.lat, a.lon, a.size, a.name)
    except Exception as exc:                       # noqa: BLE001
        print(f"실패: {type(exc).__name__}: {exc}")
        print("네트워크나 원본 주소를 확인하세요:")
        print(f"  {SOURCE}")
        return 1

    import rasterio
    with rasterio.open(path) as src:
        elev = src.read(1)
    print(f"저장 완료 -> {os.path.relpath(path, ROOT)}  "
          f"({os.path.getsize(path)/1e6:.2f} MB)")
    print(f"  고도 {np.nanmin(elev):.0f} ~ {np.nanmax(elev):.0f} m  "
          f"(기복 {np.nanmax(elev)-np.nanmin(elev):.0f} m)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
