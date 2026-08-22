"""실제로 찍은 스테레오 쌍을 이 파이프라인에 바로 넣을 수 있는지 확인한다.

왜 이 도구가 있는가
    "렌더링 말고 실제 사진을 쓰라" 는 것이 이 실험의 가장 큰 개선점이다.
    실제로 해 보니 걸림돌이 어디에 있는지가 분명해졌고, 그것을 말이 아니라
    **측정으로** 남겨 두려고 만들었다.

무엇을 확인하는가
    Kaguya(SELENE) 지형 카메라는 앞뒤로 기울어진 두 대(TC1, TC2)를 함께 실은
    스테레오 전용 카메라다. 수렴각이 약 30~34도로 이 실험의 조건과 비슷하고,
    자료는 공개되어 있다. 두 가지 형태로 받을 수 있는데 둘 다 문제가 있다.

    (가) USGS 가 지도 투영해 둔 COG — 받기 쉽고 두 장이 같은 격자에 놓인다.
         그런데 **전역 DEM 을 지형 모델로 써서 정사보정**되어 있다. 즉 고도에
         따른 어긋남이 이미 제거되어 있어 스테레오로 쓸 수 없다.
         이 스크립트가 그것을 숫자로 확인한다.

    (나) DARTS 의 원본(level 2B0) — 투영 전이라 어긋남이 살아 있다. 대신
         푸시브룸 카메라 모델(SPICE/ISIS 또는 CSM)이 있어야 어긋남을 고도로
         바꿀 수 있다. 카메라 모델 없이 특징점만으로 맞추려 해 보았으나
         실패했다. 그 기록은 아래 REPORT 에 적어 두었다.

사용법
    py -3 tools/get_kaguya_pair.py
    py -3 tools/get_kaguya_truth.py
    py -3 tools/check_real_pair.py
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data", "moon")
OUT = os.path.join(ROOT, "outputs")

GSD = 5.0
OFF_NADIR = (17.23, 17.0)     # TC1 / TC2 의 방출각 [도]. STAC 의 view:off_nadir.

#: 정사보정에 쓰인 지형 모델. 관측의 provenance.txt 에 그대로 적혀 있다.
PROVENANCE = ("spiceinit ... shape=user model=$ISISDATA/base/dems/"
              "Lunar_LRO_LOLAKaguya_DEMmerge_Global_512ppd_radius.cub")

#: 원본(level 2B0)을 카메라 모델 없이 맞춰 보려 한 기록.
RAW_ATTEMPTS = [
    ("SIFT 특징점 정합", "특징점 8000개 중 상호 최근접을 통과한 것 15개"),
    ("타일 위상 상관", "타일 88개의 응답 중앙값 0.057, 아핀 내점 4개"),
]


def load(name):
    import rasterio

    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        raise SystemExit(f"{path} 가 없습니다. tools/get_kaguya_*.py 를 먼저 실행하세요.")
    with rasterio.open(path) as src:
        return src.read(1)


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    lines = []

    def log(msg=""):
        print(msg, flush=True)
        lines.append(msg)

    tc1 = load("tycho_kaguya_tc1.tif")
    tc2 = load("tycho_kaguya_tc2.tif")
    truth = load("tycho_kaguya_lola.tif").astype(np.float64)
    h, w = tc1.shape

    log("=" * 70)
    log("실제로 찍은 스테레오 쌍을 쓸 수 있는가 — Kaguya 지형 카메라")
    log("=" * 70)
    log(f"\n  사진      Kaguya TC1 / TC2 · {w}x{h} · {GSD:.0f} m/화소 "
        f"({w*GSD/1000:.1f} km 사방)")
    log(f"  정답      LRO LOLA 를 같은 격자로 다시 샘플링 "
        f"(기복 {np.nanmax(truth)-np.nanmin(truth):.0f} m)")
    log(f"  방출각    TC1 {OFF_NADIR[0]:.1f}도 · TC2 {OFF_NADIR[1]:.1f}도 "
        f"(수렴각 약 {sum(OFF_NADIR):.0f}도)")

    factor = sum(np.tan(np.radians(a)) for a in OFF_NADIR)
    predicted = GSD / factor
    log(f"\n  구면에 투영했다면 고도 h 가 어긋남 d 로 남아야 한다")
    log(f"    d = h x (tan e1 + tan e2) / 화소  →  1 px = {predicted:.1f} m")

    log("\n[1] 실제로 얼마나 어긋나 있는가")
    flow = cv2.calcOpticalFlowFarneback(tc1, tc2, None, 0.5, 5, 41, 5, 7, 1.5, 0)
    inner = (slice(200, -200), slice(200, -200))
    fx, fy = flow[..., 0][inner], flow[..., 1][inner]
    log(f"  가로 흐름  평균 {fx.mean():+6.2f}  표준편차 {fx.std():5.2f} px")
    log(f"  세로 흐름  평균 {fy.mean():+6.2f}  표준편차 {fy.std():5.2f} px")
    log("  극궤도라 시차는 세로(궤도 방향)에 있어야 한다. 가로는 등록 오차다.")
    log(f"  세로 흔들림이 고도라면 기복은 {fy.std()*4*predicted:.0f} m 쯤이어야 하는데,")
    log(f"  정답은 {np.nanmax(truth)-np.nanmin(truth):.0f} m 다.")

    log("\n[2] 어긋남이 정답 고도를 설명하는가")
    t = truth[inner]
    ok = np.isfinite(t)
    corr = float(np.corrcoef(fy[ok].ravel(), t[ok].ravel())[0, 1])
    A = np.column_stack([fy[ok].ravel(), np.ones(int(ok.sum()))])
    slope, _ = np.linalg.lstsq(A, t[ok].ravel(), rcond=None)[0]
    log(f"  상관계수        {corr:+.3f}   (1 에 가까워야 시차다)")
    log(f"  맞춘 기울기     {abs(slope):.2f} m/px")
    log(f"  기하가 예측     {predicted:.2f} m/px")
    log(f"  어긋난 정도     {abs(abs(slope)/predicted - 1)*100:.0f}%")

    log("\n[3] 왜 이런가")
    log("  이 영상들의 처리 기록에 답이 있다.")
    log(f"    {PROVENANCE}")
    log("  전역 DEM 을 지형 모델로 놓고 정사보정한 것이다. 고도에 따른 어긋남이")
    log("  이미 제거되어 있으므로, 남은 것은 그 DEM(512 ppd, 약 59 m/화소)보다")
    log("  잘아서 지워지지 않은 잔무늬뿐이다. 스테레오로 쓸 수 없다.")

    log("\n[4] 투영 전 원본(level 2B0)은 어떤가")
    log("  DARTS 에서 받을 수 있고 크기도 25 MB 로 작다. 어긋남은 살아 있다.")
    log("  다만 푸시브룸 카메라 모델이 있어야 어긋남을 고도로 바꿀 수 있다.")
    log("  카메라 모델 없이 특징점만으로 맞춰 보았으나 실패했다.")
    for name, result in RAW_ATTEMPTS:
        log(f"    {name:16s} {result}")
    log("  티코 안쪽 벽은 태양 고도가 8도라 그림자가 지배하고, 두 카메라가")
    log("  반대쪽에서 보므로 한쪽에 보이는 사면이 다른 쪽에서는 그늘이다.")
    log("  이 지형에서 카메라 모델 없이 맞추는 것은 무리다.")

    log("\n결론")
    log("  실제 스테레오 쌍은 공개되어 있고 받기도 쉽다. 걸림돌은 자료가 아니라")
    log("  **푸시브룸 카메라 모델을 다루는 도구 체계**(ISIS3 / CSM)다. 지도 투영된")
    log("  판본은 이미 정사보정되어 시차가 없고, 원본은 카메라 모델이 필요하다.")
    log("  개선점 1번이 남아 있는 이유가 이것이다.")

    summary = {
        "instrument": "Kaguya (SELENE) Terrain Camera TC1/TC2",
        "gsd_m": GSD, "off_nadir_deg": list(OFF_NADIR),
        "predicted_m_per_px": float(predicted),
        "flow_x_std_px": float(fx.std()), "flow_y_std_px": float(fy.std()),
        "correlation_with_truth": corr,
        "fitted_m_per_px": float(abs(slope)),
        "provenance": PROVENANCE,
        "verdict": "map-projected products are ortho-corrected; unusable for stereo",
    }
    with open(os.path.join(OUT, "real_pair_check.json"), "w",
              encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(OUT, "real_pair_check.txt"), "w",
              encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    log(f"\n저장 -> outputs/real_pair_check.txt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
