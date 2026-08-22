"""실제로 찍힌 달 사진 두 장을 내려받는다 — 아폴로 15호 매핑 카메라.

무엇을 받는가
    1971년 7월 31일, 아폴로 15호 사령선이 24초 간격으로 필름에 찍은 두 장이다.
    매핑 카메라는 궤도를 돌며 78% 씩 겹치게 연속 촬영했으므로, 이웃한 두 장이
    그대로 스테레오 쌍이 된다.

    ASU 아폴로 영상 아카이브가 원본 필름 스캔을 공개하고 있다. 원본은 한 장에
    500 MB 가 넘으므로 4048 화소짜리 중간 판본을 받는다. 그것만으로도 지상
    화소가 37 m 라 이 실험에 충분하다.

왜 아폴로인가
    달 궤도 스테레오는 거의 전부 푸시브룸이라 카메라 자세 커널이 필요하다.
    매핑 카메라는 프레임 카메라여서 두 장 사이가 행렬 하나로 기술되고,
    이 파이프라인의 정렬·삼각측량이 그대로 성립한다. 자세한 사정은
    tools/check_real_pair.py 에 적어 두었다.

사용법
    py -3 tools/get_apollo_pair.py
"""

from __future__ import annotations

import os
import sys
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEST = os.path.join(ROOT, "data", "moon", "apollo")

BASE = "https://data.lroc.im-ldi.com/data/metric/AS15/png"
FRAMES = ("AS15-M-1000", "AS15-M-1001")

#: 정답 고도를 함께 받아 두면 복원한 기복이 자릿수까지 맞는지 볼 수 있다.
#: 프레임 중심은 (25.89, -6.10) 과 (25.89, -7.43) 이다.
TRUTH = dict(lat=26.4, lon=-6.1, size=1700, name="apollo_as15_lola.tif")


def download(name: str) -> str:
    out = os.path.join(DEST, f"{name}.png")
    if os.path.exists(out) and os.path.getsize(out) > 1_000_000:
        print(f"  {name}  이미 있음")
        return out
    url = f"{BASE}/{name}_MED.png"
    print(f"  {name}  받는 중 ...", flush=True)
    with urllib.request.urlopen(url, timeout=600) as r, open(out, "wb") as f:
        f.write(r.read())
    print(f"  {name}  {os.path.getsize(out)/1e6:.1f} MB")
    return out


def main() -> int:
    os.makedirs(DEST, exist_ok=True)
    print("아폴로 15호 매핑 카메라 스테레오 쌍")
    for name in FRAMES:
        download(name)

    print("\n같은 지역의 LOLA 고도 (기복을 견줄 기준)")
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    truth_path = os.path.join(ROOT, "data", "moon", TRUTH["name"])
    if os.path.exists(truth_path):
        print(f"  {TRUTH['name']}  이미 있음")
    else:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "get_moon_dtm", os.path.join(os.path.dirname(__file__),
                                         "get_moon_dtm.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        mod.fetch(TRUTH["lat"], TRUTH["lon"], TRUTH["size"], TRUTH["name"])
        print(f"  {TRUTH['name']}  저장")

    print("\n다음: py -3 run_apollo_stereo.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
