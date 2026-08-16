"""SPE3R 데이터셋에서 aqua 모델 한 종만 내려받는다.

전체 데이터셋은 64종 0.93 GB 이지만 이 과제는 한 종만 쓰므로
필요한 파일(약 18 MB)만 선택적으로 받는다.

출처 : Park, T. H. and D'Amico, S., "SPE3R: Synthetic Dataset for Satellite
        Pose Estimation and 3D Reconstruction", Stanford Digital Repository, 2024.
        https://purl.stanford.edu/pk719hm4806
라이선스: CC BY-NC-SA 4.0 (비상업적 이용)

사용법
    py -3 tools/get_spe3r_aqua.py
"""

from __future__ import annotations

import os
import sys
import urllib.parse
import urllib.request
import zipfile

BASE = "https://stacks.stanford.edu/file/druid:pk719hm4806/"
MODEL = "aqua"
DEST = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "spe3r")

FILES = [
    "camera.json",
    "splits.csv",
    "README.md",
    "LICENSE.md",
    f"{MODEL}/labels.json",
    f"{MODEL}/images.zip",
    f"{MODEL}/masks.zip",
    f"{MODEL}/models/model_normalized.obj",
]


def download(name: str) -> str:
    dest = os.path.join(DEST, *name.split("/"))
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    if os.path.exists(dest) and os.path.getsize(dest) > 0:
        print(f"  skip  {name} (이미 있음)")
        return dest

    url = BASE + urllib.parse.quote(name)
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=300) as r:
        blob = r.read()
    with open(dest, "wb") as f:
        f.write(blob)
    print(f"  ok    {name} ({len(blob) / 1e6:.2f} MB)")
    return dest


def main() -> int:
    print(f"SPE3R '{MODEL}' 내려받기 -> {DEST}")
    for name in FILES:
        try:
            path = download(name)
        except Exception as exc:
            print(f"  FAIL  {name}: {exc}")
            return 1
        if path.endswith(".zip"):
            # 압축 파일 안에는 이미지가 폴더 없이 평평하게 들어 있다.
            # images.zip -> images/, masks.zip -> masks/ 로 나눠서 푼다.
            out_dir = os.path.join(os.path.dirname(path),
                                   os.path.basename(path)[:-4])
            if not os.path.isdir(out_dir) or not os.listdir(out_dir):
                os.makedirs(out_dir, exist_ok=True)
                with zipfile.ZipFile(path) as z:
                    z.extractall(out_dir)
                print(f"        압축 해제 -> {out_dir} ({len(os.listdir(out_dir))} 개)")
    print("완료")
    return 0


if __name__ == "__main__":
    sys.exit(main())
