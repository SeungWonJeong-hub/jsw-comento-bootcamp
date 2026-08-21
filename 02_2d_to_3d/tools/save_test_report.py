"""Unit Test 를 실행하고 결과를 outputs/pytest_report.txt 로 남긴다.

왜 따로 두는가
    과제 요청은 "Unit Test 코드 및 실행 결과 문서화" 다. 실행 결과가 화면에만
    남으면 문서가 아니므로 파일로 남기고, 발표자료도 이 파일을 읽어 쓴다.

    pytest 를 그냥 리다이렉트하면 마지막 줄에 걸린 시간이 들어가 실행할 때마다
    파일이 달라진다. outputs/ 의 다른 산출물은 전부 바이트 단위로 재현되는데
    이 파일만 매번 바뀌면 "재현된다" 는 말이 반만 맞게 된다. 걸린 시간은
    측정값이 아니라 실행 환경이므로 지운 뒤 저장한다.

사용법
    py -3 tools/save_test_report.py
"""

from __future__ import annotations

import io
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "outputs")


def main() -> int:
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-v", "--no-header", "-p",
         "no:cacheprovider"],
        cwd=ROOT, env=env, capture_output=True, text=True, encoding="utf-8",
        errors="replace")

    text = proc.stdout
    # 걸린 시간과 루트 경로를 지운다. 둘 다 실행 환경이지 측정값이 아니다.
    text = re.sub(r" in \d+\.\d+s", "", text)
    text = text.replace(ROOT, ".").replace(ROOT.replace("\\", "/"), ".")
    text = "\n".join(line.rstrip() for line in text.splitlines()) + "\n"

    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, "pytest_report.txt")
    io.open(path, "w", encoding="utf-8", newline="\n").write(text)

    hit = re.search(r"(\d+) passed", text)
    print(f"저장 완료 -> {path}  ({hit.group(1) if hit else '?'}개 통과)")
    return proc.returncode


if __name__ == "__main__":
    sys.exit(main())
