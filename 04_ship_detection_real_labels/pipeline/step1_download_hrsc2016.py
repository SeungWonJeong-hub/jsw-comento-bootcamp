# -*- coding: utf-8 -*-
"""HRSC2016 내려받기와 압축 풀기.

    py pipeline/download_hrsc2016.py --dest ../hrsc2016

확인된 사실 (2026-09-04 실측)
----------------------------
* Kaggle `guofeng/hrsc2016`, 전체 8.0 GB, 다운로드 12,322회.
* 실제 자료는 **5분할 RAR** 입니다. `HRSC2016_dataset.zip`(3.7 GB)을 받아도
  안에 같은 RAR 5개가 들어 있습니다. Kaggle 은 그 위에 한 겹 더 zip 을
  씌워 `HRSC2016_dataset.zip.zip` 으로 내려줍니다.
* RAR 추출에 7-Zip·unrar 를 깔 필요가 없습니다. **Windows 기본 bsdtar**
  (`C:\\Windows\\System32\\tar.exe`, libarchive)가 이 RAR 을 읽습니다.
  리눅스/맥은 bsdtar 또는 unar 를 씁니다.
* 파트별 내용물:
      part01, part02  FullDataSet/AllImages (.bmp)
      part03          FullDataSet/Annotations (.xml), ImageSets, LandMask
      part04, part05  Test/, Train/ 사본
  라벨만 필요하면 part03 만 풀어도 됩니다(--annotations-only).
"""
import os
import sys
import glob
import shutil
import zipfile
import argparse
import subprocess

HERE = os.path.dirname(os.path.abspath(__file__))
DATASET = "guofeng/hrsc2016"
INNER = "HRSC2016_dataset.zip"


def tar_bin():
    """RAR 을 읽을 수 있는 bsdtar 를 찾습니다."""
    for c in (r"C:\Windows\System32\tar.exe", "bsdtar", "tar"):
        try:
            out = subprocess.run([c, "--version"], capture_output=True,
                                 text=True, timeout=20).stdout
            if "bsdtar" in out.lower() or "libarchive" in out.lower():
                return c
        except Exception:
            continue
    # GNU tar 는 RAR 을 못 읽습니다. unar 가 있으면 그걸 씁니다.
    if shutil.which("unar"):
        return "unar"
    return None


def kaggle_get(dest, fname):
    p = os.path.join(dest, fname)
    if os.path.exists(p) or os.path.exists(p + ".zip"):
        print("  이미 있음:", fname)
        return
    subprocess.run([sys.executable, "-m", "kaggle", "datasets", "download",
                    "-d", DATASET, "-f", fname, "-p", dest], check=True)


def unwrap(dest):
    """Kaggle 이 씌운 바깥 zip 을 벗깁니다."""
    outer = os.path.join(dest, INNER + ".zip")
    inner = os.path.join(dest, INNER)
    if os.path.exists(inner):
        return inner
    if not os.path.exists(outer):
        raise SystemExit("내려받은 파일이 없습니다: %s" % outer)
    print("바깥 zip 벗기는 중...")
    with zipfile.ZipFile(outer) as z:
        z.extract(INNER, dest)
    return inner


def extract_rars(dest, inner):
    """zip 안의 RAR 5개를 꺼냅니다."""
    got = sorted(glob.glob(os.path.join(dest, "HRSC2016.part*.rar")))
    if len(got) >= 5:
        print("  RAR 이미 있음: %d개" % len(got))
        return
    with zipfile.ZipFile(inner) as z:
        for n in z.namelist():
            if n.startswith("HRSC2016.part") and n.endswith(".rar"):
                z.extract(n, dest)
                print("  꺼냄:", n)


def untar(tb, rar, out, pattern=None):
    os.makedirs(out, exist_ok=True)
    cmd = [tb, "-xf", rar]
    if pattern:
        cmd.append(pattern)
    r = subprocess.run(cmd, cwd=out, capture_output=True, text=True)
    # 볼륨 경계에 걸친 파일은 실패할 수 있습니다. 치명적이지 않습니다.
    if r.returncode != 0 and r.stderr:
        print("    (경고) %s" % r.stderr.strip().splitlines()[-1][:120])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default=os.path.join(HERE, "..", "..", "hrsc2016"))
    ap.add_argument("--annotations-only", action="store_true",
                    help="part03 만 풀어 라벨·ImageSets 만 확보")
    a = ap.parse_args()
    dest = os.path.abspath(a.dest)
    os.makedirs(dest, exist_ok=True)

    tb = tar_bin()
    if tb is None:
        raise SystemExit(
            "RAR 을 풀 도구가 없습니다. Windows 는 기본 tar.exe(bsdtar)가 있어야 하고,\n"
            "리눅스/맥은 'bsdtar' 또는 'unar' 를 설치하세요. (GNU tar 는 RAR 을 못 읽습니다)")
    print("RAR 도구:", tb)

    print("\n[1/3] Kaggle 에서 내려받기")
    kaggle_get(dest, INNER)
    inner = unwrap(dest)

    print("\n[2/3] RAR 꺼내기")
    extract_rars(dest, inner)

    print("\n[3/3] 압축 풀기")
    ann = os.path.join(dest, "ann")
    untar(tb, os.path.join(dest, "HRSC2016.part03.rar"), ann,
          "HRSC2016/FullDataSet/Annotations/*")
    untar(tb, os.path.join(dest, "HRSC2016.part03.rar"), ann,
          "HRSC2016/ImageSets/*")
    n_xml = len(glob.glob(os.path.join(ann, "HRSC2016", "FullDataSet",
                                       "Annotations", "*.xml")))
    print("  라벨 XML %d개" % n_xml)

    if not a.annotations_only:
        full = os.path.join(dest, "full")
        for part in ("01", "02"):
            untar(tb, os.path.join(dest, "HRSC2016.part%s.rar" % part), full,
                  "HRSC2016/FullDataSet/AllImages/*")
        n_img = len(glob.glob(os.path.join(full, "HRSC2016", "FullDataSet",
                                           "AllImages", "*.bmp")))
        print("  영상 BMP %d개" % n_img)

    print("\n완료:", dest)


if __name__ == "__main__":
    main()
