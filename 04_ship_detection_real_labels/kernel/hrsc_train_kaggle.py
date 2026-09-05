# -*- coding: utf-8 -*-
"""Kaggle T4 커널 — 학습·평가·비교를 9시간 세션 하나에서 끝냅니다.

입력 데이터셋 (seungwon21/hrsc-sr-data)
    project/configs, project/scripts, project/manifests,
    project/data/{images_hr, images_A_native_lr, images_B_bicubic,
                  images_C_realesrgan, labels_obb}

순서
    1. 프로젝트를 /kaggle/working/project 로 복사 (입력은 읽기전용)
    2. official 분할  yolo11m-obb  12런  -> 평가 -> 표
    3. portwise 분할  yolo11n-obb  12런  -> 평가 -> 표  (남은 시간 안에서)
    4. results/ 를 통째로 출력에 남김

train_yolo.py 가 첫 런의 초/epoch 를 실측해 예산에 맞춰 epochs 를 조정하고,
런마다 상태를 저장하므로 세션이 죽어도 이어서 돌릴 수 있습니다.
"""
import os
import sys
import time
import shutil
import subprocess

T0 = time.time()
SESSION_H = 9.0
OUT = "/kaggle/working"
PROJ = os.path.join(OUT, "project")

subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                "ultralytics>=8.3", "PyYAML"], check=True)

# ---- 입력 찾기 --------------------------------------------------------
# 데이터셋에 이 폴더(04_ship_detection_real_labels)를 통째로 올립니다:
#   src/ · data/ (labels_obb, hrsc/manifests) · 원본 영상은 images/ 로
src = None
for root, dirs, files in os.walk("/kaggle/input"):
    if ("src" in dirs or "src" in dirs) and "data" in dirs:
        src = root
        break
if src is None:
    raise SystemExit("입력 데이터셋에서 project 트리를 못 찾았습니다: %s"
                     % os.listdir("/kaggle/input"))
print("project source:", src)

# ---- 작업폴더로 복사 (data 는 심볼릭) --------------------------------
os.makedirs(PROJ, exist_ok=True)
for d in ("configs", "src", "manifests"):
    if os.path.exists(os.path.join(PROJ, d)):
        shutil.rmtree(os.path.join(PROJ, d))
    shutil.copytree(os.path.join(src, d), os.path.join(PROJ, d))
if not os.path.exists(os.path.join(PROJ, "data")):
    os.symlink(os.path.join(src, "data"), os.path.join(PROJ, "data"))
os.makedirs(os.path.join(PROJ, "results"), exist_ok=True)
os.makedirs(os.path.join(PROJ, "runs"), exist_ok=True)

# 다른 곳(Colab)에서 끝낸 런이 데이터셋에 실려 있으면 되살립니다.
# train_yolo.py 는 train_state 의 done 항목을 건너뛰고, epochs_used 를 그대로
# 씁니다 — 모든 런이 같은 epoch 으로 학습되어야 비교가 성립하기 때문입니다.
restored = 0
for sub in ("results", "runs"):
    s = os.path.join(src, sub)
    if not os.path.isdir(s):
        continue
    for root, dirs, files in os.walk(s):
        for f in files:
            sp = os.path.join(root, f)
            dp = os.path.join(PROJ, sub, os.path.relpath(sp, s))
            os.makedirs(os.path.dirname(dp), exist_ok=True)
            shutil.copy(sp, dp)
            restored += 1
if restored:
    print("이전 실행에서 복원한 파일 %d개 (끝난 런은 건너뜁니다)" % restored)


def left_h():
    return SESSION_H - (time.time() - T0) / 3600.0


def run(step, *args, budget=None):
    cmd = [sys.executable, os.path.join(PROJ, "src", step)] + list(args)
    if budget is not None:
        cmd += ["--budget-hours", "%.2f" % budget]
    print("\n" + "=" * 70)
    print(" ".join(cmd[1:]), "  (남은 %.2f h)" % left_h())
    print("=" * 70, flush=True)
    r = subprocess.run(cmd, cwd=os.path.join(PROJ, "src"))
    if r.returncode != 0:
        print("!! %s 실패 (exit %d) — 다음 단계로 넘어갑니다" % (step, r.returncode))
    return r.returncode == 0


# ---- 주 실험: official · medium -------------------------------------
# 예산의 75% 를 주 실험에, 나머지를 portwise 부록에 둡니다.
run("train_yolo.py", "--split", "official", budget=left_h() * 0.75 - 0.3)
run("evaluate_yolo.py", "--split", "official")
run("compare_results.py", "--split", "official")

# ---- 부록: portwise · nano --------------------------------------------
if left_h() > 1.0:
    run("train_yolo.py", "--split", "portwise", "--model", "yolo11n-obb.pt",
        budget=left_h() - 0.4)
    run("evaluate_yolo.py", "--split", "portwise")
    run("compare_results.py", "--split", "portwise")
else:
    print("portwise 부록은 시간이 부족해 건너뜁니다 (남은 %.2f h)" % left_h())

# ---- 출력 --------------------------------------------------------------
dst = os.path.join(OUT, "results")
if os.path.exists(dst):
    shutil.rmtree(dst)
shutil.copytree(os.path.join(PROJ, "results"), dst)
# 가중치는 last.pt 만 남깁니다 (용량)
wdst = os.path.join(OUT, "weights")
os.makedirs(wdst, exist_ok=True)
for name in os.listdir(os.path.join(PROJ, "runs")):
    p = os.path.join(PROJ, "runs", name, "weights", "last.pt")
    if os.path.exists(p):
        shutil.copy(p, os.path.join(wdst, name + ".pt"))
print("\n총 %.2f h · 출력: %s" % ((time.time() - T0) / 3600.0, OUT))
for f in sorted(os.listdir(dst)):
    print("  ", f)
