# -*- coding: utf-8 -*-
"""YOLO OBB 학습 — 비교군·시드·분할을 돌면서 전부 같은 조건으로.

    py scripts/train_yolo.py --split official
    py scripts/train_yolo.py --split portwise --model yolo11n-obb.pt

공정성
------
하이퍼파라미터는 전부 configs/experiment.yaml 한 곳에서 옵니다. 비교군마다
다른 값을 줄 수 있는 통로를 두지 않았습니다. epochs 를 예산 때문에 낮출
때도 **모든 런에 똑같이** 낮춥니다(그 값을 결과에 기록합니다).

학습해야 하는 런
----------------
    조건 1 matched     A, B, C 각각 학습 -> 같은 비교군으로 평가
    조건 2 hr_trained  HR 로 한 번만 학습 -> A/B/C 에 적용 (추가 학습 없음)
  => 분할당 (3 + 1) x 시드 3 = 12런

9시간 예산
----------
첫 런을 calibrate_epochs 만큼만 돌려 초/epoch 를 실측하고, 남은 시간으로
전체를 끝낼 수 있는지 계산합니다. 부족하면 epochs 를 낮추고(하한 min_epochs),
그래도 모자라면 남은 런을 건너뛰고 무엇이 빠졌는지 기록합니다.
런 하나가 끝날 때마다 결과를 저장하므로 세션이 죽어도 거기까지는 남고,
다시 돌리면 끝난 런은 건너뜁니다.
"""
import os
import time
import json
import shutil
import argparse

from common import ROOT, load_cfg, rel, read_csv, dump_json, results_dir, smoke_cap

# 4차 본 실험은 원본(HR) 한 갈래입니다. 저해상도 비교군은 별도 실험 저장소에 있습니다.
TRAIN_SOURCES = ["hr"]


def build_dataset(cfg, split_id, source, work):
    """분할·비교군에 맞는 YOLO 데이터 트리를 만듭니다(심볼릭/복사).

    라벨은 정규화 좌표라 비교군 전부가 같은 파일을 씁니다.
    """
    data = rel(cfg, "data")
    # 원본은 BMP 그대로 씁니다. 파생 영상 폴더는 저해상도 실험에서만 씁니다.
    img_src = rel(cfg, "images") if source == "hr" else os.path.join(data, "images_" + source)
    ext = ".bmp" if source == "hr" else ".png"
    lab_src = os.path.join(data, "labels_obb")
    rows = read_csv(os.path.join(rel(cfg, "manifests"), "split.csv"))
    root = os.path.join(work, "ds_%s_%s" % (split_id, source))
    for sp in ("train", "val", "test"):
        os.makedirs(os.path.join(root, "images", sp), exist_ok=True)
        os.makedirs(os.path.join(root, "labels", sp), exist_ok=True)
    n = {"train": 0, "val": 0, "test": 0}
    cap = smoke_cap()
    for r in rows:
        sp = r[split_id]
        if sp not in n:
            continue
        if cap and n[sp] >= cap[sp]:
            continue
        stem = r["image_id"]
        ip = os.path.join(img_src, stem + ext)
        lp = os.path.join(lab_src, stem + ".txt")
        if not os.path.exists(ip):
            continue
        di = os.path.join(root, "images", sp, stem + ext)
        dl = os.path.join(root, "labels", sp, stem + ".txt")
        if not os.path.exists(di):
            try:
                os.link(ip, di)          # 하드링크가 되면 복사 안 함
            except OSError:
                shutil.copy(ip, di)
        if os.path.exists(lp) and not os.path.exists(dl):
            shutil.copy(lp, dl)
        n[sp] += 1
    yml = os.path.join(work, "%s_%s.yaml" % (split_id, source))
    with open(yml, "w", encoding="utf-8") as f:
        f.write("path: %s\ntrain: images/train\nval: images/val\n"
                "test: images/test\nnc: 1\nnames: ['ship']\n" % root)
    return yml, root, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", default="official", choices=["official", "portwise"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--work", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--budget-hours", type=float, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="배관 점검용: 영상 소수·1 epoch·시드 1개. 결과는 results/smoke/ 에 격리")
    a = ap.parse_args()
    cfg = load_cfg(a.config)
    tcfg = cfg["train"]
    bcfg = cfg["budget"]
    if a.smoke:
        # 세션을 태우기 전에 train->eval->compare 가 끝까지 도는지만 봅니다.
        tcfg = dict(tcfg, epochs=1, seeds=[0], batch=4)
        bcfg = dict(bcfg, calibrate_epochs=1, min_epochs=1)
        a.model = a.model or "yolo11n-obb.pt"
        a.work = os.path.join(ROOT, "runs", "smoke")
        os.environ["HRSC_SMOKE"] = "1"
    spec = next(s for s in cfg["splits"] if s["id"] == a.split)
    model_name = a.model or spec.get("model") or tcfg["model"]
    os.makedirs(a.work, exist_ok=True)

    budget_h = a.budget_hours or (float(bcfg["session_hours"]) -
                                  float(bcfg["reserve_hours"]))
    t_start = time.time()
    state_p = os.path.join(results_dir(), "train_state_%s.json" % a.split)
    state = json.load(open(state_p, encoding="utf-8")) \
        if os.path.exists(state_p) else {"done": {}, "epochs_used": None}

    runs = [(src, seed) for src in TRAIN_SOURCES for seed in tcfg["seeds"]]
    todo = [r for r in runs if "%s|%d" % r not in state["done"]]
    print("전체 %d런 · 남은 %d런 · 예산 %.2f h · 모델 %s"
          % (len(runs), len(todo), budget_h, model_name))

    from ultralytics import YOLO
    epochs = state["epochs_used"] or int(tcfg["epochs"])
    sec_per_epoch = None

    for src, seed in todo:
        key = "%s|%d" % (src, seed)
        yml, root, n = build_dataset(cfg, a.split, src, a.work)
        left_h = budget_h - (time.time() - t_start) / 3600.0

        # ---- 캘리브레이션: 첫 런에서 초/epoch 실측 -------------------
        if sec_per_epoch is None and state["epochs_used"] is None:
            ce = int(bcfg["calibrate_epochs"])
            t0 = time.time()
            YOLO(model_name).train(
                data=yml, epochs=ce, imgsz=tcfg["imgsz"], batch=tcfg["batch"],
                seed=seed, val=False, plots=False, verbose=False,
                deterministic=tcfg["deterministic"], patience=0,
                project=a.work, name="calib", exist_ok=True, **tcfg["augment"])
            sec_per_epoch = (time.time() - t0) / ce
            need = len(todo) * epochs * sec_per_epoch / 3600.0
            print("\n[캘리브레이션] %.1f 초/epoch · %d런 x %d epoch = %.2f h 필요 "
                  "(남은 %.2f h)" % (sec_per_epoch, len(todo), epochs, need, left_h))
            if need > left_h:
                fit = int(left_h * 3600 / (len(todo) * sec_per_epoch))
                epochs = max(int(bcfg["min_epochs"]), min(epochs, fit))
                print("  -> epochs 를 %d 로 낮춥니다 (모든 런에 동일 적용)" % epochs)
            state["epochs_used"] = epochs
            dump_json(state_p, state)
            shutil.rmtree(os.path.join(a.work, "calib"), ignore_errors=True)

        est_h = epochs * (sec_per_epoch or 0) / 3600.0
        if sec_per_epoch and est_h > left_h:
            print("남은 예산 %.2f h < 런 예상 %.2f h — 여기서 멈춥니다." % (left_h, est_h))
            break

        name = "%s_%s_seed%d" % (a.split, src, seed)
        print("\n=== %s  (train %d / val %d / test %d) ==="
              % (name, n["train"], n["val"], n["test"]))
        t0 = time.time()
        YOLO(model_name).train(
            data=yml, epochs=epochs, imgsz=tcfg["imgsz"], batch=tcfg["batch"],
            seed=seed, val=bool(tcfg.get("val_during_train", False)),
            plots=False, verbose=False, patience=0,
            deterministic=tcfg["deterministic"],
            project=a.work, name=name, exist_ok=True, **tcfg["augment"])
        dt = time.time() - t0
        sec_per_epoch = dt / epochs
        wpath = os.path.join(a.work, name, "weights",
                             "%s.pt" % tcfg.get("use_weights", "last"))
        state["done"][key] = dict(name=name, weights=wpath, seconds=round(dt, 1),
                                  epochs=epochs, model=model_name,
                                  split=a.split, source=src, seed=seed,
                                  counts=n)
        dump_json(state_p, state)
        print("  %.1f분 · %s" % (dt / 60, wpath))

    skipped = [r for r in runs if "%s|%d" % r not in state["done"]]
    print("\n완료 %d / %d런" % (len(state["done"]), len(runs)))
    if skipped:
        print("건너뜀:", ", ".join("%s(seed %d)" % r for r in skipped))
        print("  같은 명령을 다시 돌리면 이어서 합니다.")
    print("상태:", state_p)


if __name__ == "__main__":
    main()
