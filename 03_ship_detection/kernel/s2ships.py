"""S2Ships — 카글 GPU 학습
코멘토 3차 업무 · 정승원

핀란드 연안 Sentinel-2 선박 데이터로 회전 박스(OBB) 탐지기를 학습합니다.
한국 항만에 적용하는 것이 목표라, 여기서 나온 가중치를 그대로 가져다 씁니다.

설계 원칙
  무거운 것은 /kaggle/temp (출력 제외), 결과만 /kaggle/working.
  단계마다 결과를 즉시 저장해 중간에 죽어도 앞 결과는 남습니다.
  GPU 는 T4 로 지정한다 (P100 은 PyTorch cu128 이 sm_60 을 안 만듭니다).
"""
import os, sys, time, json, subprocess, traceback

WORK, TMP = "/kaggle/working", "/kaggle/temp"
os.makedirs(TMP, exist_ok=True)
DATA = f"{TMP}/S2Ships"

T0 = time.time()
RESULTS = {"stages": {}, "experiments": {}}


def save():
    json.dump(RESULTS, open(f"{WORK}/results.json", "w"), indent=2, ensure_ascii=False)


def stage(name):
    def deco(fn):
        print(f'\n{"="*60}\n[{time.time()-T0:7.1f}s] {name}\n{"="*60}', flush=True)
        try:
            r = fn()
            RESULTS["stages"][name] = "ok"
            save()
            return r
        except Exception as e:
            RESULTS["stages"][name] = f"FAIL: {type(e).__name__}: {e}"
            save()
            traceback.print_exc()
            raise
    return deco


def sh(cmd, check=True):
    print(f"$ {cmd}", flush=True)
    return subprocess.run(cmd, shell=True, check=check)


# ------------------------------------------------------------------ 1 환경
@stage("환경 확인")
def _():
    import torch, numpy as np
    print("torch", torch.__version__, "| numpy", np.__version__)
    ok = torch.cuda.is_available()
    print("CUDA", ok)
    if ok:
        print("GPU:", torch.cuda.get_device_name(0),
              f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
    RESULTS["env"] = {"torch": torch.__version__, "numpy": np.__version__,
                      "cuda": ok, "gpu": torch.cuda.get_device_name(0) if ok else None}
    assert ok, "GPU 미할당 — Accelerator 확인"


# ------------------------------------------------------------------ 2 데이터
@stage("데이터 구축")
def _():
    """업로드하지 않고 커널이 직접 만듭니다.

    주석(2.6MB)은 Zenodo 에서, 위성 사진은 AWS 공개 COG 에서 받습니다.
    둘 다 인증이 필요 없어서, 7천 장짜리 타일을 올릴 이유가 없습니다.
    """
    sh("pip install -q rasterio geopandas pyogrio requests")
    os.makedirs(DATA, exist_ok=True)
    for t in ["34WFT", "34VEN", "34VEM", "35VLG", "34VER"]:
        p = f"{DATA}/{t}.gpkg"
        if not os.path.exists(p):
            sh(f'curl -sL "https://zenodo.org/api/records/15019034/files/{t}.gpkg/content" -o {p}')
    # 저장소의 build_dataset.py 를 그대로 쓴다 (코드 중복을 만들지 않습니다)
    sh("git clone --depth 1 https://github.com/SeungWonJeong-hub/jsw-comento-bootcamp.git "
       f"{TMP}/repo -b feature/ship-detection", check=False)
    build = f"{TMP}/repo/03_ship_detection/src/build_dataset.py"
    assert os.path.exists(build), "build_dataset.py 를 못 찾음"
    sh(f"python {build} --gpkg {DATA} --scenes {DATA}/scenes.json --out {DATA}/yolo")

    counts = {}
    for sp in ["train", "val", "test"]:
        n_img = len(os.listdir(f"{DATA}/yolo/images/{sp}"))
        labs = [f"{DATA}/yolo/labels/{sp}/{f}" for f in os.listdir(f"{DATA}/yolo/labels/{sp}")]
        n_obj = sum(len(open(f).read().strip().splitlines())
                    for f in labs if os.path.getsize(f))
        counts[sp] = [n_img, n_obj]
        print(f"  {sp:6s} 타일 {n_img:6d}  인스턴스 {n_obj:6d}")
    RESULTS["dataset"] = counts


# ------------------------------------------------------------------ 3 코드
@stage("ultralytics 준비")
def _():
    sh("pip install -q ultralytics")
    import ultralytics
    print("ultralytics", ultralytics.__version__)
    RESULTS["env"]["ultralytics"] = ultralytics.__version__


from ultralytics.utils import SETTINGS
for k in ["raytune", "comet", "clearml", "dvc", "mlflow", "neptune", "wandb", "hub"]:
    try:
        SETTINGS[k] = False
    except Exception:
        pass
from ultralytics import YOLO

EPOCHS = int(os.environ.get("EPOCHS", 100))
YAML = f"{DATA}/yolo/s2ships.yaml"

# 항공기 때 배운 것을 그대로 적용합니다:
#   mosaic 과 HSV 가 동시에 켜지면 발산한 사례가 있었습니다.
#   여기서는 아키텍처가 달라 그대로는 아니겠지만, 대조군을 같이 돌려 확인합니다.
NOAUG = dict(hsv_h=0, hsv_s=0, hsv_v=0, degrees=0, translate=0,
             scale=0, fliplr=0, flipud=0, mosaic=0, erasing=0)

EXPS = [
    ("yolo11n_dota",  "yolo11n-obb.pt",   {}),                 # DOTA 사전학습 (기대 최고)
    ("yolo11n_scratch", "yolo11-obb.yaml", {}),                # 밑바닥 — 사전학습 효과 측정
    ("yolo11n_noaug", "yolo11n-obb.pt",   NOAUG),              # 증강 기여도
    ("yolo11s_dota",  "yolo11s-obb.pt",   {}),                 # 모델 크기
]

for name, cfg, extra in EXPS:
    print(f'\n{"#"*60}\n[{time.time()-T0:7.1f}s] 실험: {name} ({cfg})\n{"#"*60}', flush=True)
    try:
        t = time.time()
        m = YOLO(cfg)
        m.train(data=YAML, epochs=EPOCHS, imgsz=320, batch=32, device=0,
                workers=2, plots=False, project=f"{WORK}/runs", name=name,
                exist_ok=True, verbose=False, **extra)
        dt = time.time() - t
        rec = {"train_sec": dt, "sec_per_epoch": dt / EPOCHS}
        for split, key in [("val", "val"), ("test", "test")]:
            r = m.val(data=YAML, imgsz=320, device=0, split=split)
            rec[f"{key}_mAP50"] = float(r.box.map50)
            rec[f"{key}_mAP50_95"] = float(r.box.map)
            rec[f"{key}_precision"] = float(r.box.mp)
            rec[f"{key}_recall"] = float(r.box.mr)
        RESULTS["experiments"][name] = rec
        print(json.dumps(rec, indent=2))
    except Exception as e:
        RESULTS["experiments"][name] = {"error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()
    save()

print(f"\n[{time.time()-T0:7.1f}s] 전체 완료")
print(json.dumps(RESULTS["experiments"], indent=2, ensure_ascii=False))
save()
