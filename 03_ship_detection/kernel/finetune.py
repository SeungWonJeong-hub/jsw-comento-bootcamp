"""S2Ships 파인튜닝 — 코멘토 3차 업무 / 정승원

1차 학습에서 나온 진단
----------------------
  mAP50 0.863  vs  mAP50-95 0.459   ← 격차가 매우 큽니다
  평균 IoU 0.672 ~ 0.762 (크기와 함께 단조 증가)

박스가 헐겁게 맞고 있다는 뜻입니다. 선박이 3~5 px 이라 한 픽셀만 어긋나도
IoU 가 20~30% 흔들립니다. 그래서 IoU 문턱을 올리면 대부분 탈락합니다.

처방
----
  1) 입력 해상도를 올린다 (320 -> 640).
     화소 정보가 늘지는 않지만, 탐지기의 stride 구조가 작은 물체를 더 잘 나눕니다.
     소형 객체 탐지에서 표준적으로 쓰는 처방입니다.
  2) 모델을 키운다 (s -> m).
  3) 더 오래 학습한다 (100 -> 200) — 덜 학습된 것이 원인인지 가릅니다.

셋을 나눠 돌려야 무엇이 효과였는지 귀속됩니다.
"""
import os, sys, time, json, subprocess, traceback

WORK, TMP = "/kaggle/working", "/kaggle/temp"
os.makedirs(TMP, exist_ok=True)
DATA = f"{TMP}/S2Ships"
BASE = "/kaggle/input/s2ships-base-weights/yolo11s_dota.pt"

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


@stage("환경 확인")
def _():
    import torch
    ok = torch.cuda.is_available()
    print("torch", torch.__version__, "| CUDA", ok)
    if ok:
        print("GPU:", torch.cuda.get_device_name(0))
    RESULTS["env"] = {"torch": torch.__version__, "cuda": ok,
                      "gpu": torch.cuda.get_device_name(0) if ok else None}
    assert ok, "GPU 미할당"
    print("기준 가중치:", BASE, os.path.exists(BASE))


@stage("데이터 구축")
def _():
    sh("pip install -q rasterio geopandas pyogrio requests ultralytics")
    os.makedirs(DATA, exist_ok=True)
    for t in ["34WFT", "34VEN", "34VEM", "35VLG", "34VER"]:
        p = f"{DATA}/{t}.gpkg"
        if not os.path.exists(p):
            sh(f'curl -sL "https://zenodo.org/api/records/15019034/files/{t}.gpkg/content" -o {p}')
    sh("git clone --depth 1 https://github.com/SeungWonJeong-hub/jsw-comento-bootcamp.git "
       f"{TMP}/repo -b feature/ship-detection", check=False)
    sh(f"python {TMP}/repo/03_ship_detection/src/build_dataset.py "
       f"--gpkg {DATA} --scenes {DATA}/scenes.json --out {DATA}/yolo")
    counts = {}
    for sp in ["train", "val", "test"]:
        n_img = len(os.listdir(f"{DATA}/yolo/images/{sp}"))
        labs = [f"{DATA}/yolo/labels/{sp}/{f}" for f in os.listdir(f"{DATA}/yolo/labels/{sp}")]
        n_obj = sum(len(open(f).read().strip().splitlines())
                    for f in labs if os.path.getsize(f))
        counts[sp] = [n_img, n_obj]
        print(f"  {sp:6s} 타일 {n_img:6d}  인스턴스 {n_obj:6d}")
    RESULTS["dataset"] = counts


from ultralytics.utils import SETTINGS
for k in ["raytune", "comet", "clearml", "dvc", "mlflow", "neptune", "wandb", "hub"]:
    try:
        SETTINGS[k] = False
    except Exception:
        pass
from ultralytics import YOLO

YAML = f"{DATA}/yolo/s2ships.yaml"

# (이름, 시작 가중치, 입력크기, 에폭, 추가설정)
EXPS = [
    # 처방 1 — 해상도. 1차 최고 모델에서 이어서 학습합니다.
    ("ft_640",     BASE,             640, 50,  dict(lr0=0.002)),
    # 처방 2 — 모델 크기. 조건을 1차와 같게 두어 s 와 직접 비교합니다.
    ("yolo11m_320", "yolo11m-obb.pt", 320, 100, {}),
    # 처방 3 — 학습량. 덜 학습된 것이 원인인지 가릅니다.
    ("ft_320_long", BASE,             320, 100, dict(lr0=0.002)),
]

for name, wts, sz, ep, extra in EXPS:
    print(f'\n{"#"*60}\n[{time.time()-T0:7.1f}s] {name}  ({os.path.basename(wts)}, '
          f'imgsz={sz}, {ep} epoch)\n{"#"*60}', flush=True)
    try:
        t = time.time()
        m = YOLO(wts)
        bs = 32 if sz <= 320 else 12          # 640 은 메모리가 4배라 배치를 줄입니다
        m.train(data=YAML, epochs=ep, imgsz=sz, batch=bs, device=0, workers=2,
                plots=False, project=f"{WORK}/runs", name=name, exist_ok=True,
                verbose=False, **extra)
        dt = time.time() - t
        rec = {"imgsz": sz, "epochs": ep, "batch": bs, "train_sec": dt}
        for split in ["val", "test"]:
            r = m.val(data=YAML, imgsz=sz, device=0, split=split)
            rec[f"{split}_mAP50"] = float(r.box.map50)
            rec[f"{split}_mAP50_95"] = float(r.box.map)
            rec[f"{split}_precision"] = float(r.box.mp)
            rec[f"{split}_recall"] = float(r.box.mr)
        RESULTS["experiments"][name] = rec
        print(json.dumps(rec, indent=2))
    except Exception as e:
        RESULTS["experiments"][name] = {"error": f"{type(e).__name__}: {e}"}
        traceback.print_exc()
    save()

print(f"\n[{time.time()-T0:7.1f}s] 전체 완료")
print(json.dumps(RESULTS["experiments"], indent=2, ensure_ascii=False))
save()
