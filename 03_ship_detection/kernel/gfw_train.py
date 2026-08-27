"""GFW 데이터 + P2 헤드 학습 — 코멘토 3차 업무 / 정승원

무엇이 바뀌었나
--------------
1) 데이터: 핀란드(맑은 발트해, 사람이 그린 박스) -> GFW(황해·동중국해, AIS 대조)
   한국에서 인천의 배를 놓친 원인이 탁도였고, 그 조건이 학습에 들어간다.
   박스는 길이·방향 실측으로 계산한다. 폴리곤에서 추정하지 않는다.

2) 구조: P2(stride 4) 헤드 추가
   한국 근해 선박은 길이 중앙 30 m = 3 픽셀. stride 8 격자보다 작아
   위치 오차가 원리적으로 ±4 px 까지 벌어진다. 격자를 절반으로 줄인다.

평가
----
학습은 황해·동중국해, 평가는 한국. 학습에 안 쓴 한국 장면으로만 잰다.
지표는 IoU 기반과 점 기반을 함께 본다. 3 픽셀 물체에 IoU 0.75 를 요구하는 것은
지표의 함정이고, GFW 정답이 원래 점이라 점 기반이 정답의 성격에 맞다.
"""
import os, sys, time, json, subprocess, traceback

WORK, TMP = "/kaggle/working", "/kaggle/temp"
os.makedirs(TMP, exist_ok=True)
DATA = f"{TMP}/GFWShips"

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


@stage("데이터 구축")
def _():
    """타일 2.1GB 를 올리는 대신, 장면 목록(1.6MB)만 받아 커널이 사진을 내려 만든다.

    scenes.json 에 장면별 COG 주소와 탐지 좌표·길이·방향이 들어 있다.
    사진은 AWS 공개 COG 라 인증이 필요 없다.
    """
    sh("pip install -q ultralytics rasterio")
    # 마운트 경로가 슬러그와 다를 수 있으므로 input 아래를 훑는다
    src = None
    for root, dirs, files in os.walk("/kaggle/input"):
        if "scenes.json" in files:
            src = root
            break
    if src is None:
        print("사용 가능한 입력:", os.listdir("/kaggle/input") if os.path.isdir("/kaggle/input") else "없음")
    assert src, "장면 패키지를 못 찾음 — dataset_sources 확인"
    print("장면 패키지:", src)

    sh("git clone --depth 1 https://github.com/SeungWonJeong-hub/jsw-comento-bootcamp.git "
       f"{TMP}/repo -b feature/ship-detection", check=False)
    sys.path.insert(0, f"{TMP}/repo/03_ship_detection/src")
    from build_gfw_dataset import cut_scene
    import numpy as np

    S = json.load(open(f"{src}/scenes.json", encoding="utf-8"))
    split_of, cog, dets = S["split_of"], S["cog"], S["dets"]
    print(f"장면 {len(split_of)}개")

    rng = np.random.default_rng(0)
    counter = [0]
    stats = {}
    for split in ["train", "val", "test"]:
        ids = [s for s, v in split_of.items() if v == split]
        tp = to = 0
        for i, sid in enumerate(ids, 1):
            try:
                p, o = cut_scene(cog[sid], [tuple(d) for d in dets[sid]],
                                 DATA, split, f"s{abs(hash(sid)) % 10**8:08d}",
                                 counter, 0.0, rng)
                tp += p; to += o
            except Exception as e:
                print(f"  {sid[:34]} 실패: {type(e).__name__}")
            if i % 25 == 0:
                print(f"  [{split}] {i}/{len(ids)}  타일 {tp}  라벨 {to}", flush=True)
        stats[split] = [tp, to]
        print(f"[{split}] 타일 {tp}  인스턴스 {to}", flush=True)
    RESULTS["dataset"] = stats

    open(f"{TMP}/gfw.yaml", "w").write(
        f"path: {DATA}\ntrain: images/train\nval: images/val\ntest: images/test\n"
        f"names:\n  0: vessel\n")
    p2 = f"{TMP}/repo/03_ship_detection/src/yolo11-p2-obb.yaml"
    assert os.path.exists(p2), "P2 설정을 못 찾음"
    RESULTS["p2_cfg"] = p2


from ultralytics.utils import SETTINGS
for k in ["raytune", "comet", "clearml", "dvc", "mlflow", "neptune", "wandb", "hub"]:
    try:
        SETTINGS[k] = False
    except Exception:
        pass
from ultralytics import YOLO

YAML = f"{TMP}/gfw.yaml"
P2 = f"{TMP}/repo/03_ship_detection/src/yolo11-p2-obb.yaml"
EPOCHS = int(os.environ.get("EPOCHS", 80))

# 두 축을 각각 재야 무엇이 효과였는지 귀속된다.
#   데이터 효과  = 표준 구조로 GFW 학습  vs  기존 핀란드 결과
#   구조 효과    = 같은 GFW 데이터에서 P2 유무
EXPS = [
    ("gfw_std",  "yolo11s-obb.pt", 320, {}),      # GFW + 표준 구조
    ("gfw_p2",   P2,               320, {}),      # GFW + P2  <- 본 실험
    ("gfw_p2_640", P2,             640, {}),      # P2 + 해상도까지
]

for name, cfg, sz, extra in EXPS:
    print(f'\n{"#"*60}\n[{time.time()-T0:7.1f}s] {name} ({os.path.basename(cfg)}, '
          f'imgsz={sz})\n{"#"*60}', flush=True)
    try:
        t = time.time()
        m = YOLO(cfg)
        bs = 32 if sz <= 320 else 10
        m.train(data=YAML, epochs=EPOCHS, imgsz=sz, batch=bs, device=0, workers=2,
                plots=False, project=f"{WORK}/runs", name=name, exist_ok=True,
                verbose=False, **extra)
        dt = time.time() - t
        rec = {"cfg": os.path.basename(cfg), "imgsz": sz, "batch": bs,
               "epochs": EPOCHS, "train_sec": dt}
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
