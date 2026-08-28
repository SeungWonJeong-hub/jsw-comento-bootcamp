"""ONNX 내보내기와 CPU 추론 속도 — 코멘토 3차 업무 / 정승원

왜 CPU 인가
-----------
4차는 웹앱이다. 항구를 고르면 그 자리에서 탐지가 돌아야 한다. GPU 를 붙인
서버를 항상 켜 두는 것은 개인 과제에서 현실적이지 않다. 그래서 CPU 만으로
쓸 만한 지연시간이 나오는지를 먼저 재고, 그 수치 위에서 4차를 설계한다.

왜 프로세스를 나누나
--------------------
처음에는 한 프로세스에서 PyTorch 와 ONNX Runtime 을 차례로 쟀다. 그랬더니
같은 스크립트를 두 번 돌렸을 때 ONNX 8 스레드가 128.0 ms 와 25.9 ms 로
5 배 달라졌다. 노이즈가 아니라 계통 오차다.

  두 런타임이 각자 스레드 풀을 만든다. 먼저 뜬 쪽이 코어를 잡고 있으면
  나중 쪽은 남은 코어를 두고 경합한다. 18 코어에 torch 가 14 개를 잡은
  상태에서 ORT 가 8 개를 더 요구하면 초과 구독이 된다.

그래서 설정 하나당 프로세스를 새로 띄운다. 각 프로세스는 자기 런타임만
올린다. 느리지만 이 수치는 믿을 수 있다.

무엇을 재나
-----------
  1. 타일 한 장(320x320) 추론 시간 — PyTorch 대 ONNX Runtime
  2. 스레드 수에 따른 변화 — 기본값이 최적인 경우가 드물다
  3. 항만 한 곳 전체 주사 시간 — 실제로 웹앱이 기다리게 할 시간

3번이 사용자가 체감하는 값이다. 1번만 재고 "빠르다" 고 말하면 안 된다.
항만 사각형은 타일 수십 장으로 쪼개지기 때문이다.

사용법
------
  py bench_cpu.py --weights weights/yolo11s_dota.pt
"""
import os
import sys
import json
import time
import argparse
import subprocess

import numpy as np


def bench(fn, warmup=5, runs=30):
    for _ in range(warmup):
        fn()
    t = []
    for _ in range(runs):
        s = time.perf_counter()
        fn()
        t.append(time.perf_counter() - s)
    t = np.array(t) * 1000.0
    return dict(mean_ms=float(t.mean()), p50_ms=float(np.median(t)),
                p90_ms=float(np.percentile(t, 90)),
                min_ms=float(t.min()), n=runs)


def port_tiles(bbox, gsd=10.0, tile=320, overlap=64):
    """항만 사각형이 타일 몇 장으로 쪼개지나. korea_ports.detect 와 같은 격자."""
    lon0, lat0, lon1, lat1 = bbox
    # 위도 37도 부근에서 경도 1도는 약 88 km, 위도 1도는 약 111 km
    w_m = (lon1 - lon0) * 88_000.0
    h_m = (lat1 - lat0) * 111_000.0
    W, H = w_m / gsd, h_m / gsd
    stride = tile - overlap
    nx = max(1, int(np.ceil((W - overlap) / stride)))
    ny = max(1, int(np.ceil((H - overlap) / stride)))
    return int(nx * ny), int(W), int(H)


# ---------------------------------------------------------------- 자식 프로세스
def run_one(kind, path, imgsz, threads, runs):
    """설정 하나만 재고 JSON 한 줄을 찍는다. 부모가 이것을 모은다."""
    img = (np.random.rand(imgsz, imgsz, 3) * 255).astype(np.uint8)

    if kind == "torch":
        import torch
        torch.set_num_threads(threads)
        from ultralytics import YOLO
        m = YOLO(path)
        r = bench(lambda: m.predict(img, imgsz=imgsz, device="cpu",
                                    verbose=False), runs=runs)

    elif kind == "onnx_raw":
        # session.run 만. 전처리와 NMS 가 빠져 있으므로 torch 와 직접
        # 비교하면 안 된다. 런타임 자체의 상한을 보는 용도다.
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        so.inter_op_num_threads = 1
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        name = s.get_inputs()[0].name
        x = img.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        r = bench(lambda: s.run(None, {name: x}), runs=runs)

    elif kind == "onnx_e2e":
        # 전처리와 NMS 를 포함한 끝에서 끝까지. 웹앱이 실제로 쓸 값이다.
        os.environ["OMP_NUM_THREADS"] = str(threads)
        from ultralytics import YOLO
        m = YOLO(path)
        r = bench(lambda: m.predict(img, imgsz=imgsz, verbose=False), runs=runs)

    else:
        raise SystemExit(f"모르는 종류: {kind}")

    print("__RESULT__" + json.dumps(r))


# ---------------------------------------------------------------- 부모 프로세스
def spawn(kind, path, imgsz, threads, runs):
    cmd = [sys.executable, os.path.abspath(__file__), "--worker", kind,
           "--path", path, "--imgsz", str(imgsz),
           "--nthreads", str(threads), "--runs", str(runs)]
    env = dict(os.environ, PYTHONUTF8="1")
    p = subprocess.run(cmd, capture_output=True, text=True, env=env,
                       encoding="utf-8", errors="replace")
    for line in (p.stdout or "").splitlines():
        if line.startswith("__RESULT__"):
            return json.loads(line[len("__RESULT__"):])
    tail = (p.stderr or "").strip().splitlines()[-2:]
    print(f"    실패: {' / '.join(tail) if tail else '출력 없음'}")
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default="weights/yolo11s_dota.pt")
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--repeat", type=int, default=2,
                    help="설정마다 프로세스를 몇 번 띄워 재현성을 볼지")
    ap.add_argument("--out", default="outputs/cpu_bench.json")
    # 자식 전용
    ap.add_argument("--worker", default=None)
    ap.add_argument("--path", default=None)
    ap.add_argument("--nthreads", type=int, default=1)
    a = ap.parse_args()

    if a.worker:
        return run_one(a.worker, a.path, a.imgsz, a.nthreads, a.runs)

    import platform
    ncpu = os.cpu_count() or 1
    info = dict(cpu=platform.processor(), cores=ncpu,
                weights=os.path.basename(a.weights), imgsz=a.imgsz,
                runs=a.runs, repeat=a.repeat)
    print(f"CPU {info['cpu']}  ({ncpu} 코어)")
    print(f"가중치 {info['weights']}, 입력 {a.imgsz}x{a.imgsz}")
    print(f"설정마다 프로세스를 새로 띄우고 {a.repeat} 번 반복한다\n")

    # ONNX 를 먼저 만들어 둔다. 벤치 프로세스 안에서 내보내면 그 프로세스만
    # 조건이 달라진다.
    onnx_path = a.weights.replace(".pt", ".onnx")
    if not os.path.exists(onnx_path):
        print("ONNX 내보내는 중...")
        from ultralytics import YOLO
        YOLO(a.weights).export(format="onnx", imgsz=a.imgsz,
                               simplify=False, dynamic=False)
    if os.path.exists(onnx_path):
        info["onnx_mb"] = round(os.path.getsize(onnx_path) / 1e6, 1)
        print(f"ONNX {os.path.basename(onnx_path)}  {info['onnx_mb']} MB\n")

    threads = [t for t in a.threads if t <= ncpu]
    plan = ([("torch", a.weights, t) for t in threads]
            + ([("onnx_e2e", onnx_path, t) for t in threads]
               + [("onnx_raw", onnx_path, t) for t in threads]
               if os.path.exists(onnx_path) else []))

    LABEL = {"torch": "PyTorch (끝에서 끝까지)",
             "onnx_e2e": "ONNX (끝에서 끝까지)",
             "onnx_raw": "ONNX (session.run 만)"}

    res, cur = {}, None
    print(f"{'설정':<30}{'p50 (ms)':>10}{'p90':>8}{'반복 간 차이':>14}")
    for kind, path, t in plan:
        if kind != cur:
            cur = kind
            print(f"\n{LABEL[kind]}")
        got = [spawn(kind, path, a.imgsz, t, a.runs) for _ in range(a.repeat)]
        got = [g for g in got if g]
        if not got:
            continue
        p50 = [g["p50_ms"] for g in got]
        spread = (max(p50) / min(p50)) if min(p50) > 0 else float("nan")
        rec = dict(p50_ms=float(np.median(p50)),
                   p90_ms=float(np.median([g["p90_ms"] for g in got])),
                   spread=float(spread), repeats=p50)
        res[f"{kind}_{t}thread"] = rec
        flag = "  <- 편차 큼" if spread > 1.25 else ""
        print(f"{'  threads = ' + str(t):<30}{rec['p50_ms']:>10.1f}"
              f"{rec['p90_ms']:>8.1f}{spread:>13.2f}x{flag}")

    # --- 항만 한 곳 전체 주사 ---
    # 끝에서 끝까지 잰 값만 쓴다. session.run 값을 쓰면 NMS 를 공짜로 치는 셈이다.
    e2e = {k: v for k, v in res.items() if not k.startswith("onnx_raw")}
    if not e2e:
        print("\n측정된 설정이 없다.")
        return
    best = min(e2e, key=lambda k: e2e[k]["p50_ms"])
    per = e2e[best]["p50_ms"]

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    try:
        from korea_ports import PORTS
    except Exception as e:
        print(f"\n항만 목록을 못 읽음: {e}")
        PORTS = {}

    if PORTS:
        print(f"\n항만 전체 주사 — 최적 설정 {best}, 타일당 {per:.1f} ms")
        print(f"{'항만':<14}{'창 크기(px)':>16}{'타일':>7}{'추론(초)':>10}")
        scans = {}
        for key, v in list(PORTS.items())[:8]:
            label, *bbox = v
            n, W, H = port_tiles(bbox)
            sec = n * per / 1000.0
            scans[key] = dict(label=label, tiles=n, w_px=W, h_px=H,
                              infer_sec=round(sec, 2))
            print(f"{label:<14}{f'{W}x{H}':>16}{n:>7}{sec:>10.2f}")
        res["port_scan"] = scans
        med = float(np.median([v["infer_sec"] for v in scans.values()]))
        res["port_scan_median_sec"] = med
        print(f"\n항만 한 곳 추론 중앙값 {med:.2f} 초 (영상 내려받기는 별도)")
        print("웹앱에서 기다릴 만하다. GPU 없이 4차를 설계할 수 있다."
              if med < 5 else "체감이 길다. 타일을 줄이거나 더 작은 모델을 쓴다.")

    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    json.dump(dict(info=info, best=best, results=res),
              open(a.out, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    print(f"\n저장: {a.out}")


if __name__ == "__main__":
    main()
