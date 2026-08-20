"""
1차 업무 [피드백 반영 1] - 이상치 임계값을 데이터 분포에서 산출한다.

기존 코드는 밝기 하한 40, 면적비 하한 0.10 을 감으로 정해 두었다. 표본 40장에서는
밝기 필터가 한 번도 발동하지 않았고, 최저값 54.5 가 임계값과 14.5 밖에 떨어져
있지 않아 "발동하지 않았다"를 안전하다는 뜻으로 읽을 수 없는 상태였다.

이 스크립트는 대규모 표본으로 두 지표의 분포를 직접 그린 뒤, 세 가지 통계 기준으로
커트라인을 계산하고 어느 쪽을 채택할지 분포 모양을 근거로 판단한다.

  (a) 백분위수    하위 1% / 5%     - 분포 모양을 가정하지 않는다
  (b) IQR 울타리  Q1 - 1.5 * IQR   - 상자그림의 표준 이상치 기준
  (c) 정규 근사   mean - 3 * std   - 정규분포를 가정할 때의 0.13% 지점

세 기준은 분포가 정규에 가까울수록 서로 붙고, 치우칠수록 벌어진다. 어느 것이
맞다가 아니라 어느 것이 이 분포에 맞는지를 왜도로 판정한다.

사용 예)
    py -3 threshold_study.py --num-images 1000
    py -3 threshold_study.py --from-cache          # 이미 측정한 결과로 다시 계산만
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

from image_preprocessing import (
    DATASET_ID,
    DATASET_SPLIT,
    measure_mean_brightness,
    measure_object_area_ratio,
)

MEASURE_PATH = Path("outputs/threshold_study_measurements.json")
THRESHOLD_PATH = Path("thresholds.json")

# 하위 몇 %를 이상치로 볼 것인가. 백분위수 기준의 후보값이다.
PERCENTILE_CANDIDATES = (1.0, 5.0)

# 왜도의 절댓값이 이 값을 넘으면 정규 근사를 쓰지 않는다.
# 0.5 는 통계 관행상 '완만한 치우침'과 '뚜렷한 치우침'을 가르는 지점이다.
SKEW_LIMIT = 0.5

# 기존에 감으로 정했던 값. 비교 대상으로만 남긴다.
OLD = {"mean_brightness": 40.0, "object_area_ratio": 0.10}


# ----------------------------------------------------------------------------
# 1. 측정 - 지표 정의는 image_preprocessing 에서 그대로 가져와 한 곳에만 둔다
# ----------------------------------------------------------------------------
def measure_stream(num_images: int) -> list[dict]:
    """food101 을 streaming 으로 받아 밝기와 면적비만 측정한다.

    이미지 자체는 저장하지 않는다. 1,000 장이면 원본이 250 MB 가까이 되는데
    필요한 것은 숫자 두 개뿐이라 디스크에 남길 이유가 없다.
    """
    from datasets import load_dataset

    stream = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    stream = stream.shuffle(seed=42, buffer_size=1000)

    rows: list[dict] = []
    t0 = time.time()
    for idx, record in enumerate(stream):
        if idx >= num_images:
            break
        rgb = np.array(record["image"].convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        rows.append({
            "name": f"food101_{idx:04d}_label{record['label']}",
            "label": int(record["label"]),
            "mean_brightness": measure_mean_brightness(bgr),
            "object_area_ratio": measure_object_area_ratio(bgr),
        })
        if (idx + 1) % 100 == 0:
            elapsed = time.time() - t0
            rate = (idx + 1) / elapsed
            left = (num_images - idx - 1) / rate
            print(f"  {idx + 1:5d}/{num_images}  {rate:.1f} img/s  "
                  f"남은 시간 {left / 60:.1f}분", flush=True)
    return rows


# ----------------------------------------------------------------------------
# 2. 커트라인 계산
# ----------------------------------------------------------------------------
def describe(values: np.ndarray) -> dict:
    """분포의 위치, 산포, 모양을 한 번에 요약한다."""
    from scipy import stats

    q1, q2, q3 = np.percentile(values, [25, 50, 75])
    return {
        "n": int(values.size),
        "min": float(values.min()),
        "max": float(values.max()),
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)),
        "q1": float(q1),
        "median": float(q2),
        "q3": float(q3),
        "iqr": float(q3 - q1),
        # 왜도: 0 이면 대칭, 양수면 오른쪽 꼬리가 길다.
        # 첨도(초과): 0 이면 정규분포와 같은 뾰족함.
        "skew": float(stats.skew(values)),
        "kurtosis": float(stats.kurtosis(values)),
    }


def cutlines(values: np.ndarray) -> dict:
    """세 가지 기준으로 하한 커트라인을 계산한다.

    이상치 제거의 목적이 '너무 어두운' / '객체가 너무 작은' 쪽을 걸러내는 것이므로
    양측이 아니라 하한 단측만 본다.
    """
    q1, q3 = np.percentile(values, [25, 75])
    iqr = q3 - q1
    out = {
        "iqr_fence": float(q1 - 1.5 * iqr),
        "normal_3sigma": float(values.mean() - 3.0 * values.std(ddof=1)),
    }
    for p in PERCENTILE_CANDIDATES:
        out[f"percentile_{p:g}"] = float(np.percentile(values, p))
    return out


def choose(summary: dict, lines: dict) -> tuple[str, float, str]:
    """분포 모양을 보고 어떤 기준을 채택할지 정한다.

    왜도가 작으면 정규 근사와 IQR 이 비슷하게 나오므로 이상치 판정의 표준인
    IQR 울타리를 쓴다. 치우친 분포에서는 IQR 울타리가 데이터 범위 밖으로
    나가 아무것도 걸러내지 못하는 일이 잦아, 분포 가정이 없는 백분위수를 쓴다.
    """
    skewed = abs(summary["skew"]) > SKEW_LIMIT
    fence_useless = lines["iqr_fence"] < summary["min"]

    if skewed or fence_useless:
        key = "percentile_5"
        why = []
        if skewed:
            why.append(f"왜도 {summary['skew']:+.2f} 로 |{SKEW_LIMIT}| 초과 (치우친 분포)")
        if fence_useless:
            why.append(f"IQR 울타리 {lines['iqr_fence']:.4g} 가 최솟값 "
                       f"{summary['min']:.4g} 보다 낮아 아무것도 걸러내지 못함")
        reason = "; ".join(why) + " -> 분포 가정이 없는 하위 5% 백분위수 채택"
    else:
        key = "iqr_fence"
        reason = (f"왜도 {summary['skew']:+.2f} 로 대칭에 가깝고 울타리가 데이터 범위 안 "
                  f"-> 이상치 판정의 표준인 IQR 울타리(Q1-1.5*IQR) 채택")
    return key, lines[key], reason


# ----------------------------------------------------------------------------
# 3. 시각화
# ----------------------------------------------------------------------------
def plot(rows: list[dict], result: dict, path: Path) -> None:
    """히스토그램에 커트라인 후보를 모두 얹어 어디서 잘리는지 눈으로 비교한다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.dpi": 140, "savefig.dpi": 140,
        "font.family": ["Malgun Gothic", "DejaVu Sans"],
        "axes.unicode_minus": False, "font.size": 10,
        "axes.grid": True, "grid.alpha": 0.25,
    })

    specs = [
        ("mean_brightness", "밝기 (HSV V 채널 평균)", "%.1f"),
        ("object_area_ratio", "주요 객체 면적비", "%.3f"),
    ]
    colors = {"iqr_fence": "#d1495b", "normal_3sigma": "#8d99ae",
              "percentile_1": "#adb5bd", "percentile_5": "#00798c"}

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5),
                             gridspec_kw={"width_ratios": [3.2, 1]})

    for row, (key, title, fmt) in enumerate(specs):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        info = result[key]
        ax = axes[row][0]
        ax.hist(values, bins=60, color="#c8d3d5", edgecolor="#8d99ae", linewidth=0.4)
        top = ax.get_ylim()[1]

        for name, value in info["cutlines"].items():
            chosen = name == info["chosen"]
            ax.axvline(value, color=colors.get(name, "#666666"),
                       linestyle="-" if chosen else "--",
                       linewidth=2.2 if chosen else 1.1, zorder=5)
            ax.annotate(f"{name}\n{fmt % value}", xy=(value, top * 0.97),
                        fontsize=8, ha="center", va="top",
                        color=colors.get(name, "#666666"),
                        fontweight="bold" if chosen else "normal")

        ax.axvline(info["old_threshold"], color="#f4a261", linestyle=":", linewidth=2)
        ax.annotate(f"기존(감)\n{fmt % info['old_threshold']}",
                    xy=(info["old_threshold"], top * 0.55),
                    fontsize=8, ha="center", color="#e76f51", fontweight="bold")

        ax.set_title(f"{title}   n={len(values):,}   "
                     f"왜도 {info['summary']['skew']:+.2f}", fontsize=11)
        ax.set_xlabel(title)
        ax.set_ylabel("이미지 수")

        box = axes[row][1]
        box.boxplot(values, widths=0.5,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4})
        box.axhline(info["threshold"], color=colors.get(info["chosen"], "#666666"),
                    linewidth=2)
        box.set_title("상자그림", fontsize=10)
        box.set_xticks([])

    fig.suptitle("이상치 임계값을 분포에서 정한다 — food101 표본", fontsize=13, y=0.985)
    fig.tight_layout(rect=(0, 0, 1, 0.955))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)


# ----------------------------------------------------------------------------
# 4. 엔트리 포인트
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="이상치 임계값 분포 조사")
    p.add_argument("--num-images", type=int, default=1000, help="측정할 이미지 수")
    p.add_argument("--from-cache", action="store_true",
                   help="새로 받지 않고 저장된 측정값으로 계산만 다시 한다")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.from_cache:
        rows = json.loads(MEASURE_PATH.read_text(encoding="utf-8"))["rows"]
        print(f"[info] 캐시에서 {len(rows)}장 측정값 로드")
    else:
        print(f"[info] food101 {args.num_images}장 측정 시작")
        rows = measure_stream(args.num_images)
        MEASURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        MEASURE_PATH.write_text(
            json.dumps({"n": len(rows), "rows": rows}, ensure_ascii=False, indent=1),
            encoding="utf-8")
        print(f"[save] {MEASURE_PATH}")

    result: dict = {}
    for key in ("mean_brightness", "object_area_ratio"):
        values = np.array([r[key] for r in rows], dtype=np.float64)
        summary = describe(values)
        lines = cutlines(values)
        chosen, threshold, reason = choose(summary, lines)
        result[key] = {
            "summary": summary,
            "cutlines": lines,
            "chosen": chosen,
            "threshold": threshold,
            "reason": reason,
            "old_threshold": OLD[key],
            "drop_rate_new": float((values < threshold).mean()),
            "drop_rate_old": float((values < OLD[key]).mean()),
        }

    print()
    for key, info in result.items():
        s = info["summary"]
        print(f"[{key}]  n={s['n']:,}")
        print(f"  최소 {s['min']:.4g} | Q1 {s['q1']:.4g} | 중앙 {s['median']:.4g} | "
              f"Q3 {s['q3']:.4g} | 최대 {s['max']:.4g}")
        print(f"  평균 {s['mean']:.4g} | 표준편차 {s['std']:.4g} | "
              f"왜도 {s['skew']:+.3f} | 첨도 {s['kurtosis']:+.3f}")
        for name, value in info["cutlines"].items():
            mark = " <- 채택" if name == info["chosen"] else ""
            print(f"    {name:16s} {value:10.4g}{mark}")
        print(f"  근거: {info['reason']}")
        print(f"  제외율: 기존 {info['old_threshold']:.4g} -> "
              f"{info['drop_rate_old'] * 100:.2f}%  |  신규 {info['threshold']:.4g} -> "
              f"{info['drop_rate_new'] * 100:.2f}%")
        print()

    THRESHOLD_PATH.write_text(json.dumps({
        "source": {"dataset": DATASET_ID, "split": DATASET_SPLIT,
                   "num_images": len(rows), "shuffle_seed": 42},
        "dark_mean_threshold": result["mean_brightness"]["threshold"],
        "min_object_area_ratio": result["object_area_ratio"]["threshold"],
        "study": result,
    }, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[save] {THRESHOLD_PATH}")

    plot(rows, result, Path("outputs/threshold_study.png"))
    print("[save] outputs/threshold_study.png")


if __name__ == "__main__":
    main()
