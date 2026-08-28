"""
1차 업무 [피드백 반영 2] - 정답 마스크가 없을 때 검출 품질을 표본조사로 추정합니다.

기존 README 는 "정답 마스크가 없으므로 이 수치가 얼마나 정확한지 판정할 수 없습니다"
로 끝났습니다. 판정할 수 없는 것이 아니라, 전수 채점을 못 할 뿐입니다. 표본을 뽑아
사람이 판정하면 전체 정확도를 구간으로 추정할 수 있습니다.

대상이 이미지 1장(2131x1421)이므로 "무작위 100장"을 "무작위 100개 패치"로 옮깁니다.

  1. 층화 표본  검출된 화소에서 50개, 검출 안 된 화소에서 50개를 뽑습니다.
                전체에서 균등 추출하면 염전이 화면의 0.88% 뿐이라 100개 중
                한 개 남짓만 양성이라 놓친 화소를 찾을 수가 없습니다.
  2. 판정        패치 격자 이미지를 보고 CSV 의 verdict 열에 o / x 를 적습니다.
  3. 추정        정밀도와 놓침률을 윌슨 신뢰구간과 함께 계산합니다.

  --strategy uncertain 은 능동학습(active learning) 방식입니다. HSV 범위 경계에
  가장 가까운, 즉 조금만 기준이 달랐어도 판정이 뒤집혔을 화소를 우선 뽑습니다.
  같은 100개를 판정해도 기준선을 어디에 그어야 할지에 대해 더 많이 알게 됩니다.

사용 예)
    py -3 detection_audit.py --image data/saltpond_aiguesmortes_landsat8.jpg \\
        --color red-orange --strategy uncertain
    # -> outputs/audit/ 의 격자 이미지를 보고 CSV 를 채운 뒤
    py -3 detection_audit.py --score outputs/audit/audit_red-orange_uncertain.csv
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from color_pixel_filter import COLOR_RANGES, build_color_mask, load_image

PATCH = 64          # 잘라낼 패치 한 변 (원본 화소)
TILE = 96           # 격자에 그릴 때의 한 변
COLUMNS = 10
DEFAULT_N = 100


# ----------------------------------------------------------------------------
# 1. 확신도 - 판정이 얼마나 아슬아슬한가
# ----------------------------------------------------------------------------
def hsv_margin(image: np.ndarray, color: str) -> np.ndarray:
    """화소마다 HSV 범위 경계까지의 여유를 부호 있는 값으로 계산합니다.

    양수면 범위 안쪽(검출됨)이고 값이 클수록 확실합니다. 음수면 바깥쪽이고
    절댓값이 클수록 확실히 아닙니다. 0 근처가 판정이 아슬아슬한 화소입니다.

    채널마다 단위가 달라(H 0~179, S/V 0~255) 각 범위의 절반 폭으로 나눠
    무차원으로 만든 뒤 최솟값을 취합니다. 세 조건을 모두 만족해야 검출이므로
    가장 빠듯한 채널이 그 화소의 여유가 됩니다.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    best = np.full(hsv.shape[:2], -np.inf, dtype=np.float32)

    for lower, upper in COLOR_RANGES[color]:
        lo = np.array(lower, dtype=np.float32)
        hi = np.array(upper, dtype=np.float32)
        half = np.maximum((hi - lo) / 2.0, 1e-6)
        center = (lo + hi) / 2.0
        # 중심에서 떨어진 거리를 반폭으로 정규화하면 1 이 경계입니다.
        normalized = np.abs(hsv - center) / half
        margin = 1.0 - normalized.max(axis=2)   # 가장 빠듯한 채널 기준
        best = np.maximum(best, margin)

    return best


# ----------------------------------------------------------------------------
# 2. 표본 추출
# ----------------------------------------------------------------------------
def sample_points(mask: np.ndarray, margin: np.ndarray, count: int,
                  strategy: str, rng: np.random.Generator) -> list[dict]:
    """검출/미검출 두 층에서 절반씩 뽑습니다.

    strategy
      random     각 층에서 균등 추출. 정밀도의 불편추정치를 얻는 정공법입니다.
      uncertain  각 층에서 경계에 가까운 화소를 우선 추출(능동학습). 추정값은
                 전체를 대표하지 않지만, 기준선을 어디로 옮겨야 할지 알려줍니다.
    """
    half = PATCH // 2
    h, w = mask.shape
    # 패치를 온전히 잘라낼 수 있는 안쪽 영역만 후보로 둡니다.
    valid = np.zeros_like(mask, dtype=bool)
    valid[half:h - half, half:w - half] = True

    strata = {
        "detected": (mask > 0) & valid,
        "missed_candidate": (mask == 0) & valid,
    }
    per_stratum = count // len(strata)

    points: list[dict] = []
    for name, area in strata.items():
        ys, xs = np.nonzero(area)
        if len(ys) == 0:
            continue
        take = min(per_stratum, len(ys))

        if strategy == "uncertain":
            # 경계에 가까운 순서. 검출 층은 겨우 통과한 화소, 미검출 층은
            # 아깝게 탈락한 화소가 앞으로 옵니다.
            order = np.argsort(np.abs(margin[ys, xs]))
            # 상위 20배수 안에서 무작위로 뽑아 한 덩어리에 몰리지 않게 합니다.
            pool = order[:min(take * 20, len(order))]
            picked = rng.choice(pool, size=take, replace=False)
        else:
            picked = rng.choice(len(ys), size=take, replace=False)

        for rank, i in enumerate(picked):
            y, x = int(ys[i]), int(xs[i])
            points.append({
                "id": len(points),
                "stratum": name,
                "x": x,
                "y": y,
                "margin": float(margin[y, x]),
                "in_mask": int(mask[y, x] > 0),
            })

    rng.shuffle(points)          # 판정할 때 층을 모르게 섞는다 (판정 편향 방지)
    for new_id, p in enumerate(points):
        p["id"] = new_id
    return points


# ----------------------------------------------------------------------------
# 3. 판정 시트
# ----------------------------------------------------------------------------
def build_sheet(image: np.ndarray, points: list[dict]) -> np.ndarray:
    """패치를 번호와 함께 격자로 붙입니다. 가운데 십자가 판정 대상 화소입니다."""
    half = PATCH // 2
    tiles = []
    for p in points:
        crop = image[p["y"] - half:p["y"] + half, p["x"] - half:p["x"] + half]
        tile = cv2.resize(crop, (TILE, TILE), interpolation=cv2.INTER_NEAREST)

        # 판정 대상은 패치 전체가 아니라 정중앙 화소 하나다. 십자로 짚어 줍니다.
        c = TILE // 2
        cv2.line(tile, (c - 9, c), (c - 3, c), (255, 255, 255), 1)
        cv2.line(tile, (c + 3, c), (c + 9, c), (255, 255, 255), 1)
        cv2.line(tile, (c, c - 9), (c, c - 3), (255, 255, 255), 1)
        cv2.line(tile, (c, c + 3), (c, c + 9), (255, 255, 255), 1)

        cv2.rectangle(tile, (0, 0), (TILE - 1, 13), (0, 0, 0), -1)
        cv2.putText(tile, f"{p['id']:02d}", (2, 11),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.36, (255, 255, 255), 1)
        # 층을 색으로 흘리지 않기 위해 테두리는 모두 같은 회색으로 둡니다.
        cv2.rectangle(tile, (0, 0), (TILE - 1, TILE - 1), (110, 110, 110), 1)
        tiles.append(tile)

    while len(tiles) % COLUMNS:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[i:i + COLUMNS]) for i in range(0, len(tiles), COLUMNS)]
    return np.vstack(rows)


def write_csv(points: list[dict], path: Path, strategy: str) -> None:
    """판정 결과를 적어 넣을 빈 시트.

    verdict 에는 '검출이 맞았는가'가 아니라 '이 화소가 실제로 대상인가'를 적습니다
    (o = 염전이다, x = 아닙니다). 판정자가 검출 결과를 모르는 채로 봐야 편향이
    안 생기므로, 층 정보를 몰라도 답할 수 있는 질문이어야 합니다.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=["id", "strategy", "stratum", "x", "y", "margin",
                           "in_mask", "verdict", "note"])
        writer.writeheader()
        for p in points:
            writer.writerow({**p, "strategy": strategy, "verdict": "", "note": ""})


# ----------------------------------------------------------------------------
# 4. 채점
# ----------------------------------------------------------------------------
def wilson(successes: int, total: int, z: float = 1.96) -> tuple[float, float, float]:
    """윌슨 점수 구간. 표본이 작거나 비율이 0/1 에 붙어도 무너지지 않습니다.

    단순한 정규 근사(p +- z*sqrt(p(1-p)/n))는 100개 표본에서 구간이 0 아래로
    내려가거나 1 을 넘는 일이 흔해 쓸 수 없습니다.
    """
    if total == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / total
    denom = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denom
    spread = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denom
    return p, max(0.0, center - spread), min(1.0, center + spread)


def score(path: Path) -> None:
    """사람이 채운 CSV 를 읽어 정밀도와 놓침률을 구간추정합니다."""
    with path.open(encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f)]

    judged = [r for r in rows if r["verdict"].strip().lower() in ("o", "x")]
    strategy = rows[0].get("strategy", "random") if rows else "random"
    print(f"[채점] {path}  (표본 전략: {strategy})")
    print(f"  전체 {len(rows)}개 중 판정 완료 {len(judged)}개")
    if strategy == "uncertain":
        print("  주의: 경계 화소만 골라 뽑은 표본이라 아래 비율은 전체 화면의 값이")
        print("        아닙니다. '기준선 부근이 얼마나 위태로운가'를 재는 수치입니다.")
    if not judged:
        print("  verdict 열이 비어 있습니다. 격자 이미지를 보고 o / x 를 채워 주세요.")
        return

    for stratum, label, meaning in (
        ("detected", "정밀도", "검출한 화소가 실제로 대상일 확률"),
        ("missed_candidate", "놓침률", "검출 안 한 화소가 사실은 대상일 확률"),
    ):
        subset = [r for r in judged if r["stratum"] == stratum]
        if not subset:
            continue
        hits = sum(1 for r in subset if r["verdict"].strip().lower() == "o")
        p, lo, hi = wilson(hits, len(subset))
        print(f"  {label:5s} {p * 100:5.1f}%  (95% 신뢰구간 {lo * 100:.1f} ~ {hi * 100:.1f}%)"
              f"  n={len(subset)}  — {meaning}")

    # 경계 근처에서 판정이 갈리는 지점을 찾으면 임계값을 어디로 옮길지 보입니다.
    wrong = [float(r["margin"]) for r in judged
             if r["stratum"] == "detected" and r["verdict"].strip().lower() == "x"]
    right = [float(r["margin"]) for r in judged
             if r["stratum"] == "detected" and r["verdict"].strip().lower() == "o"]
    if wrong and right and strategy == "uncertain":
        # 표본 자체를 여유값으로 골랐으므로 여유의 대소를 비교해도 의미가 없습니다.
        print("  (여유값 비교는 random 표본에서만 뜻이 있습니다)")
    elif wrong and right:
        mw, mr = float(np.median(wrong)), float(np.median(right))
        print(f"  오검출 화소의 여유 중앙값 {mw:+.3f} / 정검출 {mr:+.3f}")
        if mw < mr:
            print("   -> 오검출이 경계 근처에 몰려 있습니다. 범위를 좁히면 정밀도가 오릅니다.")
        else:
            print("   -> 오검출이 경계가 아니라 범위 한가운데에 있습니다. 범위를 좁혀도")
            print("      정검출이 함께 잘려 나가므로, HSV 범위 조정으로는 못 고칩니다.")


# ----------------------------------------------------------------------------
# 5. 엔트리 포인트
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="검출 품질 표본조사")
    p.add_argument("--image", type=Path)
    p.add_argument("--color", default="red-orange", choices=sorted(COLOR_RANGES))
    p.add_argument("--strategy", default="random", choices=["random", "uncertain"])
    p.add_argument("--count", type=int, default=DEFAULT_N)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out-dir", type=Path, default=Path("outputs/audit"))
    p.add_argument("--score", type=Path, help="채워진 CSV 를 채점만 합니다")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    if args.score:
        score(args.score)
        return
    if not args.image:
        raise SystemExit("--image 또는 --score 중 하나가 필요합니다.")

    image = load_image(args.image)
    mask = build_color_mask(image, args.color)
    margin = hsv_margin(image, args.color)
    rng = np.random.default_rng(args.seed)

    detected = int(np.count_nonzero(mask))
    print(f"[info] {args.image.name}  {image.shape[1]}x{image.shape[0]}")
    print(f"       {args.color} 검출 {detected:,} px "
          f"({detected / mask.size * 100:.2f}%)")
    print(f"       표본 추출 전략: {args.strategy}")

    points = sample_points(mask, margin, args.count, args.strategy, rng)
    counts: dict[str, int] = {}
    for p in points:
        counts[p["stratum"]] = counts.get(p["stratum"], 0) + 1
    print(f"       층별 표본 수: {counts}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"audit_{args.color}_{args.strategy}"
    # 판정 시트는 위성 사진을 잘라 붙인 것이라 사진 계열입니다. 저장소의 다른
    # 결과물과 같은 규칙으로 JPEG 에 담는다 (무손실 PNG 로는 6배 커집니다).
    sheet_path = args.out_dir / f"{stem}.jpg"
    csv_path = args.out_dir / f"{stem}.csv"

    cv2.imwrite(str(sheet_path), build_sheet(image, points),
                [cv2.IMWRITE_JPEG_QUALITY, 92])
    write_csv(points, csv_path, args.strategy)
    print(f"[save] {sheet_path}  (판정용 격자, 가운데 십자가 대상 화소)")
    print(f"[save] {csv_path}  (verdict 열: 가운데 화소가 염전이면 o, 아니면 x)")
    print(f"\n채운 뒤:  py -3 detection_audit.py --score {csv_path}")


if __name__ == "__main__":
    main()
