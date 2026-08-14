"""
1차 업무 [추가 요청] - Hugging Face 데이터셋 기반 AI 학습용 이미지 전처리 파이프라인

[기본 문제]
  - 크기 조정 (224x224)
  - 색상 변환 (Grayscale & Normalize)
  - 노이즈 제거 (Blur 필터)
  - 데이터 증강 (좌우 반전, 회전, 색상 변화)

[심화 문제] 이상치(outlier) 탐지 및 필터링
  - 평균 밝기가 너무 낮은(어두운) 이미지 제거
  - 주요 객체 크기가 너무 작은 이미지 제거

데이터셋: https://huggingface.co/datasets/ethz/food101 (streaming 모드로 필요한 만큼만 수신)

사용 예)
    py -3 image_preprocessing.py --num-images 20 --num-samples 5
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# 설정값 (실무에서는 config.yaml 로 분리하는 부분)
# ----------------------------------------------------------------------------
TARGET_SIZE = (224, 224)          # 크기 조정 목표 해상도
BLUR_KERNEL = (5, 5)              # 가우시안 블러 커널
BLUR_SIGMA = 0                    # 0 이면 커널 크기로부터 자동 계산

DARK_MEAN_THRESHOLD = 40.0        # 평균 밝기(0~255) 하한. 미만이면 '너무 어두움'
MIN_OBJECT_AREA_RATIO = 0.10      # 주요 객체가 전체 화면에서 차지해야 할 최소 면적 비율

ROTATE_DEGREES = 15.0             # 증강 - 회전 각도
BRIGHTNESS_GAIN = 1.25            # 증강 - 색상(밝기) 변화 배율
SATURATION_GAIN = 1.40            # 증강 - 색상(채도) 변화 배율

DATASET_ID = "ethz/food101"
DATASET_SPLIT = "train"


# ----------------------------------------------------------------------------
# 1. 데이터 수집
# ----------------------------------------------------------------------------
def load_images_from_hf(num_images: int, cache_dir: Path) -> list[tuple[str, np.ndarray]]:
    """Hugging Face food101 데이터셋을 streaming 으로 받아 BGR ndarray 로 반환한다.

    streaming=True 를 쓰는 이유: food101 전체는 5GB 이상이라 전량 다운로드가 불필요하다.
    받은 원본은 cache_dir 에 저장해 재실행 시 네트워크 없이도 동작하게 한다.
    """
    from datasets import load_dataset

    cache_dir.mkdir(parents=True, exist_ok=True)
    stream = load_dataset(DATASET_ID, split=DATASET_SPLIT, streaming=True)
    # food101 은 클래스 순으로 정렬돼 있어 앞에서부터 그냥 받으면 전부 같은 음식이 나온다.
    # 여러 클래스가 섞이도록 버퍼 셔플을 건다(seed 고정으로 재현성 확보).
    stream = stream.shuffle(seed=42, buffer_size=1000)

    images: list[tuple[str, np.ndarray]] = []
    for idx, record in enumerate(stream):
        if idx >= num_images:
            break
        # datasets 는 PIL.Image(RGB) 로 디코딩해 주므로 OpenCV 규약(BGR)으로 변환한다.
        rgb = np.array(record["image"].convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        name = f"food101_{idx:03d}_label{record['label']}"
        cv2.imwrite(str(cache_dir / f"{name}.jpg"), bgr)
        images.append((name, bgr))
    return images


def load_images_from_disk(num_images: int, cache_dir: Path) -> list[tuple[str, np.ndarray]]:
    """네트워크가 없을 때 캐시된 원본 이미지를 대신 사용한다."""
    paths = sorted(p for p in cache_dir.glob("*.jpg"))[:num_images]
    out: list[tuple[str, np.ndarray]] = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            out.append((p.stem, img))
    return out


# ----------------------------------------------------------------------------
# 2. 이상치 탐지 (심화 문제)
# ----------------------------------------------------------------------------
@dataclass
class QualityReport:
    """이미지 1장에 대한 품질 검사 결과."""
    name: str
    mean_brightness: float
    object_area_ratio: float
    is_too_dark: bool
    has_tiny_object: bool

    @property
    def is_outlier(self) -> bool:
        return self.is_too_dark or self.has_tiny_object

    @property
    def reason(self) -> str:
        reasons = []
        if self.is_too_dark:
            reasons.append(f"too_dark(mean={self.mean_brightness:.1f}<{DARK_MEAN_THRESHOLD})")
        if self.has_tiny_object:
            reasons.append(
                f"tiny_object(ratio={self.object_area_ratio:.3f}<{MIN_OBJECT_AREA_RATIO})"
            )
        return ", ".join(reasons) if reasons else "ok"


def measure_mean_brightness(image: np.ndarray) -> float:
    """HSV 의 V(명도) 채널 평균으로 밝기를 측정한다.

    BGR 단순 평균 대신 V 채널을 쓰는 이유: 색상/채도의 영향을 배제하고
    '얼마나 밝은가'만 보기 위함이다.
    """
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 2].mean())


def measure_object_area_ratio(image: np.ndarray) -> float:
    """주요 객체가 화면에서 차지하는 면적 비율(0~1)을 추정한다.

    Otsu 이진화로 전경/배경을 자동 분리한 뒤 모폴로지로 잡음을 정리하고,
    가장 큰 윤곽선의 면적을 전체 픽셀 수로 나눈다.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, BLUR_KERNEL, BLUR_SIGMA)

    # cv2.threshold + Otsu: 임계값을 데이터로부터 자동 결정
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 전경이 어두운 경우(배경이 밝은 경우)에도 대응하도록 다수 픽셀을 배경으로 간주해 반전
    if np.count_nonzero(binary) > binary.size / 2:
        binary = cv2.bitwise_not(binary)

    kernel = np.ones((5, 5), np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    largest = max(cv2.contourArea(c) for c in contours)
    return float(largest / (image.shape[0] * image.shape[1]))


def inspect(name: str, image: np.ndarray) -> QualityReport:
    mean_brightness = measure_mean_brightness(image)
    area_ratio = measure_object_area_ratio(image)
    return QualityReport(
        name=name,
        mean_brightness=mean_brightness,
        object_area_ratio=area_ratio,
        is_too_dark=mean_brightness < DARK_MEAN_THRESHOLD,
        has_tiny_object=area_ratio < MIN_OBJECT_AREA_RATIO,
    )


# ----------------------------------------------------------------------------
# 3. 전처리 단계 (기본 문제)
# ----------------------------------------------------------------------------
def resize_image(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """224x224 로 크기 조정. 축소가 대부분이므로 INTER_AREA 가 화질 손실이 가장 적다."""
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def normalize(image: np.ndarray) -> np.ndarray:
    """픽셀 값을 0~1 실수 범위로 정규화한다(학습 시 gradient 안정화 목적)."""
    return image.astype(np.float32) / 255.0


def denoise(image: np.ndarray) -> np.ndarray:
    """가우시안 블러로 센서 노이즈/압축 잡음을 완화한다."""
    return cv2.GaussianBlur(image, BLUR_KERNEL, BLUR_SIGMA)


def augment_flip(image: np.ndarray) -> np.ndarray:
    """좌우 반전 (flipCode=1)."""
    return cv2.flip(image, 1)


def augment_rotate(image: np.ndarray, degrees: float = ROTATE_DEGREES) -> np.ndarray:
    """중심 기준 회전. 빈 영역은 경계 픽셀 복제로 채워 검은 테두리를 방지한다."""
    h, w = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)


def augment_color(image: np.ndarray) -> np.ndarray:
    """HSV 공간에서 채도(S)와 명도(V)를 조정해 조명/카메라 차이를 흉내 낸다."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * SATURATION_GAIN, 0, 255)
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * BRIGHTNESS_GAIN, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)


def preprocess(image: np.ndarray) -> dict[str, np.ndarray]:
    """전체 전처리 파이프라인. 각 단계 결과를 dict 로 반환한다(시각화/검증용).

    순서: 크기 조정 -> 블러(노이즈 제거) -> 그레이스케일 -> 정규화
    블러를 크기 조정 뒤에 두는 이유: 커널 크기가 항상 동일한 화소 비율로 동작하게 하기 위함.
    """
    resized = resize_image(image)
    blurred = denoise(resized)
    gray = to_grayscale(blurred)
    normalized = normalize(gray)  # float32, 0.0~1.0 -> 모델 입력 텐서

    return {
        "01_resized": resized,
        "02_blurred": blurred,
        "03_gray": gray,
        "04_normalized": normalized,
        "05_aug_flip": augment_flip(resized),
        "06_aug_rotate": augment_rotate(resized),
        "07_aug_color": augment_color(resized),
    }


# ----------------------------------------------------------------------------
# 4. 결과 저장 / 시각화
# ----------------------------------------------------------------------------
def to_bgr_u8(image: np.ndarray) -> np.ndarray:
    """float 정규화 결과나 grayscale 을 저장 가능한 3채널 uint8 로 되돌린다."""
    if image.dtype != np.uint8:
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.ndim == 2:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    return image


def build_contact_sheet(stages: dict[str, np.ndarray], columns: int = 4) -> np.ndarray:
    """전처리 단계별 결과를 격자 이미지 한 장으로 합쳐 비교하기 쉽게 만든다."""
    tiles = []
    for label, img in stages.items():
        tile = to_bgr_u8(img).copy()
        # 라벨 가독성을 위해 상단에 반투명 띠를 깔고 텍스트를 올린다.
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), thickness=-1)
        cv2.putText(tile, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(tile)

    while len(tiles) % columns != 0:
        tiles.append(np.zeros_like(tiles[0]))

    rows = [np.hstack(tiles[i:i + columns]) for i in range(0, len(tiles), columns)]
    return np.vstack(rows)


# ----------------------------------------------------------------------------
# 5. 엔트리 포인트
# ----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="food101 이미지 전처리 파이프라인")
    parser.add_argument("--num-images", type=int, default=20, help="수집할 원본 이미지 수")
    parser.add_argument("--num-samples", type=int, default=5, help="저장할 결과 샘플 수")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="원본 캐시 경로")
    parser.add_argument("--out-dir", type=Path, default=Path("preprocessed_samples"),
                        help="제출 항목: 전처리를 마친 이미지 저장 경로")
    parser.add_argument("--preview-dir", type=Path, default=Path("outputs/preview"),
                        help="참고용: 단계별 비교 격자와 학습 입력 텐서 저장 경로")
    parser.add_argument("--offline", action="store_true", help="캐시된 원본만 사용")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # --- 1) 수집 ---------------------------------------------------------
    if args.offline:
        images = load_images_from_disk(args.num_images, args.data_dir)
    else:
        try:
            images = load_images_from_hf(args.num_images, args.data_dir)
        except Exception as exc:  # 네트워크/인증 실패 시 캐시로 폴백
            print(f"[warn] Hugging Face 수신 실패({exc.__class__.__name__}: {exc}) -> 캐시 사용")
            images = load_images_from_disk(args.num_images, args.data_dir)

    if not images:
        raise SystemExit("처리할 이미지가 없습니다. 네트워크 상태 또는 --data-dir 을 확인하세요.")
    print(f"[info] 원본 이미지 {len(images)}장 확보")

    # --- 2) 이상치 탐지 & 필터링 -----------------------------------------
    reports = [inspect(name, img) for name, img in images]
    kept = [(name, img) for (name, img), r in zip(images, reports) if not r.is_outlier]
    dropped = [r for r in reports if r.is_outlier]

    print(f"[info] 이상치 {len(dropped)}장 제외 / 정상 {len(kept)}장")
    for r in dropped:
        print(f"       - {r.name}: {r.reason}")

    if not kept:
        raise SystemExit("모든 이미지가 이상치로 판정되었습니다. 임계값을 재검토하세요.")

    # --- 3) 전처리 & 저장 -------------------------------------------------
    args.preview_dir.mkdir(parents=True, exist_ok=True)

    saved = []
    for name, img in kept[:args.num_samples]:
        stages = preprocess(img)
        norm = stages["04_normalized"]

        # 제출 항목: 전처리를 마친 이미지 그 자체 (224x224 그레이스케일)
        # 정규화 결과는 0~1 float 이므로 저장 가능한 0~255 uint8 로 되돌린다.
        processed_path = args.out_dir / f"{name}.png"
        cv2.imwrite(str(processed_path), np.clip(norm * 255.0, 0, 255).astype(np.uint8))

        # 참고용: 단계별 비교 격자(증강 포함)와 실제 학습 입력이 되는 정규화 텐서
        cv2.imwrite(str(args.preview_dir / f"{name}_stages.png"), build_contact_sheet(stages))
        np.save(args.preview_dir / f"{name}_normalized.npy", norm)

        saved.append(processed_path.name)
        print(
            f"[save] {processed_path.name} | shape={norm.shape} dtype={norm.dtype} "
            f"range=[{norm.min():.3f}, {norm.max():.3f}]"
        )

    # --- 4) 리포트 --------------------------------------------------------
    report_path = args.out_dir / "quality_report.json"
    report_path.write_text(
        json.dumps(
            {
                "config": {
                    "target_size": list(TARGET_SIZE),
                    "dark_mean_threshold": DARK_MEAN_THRESHOLD,
                    "min_object_area_ratio": MIN_OBJECT_AREA_RATIO,
                },
                "total": len(reports),
                "kept": len(kept),
                "dropped": len(dropped),
                "saved_samples": saved,
                "details": [{**asdict(r), "outlier": r.is_outlier, "reason": r.reason}
                            for r in reports],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[done] 리포트: {report_path}")


if __name__ == "__main__":
    main()
