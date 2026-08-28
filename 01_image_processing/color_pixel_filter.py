"""
1차 업무 [요청내용 2] - OpenCV 픽셀 단위 이미지 처리

  - cv2.imread()   : 이미지 로드 및 픽셀 값 분석
  - cv2.cvtColor() : BGR -> HSV / GRAY 색공간 변환
  - cv2.threshold(): 이진화를 통한 픽셀 필터링
  - cv2.inRange()  : 특정 색상 범위의 픽셀 감지

빨간색을 기본 대상으로 하되, --color 로 다른 색도 지정할 수 있게 일반화했습니다.
과제 예제는 cv2.imshow() 로 화면 출력하지만, 여기서는 결과를 파일로 저장해
CI/원격 환경에서도 재현되도록 했습니다(--show 옵션으로 창 출력도 가능).

사용 예)
    py -3 color_pixel_filter.py --image data/sample.jpg --color red
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

# HSV 색상 범위 정의. OpenCV 의 H 는 0~179 범위임에 주의.
# 빨간색은 H=0 을 기준으로 양쪽 끝에 걸쳐 있어 범위를 두 개로 나눠야 합니다.
#
# red 는 과제 예제의 값을 그대로 유지합니다.
# red-orange 는 위성 염전 이미지의 화소 값을 직접 재서 정한 범위입니다.
#   염전 화소 실측: H 7~11, S 96~135, V 76~223
#   -> 색상(H)이 아니라 채도(S) 하한 120 에서 절반이 잘려나가고 있었습니다.
COLOR_RANGES: dict[str, list[tuple[tuple[int, int, int], tuple[int, int, int]]]] = {
    "red":        [((0, 120, 70), (10, 255, 255)), ((170, 120, 70), (180, 255, 255))],
    "red-orange": [((0, 100, 70), (15, 255, 255))],
    "green":      [((35, 80, 60), (85, 255, 255))],
    "blue":       [((90, 80, 60), (130, 255, 255))],
    "yellow":     [((20, 100, 100), (35, 255, 255))],
}

BINARY_THRESHOLD = 127  # cv2.threshold 고정 임계값
JPEG_QUALITY = 92       # 사진 계열 결과물 저장 품질


def load_image(path: Path) -> np.ndarray:
    """이미지를 로드합니다. cv2.imread 는 실패해도 예외 없이 None 을 돌려주므로 직접 검사합니다."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {path}")
    return image


def analyze_pixels(image: np.ndarray) -> dict[str, object]:
    """픽셀 값의 기초 통계를 계산합니다(채널 순서는 OpenCV 규약대로 B, G, R)."""
    b, g, r = cv2.split(image)
    return {
        "shape": image.shape,                  # (height, width, channels)
        "dtype": str(image.dtype),
        "total_pixels": int(image.shape[0] * image.shape[1]),
        "mean_bgr": [round(float(c.mean()), 2) for c in (b, g, r)],
        "std_bgr": [round(float(c.std()), 2) for c in (b, g, r)],
        "min_bgr": [int(c.min()) for c in (b, g, r)],
        "max_bgr": [int(c.max()) for c in (b, g, r)],
    }


def build_color_mask(image: np.ndarray, color: str) -> np.ndarray:
    """지정한 색상에 해당하는 픽셀만 255 로 표시된 마스크를 만듭니다."""
    if color not in COLOR_RANGES:
        raise ValueError(f"지원하지 않는 색상: {color} (가능: {', '.join(COLOR_RANGES)})")

    # 조명 변화에 강인하도록 RGB 가 아닌 HSV 공간에서 색을 판별합니다.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in COLOR_RANGES[color]:
        part = cv2.inRange(hsv, np.array(lower, np.uint8), np.array(upper, np.uint8))
        # 과제 예제의 `mask = mask1 + mask2` 형태를 그대로 따릅니다.
        # uint8 덧셈이라 두 범위에 모두 걸린 화소는 255+255 가 254 로 넘치는데,
        # 사용한 이미지에서 겹친 화소가 0 개임을 확인했습니다. red 는 두 구간이 H 양끝에
        # 떨어져 있고 red-orange 는 구간이 하나뿐이라 겹칠 여지가 없습니다.
        # 구간이 겹치는 색을 추가한다면 cv2.bitwise_or 로 바꿔야 합니다.
        mask = mask + part

    # 점 노이즈 제거 후 구멍 메우기
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask


def build_binary_map(image: np.ndarray) -> np.ndarray:
    """cv2.threshold 를 이용한 밝기 기준 이진화 결과(픽셀 단위 필터링의 기본형)."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, BINARY_THRESHOLD, 255, cv2.THRESH_BINARY)
    return binary


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """마스크에 해당하는 픽셀만 원본에서 남기고 나머지는 검게 만듭니다."""
    return cv2.bitwise_and(image, image, mask=mask)


def annotate_regions(image: np.ndarray, mask: np.ndarray, min_area: int = 200) -> np.ndarray:
    """검출된 색상 영역에 경계 상자를 그려 어디가 잡혔는지 눈으로 확인할 수 있게 합니다."""
    annotated = image.copy()
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue  # 잡음 수준의 작은 영역은 무시
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return annotated


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="특정 색상 픽셀 감지 및 필터링")
    parser.add_argument("--image", type=Path, required=True, help="입력 이미지 경로")
    parser.add_argument("--color", default="red", choices=sorted(COLOR_RANGES),
                        help="검출할 색상 (기본: red)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--show", action="store_true", help="cv2.imshow 로 창 출력")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    image = load_image(args.image)

    # --- 픽셀 값 분석 ------------------------------------------------------
    stats = analyze_pixels(image)
    print("[픽셀 분석]")
    for key, value in stats.items():
        print(f"  {key:14s}: {value}")

    # --- 색상 감지 및 필터링 ----------------------------------------------
    mask = build_color_mask(image, args.color)
    filtered = apply_mask(image, mask)
    binary = build_binary_map(image)
    annotated = annotate_regions(image, mask)

    detected = int(np.count_nonzero(mask))
    ratio = detected / stats["total_pixels"] * 100
    print(f"\n[{args.color} 검출] {detected:,} px / {stats['total_pixels']:,} px ({ratio:.2f}%)")

    # 사진 계열은 JPEG, 마스크·이진화는 PNG 로 저장합니다.
    # 사진을 PNG 무손실로 담으면 원본 JPEG 대비 5배 가까이 커지고, 반대로 마스크를
    # JPEG 로 담으면 경계에 압축 잡음이 생겨 값이 0/255 로 유지되지 않습니다.
    stem = args.image.stem
    results = [
        (f"{stem}_1_original.jpg", image),
        (f"{stem}_2_mask_{args.color}.png", mask),
        (f"{stem}_3_filtered_{args.color}.jpg", filtered),
        (f"{stem}_4_threshold.png", binary),
        (f"{stem}_5_annotated.jpg", annotated),
    ]
    for filename, result in results:
        params = [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY] if filename.endswith(".jpg") else []
        cv2.imwrite(str(args.out_dir / filename), result, params)
        print(f"[save] {args.out_dir / filename}")

    if args.show:
        cv2.imshow("Original", image)
        cv2.imshow(f"{args.color} Filtered", filtered)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
