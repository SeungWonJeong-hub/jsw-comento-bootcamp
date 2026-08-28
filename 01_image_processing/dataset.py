"""
1차 업무 [피드백 반영 3] - 증강을 미리 저장하지 않고 학습 중 실시간으로 겁니다.

기존 파이프라인은 증강 결과를 파일로 만들어 두었습니다. 그러면 두 가지가 손해입니다.

  1. 저장 공간 — 증강 종류만큼 데이터셋이 배로 불어납니다.
  2. 다양성   — 한 이미지가 평생 정해진 몇 장으로 고정됩니다. 에폭을 아무리 돌려도
               모델이 보는 변형은 저장해 둔 그 몇 장뿐입니다.

실시간 증강은 __getitem__ 이 호출될 때마다 난수를 새로 뽑으므로, 같은 이미지라도
에폭마다 다른 변형이 나옵니다. 저장 공간은 0 이고 다양성은 사실상 무제한입니다.
그 대신 CPU 연산이 늘어나므로 DataLoader 의 num_workers 로 학습과 겹쳐 돌립니다.

  구조:  경로 목록 -> __getitem__ 에서 [로드 -> 리사이즈 -> 무작위 증강 ->
         블러 -> 그레이스케일 -> 정규화] -> (1, 224, 224) float32 텐서

사용 예)
    py -3 dataset.py --num-images 40           # 실시간 증강 시연 + 용량 비교
    py -3 dataset.py --num-images 40 --workers 4
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from image_preprocessing import (
    BLUR_KERNEL,
    BLUR_SIGMA,
    TARGET_SIZE,
    inspect,
    load_thresholds,
)

try:
    import torch
    from torch.utils.data import DataLoader, Dataset
except ImportError as exc:  # pragma: no cover - 안내용
    raise SystemExit(
        "PyTorch 가 필요합니다. `pip install torch` 후 다시 실행해 주세요.\n"
        f"(원인: {exc})"
    )


# ----------------------------------------------------------------------------
# 증강 설정
# ----------------------------------------------------------------------------
@dataclass
class AugmentConfig:
    """무작위 증강의 범위와 발동 확률.

    기존 코드는 회전 15도, 채도 1.4배, 명도 1.25배를 고정값으로 썼습니다. 실시간
    증강에서는 고정값을 쓸 이유가 없으므로 구간에서 매번 뽑습니다. 고정값이면
    '15도 돌아간 사진'만 배우지만, 구간이면 그 사이 모든 각도를 배웁니다.
    """
    flip_prob: float = 0.5
    rotate_prob: float = 0.5
    rotate_range: tuple[float, float] = (-15.0, 15.0)
    color_prob: float = 0.5
    saturation_range: tuple[float, float] = (0.8, 1.4)
    brightness_range: tuple[float, float] = (0.8, 1.25)


def random_augment(image: np.ndarray, cfg: AugmentConfig,
                   rng: np.random.Generator) -> np.ndarray:
    """컬러 BGR 이미지에 무작위 증강을 적용합니다.

    난수원을 인자로 받는 이유: DataLoader 워커마다 독립적인 난수열을 주기
    위해서입니다. 전역 난수를 쓰면 워커가 fork/spawn 될 때 같은 시드를 물려받아
    모든 워커가 똑같은 증강을 내놓는, 눈에 잘 안 띄는 버그가 생깁니다.
    """
    out = image
    if rng.random() < cfg.flip_prob:
        out = cv2.flip(out, 1)

    if rng.random() < cfg.rotate_prob:
        degrees = float(rng.uniform(*cfg.rotate_range))
        h, w = out.shape[:2]
        matrix = cv2.getRotationMatrix2D((w / 2, h / 2), degrees, 1.0)
        # 빈 구석을 검게 두면 모델이 '검은 삼각형'을 특징으로 배웁니다.
        out = cv2.warpAffine(out, matrix, (w, h), borderMode=cv2.BORDER_REPLICATE)

    if rng.random() < cfg.color_prob:
        hsv = cv2.cvtColor(out, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * rng.uniform(*cfg.saturation_range), 0, 255)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * rng.uniform(*cfg.brightness_range), 0, 255)
        out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return out


# ----------------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------------
class Food101Dataset(Dataset):
    """경로 목록을 받아 학습 입력 텐서를 그때그때 만들어 내는 데이터셋.

    이미지를 미리 메모리에 다 올리지 않고 __getitem__ 에서 읽습니다. 데이터셋이
    커져도 메모리가 늘지 않고, 읽기는 워커 프로세스에서 학습과 병렬로 일어납니다.
    """

    def __init__(self, paths, labels=None, *, thresholds=None, augment: bool = True,
                 config: AugmentConfig | None = None, target_size=TARGET_SIZE,
                 seed: int = 0, filter_outliers: bool = True):
        self.config = config or AugmentConfig()
        self.target_size = target_size
        self.augment = augment
        self.seed = seed

        paths = [Path(p) for p in paths]
        labels = list(labels) if labels is not None else [0] * len(paths)

        self.reports: list = []
        if filter_outliers:
            # 품질 검사는 학습 전 1회만 합니다. __getitem__ 안에서 매번 하면
            # 에폭마다 같은 판정을 반복하는 순수한 낭비입니다.
            keep_paths, keep_labels = [], []
            for path, label in zip(paths, labels):
                image = cv2.imread(str(path))
                if image is None:
                    continue
                report = inspect(path.stem, image, thresholds)
                self.reports.append(report)
                if not report.is_outlier:
                    keep_paths.append(path)
                    keep_labels.append(label)
            self.paths, self.labels = keep_paths, keep_labels
        else:
            self.paths, self.labels = paths, labels

        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.paths)

    def __getstate__(self) -> dict:
        """워커 프로세스로 복제될 때 난수원은 빼고 보냅니다.

        메인 프로세스에서 이미 만들어진 Generator 를 그대로 피클해 보내면 모든
        워커가 동일한 난수 상태에서 출발합니다. 결과가 정상으로 보이기 때문에
        찾기 어려운 종류의 버그입니다. None 으로 비워 보내면 각 워커가 자기
        torch.initial_seed() 로 새로 만듭니다.
        """
        state = self.__dict__.copy()
        state["_rng"] = None
        return state

    def _generator(self) -> np.random.Generator:
        """워커별로 독립적인 난수원을 지연 생성합니다.

        torch.initial_seed() 는 DataLoader 가 워커마다 다르게 넣어 주는 값이라,
        이것을 섞으면 워커끼리 겹치지 않으면서 에폭마다 달라지는 난수열이 됩니다.
        """
        if self._rng is None:
            worker = torch.utils.data.get_worker_info()
            base = torch.initial_seed() % (2 ** 32)
            wid = 0 if worker is None else worker.id
            self._rng = np.random.default_rng([self.seed, base, wid])
        return self._rng

    def __getitem__(self, index: int):
        image = cv2.imread(str(self.paths[index]))
        if image is None:
            raise FileNotFoundError(f"이미지를 읽을 수 없습니다: {self.paths[index]}")

        # 순서: 리사이즈 -> 증강 -> 블러 -> 그레이스케일 -> 정규화
        # 증강을 리사이즈 뒤에 두면 회전/보간 비용이 224x224 에서만 발생합니다.
        # 블러를 증강 뒤에 두는 이유는 기존 파이프라인과 같습니다. 커널이 항상
        # 같은 화소 비율로 작동해야 합니다.
        out = cv2.resize(image, self.target_size, interpolation=cv2.INTER_AREA)
        if self.augment:
            out = random_augment(out, self.config, self._generator())
        out = cv2.GaussianBlur(out, BLUR_KERNEL, BLUR_SIGMA)
        out = cv2.cvtColor(out, cv2.COLOR_BGR2GRAY)
        tensor = torch.from_numpy(out.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor, int(self.labels[index])


def make_loader(dataset: Food101Dataset, batch_size: int = 8, workers: int = 0,
                shuffle: bool = True) -> DataLoader:
    """학습용 DataLoader. 워커가 증강을 미리 만들어 두므로 GPU 가 기다리지 않습니다."""
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=False,
                      persistent_workers=workers > 0)


# ----------------------------------------------------------------------------
# 시연 - 실시간 증강이 실제로 매번 달라지는지, 용량이 얼마나 절약되는지
# ----------------------------------------------------------------------------
def storage_comparison(dataset: Food101Dataset, num_augment: int = 3,
                       full_train_size: int = 75_750) -> dict:
    """사전 저장 방식과 실시간 방식의 디스크 사용량을 비교합니다.

    full_train_size 는 food101 학습 분할 전체 장수(101 클래스 x 750 장)다.
    """
    tensor, _ = dataset[0]
    per_tensor = tensor.numel() * tensor.element_size()
    stored = full_train_size * per_tensor * (1 + num_augment)
    return {
        "tensor_shape": tuple(tensor.shape),
        "bytes_per_tensor": int(per_tensor),
        "full_train_size": full_train_size,
        "num_augment": num_augment,
        "pre_saved_gb": stored / 1e9,
        "on_the_fly_gb": 0.0,
        "variants_pre_saved": num_augment,
        "variants_on_the_fly": "에폭마다 새로 생성 (사실상 제한 없음)",
    }


def demo_variation(dataset: Food101Dataset, index: int, epochs: int,
                   out_path: Path) -> list[float]:
    """같은 인덱스를 여러 번 꺼내 매번 다른 결과가 나오는지 확인하고 격자로 저장합니다."""
    tiles, diffs, first = [], [], None
    for epoch in range(epochs):
        tensor, _ = dataset[index]
        array = (tensor.squeeze(0).numpy() * 255.0).clip(0, 255).astype(np.uint8)
        if first is None:
            first = array.astype(np.int16)
        else:
            diffs.append(float(np.abs(array.astype(np.int16) - first).mean()))

        tile = cv2.cvtColor(array, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(tile, (0, 0), (tile.shape[1], 20), (0, 0, 0), -1)
        cv2.putText(tile, f"epoch {epoch}", (5, 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
        tiles.append(tile)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), np.hstack(tiles))
    return diffs


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="실시간 증강 DataLoader 시연")
    p.add_argument("--num-images", type=int, default=40)
    p.add_argument("--data-dir", type=Path, default=Path("data"))
    p.add_argument("--epochs", type=int, default=6, help="같은 이미지를 몇 번 꺼내볼지")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--offline", action="store_true")
    return p.parse_args()


def main() -> None:
    from image_preprocessing import load_images_from_disk, load_images_from_hf

    args = parse_args()
    args.data_dir.mkdir(parents=True, exist_ok=True)

    cached = sorted(args.data_dir.glob("food101_*.jpg"))
    if args.offline or len(cached) >= args.num_images:
        load_images_from_disk(args.num_images, args.data_dir)
    else:
        load_images_from_hf(args.num_images, args.data_dir)
    paths = sorted(args.data_dir.glob("food101_*.jpg"))[:args.num_images]
    if not paths:
        raise SystemExit("원본 이미지가 없습니다. --offline 을 빼고 다시 실행해 주세요.")

    thresholds = load_thresholds()
    print(f"[info] 임계값 출처: {thresholds.source}")
    print(f"       밝기 {thresholds.dark_mean:.2f} / 면적비 {thresholds.min_area_ratio:.4f}")

    dataset = Food101Dataset(paths, thresholds=thresholds, augment=True)
    print(f"[info] 원본 {len(paths)}장 -> 이상치 제외 후 {len(dataset)}장")

    # --- 1) 같은 이미지가 에폭마다 달라지는가 -----------------------------
    diffs = demo_variation(dataset, 0, args.epochs,
                           Path("outputs/preview/on_the_fly_variation.png"))
    print(f"\n[실시간 증강] 인덱스 0 을 {args.epochs}회 반복 조회")
    print(f"  첫 결과 대비 평균 화소 차이: "
          f"{', '.join(f'{d:.1f}' for d in diffs)}  (0 이면 매번 같은 것)")
    print("  -> outputs/preview/on_the_fly_variation.png")

    fixed = Food101Dataset(paths, thresholds=thresholds, augment=False)
    same = [float(np.abs((fixed[0][0] - fixed[0][0]).numpy()).mean()) for _ in range(2)]
    print(f"  대조: augment=False 는 매번 동일 (차이 {same[0]:.1f})")

    # --- 2) 용량 비교 -----------------------------------------------------
    comp = storage_comparison(dataset)
    print(f"\n[용량] 텐서 1장 {comp['tensor_shape']} = {comp['bytes_per_tensor'] / 1024:.0f} KB")
    print(f"  사전 저장 (원본 + 증강 {comp['num_augment']}종, "
          f"food101 학습 {comp['full_train_size']:,}장 기준): "
          f"{comp['pre_saved_gb']:.1f} GB")
    print(f"  실시간 증강: {comp['on_the_fly_gb']:.1f} GB "
          f"(변형 종류 {comp['variants_on_the_fly']})")

    # --- 3) 처리량 --------------------------------------------------------
    loader = make_loader(dataset, batch_size=args.batch_size, workers=args.workers)
    t0 = time.time()
    total = 0
    for batch, _ in loader:
        total += batch.shape[0]
    elapsed = time.time() - t0
    print(f"\n[처리량] workers={args.workers} batch={args.batch_size} "
          f"-> {total}장 {elapsed:.2f}s ({total / elapsed:.0f} img/s)")
    print(f"  배치 텐서 shape={tuple(batch.shape)} dtype={batch.dtype} "
          f"range=[{batch.min():.3f}, {batch.max():.3f}]")


if __name__ == "__main__":
    main()
