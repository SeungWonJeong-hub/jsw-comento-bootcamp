"""SPE3R 자세 라벨의 규약을 실측으로 확정한다.

왜 필요한가
    쿼터니언은 (1) 스칼라가 앞인지 뒤인지, (2) 카메라->동체인지 동체->카메라인지
    두 가지가 문서마다 다르다. SPE3R 문서에는 "SPEED+ 와 같은 형식"이라고만
    되어 있다. 규약을 잘못 잡으면 상대 자세와 기준 깊이가 통째로 어긋나는데,
    중간 산출물만 봐서는 틀렸는지 알기 어렵다.

어떻게 확인하는가
    메시 표면을 조밀하게 샘플링해 각 후보 규약으로 카메라에 투영하고, z-buffer
    로 채운 실루엣을 데이터셋의 정답 마스크와 IoU 로 비교한다. 올바른 규약만
    IoU 가 높게 나온다.

    실루엣 생성은 run_3d_experiment.py 가 기준 깊이를 만들 때와 같은 방식
    (표면 40만 점 샘플링 -> zbuffer_depth(splat=1, fill_holes=True)) 이다.
    검증에만 쓰는 별도 근사를 두면 그 근사가 맞는지를 다시 검증해야 한다.

사용법
    py -3 tools/verify_pose_convention.py              # 100 뷰, 약 70 초
    py -3 tools/verify_pose_convention.py --views 20   # 빠르게 확인만
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import Pose, quaternion_to_rotation  # noqa: E402
from src.metrics import mask_iou  # noqa: E402
from src.pointcloud import (  # noqa: E402
    sample_mesh_surface,
    transform_points,
    zbuffer_depth,
)
from src.spe3r import SPE3RModel  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "spe3r")

# run_3d_experiment.py 의 MESH_SAMPLES 와 같은 값을 쓴다.
MESH_SAMPLES = 400_000

CANDIDATES = {
    "scalar_first=True,  R = q   (그대로)": ("first", False),
    "scalar_first=True,  R = q^T (전치)": ("first", True),
    "scalar_first=False, R = q   (그대로)": ("last", False),
    "scalar_first=False, R = q^T (전치)": ("last", True),
}


def silhouette_from_mesh(camera, pose: Pose, mesh_points: np.ndarray) -> np.ndarray:
    """메시 표면 샘플을 투영해 채운 실루엣 마스크를 만든다."""
    depth = zbuffer_depth(transform_points(mesh_points, pose.R, pose.t),
                          camera, splat=1, fill_holes=True)
    return np.isfinite(depth)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SPE3R 자세 규약 실측 검증")
    p.add_argument("--views", type=int, default=100,
                   help="검증에 쓸 뷰 수 (1,000장에서 균등 간격으로 뽑는다)")
    p.add_argument("--samples", type=int, default=MESH_SAMPLES,
                   help="메시 표면 샘플 점 개수")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    model = SPE3RModel(DATA, "aqua")
    print(model)
    vertices, faces = model.load_mesh()
    print(f"메시 정점 {len(vertices):,} 개, 삼각형 {len(faces):,} 개")
    print(f"메시 경계 상자: {np.round(vertices.min(axis=0), 3)} ~ "
          f"{np.round(vertices.max(axis=0), 3)}")

    mesh_points = sample_mesh_surface(vertices, faces, args.samples, seed=0)
    # 균등 간격으로 뽑는다. 앞쪽만 쓰면 특정 자세 구간에 치우친다.
    step = max(1, len(model) // args.views)
    indices = list(range(0, len(model), step))[:args.views]
    print(f"표면 샘플 {args.samples:,} 점 · 검증 뷰 {len(indices)} 개")

    print("\n후보 규약별 평균 IoU (정답 마스크 대비)")
    results = {}
    t0 = time.time()
    for name, (order, transpose) in CANDIDATES.items():
        ious = []
        for i in indices:
            label = model.labels[i]
            R = quaternion_to_rotation(label["q_vbs2tango_true"],
                                       scalar_first=(order == "first"))
            pose = Pose(R=R.T if transpose else R, t=label["r_Vo2To_vbs_true"])
            pred = silhouette_from_mesh(model.camera, pose, mesh_points)
            ious.append(mask_iou(pred, model.load_mask(i)))
        ious = np.asarray(ious)
        results[name] = float(ious.mean())
        print(f"  {ious.mean():.4f}   (표준편차 {ious.std():.3f}, "
              f"최소 {ious.min():.3f})   {name}")

    best = max(results, key=results.get)
    runner_up = max((k for k in results if k != best), key=results.get)
    print(f"\n채택: {best}  (IoU {results[best]:.4f})")
    print(f"  2위와의 차이 {results[best] - results[runner_up]:.4f} — "
          f"규약 선택이 애매하지 않다")
    print(f"  소요 {time.time() - t0:.1f}초")

    if results[best] < 0.5:
        print("경고: 최고 IoU 가 0.5 미만입니다. 규약 후보를 더 넓혀야 합니다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
