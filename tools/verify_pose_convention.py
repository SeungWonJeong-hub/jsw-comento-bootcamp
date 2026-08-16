"""SPE3R 자세 라벨의 규약을 실측으로 확정한다.

왜 필요한가
    쿼터니언은 (1) 스칼라가 앞인지 뒤인지, (2) 카메라->동체인지 동체->카메라인지
    두 가지가 문서마다 다르다. 규약을 잘못 잡으면 복셀 카빙이 통째로 어긋나는데,
    중간 산출물만 봐서는 틀렸는지 알기 어렵다.

어떻게 확인하는가
    메시 정점을 각 후보 규약으로 카메라에 투영해 실루엣을 만들고, 데이터셋이
    제공하는 정답 마스크와 IoU 를 비교한다. 올바른 규약만 IoU 가 높게 나온다.

사용법
    py -3 tools/verify_pose_convention.py
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.camera import Pose, quaternion_to_rotation  # noqa: E402
from src.metrics import mask_iou  # noqa: E402
from src.spe3r import SPE3RModel  # noqa: E402

DATA = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                    "data", "spe3r")


def silhouette_from_mesh(camera, pose, vertices):
    """메시 정점을 투영해 만든 대략적인 실루엣 마스크."""
    p_cam = vertices @ pose.R.T + pose.t
    z = p_cam[:, 2]
    ok = z > 1e-9
    u = camera.fx * p_cam[ok, 0] / z[ok] + camera.cx
    v = camera.fy * p_cam[ok, 1] / z[ok] + camera.cy

    ui = np.round(u).astype(np.int64)
    vi = np.round(v).astype(np.int64)
    inside = (ui >= 0) & (ui < camera.width) & (vi >= 0) & (vi < camera.height)

    mask = np.zeros(camera.shape, dtype=bool)
    mask[vi[inside], ui[inside]] = True

    # 정점만 찍으면 성기므로 살짝 부풀려 면을 채운다.
    import cv2
    k = np.ones((3, 3), np.uint8)
    return cv2.dilate(mask.astype(np.uint8), k, iterations=2).astype(bool)


def main() -> int:
    model = SPE3RModel(DATA, "aqua")
    print(model)
    vertices, faces = model.load_mesh()
    print(f"메시 정점 {len(vertices):,} 개, 삼각형 {len(faces):,} 개")
    print(f"메시 경계 상자: {np.round(vertices.min(axis=0), 3)} ~ "
          f"{np.round(vertices.max(axis=0), 3)}")

    indices = [0, 137, 402, 661, 913]
    candidates = {
        "scalar_first=True,  R = q^T (동체->카메라)": ("first", True),
        "scalar_first=True,  R = q   (그대로)": ("first", False),
        "scalar_first=False, R = q^T (동체->카메라)": ("last", True),
        "scalar_first=False, R = q   (그대로)": ("last", False),
    }

    print("\n후보 규약별 평균 IoU (정답 마스크 대비)")
    results = {}
    for name, (order, transpose) in candidates.items():
        ious = []
        for i in indices:
            label = model.labels[i]
            R = quaternion_to_rotation(label["q_vbs2tango_true"],
                                       scalar_first=(order == "first"))
            R_body2cam = R.T if transpose else R
            pose = Pose(R=R_body2cam, t=label["r_Vo2To_vbs_true"])
            pred = silhouette_from_mesh(model.camera, pose, vertices)
            ious.append(mask_iou(pred, model.load_mask(i)))
        results[name] = float(np.mean(ious))
        print(f"  {np.mean(ious):.4f}   {name}")

    best = max(results, key=results.get)
    print(f"\n채택: {best}  (IoU {results[best]:.4f})")
    if results[best] < 0.5:
        print("경고: 최고 IoU 가 0.5 미만입니다. 규약 후보를 더 넓혀야 합니다.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
