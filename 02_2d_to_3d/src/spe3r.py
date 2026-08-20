"""SPE3R 공개 데이터셋 로더.

출처
    Park, T. H. and D'Amico, S., "SPE3R: Synthetic Dataset for Satellite Pose
    Estimation and 3D Reconstruction", Stanford Digital Repository, 2024.
    https://purl.stanford.edu/pk719hm4806   (CC BY-NC-SA 4.0)

데이터 구조에서 확인한 사실 (aqua 기준, 직접 측정)
    - 영상 256 x 256, fx = fy = 1277.37 px, 주점 (128, 128)
    - 병진 r_Vo2To_vbs 는 항상 (0, 0, Z). x, y 가 정확히 0 이다.
      Z 만 5.000 -> 6.000 m 로 선형 증가한다.
    - 회전은 매 프레임 무작위. 연속 프레임 간 각도 중앙값 132.9 deg.
    => 카메라가 옆으로 이동하지 않으므로 스테레오 베이스라인이 없다.
       가장 가까운 두 뷰조차 2.27 deg 차이라 삼각측량은 조건이 나쁘다.
       대신 무작위 자세가 SO(3) 를 고르게 덮으므로 실루엣 기반 복원에 유리하다.
"""

from __future__ import annotations

import json
import os

import cv2
import numpy as np

from .camera import PinholeCamera, Pose


class SPE3RModel:
    """SPE3R 위성 한 종에 대한 영상/마스크/자세/메시 묶음."""

    def __init__(self, root: str, model: str = "aqua"):
        self.root = os.path.abspath(root)
        self.model = model
        self.model_dir = os.path.join(self.root, model)

        camera_path = os.path.join(self.root, "camera.json")
        if not os.path.exists(camera_path):
            raise FileNotFoundError(
                f"camera.json 을 찾을 수 없습니다: {camera_path}\n"
                f"먼저 'py -3 tools/get_spe3r_aqua.py' 를 실행해 주세요.")
        with open(camera_path, encoding="utf-8") as f:
            self.camera_json = json.load(f)
        self.camera = PinholeCamera.from_spe3r(self.camera_json)

        labels_path = os.path.join(self.model_dir, "labels.json")
        if not os.path.exists(labels_path):
            raise FileNotFoundError(f"labels.json 을 찾을 수 없습니다: {labels_path}")
        with open(labels_path, encoding="utf-8") as f:
            self.labels = json.load(f)

    def __len__(self) -> int:
        return len(self.labels)

    def __repr__(self):
        return f"SPE3RModel({self.model}, {len(self)} views, {self.camera})"

    # ---- 개별 항목 접근 ----

    def image_path(self, index: int) -> str:
        return os.path.join(self.model_dir, "images",
                            self.labels[index]["filename"] + ".jpg")

    def mask_path(self, index: int) -> str:
        return os.path.join(self.model_dir, "masks",
                            self.labels[index]["filename"] + ".png")

    def load_image(self, index: int, grayscale: bool = False) -> np.ndarray:
        flag = cv2.IMREAD_GRAYSCALE if grayscale else cv2.IMREAD_COLOR
        img = cv2.imread(self.image_path(index), flag)
        if img is None:
            raise FileNotFoundError(f"영상을 읽지 못했습니다: {self.image_path(index)}")
        return img

    def load_mask(self, index: int) -> np.ndarray:
        """이진 실루엣 마스크 (H, W) bool."""
        m = cv2.imread(self.mask_path(index), cv2.IMREAD_GRAYSCALE)
        if m is None:
            raise FileNotFoundError(f"마스크를 읽지 못했습니다: {self.mask_path(index)}")
        return m > 127

    def pose(self, index: int, scalar_first: bool = True) -> Pose:
        return Pose.from_spe3r_label(self.labels[index], scalar_first=scalar_first)

    def distance(self, index: int) -> float:
        """카메라-타겟 거리 [m]."""
        return float(np.linalg.norm(self.labels[index]["r_Vo2To_vbs_true"]))

    # ---- 메시 ----

    def mesh_path(self) -> str:
        return os.path.join(self.model_dir, "models", "model_normalized.obj")

    def load_mesh(self):
        """watertight 메시를 (vertices, faces) 로 읽는다."""
        return load_obj(self.mesh_path())

    # ---- 뷰 선택 ----

    def select_views(self, count: int, seed: int = 0) -> list:
        """자세가 서로 최대한 멀어지도록 뷰를 고른다 (탐욕적 최원점 선택).

        실루엣 기반 복원은 시선 방향이 고르게 흩어질수록 정확해진다.
        무작위로 뽑는 것보다 같은 개수에서 더 좋은 복원을 얻는다.
        """
        if count <= 0:
            raise ValueError(f"뷰 개수는 1 이상이어야 합니다: {count}")
        n = len(self)
        count = min(count, n)

        q = np.array([lb["q_vbs2tango_true"] for lb in self.labels], dtype=np.float64)
        q /= np.linalg.norm(q, axis=1, keepdims=True)

        rng = np.random.default_rng(seed)
        chosen = [int(rng.integers(n))]
        # 각 후보와 선택된 집합 사이의 최소 각거리를 계속 갱신한다.
        best = np.degrees(2.0 * np.arccos(np.clip(np.abs(q @ q[chosen[0]]), 0, 1)))
        for _ in range(count - 1):
            best[chosen] = -1.0
            nxt = int(np.argmax(best))
            chosen.append(nxt)
            d = np.degrees(2.0 * np.arccos(np.clip(np.abs(q @ q[nxt]), 0, 1)))
            best = np.minimum(best, d)
        return chosen


def load_obj(path: str):
    """OBJ 파일에서 정점과 삼각형 면을 읽는다.

    Returns
    -------
    vertices : (V, 3) float64
    faces    : (F, 3) int64  — 0 부터 시작하는 인덱스
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"OBJ 파일이 없습니다: {path}")

    vertices = []
    faces = []
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                idx = []
                for token in line.split()[1:]:
                    # "v", "v/vt", "v//vn", "v/vt/vn" 형식 모두 허용
                    idx.append(int(token.split("/")[0]) - 1)
                # 다각형은 삼각형 부채꼴로 분할한다.
                for k in range(1, len(idx) - 1):
                    faces.append([idx[0], idx[k], idx[k + 1]])

    if not vertices:
        raise ValueError(f"정점이 없습니다: {path}")
    return np.asarray(vertices, dtype=np.float64), np.asarray(faces, dtype=np.int64)
