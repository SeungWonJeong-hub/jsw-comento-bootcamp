"""2차 업무 실험 실행기 — 스테레오 삼각측량으로 깊이 맵과 3D 포인트 클라우드 생성.

실행
    py -3 tools/get_spe3r_aqua.py      # 데이터 준비 (최초 1회)
    py -3 run_3d_experiment.py         # 결과는 outputs/ 에 저장

구성
    [1] 합성 장면 검증
        광선-도형 교차를 해석적으로 풀어 정답 깊이를 오차 없이 만든 뒤,
        스테레오 파이프라인과 과제 예시 코드(밝기->깊이)를 같은 조건에서 비교한다.
        정답이 정확하므로 남는 오차는 전부 정합 알고리즘에서 온 것이다.

    [2] SPE3R 실제 데이터 적용
        카메라는 옆으로 움직이지 않지만 타겟이 회전하므로, 두 뷰의 상대 자세를
        계산하면 유효 베이스라인이 생긴다. 쓸 만한 쌍을 골라 같은 파이프라인을
        돌리고, 메시를 z-buffer 로 투영해 만든 기준 깊이와 비교한다.
"""

from __future__ import annotations

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import figures  # noqa: E402
from src import (baseline, depth as depth_mod, metrics, pointcloud,  # noqa: E402
                 scene, stereo)
from src.camera import PinholeCamera, Pose, quaternion_to_rotation  # noqa: E402
from src.spe3r import SPE3RModel  # noqa: E402

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "data", "spe3r")
OUT = os.path.join(ROOT, "outputs")
MODEL_NAME = "aqua"

SYNTH_BASELINE = 0.40
MESH_SAMPLES = 400_000
MAX_ROTATION_DEG = 8.0
MIN_LATERAL_RATIO = 2.0



def log(msg=""):
    print(msg, flush=True)


def within(pred, ref, mask, tol=0.05):
    v = np.isfinite(pred) & np.isfinite(ref) & mask
    if v.sum() == 0:
        return 0.0
    return float((np.abs(pred[v] - ref[v]) < tol).mean())


def span(depth, mask):
    v = np.isfinite(depth) & mask
    return float(depth[v].max() - depth[v].min()) if v.sum() else 0.0


# ---------------------------------------------------------------------------
# [1] 합성 장면 검증
# ---------------------------------------------------------------------------

def run_synthetic():
    """정답 깊이가 오차 없이 주어지는 조건에서 파이프라인을 검증한다."""
    cam = PinholeCamera(640, 512, 1277.37226, cx=320.0, cy=256.0)
    prims = scene.default_satellite()
    pose_l = Pose(quaternion_to_rotation([0.94, 0.0, 0.342, 0.0]), (0.0, 0.0, 5.0))
    pose_r = Pose(pose_l.R, pose_l.t - np.array([SYNTH_BASELINE, 0.0, 0.0]))

    left = scene.render(cam, pose_l, prims, texture_strength=0.35)
    right = scene.render(cam, pose_r, prims, texture_strength=0.35)
    mask, gt = left["mask"], left["depth"]

    disparity = stereo.compute_disparity(left["image"], right["image"],
                                         num_disparities=144)
    disparity = stereo.filter_disparity(disparity)
    z_stereo = np.where(mask, stereo.disparity_to_depth(disparity, cam.fx,
                                                       SYNTH_BASELINE), np.nan)

    # 대조군 채점 화소를 두 가지로 나눈다. 스테레오는 정합에 실패한 화소를
    # NaN 으로 버리므로, 대조군을 실루엣 전체에서 채점하면 두 방법이 서로 다른
    # 화소에서 채점된다. common 이 방법 비교, full 이 커버리지 포함 비교다.
    # 아핀 정렬도 집합마다 따로 맞춰 대조군에게 매번 가장 유리한 조건을 준다.
    common = mask & np.isfinite(z_stereo)

    def bright_on(domain):
        z = depth_mod.brightness_depth(left["image"], mask=domain)
        aligned, a, b = depth_mod.align_scale_shift(z, gt, mask=domain)
        m = metrics.depth_metrics(aligned, gt, mask=domain)
        return {**m, "within_5cm": within(aligned, gt, domain),
                "span_m": span(aligned, domain),
                "affine_scale": a, "affine_shift": b}, aligned

    m_common, _ = bright_on(common)
    m_full, z_bright = bright_on(mask)
    m_s = metrics.depth_metrics(z_stereo, gt, mask=mask)

    points = cam.unproject(z_stereo, mask=mask)
    result = {
        "camera": [cam.width, cam.height, cam.fx],
        "baseline_m": SYNTH_BASELINE,
        "expected_disparity_px": cam.fx * SYNTH_BASELINE / 5.0,
        "depth_resolution_m_per_px": stereo.depth_resolution(5.0, cam.fx, SYNTH_BASELINE),
        "gt_span_m": span(gt, mask),
        "stereo": {**m_s, "within_5cm": within(z_stereo, gt, mask),
                   "span_m": span(z_stereo, mask), "n_points": int(len(points))},
        "example_code": {"common": m_common, "full": m_full},
    }
    return result, {"cam": cam, "left": left, "right": right,
                    "disparity": disparity, "z_stereo": z_stereo,
                    "z_bright": z_bright, "points": points}


# ---------------------------------------------------------------------------
# [2] SPE3R 적용
# ---------------------------------------------------------------------------

# 기준 깊이 생성은 stereo.reference_depth() 로 옮겼다. reconstruct() 와 좌표
# 규약을 공유하므로 같은 모듈에 두고 테스트로 묶어야 한다.
#
# SPE3R 은 화소 단위 정답 깊이를 제공하지 않는다. 대신 watertight 메시를
# 조밀하게 샘플링해 z-buffer 로 투영하면 기준값을 만들 수 있다. 샘플링
# 밀도에서 오는 오차가 있으므로 '정답'이 아니라 '기준'으로 부른다.


def surface_coverage(pred_body, mesh_points, threshold: float = 0.02) -> dict:
    """복원한 점구름이 정답 표면을 얼마나 덮었는가.

    깊이 맵의 유효화소 비율과는 다른 것을 잰다. 유효화소는 '보이는 실루엣 안에서
    값이 나온 비율' 이고, 이것은 '타겟 표면 전체 중 복원된 비율' 이다. 단일
    시점 스테레오는 뒷면을 원리적으로 못 보므로 유효화소가 100% 여도 이 값은
    절반을 넘을 수 없다. 3D 복원이라고 말하려면 이쪽을 보고해야 한다.

    정규화 기준은 정답 메시로 잡는다 (SPE3R 논문과 같은 정의).
    """
    from scipy.spatial import cKDTree

    gt, center, scale = pointcloud.normalize_scale(mesh_points)
    pred = (np.asarray(pred_body, dtype=np.float64) - center) / scale
    d_gt, _ = cKDTree(pred).query(gt, k=1)
    d_pr, _ = cKDTree(gt).query(pred, k=1)
    recall = float((d_gt < threshold).mean())
    precision = float((d_pr < threshold).mean())
    f = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {"n_points": int(len(pred)), "threshold": threshold,
            "surface_coverage": recall, "precision": precision, "f_score": f}


def run_multiview_fusion(model, mesh_points, target_radius,
                         max_rotation_deg: float = 12.0,
                         min_lateral_ratio: float = 1.5) -> dict:
    """스테레오를 전방위로 밀면 어디까지 가는가.

    과제는 영상 장수를 정해 주지 않았다. 깊이 맵을 거쳐 3D 로 가기만 하면 되므로
    깊이 맵을 여러 장 만들어 융합하는 것도 같은 경로다. 단일 시점이 뒷면을 못
    본다는 한계를 시점을 늘려 넘을 수 있는지 직접 확인한다.

    쌍 선별 기준을 본 실험(8도/2.0)보다 풀어 후보를 늘리고, 복원된 점구름을 전부
    동체 좌표계로 모은 뒤, 다중 뷰 일관성 필터를 단계별로 걸어 본다. 필터는
    MVS 의 표준 후처리다 — 다른 쌍이 독립적으로 확인해 준 점만 남긴다.
    """
    from scipy.spatial import cKDTree

    geos = stereo.find_pairs(model, max_rotation_deg, min_lateral_ratio)
    clouds = []
    used = []
    for g in geos:
        i, j = g["i"], g["j"]
        try:
            pair = stereo.RectifiedPair(model.camera, g["R_ij"], g["t_ij"])
        except (ValueError, cv2.error):
            continue
        d0 = g["distance"]
        if not (6.0 < pair.expected_disparity(d0) < 0.85 * pair.match_width):
            continue
        sil = cv2.erode(model.load_mask(i).astype(np.uint8),
                        np.ones((3, 3), np.uint8), iterations=2) > 0
        try:
            out = stereo.reconstruct(pair, model.load_image(i, grayscale=True),
                                     model.load_image(j, grayscale=True),
                                     mask=sil, distance=d0,
                                     depth_range=(d0 - target_radius,
                                                  d0 + target_radius))
        except (ValueError, cv2.error):
            continue
        if out["points"] is None or len(out["points"]) < 50:
            continue
        clouds.append(pair.to_body(out["points"], model.pose(i)))
        used.append((i, j))

    if not clouds:
        return {"candidates": len(geos), "pairs_used": 0, "stages": []}

    allp = np.vstack(clouds)
    src = np.concatenate([np.full(len(x), k) for k, x in enumerate(clouds)])
    tree = cKDTree(allp)

    stages = [{"filter": "없음", **surface_coverage(allp, mesh_points)}]
    for radius, need in ((0.02, 1), (0.02, 2)):
        keep = np.zeros(len(allp), dtype=bool)
        for idx, nb in enumerate(tree.query_ball_point(allp, radius)):
            if len(set(src[nb]) - {src[idx]}) >= need:
                keep[idx] = True
        if keep.sum() < 10:
            continue
        stages.append({"filter": f"다른 쌍 {need}개 이상이 확인 (반경 {radius})",
                       **surface_coverage(allp[keep], mesh_points)})

    # 왜 47% 에서 멈추는지는 총계만 보면 알 수 없다. 쌍을 하나씩 더할 때
    # 커버리지가 얼마나 오르는지를 함께 남긴다. "쌍이 12개나 되는데 왜" 가
    # 아니라 "그 12개 중 실질적으로 몇 개가 일하는가" 가 답이기 때문이다.
    incremental = []
    prev = 0.0
    for k in range(1, len(clouds) + 1):
        cum = surface_coverage(np.vstack(clouds[:k]), mesh_points)["surface_coverage"]
        alone = surface_coverage(clouds[k - 1], mesh_points)["surface_coverage"]
        incremental.append({
            "pair_index": k, "i": used[k - 1][0], "j": used[k - 1][1],
            "n_points": int(len(clouds[k - 1])),
            "cumulative_coverage": cum, "gain": cum - prev, "alone": alone,
        })
        prev = cum

    # 시점이 방향을 얼마나 덮는지. 최대각만 보면 넓어 보이지만 한쪽에 뭉쳐
    # 있을 수 있어, 구면을 균등 분할해 몇 칸이 채워지는지 함께 센다.
    dirs = np.array([model.pose(i).R[:, 2] for i, _ in used])
    rng = np.random.default_rng(0)
    cells = rng.normal(size=(32, 3))
    cells /= np.linalg.norm(cells, axis=1, keepdims=True)
    filled = len({int(np.argmax(cells @ d)) for d in dirs})
    spread = {
        "max_angle_deg": float(np.degrees(np.arccos(
            np.clip(dirs @ dirs.T, -1, 1))).max()),
        "sphere_cells_filled": filled, "sphere_cells_total": len(cells),
    }

    return {"candidates": len(geos), "pairs_used": len(clouds),
            "max_rotation_deg": max_rotation_deg,
            "min_lateral_ratio": min_lateral_ratio, "stages": stages,
            "incremental": incremental, "view_spread": spread,
            "_points": allp}


def reference_sensitivity(model, mesh_points, best, vertices, faces) -> dict:
    """기준 깊이를 만드는 선택들이 결과를 얼마나 흔드는지 잰다.

    SPE3R 은 화소 단위 정답 깊이가 없어 메시를 투영해 기준을 만든다. 그 기준을
    만드는 데 세 가지 임의 선택이 들어간다 - 메시 샘플 수, z-buffer 의 splat,
    실루엣 침식량. "근사값이라 오차가 있다" 로 끝내면 그 오차가 결론을 뒤집을
    크기인지 알 수 없으므로 직접 잰다.
    """
    i, j = best["i"], best["j"]
    pair, out, mask = best["_pair"], best["_out"], best["_mask"]
    d0 = best["distance_m"]
    lo, hi = best["_depth_range"]

    def score(ref, dmap, dmask):
        mm = metrics.depth_metrics(dmap, ref, mask=dmask)
        return {"n_domain": mm["n_domain"], "median_abs": mm["median_abs"],
                "rmse": mm["rmse"], "within_5cm": within(dmap, ref, dmask),
                "valid_ratio": mm["valid_ratio"]}

    samples = []
    for n_pt in (100_000, 200_000, 400_000, 800_000):
        ref = stereo.reference_depth(
            pair, model.pose(i),
            pointcloud.sample_mesh_surface(vertices, faces, n_pt, seed=0))
        samples.append({"n_samples": n_pt, **score(ref, out["depth"], mask)})

    p_rect = model.pose(i).apply(mesh_points) @ pair.R1.T
    splats = []
    for sp in (0, 1, 2):
        ref = pointcloud.zbuffer_depth(p_rect, pair.camera, splat=sp, fill_holes=True)
        splats.append({"splat": sp, **score(ref, out["depth"], mask)})

    ref = stereo.reference_depth(pair, model.pose(i), mesh_points)
    erodes = []
    for e in (0, 1, 2, 3, 4):
        sil = model.load_mask(i).astype(np.uint8)
        if e:
            sil = cv2.erode(sil, np.ones((3, 3), np.uint8), iterations=e)
        o = stereo.reconstruct(pair, model.load_image(i, grayscale=True),
                               model.load_image(j, grayscale=True),
                               mask=sil > 0, distance=d0, depth_range=(lo, hi))
        erodes.append({"erode_px": e, **score(ref, o["depth"], o["mask"])})

    return {"mesh_samples": samples, "zbuffer_splat": splats,
            "silhouette_erosion": erodes}


def filter_ablation(raw, pair, ref, mask, depth_range) -> list:
    """이상치 필터를 단계별로 켜 가며 효과를 분리한다 (README 5-2 절 표).

    입력은 필터를 끄고 복원한 결과의 시차 맵이다. 나머지 조건(침식, 마스크,
    depth_range, 기준 깊이)은 전부 고정하고 필터만 바꾼다.
    """
    lo, hi = depth_range
    rows = []
    for tag, disparity in (
            ("필터 없음", raw["disparity"]),
            ("filterSpeckles(400, 1px)",
             stereo.filter_disparity(raw["disparity"], median_kernel=0)),
            ("+ 중앙값 3x3", stereo.filter_disparity(raw["disparity"]))):
        z = stereo.disparity_to_depth(disparity, pair.focal, pair.baseline)
        z = np.where(mask, z, np.nan)
        z = np.where((z >= lo) & (z <= hi), z, np.nan)
        m = metrics.depth_metrics(z, ref, mask=mask)
        rows.append({"stage": tag, "rmse": m["rmse"],
                     "median_abs": m["median_abs"],
                     "within_5cm": within(z, ref, mask),
                     "valid_ratio": m["valid_ratio"]})
    return rows


def disparity_range_ablation(model, results) -> list:
    """시차 탐색 범위를 물리적으로 가능한 구간으로 좁히면 어떻게 되는가.

    현재는 minDisparity=0 에서 기대 시차의 1.6 배까지 훑는다. 그런데 타겟의
    경계 반지름 R 을 알면 깊이가 [d0-R, d0+R] 안이므로 시차도
    [f·B/(d0+R), f·B/(d0-R)] 안이다. 최적 쌍에서는 이 폭이 24 px 인데 144 px
    를 훑고 있다.

    좁히면 커버리지가 오르지만 정확도가 함께 오르지는 않는다. 탐색 후보가
    줄면 uniquenessRatio 검사를 통과하기 쉬워져, 원래는 기각됐을 애매한
    대응이 살아남기 때문이다. 개선이 아니라 트레이드오프이므로 채택하지 않고
    측정값만 남긴다 (README 8절).
    """
    rows = []
    for r in results:
        pair, ref, mask = r["_pair"], r["_ref"], r["_mask"]
        lo, hi = r["_depth_range"]
        L, R_img, _ = pair.remap(model.load_image(r["i"], grayscale=True),
                                 model.load_image(r["j"], grayscale=True))
        d_lo = pair.focal * pair.baseline / hi
        d_hi = pair.focal * pair.baseline / lo
        mind = max(0, int(np.floor(d_lo / 16)) * 16)
        nd = max(16, int(np.ceil((d_hi - mind) / 16)) * 16)

        disparity = stereo.compute_disparity(L, R_img, num_disparities=nd,
                                             min_disparity=mind)
        disparity = stereo.filter_disparity(disparity)
        if not pair.horizontal:
            disparity = pair.unrotate(disparity)
        z = stereo.disparity_to_depth(disparity, pair.focal, pair.baseline)
        z = np.where(mask, z, np.nan)
        z = np.where((z >= lo) & (z <= hi), z, np.nan)
        m = metrics.depth_metrics(z, ref, mask=mask)
        rows.append({
            "i": r["i"], "j": r["j"],
            "current": {"min_disparity": 0, "num_disparities": r["num_disparities"],
                        "median_abs": r["median_abs"], "within_5cm": r["within_5cm"],
                        "valid_ratio": r["valid_ratio"]},
            "narrowed": {"min_disparity": mind, "num_disparities": nd,
                         "median_abs": m["median_abs"],
                         "within_5cm": within(z, ref, mask),
                         "valid_ratio": m["valid_ratio"]},
        })
    return rows


def block_size_ablation(model, geos, mesh_points, target_radius,
                        sizes=(3, 5, 7, 9, 11, 13, 15, 17)) -> list:
    """SGBM 정합 블록 크기를 실데이터에서 고른 근거 (stereo.DEFAULT_BLOCK_SIZE).

    처음 기본값은 3 이었다. **합성 장면**에서 RMSE 가 가장 낮았기 때문이다.
    합성 장면은 무늬가 충분해 어느 블록이든 정합이 되고, 그때는 작은 블록이
    기울어진 면을 덜 뭉갠다. 실제 위성 영상은 반대다 - 정합할 단서가 부족해서
    작은 블록은 아예 대응을 못 찾는다.

    "합성에서 고른 값이 실데이터에서도 최선인가" 는 검증되지 않은 가정이었다.
    후보 기하 전체를 고정해 두고 블록만 바꿔 다시 복원한다. 기본값이 바뀌어도
    비교 대상이 흔들리지 않게 `results` 가 아니라 `geos` 에서 출발한다.

    블록을 키우면 정확도가 커버리지와 맞바뀌는지 보이도록, 복원에 성공한 쌍
    수와 그 쌍들을 융합한 표면 커버리지까지 함께 남긴다.
    """
    rows = []
    for bs in sizes:
        rs = [r for r in (evaluate_pair(model, g, mesh_points, target_radius,
                                        block_size=bs) for g in geos)
              if r is not None]
        if not rs:
            continue
        clouds = [r["_pair"].to_body(r["_out"]["points"], model.pose(r["i"]))
                  for r in rs]
        cov = surface_coverage(np.vstack(clouds), mesh_points)
        best = min(rs, key=lambda r: r["median_abs"])
        rows.append({
            "block_size": bs,
            "candidates": len(geos), "pairs_reconstructed": len(rs),
            "median_abs_mean": float(np.mean([r["median_abs"] for r in rs])),
            "within_5cm_mean": float(np.mean([r["within_5cm"] for r in rs])),
            "valid_ratio_mean": float(np.mean([r["valid_ratio"] for r in rs])),
            "best_pair": {"i": best["i"], "j": best["j"],
                          "median_abs": best["median_abs"],
                          "within_5cm": best["within_5cm"],
                          "valid_ratio": best["valid_ratio"],
                          "rmse": best["rmse"]},
            "fused_surface_coverage": cov["surface_coverage"],
            "fused_precision": cov["precision"],
            "fused_f_score": cov["f_score"],
            "fused_points": cov["n_points"],
        })
    return rows


def evaluate_pair(model, geo, mesh_points, target_radius, erode_px=2,
                  block_size=stereo.DEFAULT_BLOCK_SIZE):
    """뷰 쌍 하나에 대해 스테레오 복원을 수행하고 기준 깊이와 비교한다.

    두 가지 표준 후처리를 적용하고, 적용 전후를 모두 기록한다.

    1) 실루엣 경계 침식 (erode_px)
       경계 화소는 전경과 배경이 섞여 있어 정합이 신뢰할 수 없다.
    2) 물리적 깊이 범위 제한 (target_radius)
       타겟의 경계 반지름을 알고 있으므로 [거리-R, 거리+R] 밖의 값은
       정합 실패다. 창의 폭 2R 은 메시에서 유도하지만(pointcloud.bounding_radius)
       창의 중심은 정답 거리다. 7절 한계에 함께 적었다.
    """
    i, j = geo["i"], geo["j"]
    try:
        pair = stereo.RectifiedPair(model.camera, geo["R_ij"], geo["t_ij"], alpha=-1.0)
    except (ValueError, cv2.error):
        return None

    expected = pair.expected_disparity(geo["distance"])
    if not (6.0 < expected < 0.85 * pair.match_width):
        return None

    silhouette = model.load_mask(i).astype(np.uint8)
    if erode_px > 0:
        silhouette = cv2.erode(silhouette, np.ones((3, 3), np.uint8),
                               iterations=erode_px)

    d0 = geo["distance"]
    out = stereo.reconstruct(pair, model.load_image(i, grayscale=True),
                             model.load_image(j, grayscale=True),
                             mask=silhouette > 0, distance=d0,
                             depth_range=(d0 - target_radius, d0 + target_radius),
                             block_size=block_size)
    # 이상치 필터의 효과만 분리하려면 나머지 조건이 전부 같아야 한다. 침식과
    # depth_range 까지 함께 빼면 필터가 아니라 그 셋의 합을 재게 된다.
    raw = stereo.reconstruct(pair, model.load_image(i, grayscale=True),
                             model.load_image(j, grayscale=True),
                             mask=silhouette > 0, distance=d0,
                             depth_range=(d0 - target_radius, d0 + target_radius),
                             postfilter=False, block_size=block_size)

    ref = stereo.reference_depth(pair, model.pose(i), mesh_points)
    mask = out["mask"] if out["mask"] is not None else np.isfinite(ref)
    raw_mask = raw["mask"] if raw["mask"] is not None else np.isfinite(ref)

    m = metrics.depth_metrics(out["depth"], ref, mask=mask)
    if m["n_valid"] < 50:
        return None
    m_raw = metrics.depth_metrics(raw["depth"], ref, mask=raw_mask)

    return {
        "i": i, "j": j,
        "rotation_deg": geo["rotation_deg"],
        "baseline_m": pair.baseline,
        "lateral_ratio": geo["lateral_ratio"],
        "focal_px": pair.focal,
        "horizontal": bool(pair.horizontal),
        "expected_disparity_px": expected,
        "num_disparities": out["num_disparities"],
        "distance_m": d0,
        "rmse": m["rmse"], "median_abs": m["median_abs"],
        "valid_ratio": m["valid_ratio"],
        "within_5cm": within(out["depth"], ref, mask),
        # 필터만 끈 값. 나머지 조건(침식, depth_range)은 위와 동일하다.
        "rmse_nofilter": m_raw["rmse"],
        "median_abs_nofilter": m_raw["median_abs"],
        "within_5cm_nofilter": within(raw["depth"], ref, raw_mask),
        "valid_ratio_nofilter": m_raw["valid_ratio"],
        "depth_resolution_m_per_px": stereo.depth_resolution(
            d0, pair.focal, pair.baseline),
        "_pair": pair, "_out": out, "_ref": ref, "_mask": mask, "_raw": raw,
        "_silhouette": silhouette > 0,
        "_depth_range": (d0 - target_radius, d0 + target_radius),
    }


def run_spe3r(model, mesh_points, target_radius):
    geos = stereo.find_pairs(model, max_rotation_deg=MAX_ROTATION_DEG,
                             min_lateral_ratio=MIN_LATERAL_RATIO)
    log(f"  회전 {MAX_ROTATION_DEG:.0f}도 이내 · 횡방향비 {MIN_LATERAL_RATIO:.0f} 이상 "
        f"후보 {len(geos)}쌍")

    # 걸린 시간은 로그에 적지 않는다. 실행할 때마다 달라져서 저장소에 남는
    # run_log.txt 가 재현되지 않는 유일한 원인이었다. 로컬 경로를 빼는 것과 같은
    # 이유다. 대략적인 소요 시간은 README 1절에 적어 둔다.
    results = [r for r in (evaluate_pair(model, g, mesh_points, target_radius)
               for g in geos)
               if r is not None]
    log(f"  복원 성공 {len(results)}쌍")
    if not results:
        return geos, [], None

    results.sort(key=lambda r: r["median_abs"])
    good = [r for r in results if r["median_abs"] < 0.10]
    log(f"  깊이 오차 중앙값 10 cm 이내 {len(good)}쌍")
    for r in results[:5]:
        log(f"    img{r['i']+1:06d}/img{r['j']+1:06d}  {r['rotation_deg']:5.2f}deg  "
            f"B={r['baseline_m']:.3f} m  med={r['median_abs']:.4f} m  "
            f"<5cm {r['within_5cm']*100:5.1f}%  cov {r['valid_ratio']*100:5.1f}%")
    return geos, results, results[0]


def run_example_code(model, best):
    """과제 예시 코드를 같은 정렬 좌표계에서 돌려 동일 기준으로 비교한다.

    채점 화소를 두 가지로 나눠 보고한다. 스테레오는 정합에 실패한 화소를
    NaN 으로 버리므로(최적 쌍에서 80.8%), 대조군을 실루엣 전체에서 채점하면
    두 방법이 서로 다른 화소에서 채점되는 셈이 된다. 스테레오만 어려운
    화소를 빼고 채점받는 비교는 성립하지 않는다.

        common : 스테레오가 값을 낸 화소에서만 채점 — 같은 화소, 같은 기준
        full   : 실루엣 전체에서 채점 — 대조군은 모든 화소에 값을 내므로
                 커버리지까지 포함한 비교

    아핀 정렬도 채점 집합마다 따로 맞춘다. 각 집합에서 대조군에게 가능한
    가장 유리한 스케일·오프셋을 준 뒤에도 지는지를 봐야 하기 때문이다.
    """
    out, ref, mask = best["_out"], best["_ref"], best["_mask"]
    left_rect = out["left"]
    common = mask & np.isfinite(out["depth"])

    def score(domain):
        z = depth_mod.brightness_depth(left_rect, mask=domain)
        aligned, a, b = depth_mod.align_scale_shift(z, ref, mask=domain)
        m = metrics.depth_metrics(aligned, ref, mask=domain)
        return {**m, "within_5cm": within(aligned, ref, domain),
                "span_m": span(aligned, domain),
                "affine_scale": a, "affine_shift": b}, aligned

    m_common, aligned_common = score(common)
    m_full, aligned_full = score(mask)

    grid = baseline.image_to_points_3d(cv2.cvtColor(left_rect, cv2.COLOR_GRAY2BGR))
    cloud = baseline.points_3d_to_cloud(grid, mask)
    return ({"common": m_common, "full": {**m_full, "n_points": int(len(cloud))}},
            aligned_full, cloud)


# ---------------------------------------------------------------------------

def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    log("=" * 70)
    log("2차 업무 — 스테레오 삼각측량으로 깊이 맵과 3D 포인트 클라우드 생성")
    log("=" * 70)

    log("\n[1] 합성 장면 검증 (정답 깊이 오차 0)")
    synth, art = run_synthetic()
    s = synth["stereo"]
    e_c, e_f = synth["example_code"]["common"], synth["example_code"]["full"]
    log(f"  베이스라인 {SYNTH_BASELINE} m · 기대 시차 "
        f"{synth['expected_disparity_px']:.1f} px · 깊이 분해능 "
        f"{synth['depth_resolution_m_per_px']*100:.1f} cm/px")
    log(f"  채점 화소 — 스테레오가 값을 낸 {e_c['n_valid']:,}개 / "
        f"실루엣 {s['n_domain']:,}개 ({s['valid_ratio']*100:.1f}%)")
    log(f"  {'':28s}{'RMSE':>9s}{'중앙값':>10s}{'<5cm':>9s}{'깊이폭':>9s}{'n':>8s}")
    log(f"  {'[같은 화소] 스테레오':28s}{s['rmse']:9.4f}{s['median_abs']:10.4f}"
        f"{s['within_5cm']*100:8.1f}%{s['span_m']:9.3f}{s['n_valid']:8,d}")
    log(f"  {'[같은 화소] 과제 예시':28s}{e_c['rmse']:9.4f}{e_c['median_abs']:10.4f}"
        f"{e_c['within_5cm']*100:8.1f}%{e_c['span_m']:9.3f}{e_c['n_valid']:8,d}")
    log(f"  {'[실루엣 전체] 과제 예시':28s}{e_f['rmse']:9.4f}{e_f['median_abs']:10.4f}"
        f"{e_f['within_5cm']*100:8.1f}%{e_f['span_m']:9.3f}{e_f['n_valid']:8,d}")
    log(f"  정답 깊이폭 {synth['gt_span_m']:.3f} m  →  같은 화소에서 중앙값 오차 "
        f"{e_c['median_abs']/s['median_abs']:.1f}배 개선")

    log("\n[2] SPE3R 실제 데이터")
    model = SPE3RModel(DATA, MODEL_NAME)
    log(f"  {model}")
    vertices, faces = model.load_mesh()
    mesh_points = pointcloud.sample_mesh_surface(vertices, faces, MESH_SAMPLES, seed=0)
    log(f"  메시 표면 샘플 {len(mesh_points):,}점 → 기준 깊이 맵 생성용")

    target_radius = pointcloud.bounding_radius(vertices)
    log(f"  타겟 경계 반지름 {target_radius:.4f} m (메시에서 유도, 여유 5%)")

    geos, results, best = run_spe3r(model, mesh_points, target_radius)
    if best is None:
        log("복원 가능한 쌍이 없습니다.")
        return 1

    ex, ex_depth, ex_cloud = run_example_code(model, best)
    log(f"\n  최적 쌍 img{best['i']+1:06d}/img{best['j']+1:06d}")
    log(f"  채점 화소 — 스테레오가 값을 낸 {ex['common']['n_valid']:,}개 / "
        f"실루엣 {int(best['_mask'].sum()):,}개 ({best['valid_ratio']*100:.1f}%)")
    log(f"  {'':28s}{'RMSE':>9s}{'중앙값':>10s}{'<5cm':>9s}{'n':>8s}")
    log(f"  {'[같은 화소] 스테레오':28s}{best['rmse']:9.4f}"
        f"{best['median_abs']:10.4f}{best['within_5cm']*100:8.1f}%"
        f"{ex['common']['n_valid']:8,d}")
    log(f"  {'[같은 화소] 과제 예시':28s}{ex['common']['rmse']:9.4f}"
        f"{ex['common']['median_abs']:10.4f}"
        f"{ex['common']['within_5cm']*100:8.1f}%{ex['common']['n_valid']:8,d}")
    log(f"  {'[실루엣 전체] 과제 예시':28s}{ex['full']['rmse']:9.4f}"
        f"{ex['full']['median_abs']:10.4f}"
        f"{ex['full']['within_5cm']*100:8.1f}%{ex['full']['n_valid']:8,d}")
    log("  스테레오는 정합 실패 화소를 NaN 으로 버린다. 방법 비교는 같은 화소에서")
    log("  채점한 앞의 두 줄이고, 셋째 줄은 커버리지까지 포함한 비교다.")

    pair = best["_pair"]
    stereo_body = pair.to_body(best["_out"]["points"], model.pose(best["i"]))
    chamfer = metrics.chamfer_distance(stereo_body, mesh_points, norm=1)
    log(f"  포인트 클라우드 {len(stereo_body):,}점 · Chamfer pred→GT "
        f"{chamfer['pred_to_target']:.4f} / GT→pred {chamfer['target_to_pred']:.4f}")
    log("  GT→pred 가 큰 것은 단일 뷰라 뒷면이 비어 있기 때문이다.")

    log("\n[3] 이상치 필터 단계별 효과 (최적 쌍, 나머지 조건 고정)")
    ablation = filter_ablation(best["_raw"], pair, best["_ref"], best["_mask"],
                               best["_depth_range"])
    log(f"  {'':26s}{'RMSE':>9s}{'중앙값':>10s}{'<5cm':>9s}{'유효화소':>9s}")
    for r in ablation:
        log(f"  {r['stage']:26s}{r['rmse']:9.4f}{r['median_abs']:10.4f}"
            f"{r['within_5cm']*100:8.1f}%{r['valid_ratio']*100:8.1f}%")

    log("\n[4] 시차 탐색 범위를 좁히면 (개선 후보, 채택하지 않음)")
    narrow = disparity_range_ablation(model, results)
    log(f"  {'쌍':22s}{'탐색범위':16s}{'중앙값':>10s}{'<5cm':>9s}{'유효화소':>9s}")
    for r in narrow:
        for tag, key in (("현재", "current"), ("좁힘", "narrowed")):
            c = r[key]
            rng = f"{c['min_disparity']}~{c['min_disparity']+c['num_disparities']}"
            name = f"img{r['i']+1:06d}/img{r['j']+1:06d}" if tag == "현재" else ""
            log(f"  {name:22s}{tag+' '+rng:16s}{c['median_abs']:10.4f}"
                f"{c['within_5cm']*100:8.1f}%{c['valid_ratio']*100:8.1f}%")
    log("  커버리지는 오르지만 정확도가 함께 오르지는 않는다. 트레이드오프다.")

    log("\n[5] 기준 깊이를 만드는 선택이 결과를 얼마나 흔드는가")
    sens = reference_sensitivity(model, mesh_points, best, vertices, faces)
    log(f"  {'메시 샘플':>12s}  {'중앙값':>9s}{'5cm':>8s}")
    for r in sens["mesh_samples"]:
        log(f"  {format(r['n_samples'], ','):>11s}점  {r['median_abs']:9.5f}"
            f"{r['within_5cm']*100:7.1f}%")
    log(f"  {'z-buffer':>12s}  {'중앙값':>9s}{'5cm':>8s}{'RMSE':>9s}")
    for r in sens["zbuffer_splat"]:
        log(f"  {'splat=' + str(r['splat']):>12s}  {r['median_abs']:9.5f}"
            f"{r['within_5cm']*100:7.1f}%{r['rmse']:9.5f}")
    log(f"  {'실루엣 침식':>12s}  {'채점영역':>8s}{'중앙값':>10s}{'5cm':>8s}{'유효화소':>9s}")
    for r in sens["silhouette_erosion"]:
        log(f"  {'erode=' + str(r['erode_px']):>12s}  {r['n_domain']:8,d}"
            f"{r['median_abs']:10.5f}{r['within_5cm']*100:7.1f}%"
            f"{r['valid_ratio']*100:8.1f}%")
    log("  샘플 수는 수렴했고, splat 과 침식량은 쌍마다 최적이 달라 고르지 않는다.")

    log("\n[6] 표면을 얼마나 덮었는가")
    log("  유효화소는 '보이는 실루엣 안에서 값이 나온 비율' 이고, 표면 커버리지는")
    log("  '타겟 표면 전체 중 복원된 비율' 이다. 단일 시점은 뒷면을 원리적으로")
    log("  못 보므로 유효화소가 100% 여도 이 값은 절반을 넘지 못한다.")
    cov_single = surface_coverage(stereo_body, mesh_points)
    fusion = run_multiview_fusion(model, mesh_points, target_radius)
    fusion_points = fusion.pop("_points")

    log(f"  {'':32s}{'점 수':>9s}{'정밀도':>9s}{'표면 커버리지':>13s}{'F':>8s}")

    def cov_line(tag, r):
        log(f"  {tag:32s}{r['n_points']:9,d}{r['precision']*100:8.1f}%"
            f"{r['surface_coverage']*100:12.1f}%{r['f_score']*100:8.1f}")

    cov_line("1쌍 (과제 지정 경로)", cov_single)
    for st in fusion["stages"]:
        cov_line(f"{fusion['pairs_used']}쌍 융합 · {st['filter']}", st)
    log(f"  후보를 회전 {fusion['max_rotation_deg']:.0f}도까지 풀어 "
        f"{fusion['candidates']}쌍을 훑어도 {fusion['pairs_used']}쌍만 복원된다.")
    inc = fusion["incremental"]
    top = sorted(inc, key=lambda r: -r["gain"])[:2]
    log(f"  그 {fusion['pairs_used']}쌍 중에서도 상위 2쌍이 커버리지의 "
        f"{sum(r['gain'] for r in top)/inc[-1]['cumulative_coverage']*100:.0f}% 를 만든다:")
    for r in inc:
        mark = " <<<" if r in top else ""
        log(f"    {r['pair_index']:2d}쌍 누적  {r['cumulative_coverage']*100:5.1f}%"
            f"  (+{r['gain']*100:4.1f}%p · 단독 {r['alone']*100:4.1f}%)"
            f"  img{r['i']+1:06d}/img{r['j']+1:06d}{mark}")
    sp = fusion["view_spread"]
    log(f"  시점 간 최대각 {sp['max_angle_deg']:.0f}도인데 구면 "
        f"{sp['sphere_cells_total']}칸 중 {sp['sphere_cells_filled']}칸에만 있다. "
        f"넓어 보여도 한쪽에 뭉쳐 있다.")
    log("  그래서 벽은 '쌍이 모자라서' 가 아니다. 쓸 만한 쌍이 두어 개고 그")
    log("  둘이 비슷한 데를 본다. 남은 커버리지는 쌍을 더 모아서가 아니라")
    log("  시점 배치와 정합 설정에서 나온다 ([7] 절).")

    log("\n[7] 정합 블록 크기를 실데이터에서 다시 고른 근거")
    blocks = block_size_ablation(model, geos, mesh_points, target_radius)
    log("  처음 기본값은 3 이었다. 합성 장면 RMSE 가 가장 낮았기 때문이다.")
    log("  같은 후보 기하 위에서 블록만 바꿔 다시 복원하면 그림이 뒤집힌다.")
    log(f"  {'블록':>4s}{'복원쌍':>7s}{'최적쌍 중앙값':>14s}{'<5cm':>8s}"
        f"{'유효화소':>9s}{'융합 커버리지':>14s}{'정밀도':>9s}")
    for bl in blocks:
        bp = bl["best_pair"]
        mark = " <- 채택" if bl["block_size"] == stereo.DEFAULT_BLOCK_SIZE else ""
        log(f"  {bl['block_size']:4d}{bl['pairs_reconstructed']:7d}"
            f"{bp['median_abs']:14.5f}{bp['within_5cm']*100:7.1f}%"
            f"{bp['valid_ratio']*100:8.1f}%"
            f"{bl['fused_surface_coverage']*100:13.1f}%"
            f"{bl['fused_precision']*100:8.1f}%{mark}")
    log(f"  후보 {blocks[0]['candidates']}쌍 고정. 융합 커버리지는 이 후보만 쓴 값이라")
    log("  [6] 의 121쌍 융합과는 다른 집합이다. 블록끼리의 비교만 읽으면 된다.")
    log("  11 에서 5cm 이내 비율이 최고이고 중앙값도 사실상 최저다. 더 키우면")
    log("  유효화소만 오르고 정확도는 다시 나빠진다. 격자 끝이 아니라 안쪽이다.")
    log("  합성으로 고른 하이퍼파라미터를 실데이터에 검증 없이 옮기면 안 된다는")
    log("  뜻이다. 두 데이터에서 병목이 서로 다르기 때문이다.")

    log("\n그림 생성")
    figures.figure_concept(best, os.path.join(OUT, "00_concept.png"))
    figures.figure_synthetic(art, synth, os.path.join(OUT, "01_synthetic_validation.png"))
    figures.figure_survey(results, os.path.join(OUT, "02_pair_survey.png"))
    best["_example_median"] = ex["full"]["median_abs"]
    figures.figure_spe3r(best, ex_depth, os.path.join(OUT, "03_spe3r_stereo.png"))
    figures.figure_pointclouds(
        mesh_points, stereo_body, fusion_points, ex_cloud,
        os.path.join(OUT, "04_pointclouds.png"),
        coverage={"single": cov_single["surface_coverage"],
                  "fusion": fusion["stages"][0]["surface_coverage"],
                  "pairs": fusion["pairs_used"]})

    log("PLY 저장")
    pointcloud.write_ply(os.path.join(OUT, "pointcloud_ground_truth.ply"),
                         mesh_points[::13])
    pointcloud.write_ply(os.path.join(OUT, "pointcloud_stereo.ply"), stereo_body)
    pointcloud.write_ply(os.path.join(OUT, "pointcloud_stereo_fusion.ply"),
                         fusion_points)
    pointcloud.write_ply(os.path.join(OUT, "pointcloud_example_code.ply"),
                         pointcloud.normalize_scale(ex_cloud)[0])

    med = np.array([r["median_abs"] for r in results])
    summary = {
        "dataset": {
            "name": "SPE3R", "model": MODEL_NAME, "views": len(model),
            "image_size": [model.camera.width, model.camera.height],
            "fx": model.camera.fx,
            "camera_translation": "always (0, 0, Z); Z sweeps 5.000 to 6.000 m",
            "note": ("카메라는 옆으로 움직이지 않지만 타겟이 회전하므로 두 뷰의 "
                     "상대 자세에서 유효 베이스라인이 생긴다"),
            "license": "CC BY-NC-SA 4.0",
            "source": "https://purl.stanford.edu/pk719hm4806",
        },
        "synthetic_validation": synth,
        "spe3r_pair_survey": {
            "max_rotation_deg": MAX_ROTATION_DEG,
            "min_lateral_ratio": MIN_LATERAL_RATIO,
            "pairs_reconstructed": len(results),
            "pairs_within_10cm": int((med < 0.10).sum()),
            "median_error_m": {"best": float(med.min()),
                               "median": float(np.median(med)),
                               "worst": float(med.max())},
            "pairs": [{k: v for k, v in r.items() if not k.startswith("_")}
                      for r in results],
        },
        "best_pair": {k: v for k, v in best.items() if not k.startswith("_")},
        "best_pair_example_code": ex,
        "best_pair_chamfer": chamfer,
        "best_pair_points": int(len(stereo_body)),
        "filter_ablation": ablation,
        "disparity_range_ablation": narrow,
        "block_size_ablation": blocks,
        "surface_coverage": {"stereo_single_pair": cov_single,
                             "multiview_stereo_fusion": fusion},
        "reference_sensitivity": sens,
    }
    with open(os.path.join(OUT, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    # 저장소에 남는 로그이므로 실행한 사람의 로컬 경로를 적지 않는다.
    log(f"\n완료 -> {os.path.relpath(OUT, ROOT).replace(os.sep, '/')}/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
