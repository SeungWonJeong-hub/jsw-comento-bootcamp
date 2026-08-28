"""촬영 모델과 정답 없는 신뢰도의 Unit Test.

여기 있는 것들은 모두 **정답 고도를 쓰지 않고** 판단을 내리는 도구이거나,
렌더링을 실제 촬영에 가깝게 만드는 도구입니다. 그래서 "그럴듯한 값이 나옵니다"
로는 검증이 안 됩니다. 손으로 답을 낼 수 있는 조건을 만들어 숫자까지 맞춥니다.

  해석해      평평한 면에 달 광도함수를 직접 계산해 렌더링 값과 대조합니다
  통계        잡음을 넣은 만큼 추정되는가, 광자 잡음이 밝기를 따라 커지는가
  불변식      같은 시차끼리는 좌우 일관성을 통과하고, 어긋난 곳은 걸립니다
  대조군      신뢰도가 무작위보다 나은지를 같은 자로 잽니다
  경계        잘못된 입력은 조용히 넘어가지 말고 예외를 냅니다
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import stereo, terrain  # noqa: E402


# ---------------------------------------------------------------------------
# 촬영 모델 — 시선 방향이 들어간 달 반사
# ---------------------------------------------------------------------------


def test_lambert_shading_is_identical_from_every_viewpoint():
    """램버트 반사는 보는 방향과 무관하다 — 그래서 실제보다 쉬운 문제입니다.

    이 테스트는 통과하는 것이 목적이 아니라, **왜 램버트로는 안 되는지**를
    코드로 못 박아 두는 것이 목적입니다. 두 사진의 같은 지면이 똑같은 밝기로
    찍히면 정합기는 실제로 겪지 않을 쉬운 문제를 푸는 셈이 됩니다.
    """
    elev, gsd = terrain.synthetic_dtm(size=64, gsd=5.0, relief=200.0, seed=1)
    a = terrain.shade(elev, gsd)
    b = terrain.shade(elev, gsd)
    assert np.array_equal(a, b)


def test_lunar_lambert_differs_between_the_two_stereo_viewpoints():
    """달 광도함수는 시선이 다르면 밝기가 다르다 — 정합의 진짜 난이도."""
    elev, gsd = terrain.synthetic_dtm(size=64, gsd=5.0, relief=200.0, seed=1)
    left, right, _ = terrain.stereo_cameras(500.0 * gsd, 30.0)

    a = terrain.shade(elev, gsd, viewer=terrain.camera_center(left))
    b = terrain.shade(elev, gsd, viewer=terrain.camera_center(right))
    assert not np.allclose(a, b), "두 시점의 밝기가 같으면 시선이 안 들어간 것입니다"

    # 램버트와도 달라야 합니다. 같다면 mu 가 식에서 빠진 것입니다.
    assert not np.allclose(a, terrain.shade(elev, gsd))


def test_lunar_lambert_matches_the_formula_computed_by_hand():
    """평평한 면에서는 각도가 전부 손으로 나오므로 값까지 대조할 수 있습니다.

    수평면의 법선은 (0, 0, 1) 입니다. 카메라를 정확히 머리 위에 두면 방출각이
    0 이라 mu = 1 이고, mu0 은 태양 고도의 sin 입니다. 위상각은 태양과 천정이
    이루는 각이므로 90도 - 태양고도입니다. 그러면 다음 값이 나와야 합니다.

        I = A [ 2 L mu0 / (mu0 + 1) + (1 - L) mu0 ] + ambient
    """
    gsd, sun_el, albedo, ambient = 10.0, 40.0, 0.12, 0.06
    # 격자 한가운데 화소가 (0, 0) 이 되도록 홀수 크기를 씁니다.
    elev = np.zeros((31, 31))
    viewer = np.array([0.0, 0.0, 5000.0])

    got = terrain.shade(elev, gsd, sun_elevation_deg=sun_el, albedo=albedo,
                        ambient=ambient, viewer=viewer)

    mu0 = np.sin(np.radians(sun_el))
    mu = 1.0
    g = 90.0 - sun_el
    a1, a2, a3 = terrain.LUNAR_LAMBERT_COEFFS
    L = 1.0 + a1 * g + a2 * g ** 2 + a3 * g ** 3
    want = albedo * (L * 2.0 * mu0 / (mu0 + mu) + (1.0 - L) * mu0) + ambient

    centre = got[got.shape[0] // 2, got.shape[1] // 2]
    assert centre == pytest.approx(want, abs=1e-6)


def test_camera_center_matches_the_stereo_geometry():
    """카메라 중심 두 개의 거리가 곧 베이스라인이어야 합니다."""
    altitude = 3000.0
    left, right, baseline = terrain.stereo_cameras(altitude, 20.0)
    cl = terrain.camera_center(left)
    cr = terrain.camera_center(right)

    assert cl[2] == pytest.approx(altitude)
    assert cr[2] == pytest.approx(altitude)
    assert np.linalg.norm(cr - cl) == pytest.approx(baseline)
    # 수렴 촬영이므로 두 대가 원점을 기준으로 좌우 대칭입니다.
    assert cl[0] == pytest.approx(-cr[0])


# ---------------------------------------------------------------------------
# 센서 모델
# ---------------------------------------------------------------------------


def test_photon_noise_grows_with_brightness():
    """광자 잡음은 밝기의 제곱근에 비례한다 — 어두운 곳이 상대적으로 더 나쁩니다.

    이것이 크레이터 그늘에서 정합이 어려운 물리적 이유입니다. 잡음을 밝기와
    무관한 상수로 넣으면 그 사실이 실험에서 사라집니다.
    """
    bright = np.full((256, 256), 240, dtype=np.uint8)
    dark = np.full((256, 256), 30, dtype=np.uint8)

    sb = stereo.estimate_noise_sigma(
        terrain.sensor_image(bright, snr=60.0, blur_px=0.0, seed=0))
    sd = stereo.estimate_noise_sigma(
        terrain.sensor_image(dark, snr=60.0, blur_px=0.0, seed=0))

    assert sb > sd, "밝은 곳의 잡음이 더 커야 광자 잡음입니다"
    # 세기의 제곱근 비만큼. 읽기 잡음이 섞여 있으므로 느슨하게 봅니다.
    assert sb / sd == pytest.approx(np.sqrt(240 / 30), rel=0.5)


def test_sensor_image_is_reproducible_and_quantised():
    """같은 씨앗이면 같은 영상, 비트 수를 줄이면 값의 가짓수가 줍니다."""
    img = np.linspace(0, 255, 64 * 64).reshape(64, 64).astype(np.uint8)

    a = terrain.sensor_image(img, seed=3)
    b = terrain.sensor_image(img, seed=3)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, terrain.sensor_image(img, seed=4))

    coarse = terrain.sensor_image(img, bits=4, blur_px=0.0, seed=0)
    assert len(np.unique(coarse)) <= 2 ** 4


@pytest.mark.parametrize("kwargs", [
    {"snr": 0.0}, {"snr": -1.0}, {"bits": 0}, {"bits": 17}, {"blur_px": -0.5},
])
def test_sensor_image_rejects_impossible_settings(kwargs):
    with pytest.raises(ValueError):
        terrain.sensor_image(np.zeros((8, 8), dtype=np.uint8), **kwargs)


# ---------------------------------------------------------------------------
# 정답 없는 신뢰도
# ---------------------------------------------------------------------------


def test_noise_estimate_recovers_the_noise_that_was_added():
    """넣은 잡음만큼 추정되어야 대비 하한을 잡음의 배수로 둘 수 있습니다."""
    rng = np.random.default_rng(0)
    smooth = np.tile(np.linspace(40, 200, 128), (128, 1))
    for sigma in (1.0, 3.0, 7.0):
        noisy = smooth + rng.normal(0.0, sigma, smooth.shape)
        assert stereo.estimate_noise_sigma(noisy) == pytest.approx(sigma, rel=0.15)


def test_contrast_floor_follows_the_noise_level():
    """대비 하한은 절대값이 아니라 잡음의 배수여야 영상이 바뀌어도 뜻이 같습니다."""
    rng = np.random.default_rng(1)
    flat = np.zeros((128, 128))
    quiet = stereo.contrast_floor(flat + rng.normal(0, 1.0, flat.shape), k=2.0)
    loud = stereo.contrast_floor(flat + rng.normal(0, 4.0, flat.shape), k=2.0)
    assert loud / quiet == pytest.approx(4.0, rel=0.2)

    with pytest.raises(ValueError):
        stereo.contrast_floor(flat, k=0.0)


def test_autocorrelation_is_longer_for_smoother_texture():
    """무늬가 완만할수록 자기상관이 오래 간다 — 블록이 커야 한다는 뜻입니다."""
    import cv2
    rng = np.random.default_rng(2)
    base = rng.normal(0, 1, (128, 128))
    fine = cv2.GaussianBlur(base, (0, 0), 1.0)
    coarse = cv2.GaussianBlur(base, (0, 0), 6.0)

    assert (stereo.autocorrelation_length(coarse)
            > stereo.autocorrelation_length(fine))
    for img in (fine, coarse):
        block = stereo.suggest_block_size(img)
        assert block % 2 == 1 and 3 <= block <= 15


def test_left_right_consistency_passes_matching_disparities():
    """같은 시차끼리는 통과합니다. 단, 왼쪽 가장자리는 짝이 영상 밖이라 걸립니다."""
    d = np.full((8, 20), 4.0)
    ok = stereo.left_right_consistency(d, d)
    assert ok[:, 4:].all(), "짝이 영상 안에 있으면 통과해야 합니다"
    assert not ok[:, :4].any(), "x - d 가 음수면 짝이 없으므로 걸러야 합니다"


def test_left_right_consistency_catches_a_mismatched_region():
    """한쪽만 다른 값을 가리키면 그 왕복은 깨진다 — 가림이 잡히는 원리입니다."""
    d_left = np.full((8, 20), 4.0)
    d_right = d_left.copy()
    d_right[:, 8:12] = 11.0        # 이 구간은 왼쪽을 도로 가리키지 않습니다

    ok = stereo.left_right_consistency(d_left, d_right)
    # 왼쪽 x 는 오른쪽 x-4 를 봅니다. 오른쪽 8~11 이 어긋났으니 왼쪽 12~15 가 걸립니다.
    assert not ok[:, 12:16].any()
    assert ok[:, 16:].all()


def test_left_right_consistency_rejects_mismatched_shapes():
    with pytest.raises(ValueError):
        stereo.left_right_consistency(np.zeros((4, 5)), np.zeros((4, 6)))


def test_photometric_residual_is_lowest_at_the_true_disparity():
    """옳은 시차로 겹치면 두 영상이 포개집니다. 틀린 시차는 잔차를 남깁니다."""
    rng = np.random.default_rng(3)
    left = rng.integers(0, 255, (64, 96)).astype(np.float32)
    shift = 5
    # 시차 d 는 왼쪽 x 가 오른쪽 x-d 에 대응한다는 뜻이므로, 오른쪽 영상은
    # 왼쪽으로 밀린 것이어야 합니다. 부호를 뒤집으면 탐색 구간 밖이 됩니다.
    right = np.roll(left, -shift, axis=1).astype(np.float32)

    true_d = np.full(left.shape, float(shift))
    wrong_d = np.full(left.shape, float(shift + 3))

    good = stereo.photometric_residual(left, right, true_d)
    bad = stereo.photometric_residual(left, right, wrong_d)
    inner = (slice(8, -8), slice(16, -16))
    assert np.nanmedian(good[inner]) == pytest.approx(0.0, abs=1e-3)
    assert np.nanmedian(bad[inner]) > 10.0


def test_photometric_residual_rejects_even_windows():
    with pytest.raises(ValueError):
        stereo.photometric_residual(np.zeros((8, 8), np.float32),
                                    np.zeros((8, 8), np.float32),
                                    np.zeros((8, 8)), ksize=4)


def test_sparsification_separates_a_perfect_signal_from_a_random_one():
    """희소화 곡선이 실제로 신뢰도의 좋고 나쁨을 가려내는지 봅니다.

    오차를 그대로 신뢰도로 쓰면 오라클과 같아지므로 AUSE 가 0 이어야 합니다.
    무작위 신뢰도는 곡선이 평평해 넓이가 커집니다. 이 두 극단을 구분하지
    못하면 이 자로 잰 어떤 값도 뜻이 없습니다.
    """
    rng = np.random.default_rng(4)
    err = rng.gamma(2.0, 20.0, 4096)

    perfect = stereo.sparsification(err, -err)
    random = stereo.sparsification(err, rng.random(4096))

    assert perfect["ause"] == pytest.approx(0.0, abs=1e-9)
    assert random["ause"] > 0.05
    assert perfect["curve"][-1] < perfect["curve"][0]


def test_sparsification_rejects_too_few_steps():
    with pytest.raises(ValueError):
        stereo.sparsification(np.ones(100), np.ones(100), steps=1)


def test_rank_correlation_reads_monotone_relations():
    """순위 상관은 직선이 아니어도 같은 방향이면 1 이 나와야 합니다."""
    x = np.linspace(0.1, 5.0, 200)
    assert stereo.rank_correlation(x, np.exp(x)) == pytest.approx(1.0)
    assert stereo.rank_correlation(x, -np.exp(x)) == pytest.approx(-1.0)


def test_compute_disparity_both_recovers_a_known_shift():
    """좌우 두 기준의 시차가 모두 실제 이동량을 되찾아야 합니다."""
    rng = np.random.default_rng(5)
    left = rng.integers(0, 255, (96, 160)).astype(np.uint8)
    shift = 12
    right = np.roll(left, -shift, axis=1)     # 위와 같은 부호 규약

    dl, dr = stereo.compute_disparity_both(left, right, num_disparities=32,
                                           block_size=7)
    inner = (slice(12, -12), slice(40, -12))
    assert np.nanmedian(dl[inner]) == pytest.approx(shift, abs=0.5)
    assert np.nanmedian(dr[inner]) == pytest.approx(shift, abs=0.5)
