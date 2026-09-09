"""The quality pipeline is what stands between a photoreal face and a demo that
gets laughed at, and it is all pure numpy — so it gets tested properly here,
GPU or no GPU."""

from __future__ import annotations

import numpy as np
import pytest

from qav_face.quality import (
    MOUTH_BOX,
    TemporalSmoother,
    crop_fraction,
    find_seamless_loop,
    region_report,
    sharpness,
    temporal_jitter,
)

RNG = np.random.default_rng(7)


def _face(seed: int = 0, mouth_blur: bool = False, size: int = 128) -> np.ndarray:
    """A synthetic 'face' — spatially smooth like a photograph, with fine detail
    on top. White noise would break every metric here, and isn't what a camera
    produces; `seed` shifts the content slightly, as consecutive frames do."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    phase = seed * 0.15
    base = (
        120
        + 55 * np.sin(2 * np.pi * (xx / size) * 2 + phase)
        + 40 * np.cos(2 * np.pi * (yy / size) * 3 - phase)
    )
    detail = 18 * np.sin(2 * np.pi * (xx + yy) / 6.0) + rng.normal(0, 4, (size, size))
    img = np.clip(base + detail, 0, 255).astype(np.uint8)
    img = np.repeat(img[..., None], 3, axis=2)
    if mouth_blur:
        y0, y1 = int(MOUTH_BOX[1] * size), int(MOUTH_BOX[3] * size)
        x0, x1 = int(MOUTH_BOX[0] * size), int(MOUTH_BOX[2] * size)
        region = img[y0:y1, x0:x1].astype(np.float32)
        for _ in range(6):  # cheap box blur
            region[1:-1, 1:-1] = (region[:-2, 1:-1] + region[2:, 1:-1] + region[1:-1, :-2] + region[1:-1, 2:]) / 4
        img[y0:y1, x0:x1] = region.astype(np.uint8)
    return img


# --------------------------------------------------------------------- metrics
def test_sharpness_separates_sharp_from_blurred() -> None:
    sharp = _face(1)
    blurred = _face(1, mouth_blur=True)
    assert sharpness(crop_fraction(sharp, MOUTH_BOX)) > 5 * sharpness(crop_fraction(blurred, MOUTH_BOX))


def test_region_report_flags_a_soft_mouth() -> None:
    # a near-still talking head: same face, drifting a pixel at a time
    good = region_report([np.roll(_face(1), i, axis=1) for i in range(6)])
    bad = region_report([np.roll(_face(1, mouth_blur=True), i, axis=1) for i in range(6)])
    assert good.sharpness_ratio > 0.7, "an evenly detailed face should score near 1.0"
    assert bad.sharpness_ratio < 0.45
    assert "restoration" in bad.verdict
    assert good.verdict == "good"


def test_temporal_jitter_ignores_smooth_motion_but_catches_flicker() -> None:
    base = _face(3)
    # a slow pan, as a real head drifts: consistent direction, ~1 px per frame
    panning = [np.roll(base, i, axis=1) for i in range(8)]
    # flicker: alternating brightness with no real motion
    flicker = [np.clip(base.astype(np.int16) + (25 if i % 2 else -25), 0, 255).astype(np.uint8) for i in range(8)]
    assert temporal_jitter(flicker) > 5 * temporal_jitter(panning)


def test_metrics_are_safe_on_degenerate_input() -> None:
    assert sharpness(np.zeros((2, 2, 3), dtype=np.uint8)) == 0.0
    assert temporal_jitter([]) == 0.0
    assert temporal_jitter([_face(0)]) == 0.0
    assert region_report([]).sharpness_ratio == 0.0


# ------------------------------------------------------------------- smoothing
def test_smoother_removes_flicker_from_static_content() -> None:
    base = _face(5)
    flicker = [np.clip(base.astype(np.int16) + (18 if i % 2 else -18), 0, 255).astype(np.uint8) for i in range(10)]
    smoothed = TemporalSmoother(strength=0.9)(flicker)
    assert temporal_jitter(smoothed) < 0.5 * temporal_jitter(flicker)


def test_smoother_preserves_real_mouth_motion() -> None:
    """The whole point of the frequency split: articulation at a speech rate
    (here period 6 ≈ 4 Hz at 25 fps) must survive at close to full amplitude."""
    size = 96
    frames = []
    for i in range(18):
        f = np.full((size, size, 3), 120, dtype=np.uint8)
        open_px = 4 + (20 if i % 6 < 3 else 0)  # mouth opening ~4 times a second
        f[size // 2 : size // 2 + open_px, 20:76] = 10
        frames.append(f)
    smoothed = TemporalSmoother(strength=0.9)(frames)
    area = lambda fs: [int((f < 60).sum()) for f in fs]  # noqa: E731
    swing_in = max(area(frames)) - min(area(frames))
    swing_out = max(area(smoothed)) - min(area(smoothed))
    assert swing_out > 0.6 * swing_in, f"articulation was flattened: {swing_out} vs {swing_in}"


def test_smoother_is_a_noop_at_zero_strength_and_across_resets() -> None:
    frames = [_face(i) for i in range(4)]
    assert all(np.array_equal(a, b) for a, b in zip(TemporalSmoother(strength=0.0)(frames), frames))
    s = TemporalSmoother(strength=0.8)
    first = s(frames)[0]
    s.reset()
    assert np.array_equal(s(frames)[0], first), "after reset the first frame passes through untouched"


def test_smoother_handles_a_size_change_without_crashing() -> None:
    s = TemporalSmoother(strength=0.8)
    s([_face(0, size=64)])
    out = s([_face(1, size=128)])
    assert out[0].shape == (128, 128, 3)


# ----------------------------------------------------------------- loop finder
def test_seamless_loop_finds_the_repeat_in_a_periodic_clip() -> None:
    # a 40-frame cycle repeated: the ideal loop is one full period
    period = 40
    frames = []
    for i in range(140):
        f = np.zeros((48, 48, 3), dtype=np.uint8)
        col = int(20 + 15 * np.sin(2 * np.pi * (i % period) / period))
        f[:, :, :] = col
        f[10:20, (i % period) : (i % period) + 6] = 220
        frames.append(f)
    start, end = find_seamless_loop(frames, min_frames=20)
    assert end > start
    assert abs((end - start) - period) <= 3, f"expected a ~{period}-frame loop, got {end - start}"


def test_seamless_loop_end_matches_start_better_than_a_naive_cut() -> None:
    frames = [np.full((40, 40, 3), int(30 + 60 * (i % 25) / 25), dtype=np.uint8) for i in range(120)]
    start, end = find_seamless_loop(frames, min_frames=15)
    seam = abs(int(frames[start][0, 0, 0]) - int(frames[end % len(frames)][0, 0, 0]))
    worst = abs(int(frames[0][0, 0, 0]) - int(frames[len(frames) - 1][0, 0, 0]))
    assert seam <= worst


def test_seamless_loop_degrades_gracefully_on_short_input() -> None:
    frames = [_face(i, size=32) for i in range(5)]
    assert find_seamless_loop(frames, min_frames=50) == (0, 5)


# --------------------------------------------------------------- restorer glue
def test_restorer_is_a_transparent_noop_when_unavailable() -> None:
    from qav_face.restore import FaceRestorer

    r = FaceRestorer(mode="auto", weight_path="/nonexistent.pth")
    r.warmup(_face(0))
    assert r.enabled is False
    crops = [_face(1), _face(2)]
    assert all(np.array_equal(a, b) for a, b in zip(r(crops), crops))


def test_restorer_respects_an_explicit_off(monkeypatch: pytest.MonkeyPatch) -> None:
    from qav_face.restore import FaceRestorer

    monkeypatch.setenv("QAV_FACE_RESTORE", "0")
    r = FaceRestorer.from_env(fps=25)
    assert r.mode == "off"
    r.warmup(_face(0))
    assert r.enabled is False


def test_restorer_budget_defaults_to_half_a_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    from qav_face.restore import FaceRestorer

    monkeypatch.delenv("QAV_FACE_RESTORE_BUDGET_MS", raising=False)
    assert FaceRestorer.from_env(fps=25).frame_budget_ms == pytest.approx(20.0)
