"""Quality tooling for generated faces — no GPU, no vendor code, pure numpy.

Three things live here, all testable on a laptop:

* :class:`TemporalSmoother` — kills the single-frame flicker MuseTalk's own
  README lists as a known issue, without smearing real mouth motion.
* :func:`find_seamless_loop` — picks an idle loop whose last frame matches its
  first, instead of the naive ping-pong that plays the clip backwards.
* metrics (:func:`sharpness`, :func:`temporal_jitter`, :func:`region_report`)
  that turn "does it look good" into numbers the selftest can print.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Mouth region as a fraction of a face crop (x0, y0, x1, y1). MuseTalk only
# regenerates the lower half, so this is where its artifacts live.
MOUTH_BOX = (0.22, 0.55, 0.78, 0.95)


def _gray(img: np.ndarray) -> np.ndarray:
    a = img.astype(np.float32)
    if a.ndim == 2:
        return a
    return 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]


def crop_fraction(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian — the standard blur metric. Higher is sharper."""
    g = _gray(img)
    if g.shape[0] < 3 or g.shape[1] < 3:
        return 0.0
    lap = (
        -4.0 * g[1:-1, 1:-1]
        + g[:-2, 1:-1]
        + g[2:, 1:-1]
        + g[1:-1, :-2]
        + g[1:-1, 2:]
    )
    return float(lap.var())


def temporal_jitter(frames: list[np.ndarray], box: tuple[float, float, float, float] | None = None) -> float:
    """How much a region flickers frame to frame beyond its smooth motion.

    Each frame is compared against the average of its neighbours, so steady
    motion cancels and per-frame noise doesn't. The result is divided by the
    region's own spatial contrast, which makes it comparable across different
    footage — a dark, flat face and a bright, detailed one are judged the same.

    Measured reference points (normalized):

    ==========================  ======
    still head + sensor noise     0.03
    1 px/frame drift              0.14
    visible flicker               0.34
    severe flicker                1.05
    ==========================  ======
    """
    if len(frames) < 3:
        return 0.0
    seq = [_gray(crop_fraction(f, box)) if box else _gray(f) for f in frames]
    shape = seq[0].shape
    if any(s.shape != shape for s in seq):
        return 0.0
    stack = np.stack(seq)
    mid = stack[1:-1]
    smooth = 0.5 * (stack[:-2] + stack[2:])
    contrast = float(stack[0].std())
    if contrast < 1e-3:
        return 0.0
    return float(np.abs(mid - smooth).mean() / contrast)


@dataclass
class RegionReport:
    mouth_sharpness: float
    face_sharpness: float
    #: mouth / face sharpness. < 1 means the generated mouth is softer than the
    #: real footage around it — the artifact face restoration is meant to fix.
    sharpness_ratio: float
    #: contrast-normalized; see :func:`temporal_jitter` for reference points
    jitter: float

    @property
    def verdict(self) -> str:
        if self.sharpness_ratio < 0.45:
            return "soft mouth — enable face restoration (QAV_FACE_RESTORE=1)"
        if self.jitter > 0.25:
            return "visible flicker — raise QAV_FACE_SMOOTH"
        if self.sharpness_ratio < 0.7:
            return "acceptable, restoration would still help"
        return "good"


def region_report(frames: list[np.ndarray]) -> RegionReport:
    """Score a rendered sequence the way a viewer would: is the mouth as sharp as
    the face around it, and does it sit still when it should?"""
    if not frames:
        return RegionReport(0.0, 0.0, 0.0, 0.0)
    mouth = float(np.mean([sharpness(crop_fraction(f, MOUTH_BOX)) for f in frames]))
    face = float(np.mean([sharpness(f) for f in frames]))
    return RegionReport(
        mouth_sharpness=mouth,
        face_sharpness=face,
        sharpness_ratio=mouth / face if face > 1e-6 else 0.0,
        jitter=temporal_jitter(frames, MOUTH_BOX),
    )


class TemporalSmoother:
    """Adaptive temporal filter for the generated face crop.

    MuseTalk generates every frame independently, so its output flickers — its
    own README lists this as a known issue. A flat 3-tap average would fix the
    flicker and destroy the lip sync, because fast mouth transitions are exactly
    the signal we must keep.

    Amplitude can't tell the two apart: flicker is often *larger* than real
    articulation. Frequency can. Independent-per-frame noise alternates every
    frame (period 2 = 12.5 Hz at 25 fps); speech articulates at 4-8 Hz, i.e.
    period 3-6 frames. The 3-tap [1,2,1]/4 kernel nulls period 2 exactly while
    passing period 6 at ~75%, which is precisely the split we want.

    On top of that the filter is gated per pixel: it only applies where frame
    i-1 and i+1 agree with each other. Where they differ something is genuinely
    moving, and that pixel is passed through untouched — so a head turn or a
    fast jaw drop keeps its edges.

    Works within a batch (no added latency): every frame but the last has a real
    successor, and the last frame carries over as the next batch's history.
    `strength` 0 disables it entirely.
    """

    def __init__(self, strength: float = 0.6, motion_threshold: float = 10.0) -> None:
        self.strength = float(np.clip(strength, 0.0, 1.0))
        #: per-pixel neighbour disagreement (0-255) above which we assume real motion
        self.motion_threshold = motion_threshold
        self._prev: np.ndarray | None = None

    def reset(self) -> None:
        self._prev = None

    def __call__(self, frames: list[np.ndarray]) -> list[np.ndarray]:
        if self.strength <= 0.0 or not frames:
            if frames:
                self._prev = frames[-1]
            return frames

        prev = self._prev
        if prev is not None and prev.shape != frames[0].shape:
            prev = None  # resolution changed; start fresh rather than blend garbage
        self._prev = frames[-1]

        out: list[np.ndarray] = []
        for i, frame in enumerate(frames):
            before = prev if i == 0 else frames[i - 1]
            after = frames[i + 1] if i + 1 < len(frames) else None
            if before is None or after is None:
                # first frame of the very first batch, or the last of this one:
                # no bracketing pair, so pass it through untouched
                out.append(frame)
                continue
            cur = frame.astype(np.float32)
            b = before.astype(np.float32)
            a = after.astype(np.float32)
            # [1,2,1]/4 — nulls per-frame alternation, keeps speech-rate motion
            filtered = 0.25 * b + 0.5 * cur + 0.25 * a
            # how much the neighbours disagree with *each other* = real motion
            motion = np.abs(a - b)
            motion = motion.mean(axis=2, keepdims=True) if motion.ndim == 3 else motion[..., None]
            weight = np.clip(1.0 - motion / self.motion_threshold, 0.0, 1.0) * self.strength
            blended = cur * (1.0 - weight) + filtered * weight
            out.append(np.clip(blended, 0, 255).astype(frame.dtype))
        return out


def find_seamless_loop(
    frames: list[np.ndarray],
    *,
    min_frames: int = 50,
    thumb: int = 32,
) -> tuple[int, int]:
    """Pick ``[start, end)`` whose wrap-around is least visible.

    The reference implementation loops a clip by appending it reversed, which
    plays the person's head motion backwards — clearly unnatural on a talking
    head. Instead we find the pair of frames that look most alike (and are
    moving alike, so the loop doesn't stutter) and cut there, giving a clip
    that plays forward forever.

    Returns the whole range when no good seam exists, so callers always get
    something usable.
    """
    n = len(frames)
    if n <= min_frames + 2:
        return 0, n

    thumbs = np.stack([_downsample(_gray(f), thumb) for f in frames])
    thumbs = (thumbs - thumbs.mean()) / (thumbs.std() + 1e-6)
    flat = thumbs.reshape(n, -1)
    # velocity, so we match "moving the same way", not just "looking the same"
    vel = np.zeros_like(flat)
    vel[1:] = flat[1:] - flat[:-1]

    best = (0, n)
    best_cost = np.inf
    # appearance + motion distance between every candidate start and end
    for start in range(0, n - min_frames):
        ends = np.arange(start + min_frames, n)
        app = np.abs(flat[ends] - flat[start]).mean(axis=1)
        mot = np.abs(vel[ends] - vel[start]).mean(axis=1)
        cost = app + 0.5 * mot
        # mild preference for longer loops: a 2 s loop reads as a loop, 10 s doesn't
        cost = cost * (1.0 + 0.35 * (1.0 - (ends - start) / n))
        k = int(np.argmin(cost))
        if cost[k] < best_cost:
            best_cost = float(cost[k])
            best = (start, int(ends[k]))
    return best


def _downsample(g: np.ndarray, size: int) -> np.ndarray:
    """Box-downsample a 2-D array to `size`×`size` without scipy/cv2."""
    h, w = g.shape
    ys = np.linspace(0, h, size + 1).astype(int)
    xs = np.linspace(0, w, size + 1).astype(int)
    out = np.empty((size, size), dtype=np.float32)
    for i in range(size):
        for j in range(size):
            block = g[ys[i] : max(ys[i] + 1, ys[i + 1]), xs[j] : max(xs[j] + 1, xs[j + 1])]
            out[i, j] = block.mean()
    return out
