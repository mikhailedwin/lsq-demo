"""Audio → face-state parameters, plus idle life (blinks, sway) when silent.

This is the "animation brain" every renderer consumes. It is deliberately
renderer-agnostic so a neural face (MuseTalk, Wav2Lip, ...) can either use
the audio directly or reuse these parameters for head/eye motion.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceState:
    t: float = 0.0
    speaking: bool = False
    mouth_open: float = 0.0
    """0 = closed, 1 = fully open."""
    mouth_width: float = 0.5
    """0 = rounded ("oo"), 1 = wide ("ee")."""
    teeth: float = 0.0
    """0..1 amount of upper teeth visible (fricatives / wide vowels)."""
    blink: float = 0.0
    """0 = eyes open, 1 = eyelids closed."""
    brow_raise: float = 0.0
    head_yaw: float = 0.0
    """-1..1, positive = turned to viewer's right."""
    head_pitch: float = 0.0
    """-1..1, positive = nod down."""
    head_roll: float = 0.0
    gaze_x: float = 0.0
    gaze_y: float = 0.0
    energy: float = 0.0
    """Smoothed loudness 0..1, handy for extra effects."""
    extras: dict[str, float] = field(default_factory=dict)


class LipSyncAnalyzer:
    """Stateful per-frame analysis. Call `update_audio` when speaking, `update_idle` otherwise."""

    def __init__(self, sample_rate: int, fps: float, seed: int | None = None) -> None:
        self.sample_rate = sample_rate
        self.fps = fps
        self.dt = 1.0 / fps
        self._rng = random.Random(seed)
        self.state = FaceState()

        self._open_s = 0.0
        self._width_s = 0.5
        self._teeth_s = 0.0
        self._energy_s = 0.0
        self._noise_floor = 0.004
        self._peak = 0.08

        self._t = 0.0
        self._blink_t = 0.0
        self._next_blink = self._rng.uniform(1.5, 4.5)
        self._blink_phase = -1.0  # <0 idle, else progress 0..1
        self._gaze_target = (0.0, 0.0)
        self._gaze = (0.0, 0.0)
        self._next_gaze = self._rng.uniform(1.0, 3.0)
        self._nod = 0.0
        self._sway_seed = self._rng.uniform(0, 100)

    # ------------------------------------------------------------------ audio
    def update_audio(self, samples: np.ndarray) -> FaceState:
        """`samples`: int16 mono for exactly one video frame of audio."""
        x = samples.astype(np.float32) / 32768.0
        if x.ndim > 1:
            x = x.mean(axis=1)
        rms = float(np.sqrt(np.mean(x * x) + 1e-12))

        # adaptive normalisation: track the utterance peak, decay slowly
        self._peak = max(self._peak * 0.985, rms, 0.02)
        self._noise_floor = min(self._noise_floor * 1.02, rms) if rms > 0 else self._noise_floor
        level = (rms - self._noise_floor) / max(self._peak - self._noise_floor, 1e-4)
        level = float(np.clip(level, 0.0, 1.0)) ** 0.75

        # spectral features for a crude viseme: centroid → mouth width, HF ratio → teeth
        width_target, teeth_target = self._spectral_shape(x, level)

        # asymmetric smoothing: mouths open fast and close a little slower
        a_open = 0.65 if level > self._open_s else 0.35
        self._open_s += (level - self._open_s) * a_open
        self._width_s += (width_target - self._width_s) * 0.35
        self._teeth_s += (teeth_target - self._teeth_s) * 0.4
        self._energy_s += (level - self._energy_s) * 0.2

        # small nod on syllable onsets
        if level - self._energy_s > 0.35:
            self._nod = min(1.0, self._nod + 0.6)

        return self._finish(speaking=True)

    def _spectral_shape(self, x: np.ndarray, level: float) -> tuple[float, float]:
        if level < 0.05 or len(x) < 64:
            return 0.5, 0.0
        win = np.hanning(len(x))
        spec = np.abs(np.fft.rfft(x * win))
        freqs = np.fft.rfftfreq(len(x), 1.0 / self.sample_rate)
        total = float(spec.sum()) + 1e-9
        centroid = float((spec * freqs).sum() / total)
        hf = float(spec[freqs > 3500].sum() / total)
        # 250 Hz → 0 (round), 2200 Hz → 1 (wide)
        width = float(np.clip((centroid - 250.0) / 1950.0, 0.0, 1.0))
        teeth = float(np.clip((hf - 0.15) / 0.35, 0.0, 1.0))
        return width, teeth

    # ------------------------------------------------------------------- idle
    def update_idle(self) -> FaceState:
        self._open_s *= 0.55
        self._width_s += (0.5 - self._width_s) * 0.2
        self._teeth_s *= 0.6
        self._energy_s *= 0.8
        return self._finish(speaking=False)

    def reset(self) -> None:
        self._open_s = 0.0
        self._teeth_s = 0.0
        self._energy_s = 0.0
        self._peak = 0.08

    # ----------------------------------------------------------------- shared
    def _finish(self, *, speaking: bool) -> FaceState:
        self._t += self.dt
        t = self._t

        # blinks: quick close, slightly slower open
        self._blink_t += self.dt
        if self._blink_phase < 0 and self._blink_t >= self._next_blink:
            self._blink_phase = 0.0
        blink = 0.0
        if self._blink_phase >= 0:
            self._blink_phase += self.dt / 0.16
            p = self._blink_phase
            blink = math.sin(min(p, 1.0) * math.pi) ** 0.8
            if p >= 1.0:
                self._blink_phase = -1.0
                self._blink_t = 0.0
                self._next_blink = self._rng.uniform(1.8, 5.0)

        # gaze wanders a little when listening, locks on when speaking
        if t >= self._next_gaze:
            self._next_gaze = t + self._rng.uniform(1.2, 3.5)
            self._gaze_target = (
                (self._rng.uniform(-0.35, 0.35), self._rng.uniform(-0.2, 0.2))
                if not speaking
                else (self._rng.uniform(-0.1, 0.1), self._rng.uniform(-0.05, 0.05))
            )
        gx = self._gaze[0] + (self._gaze_target[0] - self._gaze[0]) * 0.12
        gy = self._gaze[1] + (self._gaze_target[1] - self._gaze[1]) * 0.12
        self._gaze = (gx, gy)

        # slow layered sway; speaking adds a touch more life
        s = self._sway_seed
        amp = 1.0 if speaking else 0.7
        yaw = amp * (0.35 * math.sin(0.45 * t + s) + 0.15 * math.sin(1.3 * t + 2 * s))
        pitch = amp * (0.2 * math.sin(0.6 * t + 0.7 * s)) + 0.35 * self._nod
        roll = amp * 0.15 * math.sin(0.3 * t + 1.9 * s)
        self._nod *= 0.85

        brow = 0.0
        if speaking:
            brow = float(np.clip((self._energy_s - 0.45) * 1.6, 0.0, 1.0))
        if self._rng.random() < 0.002:
            self._nod = min(1.0, self._nod + 0.5)  # occasional acknowledging nod

        st = self.state
        st.t = t
        st.speaking = speaking
        st.mouth_open = float(np.clip(self._open_s, 0.0, 1.0))
        st.mouth_width = float(np.clip(self._width_s, 0.0, 1.0))
        st.teeth = float(np.clip(self._teeth_s, 0.0, 1.0))
        st.blink = float(np.clip(blink, 0.0, 1.0))
        st.brow_raise = brow
        st.head_yaw = float(np.clip(yaw, -1.0, 1.0))
        st.head_pitch = float(np.clip(pitch, -1.0, 1.0))
        st.head_roll = float(np.clip(roll, -1.0, 1.0))
        st.gaze_x = gx
        st.gaze_y = gy
        st.energy = float(np.clip(self._energy_s, 0.0, 1.0))
        return st
