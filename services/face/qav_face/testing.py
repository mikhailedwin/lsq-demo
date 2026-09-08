"""Synthetic speech for tests and offline demos: a syllable tone envelope with
enough energy variation for lip-sync, clearly audible, and deterministic."""

from __future__ import annotations

import math
import re

import numpy as np


def synth_speech(text: str, *, sample_rate: int = 24_000, base_pitch_hz: float = 180.0, wpm: float = 170.0) -> np.ndarray:
    """Return int16 mono PCM for `text`."""
    sr = sample_rate
    words = [w for w in re.split(r"\s+", text.strip()) if w]
    if not words:
        return np.zeros(0, dtype=np.int16)
    word_len = 60.0 / wpm  # seconds per word incl. gap
    chunks: list[np.ndarray] = []
    for i, word in enumerate(words):
        syllables = max(1, len(re.findall(r"[aeiouy]+", word.lower())))
        voiced = word_len * 0.72
        syl = voiced / syllables
        pitch = base_pitch_hz * (1.0 + 0.08 * math.sin(i * 0.9)) * (1.15 if word.endswith("?") else 1.0)
        for s in range(syllables):
            n = int(sr * syl)
            t = np.arange(n) / sr
            env = np.minimum(t / (syl * 0.25), 1.0) * np.minimum((syl - t) / (syl * 0.35), 1.0)
            env = np.clip(env, 0.0, 1.0)
            f = pitch * (1.0 + 0.05 * s)
            wave = (
                0.55 * np.sin(2 * np.pi * f * t)
                + 0.25 * np.sin(2 * np.pi * 2 * f * t)
                + 0.12 * np.sin(2 * np.pi * 3 * f * t)
            )
            chunks.append(0.6 * env * wave)
        gap = np.zeros(int(sr * (word_len - voiced)))
        if word.endswith((".", "!", "?", ",")):
            gap = np.zeros(int(sr * word_len * 0.8))
        chunks.append(gap)
    audio = np.concatenate(chunks)
    return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
