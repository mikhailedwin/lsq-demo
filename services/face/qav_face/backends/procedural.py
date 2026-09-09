"""CPU placeholder backend: audio → animation parameters → stylised 2D face."""

from __future__ import annotations

import numpy as np

from ..avatars import AvatarSpec
from ..lipsync import LipSyncAnalyzer
from ..renderers import get_renderer
from .base import FaceBackend


class ProceduralBackend(FaceBackend):
    batch_frames = 1
    lookahead_frames = 0
    model_sample_rate = 24_000

    def __init__(self, spec: AvatarSpec) -> None:
        super().__init__(spec)
        self._renderer = get_renderer("procedural", width=spec.width, height=spec.height, style=spec.style)
        self._analyzer = LipSyncAnalyzer(self.model_sample_rate, spec.fps)
        self._spf = int(self.model_sample_rate / spec.fps)

    def warmup(self) -> None:
        self._renderer.warmup()

    def idle_frame(self) -> np.ndarray:
        return self._renderer.render(self._analyzer.update_idle())

    def speech_frames(self, audio: np.ndarray, n_frames: int) -> list[np.ndarray]:
        out = []
        for i in range(n_frames):
            chunk = audio[i * self._spf : (i + 1) * self._spf]
            if len(chunk) < self._spf:
                chunk = np.pad(chunk, (0, self._spf - len(chunk)))
            out.append(self._renderer.render(self._analyzer.update_audio(chunk)))
        return out

    def reset(self) -> None:
        self._analyzer.reset()

    def close(self) -> None:
        self._renderer.close()
