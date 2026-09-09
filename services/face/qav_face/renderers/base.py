from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..avatars import AvatarStyle
from ..lipsync import FaceState


class FaceRenderer(ABC):
    """Produces one RGBA frame per FaceState. Implementations must be fast enough for real time
    on the hardware they target: the generator calls `render` once per video frame."""

    def __init__(self, *, width: int, height: int, style: AvatarStyle, seed: int | None = None) -> None:
        self.width = width
        self.height = height
        self.style = style
        self.seed = seed

    def warmup(self) -> None:
        """Pre-build caches / load weights before the first frame is due."""

    @abstractmethod
    def render(self, state: FaceState) -> np.ndarray:
        """Return an (height, width, 4) uint8 RGBA array."""

    def close(self) -> None:
        """Release GPU memory / file handles."""
