"""The face-model contract.

A backend never touches LiveKit. It gets 16 kHz (or whatever it asks for)
speech audio in fixed batches and returns RGBA frames; the generator in
:mod:`qav_face.stream` does buffering, pacing, silence and interruptions.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..avatars import AvatarSpec


class FaceBackend(ABC):
    #: Frames rendered per :meth:`speech_frames` call. GPU models batch for throughput
    #: (MuseTalk ~8 = 320 ms); the CPU placeholder renders one at a time.
    batch_frames: int = 1
    #: Extra frames of *future* audio the model wants for the last frame of a batch
    #: (MuseTalk's whisper window looks 2 frames ahead). Adds that much latency.
    lookahead_frames: int = 0
    #: Sample rate the model wants; the generator resamples the analysis copy.
    model_sample_rate: int = 16_000
    #: Run :meth:`speech_frames` / :meth:`idle_frame` in a worker thread so slow GPU
    #: inference never blocks the LiveKit event loop (RPC, data stream reads).
    run_in_thread: bool = False

    def __init__(self, spec: AvatarSpec) -> None:
        self.spec = spec

    @property
    def width(self) -> int:
        return self.spec.width

    @property
    def height(self) -> int:
        return self.spec.height

    @property
    def fps(self) -> float:
        return self.spec.fps

    def warmup(self) -> None:
        """Load weights / assets. Called once before the first frame is due."""

    @abstractmethod
    def idle_frame(self) -> np.ndarray:
        """One (H, W, 4) uint8 RGBA frame of the avatar *not* speaking. Called at fps while silent."""

    @abstractmethod
    def speech_frames(self, audio: np.ndarray, n_frames: int) -> list[np.ndarray]:
        """Render exactly `n_frames` RGBA frames.

        `audio` is int16 mono at :attr:`model_sample_rate` covering
        ``n_frames + lookahead_frames`` video frames, starting at the first frame
        of this batch. Backends that need *past* context keep it themselves.
        """

    def segment_end(self) -> None:
        """The current utterance finished (normally or interrupted)."""

    def reset(self) -> None:
        """Hard reset after an interruption: drop any internal audio history."""

    def close(self) -> None:
        """Release GPU memory / file handles."""
