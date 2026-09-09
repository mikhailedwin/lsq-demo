"""Drives BackendVideoGenerator exactly like AvatarRunner does and checks the contract."""

from __future__ import annotations

import asyncio

import numpy as np
import pytest
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, AvatarOptions

from qav_face.avatars import AvatarSpec
from qav_face.backends.base import FaceBackend
from qav_face.stream import BackendVideoGenerator
from qav_face.testing import synth_speech

SR = 24_000
FPS = 25


class RecordingBackend(FaceBackend):
    """Batching backend that records what audio it was handed."""

    batch_frames = 4
    lookahead_frames = 2
    model_sample_rate = 16_000

    def __init__(self, spec: AvatarSpec) -> None:
        super().__init__(spec)
        self.calls: list[tuple[int, int]] = []
        self.idle = 0
        self.ends = 0
        self.resets = 0

    def idle_frame(self) -> np.ndarray:
        self.idle += 1
        return np.zeros((self.height, self.width, 4), dtype=np.uint8)

    def speech_frames(self, audio: np.ndarray, n_frames: int) -> list[np.ndarray]:
        self.calls.append((len(audio), n_frames))
        return [np.full((self.height, self.width, 4), 255, dtype=np.uint8) for _ in range(n_frames)]

    def segment_end(self) -> None:
        self.ends += 1

    def reset(self) -> None:
        self.resets += 1


def _spec(renderer: str = "recording") -> AvatarSpec:
    return AvatarSpec(id="t", name="t", renderer=renderer, width=64, height=64, fps=FPS)


def _options() -> AvatarOptions:
    return AvatarOptions(video_width=64, video_height=64, video_fps=FPS, audio_sample_rate=SR, audio_channels=1)


async def _drive(gen: BackendVideoGenerator, pcm: np.ndarray, *, frame_ms: int = 20, stop_after_end: bool = True):
    step = SR * frame_ms // 1000
    for i in range(0, len(pcm), step):
        chunk = pcm[i : i + step]
        await gen.push_audio(rtc.AudioFrame(data=chunk.tobytes(), sample_rate=SR, num_channels=1, samples_per_channel=len(chunk)))
    await gen.push_audio(AudioSegmentEnd())

    items = []
    async for item in gen:
        items.append(item)
        if isinstance(item, AudioSegmentEnd) and stop_after_end:
            break
    return items


def test_speech_is_emitted_as_video_audio_pairs_then_segment_end() -> None:
    backend = RecordingBackend(_spec())
    gen = BackendVideoGenerator(_options(), backend)
    pcm = synth_speech("one two three four five six", sample_rate=SR)
    n_frames_expected = len(pcm) // (SR // FPS)

    items = asyncio.run(_drive(gen, pcm))

    videos = [i for i in items if isinstance(i, rtc.VideoFrame)]
    audios = [i for i in items if isinstance(i, rtc.AudioFrame)]
    assert isinstance(items[-1], AudioSegmentEnd)
    # every speech frame carries exactly one frame of audio; idle frames carry none
    assert len(audios) >= n_frames_expected - 1
    assert len(videos) >= len(audios)
    total_out = sum(a.samples_per_channel for a in audios)
    assert abs(total_out - len(pcm)) <= SR // FPS * 2, "output audio must be the input audio (whole frames)"
    # pairs are ordered video, audio
    seq = [type(i).__name__ for i in items if not isinstance(i, AudioSegmentEnd)]
    for k in range(len(seq) - 1):
        if seq[k] == "AudioFrame":
            assert seq[k - 1] == "VideoFrame"
    assert backend.ends == 1


def test_backend_gets_batches_with_lookahead_at_model_rate() -> None:
    backend = RecordingBackend(_spec())
    gen = BackendVideoGenerator(_options(), backend)
    pcm = synth_speech("alpha beta gamma delta epsilon zeta eta theta", sample_rate=SR)
    asyncio.run(_drive(gen, pcm))

    spf_model = 16_000 // FPS
    full = [c for c in backend.calls if c[1] == backend.batch_frames]
    assert full, "should have rendered full batches"
    for n_samples, n in full:
        assert n_samples == (n + backend.lookahead_frames) * spf_model
    # the tail batch may be shorter but never empty
    assert all(n > 0 for _, n in backend.calls)
    assert sum(n for _, n in backend.calls) == len(pcm) // (SR // FPS) or sum(n for _, n in backend.calls) == len(pcm) // (SR // FPS) + 1


def test_idle_frames_when_silent_and_reset_on_interrupt() -> None:
    backend = RecordingBackend(_spec())
    gen = BackendVideoGenerator(_options(), backend)

    async def run():
        items = []
        async for item in gen:
            items.append(item)
            if len(items) >= 5:
                break
        gen.clear_buffer()
        return items

    items = asyncio.run(run())
    assert all(isinstance(i, rtc.VideoFrame) for i in items)
    assert backend.idle >= 5
    assert backend.resets == 1


def test_procedural_backend_end_to_end() -> None:
    from qav_face.backends import create_backend

    backend = create_backend(_spec("procedural"))
    gen = BackendVideoGenerator(_options(), backend)
    pcm = synth_speech("hello there", sample_rate=SR)
    items = asyncio.run(_drive(gen, pcm))
    videos = [i for i in items if isinstance(i, rtc.VideoFrame)]
    assert videos and videos[0].width == 64 and videos[0].height == 64


def test_unknown_backend_is_a_clear_error() -> None:
    from qav_face.backends import create_backend

    with pytest.raises(ValueError, match="unknown face renderer"):
        create_backend(_spec("nope"))
