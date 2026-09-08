"""Offline stand-ins so the whole WebRTC/avatar path runs with zero vendor keys.

MockTTS turns text into a "syllable" tone envelope — enough energy variation
for the face renderer to lip-sync against, and clearly audible in the browser.
"""

from __future__ import annotations

import math
import re
import uuid

import numpy as np
from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS

from .protocol import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE


class MockTTS(tts.TTS):
    def __init__(self, *, base_pitch_hz: float = 180.0, wpm: float = 170.0) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=False),
            sample_rate=AUDIO_SAMPLE_RATE,
            num_channels=AUDIO_CHANNELS,
        )
        self._base_pitch = base_pitch_hz
        self._wpm = wpm

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> tts.ChunkedStream:
        return _MockChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def render_pcm(self, text: str) -> bytes:
        """Deterministic int16 mono PCM for `text`. Public so tests can reuse it."""
        sr = self.sample_rate
        words = [w for w in re.split(r"\s+", text.strip()) if w]
        if not words:
            return b""
        word_len = 60.0 / self._wpm  # seconds per word incl. gap
        chunks: list[np.ndarray] = []
        for i, word in enumerate(words):
            syllables = max(1, len(re.findall(r"[aeiouy]+", word.lower())))
            voiced = word_len * 0.72
            syl = voiced / syllables
            pitch = self._base_pitch * (1.0 + 0.08 * math.sin(i * 0.9)) * (1.15 if word.endswith("?") else 1.0)
            for s in range(syllables):
                n = int(sr * syl)
                t = np.arange(n) / sr
                # attack/decay envelope so every syllable reads as a distinct mouth pulse
                env = np.minimum(t / (syl * 0.25), 1.0) * np.minimum((syl - t) / (syl * 0.35), 1.0)
                env = np.clip(env, 0.0, 1.0)
                f = pitch * (1.0 + 0.05 * s)
                wave = (
                    0.55 * np.sin(2 * np.pi * f * t)
                    + 0.25 * np.sin(2 * np.pi * 2 * f * t)
                    + 0.12 * np.sin(2 * np.pi * 3 * f * t)
                )
                chunks.append((0.6 * env * wave))
            gap = np.zeros(int(sr * (word_len - voiced)))
            if word.endswith((".", "!", "?", ",")):
                gap = np.zeros(int(sr * word_len * 0.8))
            chunks.append(gap)
        audio = np.concatenate(chunks)
        return (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


class _MockChunkedStream(tts.ChunkedStream):
    def __init__(self, *, tts: MockTTS, input_text: str, conn_options: APIConnectOptions) -> None:
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._mock = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=uuid.uuid4().hex,
            sample_rate=self._mock.sample_rate,
            num_channels=self._mock.num_channels,
            mime_type="audio/pcm",
        )
        pcm = self._mock.render_pcm(self.input_text)
        # push in ~100 ms slices so playback can start before the whole utterance is rendered
        step = self._mock.sample_rate // 10 * 2
        for i in range(0, len(pcm), step):
            output_emitter.push(pcm[i : i + step])
        output_emitter.flush()
