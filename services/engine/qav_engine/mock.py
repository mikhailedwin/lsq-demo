"""Offline stand-ins so the whole WebRTC/avatar path runs with zero vendor keys.

MockTTS turns text into a "syllable" tone envelope — enough energy variation
for the face renderer to lip-sync against, and clearly audible in the browser.
"""

from __future__ import annotations

import uuid

from livekit.agents import APIConnectOptions, tts
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS
from qav_face.testing import synth_speech

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
        """Deterministic int16 mono PCM for `text`."""
        return synth_speech(text, sample_rate=self.sample_rate, base_pitch_hz=self._base_pitch, wpm=self._wpm).tobytes()


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
