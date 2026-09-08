"""Audio-in → synchronised (audio, video) frames out, for LiveKit's AvatarRunner.

Modelled on the reference `audio_wave` avatar example from livekit/agents: audio
is buffered and sliced into exactly one video frame's worth of samples, each
slice is analysed for mouth shape, and a frame is rendered for it. When no
audio is queued we emit idle frames so the face keeps breathing and blinking.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator, Generator

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, AvatarOptions, VideoGenerator

from .lipsync import LipSyncAnalyzer
from .renderers.base import FaceRenderer

logger = logging.getLogger("qav.avatar.generator")


class FaceVideoGenerator(VideoGenerator):
    def __init__(self, options: AvatarOptions, renderer: FaceRenderer) -> None:
        self._options = options
        self._renderer = renderer
        self._analyzer = LipSyncAnalyzer(options.audio_sample_rate, options.video_fps)
        self._audio_queue: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()
        self._resampler: rtc.AudioResampler | None = None
        self._buffer = np.zeros((0, options.audio_channels), dtype=np.int16)
        self._samples_per_frame = int(options.audio_sample_rate / options.video_fps)
        self._av_sync: rtc.AVSynchronizer | None = None
        self._render_ms_ema = 0.0
        self._frames = 0

    # ---------------------------------------------------------------- inputs
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            if self._resampler is not None:
                for f in self._resampler.flush():
                    await self._audio_queue.put(f)
            await self._audio_queue.put(frame)
            return

        needs_resample = (
            frame.sample_rate != self._options.audio_sample_rate
            or frame.num_channels != self._options.audio_channels
        )
        if needs_resample and self._resampler is None:
            self._resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=self._options.audio_sample_rate,
                num_channels=self._options.audio_channels,
            )
        if self._resampler is not None:
            for f in self._resampler.push(frame):
                await self._audio_queue.put(f)
        else:
            await self._audio_queue.put(frame)

    def clear_buffer(self) -> None:
        while not self._audio_queue.empty():
            try:
                self._audio_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._reset_buffer()
        self._analyzer.reset()

    def set_av_sync(self, av_sync: rtc.AVSynchronizer | None) -> None:
        self._av_sync = av_sync

    @property
    def render_ms(self) -> float:
        return self._render_ms_ema

    # --------------------------------------------------------------- outputs
    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream()

    async def _stream(self) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        self._renderer.warmup()
        frame_interval = 1.0 / self._options.video_fps
        while True:
            try:
                item = await asyncio.wait_for(self._audio_queue.get(), timeout=0.5 * frame_interval)
            except asyncio.TimeoutError:
                # nothing to say: keep the face alive, but don't outrun the synchroniser
                if self._av_sync is not None and self._av_sync._video_queue.qsize() > 1:
                    continue
                yield self._render(self._analyzer.update_idle())
                await asyncio.sleep(0)
                continue

            for video_frame, audio_frame in self._active_frames(item):
                yield video_frame
                yield audio_frame

            if isinstance(item, AudioSegmentEnd):
                yield item
                self._reset_buffer()

    def _active_frames(
        self, item: rtc.AudioFrame | AudioSegmentEnd
    ) -> Generator[tuple[rtc.VideoFrame, rtc.AudioFrame], None, None]:
        spf = self._samples_per_frame
        if isinstance(item, rtc.AudioFrame):
            samples = np.frombuffer(item.data, dtype=np.int16).reshape(-1, item.num_channels)
        else:
            # pad the tail so the last partial frame still gets drawn
            fill = (spf - len(self._buffer) % spf) if len(self._buffer) % spf else 0
            samples = np.zeros((fill, self._buffer.shape[1]), dtype=np.int16)
        self._buffer = np.concatenate([self._buffer, samples], axis=0)

        while len(self._buffer) >= spf:
            chunk = self._buffer[:spf]
            self._buffer = self._buffer[spf:]
            state = self._analyzer.update_audio(chunk[:, 0])
            video = self._render(state)
            audio = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=self._options.audio_sample_rate,
                num_channels=chunk.shape[1],
                samples_per_channel=chunk.shape[0],
            )
            yield video, audio

    def _render(self, state) -> rtc.VideoFrame:  # type: ignore[no-untyped-def]
        t0 = time.perf_counter()
        rgba = self._renderer.render(state)
        ms = (time.perf_counter() - t0) * 1000.0
        self._render_ms_ema = ms if self._frames == 0 else self._render_ms_ema * 0.95 + ms * 0.05
        self._frames += 1
        if self._frames % 500 == 0:
            logger.debug("render avg %.1f ms/frame", self._render_ms_ema)
        if not rgba.flags["C_CONTIGUOUS"]:
            rgba = np.ascontiguousarray(rgba)
        return rtc.VideoFrame(
            width=rgba.shape[1],
            height=rgba.shape[0],
            type=rtc.VideoBufferType.RGBA,
            data=rgba.tobytes(),
        )

    def _reset_buffer(self) -> None:
        self._buffer = np.zeros((0, self._options.audio_channels), dtype=np.int16)
