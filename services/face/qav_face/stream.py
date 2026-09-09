"""LiveKit audio in → synchronised (video, audio) frames out, for any FaceBackend.

Modelled on the reference `audio_wave` avatar example from livekit/agents:
audio is buffered and cut into whole video frames; each batch of frames is
handed to the backend together with the lookahead it asked for; when nothing
is queued we emit idle frames so the face keeps living. Output audio stays
at the wire rate (24 kHz); the backend gets a resampled analysis copy.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncGenerator, AsyncIterator

import numpy as np
from livekit import rtc
from livekit.agents.voice.avatar import AudioSegmentEnd, AvatarOptions, VideoGenerator

from .backends.base import FaceBackend

logger = logging.getLogger("qav.face.stream")


class BackendVideoGenerator(VideoGenerator):
    def __init__(self, options: AvatarOptions, backend: FaceBackend) -> None:
        self._options = options
        self._backend = backend
        self._queue: asyncio.Queue[rtc.AudioFrame | AudioSegmentEnd] = asyncio.Queue()
        self._out_resampler: rtc.AudioResampler | None = None
        self._model_resampler: rtc.AudioResampler | None = None
        self._out_buf = np.zeros(0, dtype=np.int16)  # wire-rate mono
        self._model_buf = np.zeros(0, dtype=np.int16)  # model-rate mono
        self._spf_out = int(options.audio_sample_rate / options.video_fps)
        self._spf_model = int(backend.model_sample_rate / options.video_fps)
        self._av_sync: rtc.AVSynchronizer | None = None
        self._render_ms = 0.0
        self._frames = 0

    # ---------------------------------------------------------------- inputs
    async def push_audio(self, frame: rtc.AudioFrame | AudioSegmentEnd) -> None:
        if isinstance(frame, AudioSegmentEnd):
            if self._out_resampler is not None:
                for f in self._out_resampler.flush():
                    await self._queue.put(f)
            await self._queue.put(frame)
            return

        needs_resample = (
            frame.sample_rate != self._options.audio_sample_rate
            or frame.num_channels != self._options.audio_channels
        )
        if needs_resample and self._out_resampler is None:
            self._out_resampler = rtc.AudioResampler(
                input_rate=frame.sample_rate,
                output_rate=self._options.audio_sample_rate,
                num_channels=self._options.audio_channels,
            )
        if self._out_resampler is not None:
            for f in self._out_resampler.push(frame):
                await self._queue.put(f)
        else:
            await self._queue.put(frame)

    def clear_buffer(self) -> None:
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                break
        self._reset_buffers()
        self._backend.reset()

    def set_av_sync(self, av_sync: rtc.AVSynchronizer | None) -> None:
        self._av_sync = av_sync

    @property
    def render_ms(self) -> float:
        return self._render_ms

    # --------------------------------------------------------------- outputs
    def __aiter__(self) -> AsyncIterator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd]:
        return self._stream()

    async def _stream(self) -> AsyncGenerator[rtc.VideoFrame | rtc.AudioFrame | AudioSegmentEnd, None]:
        loop = asyncio.get_running_loop()
        threaded = bool(getattr(self._backend, "run_in_thread", False))
        if threaded:
            await loop.run_in_executor(None, self._backend.warmup)
        else:
            self._backend.warmup()
        frame_interval = 1.0 / self._options.video_fps
        batch = max(1, self._backend.batch_frames)
        lookahead = max(0, self._backend.lookahead_frames)

        async def idle() -> rtc.VideoFrame:
            rgba = await loop.run_in_executor(None, self._backend.idle_frame) if threaded else self._backend.idle_frame()
            return self._to_video(rgba)

        async def render(n: int) -> list[tuple[rtc.VideoFrame, rtc.AudioFrame]]:
            if threaded:
                return await loop.run_in_executor(None, self._render_batch, n)
            return self._render_batch(n)

        while True:
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=0.5 * frame_interval)
            except asyncio.TimeoutError:
                if self._av_sync is not None and self._av_sync._video_queue.qsize() > 1:
                    continue  # synchroniser is still busy with frames we already produced
                yield await idle()
                await asyncio.sleep(0)
                continue

            if isinstance(item, rtc.AudioFrame):
                self._append(item)
                while self._frames_buffered() >= batch + lookahead:
                    for v, a in await render(batch):
                        yield v
                        yield a
                continue

            # AudioSegmentEnd: drain whatever is left, padding the tail with silence
            while self._frames_buffered() > 0:
                n = min(batch, self._frames_buffered())
                self._pad_model_buffer((n + lookahead) * self._spf_model)
                for v, a in await render(n):
                    yield v
                    yield a
            self._backend.segment_end()
            self._reset_buffers()
            yield item

    # ------------------------------------------------------------- internals
    def _append(self, frame: rtc.AudioFrame) -> None:
        samples = np.frombuffer(frame.data, dtype=np.int16).reshape(-1, frame.num_channels)[:, 0]
        self._out_buf = np.concatenate([self._out_buf, samples])
        if self._backend.model_sample_rate == self._options.audio_sample_rate:
            self._model_buf = np.concatenate([self._model_buf, samples])
            return
        if self._model_resampler is None:
            self._model_resampler = rtc.AudioResampler(
                input_rate=self._options.audio_sample_rate,
                output_rate=self._backend.model_sample_rate,
                num_channels=1,
            )
        for f in self._model_resampler.push(frame):
            self._model_buf = np.concatenate([self._model_buf, np.frombuffer(f.data, dtype=np.int16)])

    def _frames_buffered(self) -> int:
        return len(self._out_buf) // self._spf_out

    def _pad_model_buffer(self, n_samples: int) -> None:
        if self._model_resampler is not None:
            for f in self._model_resampler.flush():
                self._model_buf = np.concatenate([self._model_buf, np.frombuffer(f.data, dtype=np.int16)])
            self._model_resampler = None
        if len(self._model_buf) < n_samples:
            self._model_buf = np.pad(self._model_buf, (0, n_samples - len(self._model_buf)))

    def _render_batch(self, n: int) -> list[tuple[rtc.VideoFrame, rtc.AudioFrame]]:
        lookahead = max(0, self._backend.lookahead_frames)
        need = (n + lookahead) * self._spf_model
        if len(self._model_buf) < need:
            # resampler latency can leave the model copy a few samples short; pad, don't stall
            self._model_buf = np.pad(self._model_buf, (0, need - len(self._model_buf)))
        model_audio = self._model_buf[:need]
        t0 = time.perf_counter()
        frames = self._backend.speech_frames(model_audio, n)
        self._track(t0, n)
        if len(frames) != n:
            raise RuntimeError(f"{type(self._backend).__name__} returned {len(frames)} frames for a batch of {n}")

        out: list[tuple[rtc.VideoFrame, rtc.AudioFrame]] = []
        for i, rgba in enumerate(frames):
            chunk = self._out_buf[i * self._spf_out : (i + 1) * self._spf_out]
            audio = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=self._options.audio_sample_rate,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            out.append((self._to_video(rgba), audio))
        self._out_buf = self._out_buf[n * self._spf_out :]
        self._model_buf = self._model_buf[n * self._spf_model :]
        return out

    def _track(self, t0: float, n: int) -> None:
        ms = (time.perf_counter() - t0) * 1000.0 / max(1, n)
        self._render_ms = ms if self._frames == 0 else self._render_ms * 0.9 + ms * 0.1
        self._frames += n
        if self._frames % 250 < n:
            logger.debug("render %.1f ms/frame (%s)", self._render_ms, type(self._backend).__name__)

    def _reset_buffers(self) -> None:
        self._out_buf = np.zeros(0, dtype=np.int16)
        self._model_buf = np.zeros(0, dtype=np.int16)
        self._model_resampler = None

    def _to_video(self, rgba: np.ndarray) -> rtc.VideoFrame:
        if rgba.shape[0] != self._options.video_height or rgba.shape[1] != self._options.video_width:
            raise RuntimeError(
                f"backend frame is {rgba.shape[1]}x{rgba.shape[0]}, expected "
                f"{self._options.video_width}x{self._options.video_height}"
            )
        if not rgba.flags["C_CONTIGUOUS"]:
            rgba = np.ascontiguousarray(rgba)
        return rtc.VideoFrame(
            width=rgba.shape[1],
            height=rgba.shape[0],
            type=rtc.VideoBufferType.RGBA,
            data=rgba.tobytes(),
        )
