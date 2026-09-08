"""QAV's counterpart to `livekit.plugins.anam.AvatarSession`.

Anam's plugin asks Anam's cloud to join the room with a face worker and routes
the agent's TTS audio to it over a data stream. Ours does the same job locally:
it joins the room as a second participant (`qav-avatar`) that publishes video
and audio *on behalf of* the engine, and feeds it the agent's audio through an
in-process queue. Swap `renderer` for any `FaceRenderer` — the plumbing stays.
"""

from __future__ import annotations

import logging

from livekit import api, rtc
from livekit.agents import AgentSession, get_job_context, utils
from livekit.agents.voice.avatar import AvatarOptions, AvatarRunner, QueueAudioOutput
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from ..protocol import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, AVATAR_IDENTITY, AVATAR_NAME
from .generator import FaceVideoGenerator
from .renderers.base import FaceRenderer

logger = logging.getLogger("qav.avatar")


def video_bitrate_for(width: int, height: int, fps: float) -> int:
    """Bits/s that keep a synthetic talking head sharp. libwebrtc's default table
    for small frames (~225 kbps at 384²) makes the encoder *downscale* the
    picture; a face is mostly static so this is still cheap on the wire."""
    pixels = width * height
    bps = int(pixels * fps * 0.22)  # ≈ 0.22 bits per pixel per frame
    return max(600_000, min(bps, 4_000_000))


class QavAvatarRunner(AvatarRunner):
    """AvatarRunner with explicit video publish options.

    The base class publishes with LiveKit's defaults (auto bitrate, simulcast on,
    balanced degradation). Overriding the private ``_publish_track`` is the only
    hook it offers (livekit-agents 1.8); revisit if a public option appears.
    """

    def __init__(self, room: rtc.Room, *, video_options: rtc.TrackPublishOptions, **kwargs) -> None:  # type: ignore[no-untyped-def]
        super().__init__(room, **kwargs)
        self._qav_video_options = video_options

    async def _publish_track(self) -> None:
        async with self._lock:
            await self._room_connected_fut

            audio_track = rtc.LocalAudioTrack.create_audio_track("avatar_audio", self._audio_source)
            audio_options = rtc.TrackPublishOptions(source=rtc.TrackSource.SOURCE_MICROPHONE)
            self._audio_publication = await self._room.local_participant.publish_track(audio_track, audio_options)
            await self._audio_publication.wait_for_subscription()

            video_track = rtc.LocalVideoTrack.create_video_track("avatar_video", self._video_source)
            self._video_publication = await self._room.local_participant.publish_track(
                video_track, self._qav_video_options
            )


class AvatarSession(BaseAvatarSession):
    def __init__(
        self,
        *,
        renderer: FaceRenderer,
        video_fps: float = 25.0,
        livekit_url: str,
        livekit_api_key: str,
        livekit_api_secret: str,
        avatar_identity: str = AVATAR_IDENTITY,
        avatar_name: str = AVATAR_NAME,
    ) -> None:
        super().__init__()
        self._renderer = renderer
        self._options = AvatarOptions(
            video_width=renderer.width,
            video_height=renderer.height,
            video_fps=video_fps,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_channels=AUDIO_CHANNELS,
        )
        self._lk_url = livekit_url
        self._lk_key = livekit_api_key
        self._lk_secret = livekit_api_secret
        self._identity = avatar_identity
        self._name = avatar_name
        self._avatar_room: rtc.Room | None = None
        self._runner: AvatarRunner | None = None
        self._generator: FaceVideoGenerator | None = None

    @property
    def avatar_identity(self) -> str:
        return self._identity

    @property
    def provider(self) -> str:
        return "qav"

    @property
    def generator(self) -> FaceVideoGenerator | None:
        return self._generator

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        await super().start(agent_session, room)

        job_ctx = get_job_context()
        token = (
            api.AccessToken(api_key=self._lk_key, api_secret=self._lk_secret)
            .with_kind("agent")
            .with_identity(self._identity)
            .with_name(self._name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            # tells clients the tracks belong to the engine participant
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: job_ctx.local_participant_identity})
            .to_jwt()
        )

        self._avatar_room = rtc.Room()
        await self._avatar_room.connect(self._lk_url, token)
        logger.info("avatar participant joined room %s", room.name)

        # Agent audio → queue → runner (renders video, publishes AV-synced tracks).
        queue = QueueAudioOutput(sample_rate=AUDIO_SAMPLE_RATE)
        self._generator = FaceVideoGenerator(self._options, self._renderer)
        video_options = rtc.TrackPublishOptions(
            source=rtc.TrackSource.SOURCE_CAMERA,
            simulcast=False,  # one avatar per session; always serve the full layer
            video_encoding=rtc.VideoEncoding(
                max_bitrate=video_bitrate_for(self._options.video_width, self._options.video_height, self._options.video_fps),
                max_framerate=int(self._options.video_fps),
            ),
            degradation_preference=rtc.DegradationPreference.MAINTAIN_RESOLUTION,
        )
        self._runner = QavAvatarRunner(
            self._avatar_room,
            video_options=video_options,
            audio_recv=queue,
            video_gen=self._generator,
            options=self._options,
            _lazy_publish=False,  # publish immediately so the browser sees a face before the first word
        )
        self._generator.set_av_sync(self._runner.av_sync)
        await self._runner.start()

        agent_session.output.replace_audio_tail(queue)

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.aclose()
            self._runner = None
        if self._avatar_room is not None:
            await self._avatar_room.disconnect()
            self._avatar_room = None
        self._renderer.close()
        await super().aclose()


__all__ = ["AvatarSession", "utils"]
