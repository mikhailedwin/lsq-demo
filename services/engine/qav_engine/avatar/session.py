"""Where the persona's face is rendered, and how its audio gets there.

A hosted avatar plugin would ask a vendor's cloud to send a face worker into
the room and route the agent's TTS audio to it over a data stream. Ours has
two modes:

* ``local``  — the CPU placeholder face runs inside the engine process, on a
  second Room connection (``qav-avatar``) that publishes on behalf of the engine.
* ``remote`` — dispatch the ``qav-face`` GPU worker into the room and stream the
  audio to it (``DataStreamAudioOutput``).

Which one is used follows the avatar: neural renderers are always remote.
"""

from __future__ import annotations

import json
import logging
from typing import Any

from livekit import api, rtc
from livekit.agents import AgentSession, get_job_context
from livekit.agents.voice.avatar import AvatarOptions, DataStreamAudioOutput, QueueAudioOutput
from livekit.agents.voice.avatar import AvatarSession as BaseAvatarSession
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF
from qav_face.avatars import AvatarSpec
from qav_face.backends import create_backend
from qav_face.protocol import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, AVATAR_IDENTITY, AVATAR_NAME, FACE_AGENT_NAME
from qav_face.publish import QavAvatarRunner
from qav_face.stream import BackendVideoGenerator

logger = logging.getLogger("qav.avatar")

LOCAL_RENDERERS = {"procedural"}


def choose_mode(spec: AvatarSpec, configured: str) -> str:
    """`configured` is QAV_FACE_MODE: auto | local | remote."""
    if configured == "remote":
        return "remote"
    if configured == "local":
        if spec.renderer not in LOCAL_RENDERERS:
            logger.warning("QAV_FACE_MODE=local but %s is a GPU renderer; dispatching qav-face instead", spec.renderer)
            return "remote"
        return "local"
    return "local" if spec.renderer in LOCAL_RENDERERS else "remote"


class AvatarSession(BaseAvatarSession):
    def __init__(
        self,
        *,
        spec: AvatarSpec,
        mode: str = "auto",
        livekit_url: str,
        livekit_api_key: str,
        livekit_api_secret: str,
        avatar_identity: str = AVATAR_IDENTITY,
        avatar_name: str = AVATAR_NAME,
        catalog_avatar: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self._spec = spec
        self._mode = choose_mode(spec, mode)
        self._catalog_avatar = catalog_avatar
        self._options = AvatarOptions(
            video_width=spec.width,
            video_height=spec.height,
            video_fps=spec.fps,
            audio_sample_rate=AUDIO_SAMPLE_RATE,
            audio_channels=AUDIO_CHANNELS,
        )
        self._lk_url = livekit_url
        self._lk_key = livekit_api_key
        self._lk_secret = livekit_api_secret
        self._identity = avatar_identity
        self._name = avatar_name
        self._avatar_room: rtc.Room | None = None
        self._runner: QavAvatarRunner | None = None
        self._generator: BackendVideoGenerator | None = None
        self._backend = None
        self.dispatch_id: str | None = None

    @property
    def avatar_identity(self) -> str:
        return self._identity

    @property
    def provider(self) -> str:
        return f"qav-{self._spec.renderer}"

    @property
    def mode(self) -> str:
        return self._mode

    async def start(self, agent_session: AgentSession, room: rtc.Room) -> None:
        await super().start(agent_session, room)
        if self._mode == "remote":
            await self._start_remote(agent_session, room)
        else:
            await self._start_local(agent_session, room)

    # ------------------------------------------------------------ local mode
    async def _start_local(self, agent_session: AgentSession, room: rtc.Room) -> None:
        job_ctx = get_job_context()
        token = (
            api.AccessToken(api_key=self._lk_key, api_secret=self._lk_secret)
            .with_kind("agent")
            .with_identity(self._identity)
            .with_name(self._name)
            .with_grants(api.VideoGrants(room_join=True, room=room.name))
            .with_attributes({ATTRIBUTE_PUBLISH_ON_BEHALF: job_ctx.local_participant_identity})
            .to_jwt()
        )
        self._avatar_room = rtc.Room()
        await self._avatar_room.connect(self._lk_url, token)
        logger.info("avatar participant joined room %s (local %s renderer)", room.name, self._spec.renderer)

        queue = QueueAudioOutput(sample_rate=AUDIO_SAMPLE_RATE)
        self._backend = create_backend(self._spec)
        self._generator = BackendVideoGenerator(self._options, self._backend)
        self._runner = QavAvatarRunner(
            self._avatar_room,
            audio_recv=queue,
            video_gen=self._generator,
            options=self._options,
            _lazy_publish=False,  # publish immediately so the browser sees a face before the first word
        )
        self._generator.set_av_sync(self._runner.av_sync)
        await self._runner.start()
        agent_session.output.replace_audio_tail(queue)

    # ----------------------------------------------------------- remote mode
    async def _start_remote(self, agent_session: AgentSession, room: rtc.Room) -> None:
        job_ctx = get_job_context()
        metadata = {
            "avatar": self._catalog_avatar or {"id": self._spec.id, "name": self._spec.name, "renderer": self._spec.renderer, "assets": self._spec.assets},
            "avatarId": self._spec.id,
            "videoWidth": self._spec.width,
            "videoHeight": self._spec.height,
            "videoFps": self._spec.fps,
            "engineIdentity": job_ctx.local_participant_identity,
        }
        req = api.CreateAgentDispatchRequest(agent_name=FACE_AGENT_NAME, room=room.name, metadata=json.dumps(metadata))
        dispatch = await job_ctx.api.agent_dispatch.create_dispatch(req)
        self.dispatch_id = dispatch.id
        logger.info("dispatched %s for avatar %s (%s) into %s", FACE_AGENT_NAME, self._spec.id, self._spec.renderer, room.name)

        agent_session.output.replace_audio_tail(
            DataStreamAudioOutput(
                room=room,
                destination_identity=self._identity,
                sample_rate=AUDIO_SAMPLE_RATE,
                wait_remote_track=rtc.TrackKind.KIND_VIDEO,
                # the face tells us when audio actually starts playing, so "speaking"
                # state and interruption timing follow the rendered video, not the TTS
                wait_playback_start=True,
            )
        )

    async def aclose(self) -> None:
        if self._runner is not None:
            await self._runner.aclose()
            self._runner = None
        if self._avatar_room is not None:
            await self._avatar_room.disconnect()
            self._avatar_room = None
        if self._backend is not None:
            self._backend.close()
            self._backend = None
        await super().aclose()
