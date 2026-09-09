"""The `qav-face` worker: a GPU face renderer that joins a session room.

    python -m qav_face.worker start

Dispatched by the engine (explicit dispatch, agent_name = "qav-face") with the
avatar spec as job metadata. It joins as ``qav-avatar``, publishing on behalf
of the engine participant, receives the engine's TTS audio over a LiveKit
data stream, and publishes AV-synced video — the standard LiveKit Agents
avatar-worker contract.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os

from dotenv import load_dotenv
from livekit import rtc
from livekit.agents import JobContext, JobProcess, JobRequest, WorkerOptions, cli
from livekit.agents.voice.avatar import AvatarOptions, DataStreamAudioReceiver
from livekit.agents.voice.room_io import ATTRIBUTE_PUBLISH_ON_BEHALF

from .avatars import AvatarSpec
from .backends import create_backend
from .protocol import AUDIO_CHANNELS, AUDIO_SAMPLE_RATE, AVATAR_IDENTITY, AVATAR_NAME, ENGINE_IDENTITY, FACE_AGENT_NAME
from .publish import QavAvatarRunner
from .stream import BackendVideoGenerator

load_dotenv(override=False)
for parent in ("..", "../..", "../../.."):
    load_dotenv(os.path.join(parent, ".env"), override=False)

logger = logging.getLogger("qav.face")


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


def parse_job(metadata: str) -> tuple[AvatarSpec, str]:
    """Job metadata written by the engine (see qav_engine.avatar.session)."""
    d = json.loads(metadata or "{}")
    spec = AvatarSpec.from_catalog(
        d.get("avatar"),
        avatar_id=d.get("avatarId", ""),
        width=int(d.get("videoWidth") or _env("QAV_VIDEO_WIDTH", "512")),
        height=int(d.get("videoHeight") or _env("QAV_VIDEO_HEIGHT", "512")),
        fps=float(d.get("videoFps") or _env("QAV_VIDEO_FPS", "25")),
        default_renderer=_env("QAV_RENDERER", "musetalk"),
    )
    return spec, d.get("engineIdentity") or ENGINE_IDENTITY


async def on_request(req: JobRequest) -> None:
    _, engine_identity = parse_job(req.job.metadata)
    await req.accept(
        identity=AVATAR_IDENTITY,
        name=AVATAR_NAME,
        # tells clients (and RoomIO) the tracks belong to the engine participant
        attributes={ATTRIBUTE_PUBLISH_ON_BEHALF: engine_identity},
    )


def prewarm(proc: JobProcess) -> None:
    # Nothing global to preload: backends own their weights and are created per
    # job. A GPU box runs one face per process; set --num-idle-processes 1.
    proc.userdata["ready"] = True


async def entrypoint(ctx: JobContext) -> None:
    spec, engine_identity = parse_job(ctx.job.metadata)
    ctx.log_context_fields = {"avatar": spec.id, "renderer": spec.renderer}
    logger.info("rendering avatar %s with %s at %dx%d@%.0f", spec.id, spec.renderer, spec.width, spec.height, spec.fps)

    await ctx.connect()

    backend = create_backend(spec)
    options = AvatarOptions(
        video_width=spec.width,
        video_height=spec.height,
        video_fps=spec.fps,
        audio_sample_rate=AUDIO_SAMPLE_RATE,
        audio_channels=AUDIO_CHANNELS,
    )
    generator = BackendVideoGenerator(options, backend)
    runner = QavAvatarRunner(
        ctx.room,
        audio_recv=DataStreamAudioReceiver(ctx.room, sender_identity=engine_identity),
        video_gen=generator,
        options=options,
        _lazy_publish=False,
    )
    generator.set_av_sync(runner.av_sync)
    await runner.start()
    logger.info("avatar tracks published")

    done = asyncio.Event()

    def _on_left(p: rtc.RemoteParticipant) -> None:
        if p.identity == engine_identity:
            logger.info("engine left; stopping renderer")
            done.set()

    ctx.room.on("participant_disconnected", _on_left)
    ctx.room.on("disconnected", lambda *_: done.set())

    async def _cleanup() -> None:
        await runner.aclose()
        backend.close()

    ctx.add_shutdown_callback(_cleanup)
    await done.wait()
    ctx.shutdown(reason="engine left")


def main() -> None:
    opts: dict = {}
    if os.getenv("QAV_WORKER_LOAD_THRESHOLD"):
        opts["load_threshold"] = float(os.environ["QAV_WORKER_LOAD_THRESHOLD"])
    cli.run_app(
        WorkerOptions(
            **opts,
            entrypoint_fnc=entrypoint,
            request_fnc=on_request,
            prewarm_fnc=prewarm,
            agent_name=FACE_AGENT_NAME,
            ws_url=_env("LIVEKIT_URL", "ws://localhost:7880"),
            api_key=_env("LIVEKIT_API_KEY", "devkey"),
            api_secret=_env("LIVEKIT_API_SECRET", "secret"),
            # a face worker is heavyweight: one concurrent job per process by default
            num_idle_processes=int(_env("QAV_FACE_IDLE_PROCESSES", "1")),
            port=int(_env("QAV_FACE_HEALTH_PORT", "8082")),
        )
    )


if __name__ == "__main__":
    main()
