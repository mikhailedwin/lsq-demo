"""QAV engine worker. One LiveKit Agents job == one persona session.

    python -m qav_engine.worker dev      # hot-reload, verbose
    python -m qav_engine.worker start    # production

The API dispatches jobs explicitly (agent_name = "qav-engine") with the persona
as job metadata; nothing runs for rooms we weren't asked to join.
"""

from __future__ import annotations

import asyncio
import logging

from livekit import rtc
from livekit.agents import (
    Agent,
    AgentSession,
    AgentStateChangedEvent,
    CloseEvent,
    ConversationItemAddedEvent,
    ErrorEvent,
    InterruptionOptions,
    JobContext,
    JobProcess,
    JobRequest,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
)
from livekit.agents.voice.room_io import RoomOptions

from .avatar import AvatarSession, get_renderer
from .callbacks import ApiCallbacks
from .config import CONFIG
from .persona import Persona
from .protocol import AGENT_NAME, ENGINE_IDENTITY
from .providers import build_pipeline, load_vad
from .rpc import register_rpc

logger = logging.getLogger("qav.engine")


async def on_request(req: JobRequest) -> None:
    # Join under a stable identity: the SDK addresses RPCs to it and the API
    # advertises it. (The framework's default would be "agent-<job id>".)
    await req.accept(identity=ENGINE_IDENTITY, name="QAV Engine")


def prewarm(proc: JobProcess) -> None:
    # VAD weights load once per process, not per session.
    if CONFIG.stt != "mock":
        proc.userdata["vad"] = load_vad()


async def entrypoint(ctx: JobContext) -> None:
    persona = Persona.from_job_metadata(ctx.job.metadata)
    ctx.log_context_fields = {"session": persona.session_id, "persona": persona.name}
    logger.info("starting session for persona %r (avatar %s)", persona.name, persona.avatar_id)

    callbacks = ApiCallbacks(persona)
    ctx.add_shutdown_callback(callbacks.aclose)

    await ctx.connect()

    # --- face ------------------------------------------------------------
    width = persona.video_width or CONFIG.video_width
    height = persona.video_height or CONFIG.video_height
    renderer = get_renderer(CONFIG.renderer, width=width, height=height, style=persona.style)
    avatar = AvatarSession(
        renderer=renderer,
        video_fps=CONFIG.video_fps,
        livekit_url=CONFIG.livekit_url,
        livekit_api_key=CONFIG.livekit_api_key,
        livekit_api_secret=CONFIG.livekit_api_secret,
    )

    # --- ears / brain / voice ------------------------------------------------
    pipeline = build_pipeline(CONFIG, persona, ctx.proc.userdata.get("vad"))
    # The queue-backed avatar output can't pause mid-utterance, so the
    # "resume after false interruption" feature would only log a warning.
    session_kwargs: dict = {
        "tts": pipeline.tts,
        "turn_handling": TurnHandlingOptions(interruption=InterruptionOptions(resume_false_interruption=False)),
    }
    if pipeline.stt is not None:
        session_kwargs["stt"] = pipeline.stt
    if pipeline.llm is not None:
        session_kwargs["llm"] = pipeline.llm
    if pipeline.vad is not None:
        session_kwargs["vad"] = pipeline.vad
    if persona.idle_timeout_s:
        session_kwargs["user_away_timeout"] = float(persona.idle_timeout_s)
    session = AgentSession(**session_kwargs)

    # --- wiring ------------------------------------------------------------------
    @session.on("conversation_item_added")
    def _on_item(ev: ConversationItemAddedEvent) -> None:
        item = ev.item
        if getattr(item, "type", None) == "message":
            role = "persona" if item.role == "assistant" else "user"
            callbacks.transcript(role, item.text_content or "")

    @session.on("agent_state_changed")
    def _on_state(ev: AgentStateChangedEvent) -> None:
        logger.debug("agent state: %s -> %s", ev.old_state, ev.new_state)

    @session.on("error")
    def _on_error(ev: ErrorEvent) -> None:
        logger.error("pipeline error from %s: %s", ev.source, ev.error)

    @session.on("close")
    def _on_close(ev: CloseEvent) -> None:
        callbacks.status("ended", str(ev.error) if ev.error else None)

    # Face first: publishing the video track before the agent starts means the
    # browser's `streamToVideoElement` resolves as soon as it joins.
    await avatar.start(session, ctx.room)
    register_rpc(ctx.room, session, echo_without_llm=pipeline.llm is None)

    await session.start(
        agent=Agent(instructions=persona.system_prompt),
        room=ctx.room,
        session_host=False,  # inspector byte-streams would just be noise on the avatar's room connection
        room_options=RoomOptions(
            audio_output=False,  # the avatar participant carries the voice
            audio_input=pipeline.stt is not None,
        ),
    )
    callbacks.status("active")

    try:
        await avatar.wait_for_join(timeout=15.0)
    except asyncio.TimeoutError:
        logger.warning("avatar video track not observed within 15s; continuing")

    if persona.greet_on_join:
        if pipeline.llm is not None:
            session.generate_reply(
                instructions=f"Greet the user in one short sentence as {persona.name}, then ask how you can help."
            )
        else:
            session.say(f"Hi, I'm {persona.name}. I'm running in offline mode, so type to me and I'll answer.")

    # Keep the job alive until the user leaves; the room's empty timeout ends it otherwise.
    async def _wait_for_user_leave() -> None:
        done = asyncio.Event()

        def _on_left(p: rtc.RemoteParticipant) -> None:
            if p.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT:
                humans = [
                    rp
                    for rp in ctx.room.remote_participants.values()
                    if rp.kind != rtc.ParticipantKind.PARTICIPANT_KIND_AGENT
                ]
                if not humans:
                    done.set()

        ctx.room.on("participant_disconnected", _on_left)
        await done.wait()
        logger.info("last user left; shutting down session")
        ctx.shutdown(reason="user left")

    asyncio.create_task(_wait_for_user_leave())


def main() -> None:
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
            request_fnc=on_request,
            agent_name=AGENT_NAME,
            ws_url=CONFIG.livekit_url,
            api_key=CONFIG.livekit_api_key,
            api_secret=CONFIG.livekit_api_secret,
        )
    )


if __name__ == "__main__":
    main()

__all__ = ["main", "entrypoint", "ENGINE_IDENTITY"]
