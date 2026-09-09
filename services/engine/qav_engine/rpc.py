"""RPC surface the browser SDK calls on the engine participant."""

from __future__ import annotations

import json
import logging

from livekit import rtc
from livekit.agents import AgentSession

from .protocol import RPC_INTERRUPT, RPC_TALK, RPC_USER_MESSAGE

logger = logging.getLogger("qav.rpc")


def register_rpc(room: rtc.Room, session: AgentSession, *, echo_without_llm: bool) -> None:
    lp = room.local_participant

    def _text(data: rtc.RpcInvocationData) -> str:
        try:
            payload = json.loads(data.payload or "{}")
        except json.JSONDecodeError as e:
            raise rtc.RpcError(1400, "payload must be JSON") from e
        text = str(payload.get("text", "")).strip()
        if not text:
            raise rtc.RpcError(1400, "text is required")
        if len(text) > 4000:
            raise rtc.RpcError(1413, "text too long")
        return text

    @lp.register_rpc_method(RPC_TALK)
    async def _talk(data: rtc.RpcInvocationData) -> str:
        text = _text(data)
        logger.info("talk: %r", text[:80])
        session.say(text)
        return json.dumps({"ok": True})

    @lp.register_rpc_method(RPC_USER_MESSAGE)
    async def _user_message(data: rtc.RpcInvocationData) -> str:
        text = _text(data)
        logger.info("user_message: %r", text[:80])
        if echo_without_llm:
            session.say(f"You said: {text}")
        else:
            session.generate_reply(user_input=text)
        return json.dumps({"ok": True})

    @lp.register_rpc_method(RPC_INTERRUPT)
    async def _interrupt(_: rtc.RpcInvocationData) -> str:
        session.interrupt()
        return json.dumps({"ok": True})
