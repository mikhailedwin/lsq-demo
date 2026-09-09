"""Job metadata → typed persona description. Shape is produced by apps/api (EngineJobMetadata)."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

DEFAULT_SYSTEM_PROMPT = (
    "You are a helpful, warm conversational assistant with a face and a voice. "
    "You are speaking out loud in real time, so keep replies short (one to three sentences), "
    "avoid lists, markdown, emojis and URLs, and never mention that you are an AI unless asked."
)


@dataclass(frozen=True)
class VoiceRef:
    provider: str = "mock"
    provider_voice_id: str = "tone"
    language: str = "en"


@dataclass(frozen=True)
class LlmRef:
    provider: str | None = None
    model: str | None = None


@dataclass(frozen=True)
class Persona:
    session_id: str
    name: str
    avatar_id: str
    avatar_model: str
    avatar: dict[str, Any]
    """The catalog entry as the API sent it (renderer, style, assets) — handed to the face."""
    voice: VoiceRef
    llm: LlmRef
    system_prompt: str
    greet_on_join: bool
    idle_timeout_s: int
    video_width: int | None
    video_height: int | None
    callback_url: str
    callback_token: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # BYO-LiveKit ("face only") mode: the caller runs the voice agent, we only render.
    environment: dict[str, str] | None = None

    @classmethod
    def from_job_metadata(cls, raw: str) -> Persona:
        if not raw:
            raise ValueError("job metadata is empty; dispatch the engine through the QAV API")
        d = json.loads(raw)
        pc = d.get("personaConfig") or {}
        so = d.get("sessionOptions") or {}
        avatar = d.get("avatar") or {}
        voice = d.get("voice") or {}
        llm = d.get("llm") or {}
        return cls(
            session_id=d.get("sessionId", ""),
            name=pc.get("name", "QAV"),
            avatar_id=pc.get("avatarId", avatar.get("id", "qav-nova")),
            avatar_model=pc.get("avatarModel") or avatar.get("model") or avatar.get("renderer") or "procedural",
            avatar=dict(avatar) if avatar else {"id": pc.get("avatarId", "qav-nova"), "renderer": "procedural"},
            voice=VoiceRef(
                provider=voice.get("provider", "mock"),
                provider_voice_id=voice.get("providerVoiceId", "tone"),
                language=voice.get("language", "en"),
            ),
            llm=LlmRef(provider=llm.get("provider"), model=llm.get("model")),
            system_prompt=pc.get("systemPrompt") or DEFAULT_SYSTEM_PROMPT,
            greet_on_join=bool(so.get("greetOnJoin", True)),
            idle_timeout_s=int(so.get("idleTimeoutSeconds", 0) or 0),
            video_width=so.get("videoWidth"),
            video_height=so.get("videoHeight"),
            callback_url=d.get("callbackUrl", ""),
            callback_token=d.get("callbackToken", ""),
            metadata=pc.get("metadata") or {},
            environment=d.get("environment"),
        )
