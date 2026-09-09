"""Turn (env config + persona) into concrete STT / LLM / TTS / VAD instances.

Provider choice is per-deployment (env); voice/model choice is per-persona
(catalog entries from the API). A persona whose voice provider doesn't match
the deployed TTS falls back to that TTS's default voice with a warning, so a
misconfigured catalog degrades instead of failing the session.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from livekit.agents import llm as llm_mod
from livekit.agents import stt as stt_mod
from livekit.agents import tts as tts_mod
from livekit.agents import vad as vad_mod

from .config import EngineConfig
from .mock import MockTTS
from .persona import Persona

logger = logging.getLogger("qav.providers")


@dataclass
class Pipeline:
    stt: stt_mod.STT | None
    llm: llm_mod.LLM | None
    tts: tts_mod.TTS
    vad: vad_mod.VAD | None
    text_only_input: bool
    """True when there is no STT: the persona can only be driven by text/RPC."""


def load_vad() -> vad_mod.VAD:
    from livekit.plugins import silero

    return silero.VAD.load()


def build_pipeline(cfg: EngineConfig, persona: Persona, vad: vad_mod.VAD | None) -> Pipeline:
    return Pipeline(
        stt=_build_stt(cfg, persona),
        llm=_build_llm(cfg, persona),
        tts=_build_tts(cfg, persona),
        vad=vad,
        text_only_input=cfg.stt == "mock",
    )


def _build_stt(cfg: EngineConfig, persona: Persona) -> stt_mod.STT | None:
    lang = persona.voice.language or "en"
    if cfg.stt == "deepgram":
        from livekit.plugins import deepgram

        return deepgram.STT(model="nova-3", language="multi" if lang == "multi" else f"{lang}-US" if lang == "en" else lang)
    if cfg.stt == "openai":
        from livekit.plugins import openai

        return openai.STT(model="gpt-4o-mini-transcribe", language=lang)
    if cfg.stt == "mock":
        logger.warning("QAV_STT=mock: no speech recognition, persona responds to text input only")
        return None
    raise ValueError(f"unknown QAV_STT provider: {cfg.stt}")


def _build_llm(cfg: EngineConfig, persona: Persona) -> llm_mod.LLM | None:
    # Persona-level provider wins when it is deployable here; otherwise use the env default.
    provider = persona.llm.provider or cfg.llm
    model = persona.llm.model if persona.llm.provider else cfg.llm_model
    if provider == "mock" or cfg.llm == "mock":
        logger.warning("LLM=mock: persona echoes user input instead of thinking")
        return None
    if provider == "anthropic":
        from livekit.plugins import anthropic

        return anthropic.LLM(model=model or "claude-opus-5", caching="ephemeral")
    if provider == "openai":
        from livekit.plugins import openai

        return openai.LLM(model=model or "gpt-4.1-mini")
    raise ValueError(f"unknown LLM provider: {provider}")


def _build_tts(cfg: EngineConfig, persona: Persona) -> tts_mod.TTS:
    voice = persona.voice
    voice_id = voice.provider_voice_id if voice.provider == cfg.tts else None
    if voice.provider != cfg.tts and cfg.tts != "mock":
        logger.warning(
            "persona voice provider %r != deployed QAV_TTS %r; using %s default voice",
            voice.provider,
            cfg.tts,
            cfg.tts,
        )

    if cfg.tts == "elevenlabs":
        from livekit.plugins import elevenlabs

        kwargs = {"voice_id": voice_id} if voice_id else {}
        return elevenlabs.TTS(model="eleven_turbo_v2_5", **kwargs)
    if cfg.tts == "cartesia":
        from livekit.plugins import cartesia

        kwargs = {"voice": voice_id} if voice_id else {}
        return cartesia.TTS(model="sonic-3", **kwargs)
    if cfg.tts == "openai":
        from livekit.plugins import openai

        kwargs = {"voice": voice_id} if voice_id else {}
        return openai.TTS(**kwargs)
    if cfg.tts == "mock":
        return MockTTS()
    raise ValueError(f"unknown QAV_TTS provider: {cfg.tts}")
