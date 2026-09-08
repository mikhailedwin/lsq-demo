"""Wire constants — single source of truth lives in qav_face.protocol."""

from qav_face.protocol import (
    AUDIO_CHANNELS,
    AUDIO_SAMPLE_RATE,
    AVATAR_IDENTITY,
    AVATAR_NAME,
    ENGINE_AGENT_NAME as AGENT_NAME,
    ENGINE_IDENTITY,
    FACE_AGENT_NAME,
    RPC_INTERRUPT,
    RPC_TALK,
    RPC_USER_MESSAGE,
)

__all__ = [
    "AGENT_NAME",
    "AUDIO_CHANNELS",
    "AUDIO_SAMPLE_RATE",
    "AVATAR_IDENTITY",
    "AVATAR_NAME",
    "ENGINE_IDENTITY",
    "FACE_AGENT_NAME",
    "RPC_INTERRUPT",
    "RPC_TALK",
    "RPC_USER_MESSAGE",
]
