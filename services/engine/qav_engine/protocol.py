"""Wire constants shared with the API (apps/api) and the browser SDK (packages/js-sdk)."""

# Participant identities inside a session room.
ENGINE_IDENTITY = "qav-engine"
AVATAR_IDENTITY = "qav-avatar"
AVATAR_NAME = "QAV Avatar"

# Agent name the worker registers under; the API dispatches to it explicitly.
AGENT_NAME = "qav-engine"

# RPC methods the browser SDK invokes on the engine participant.
RPC_TALK = "qav.talk"
RPC_USER_MESSAGE = "qav.user_message"
RPC_INTERRUPT = "qav.interrupt"

# Audio format between TTS, the queue, and the face renderer.
AUDIO_SAMPLE_RATE = 24_000
AUDIO_CHANNELS = 1
