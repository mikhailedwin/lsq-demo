"""Wire constants shared by the engine, the face worker, the API and the browser SDK."""

ENGINE_IDENTITY = "qav-engine"
AVATAR_IDENTITY = "qav-avatar"
AVATAR_NAME = "QAV Avatar"

# Agent names for explicit dispatch.
ENGINE_AGENT_NAME = "qav-engine"
FACE_AGENT_NAME = "qav-face"

# Audio format between TTS, the data stream and the face backends.
AUDIO_SAMPLE_RATE = 24_000
AUDIO_CHANNELS = 1

# RPC methods the browser SDK invokes on the engine participant.
RPC_TALK = "qav.talk"
RPC_USER_MESSAGE = "qav.user_message"
RPC_INTERRUPT = "qav.interrupt"
