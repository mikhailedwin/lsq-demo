from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Walk up from CWD so `.env` at the monorepo root is picked up when running from services/engine.
load_dotenv(override=False)
for parent in ("..", "../..", "../../.."):
    load_dotenv(os.path.join(parent, ".env"), override=False)


def _env(name: str, default: str) -> str:
    v = os.getenv(name)
    return v if v not in (None, "") else default


@dataclass(frozen=True)
class EngineConfig:
    livekit_url: str
    livekit_api_key: str
    livekit_api_secret: str

    stt: str
    llm: str
    tts: str
    llm_model: str

    renderer: str
    video_width: int
    video_height: int
    video_fps: float

    @classmethod
    def from_env(cls) -> EngineConfig:
        return cls(
            livekit_url=_env("LIVEKIT_URL", "ws://localhost:7880"),
            livekit_api_key=_env("LIVEKIT_API_KEY", "devkey"),
            livekit_api_secret=_env("LIVEKIT_API_SECRET", "secret"),
            stt=_env("QAV_STT", "deepgram").lower(),
            llm=_env("QAV_LLM", "anthropic").lower(),
            tts=_env("QAV_TTS", "elevenlabs").lower(),
            llm_model=_env("QAV_LLM_MODEL", "claude-opus-5"),
            renderer=_env("QAV_RENDERER", "procedural").lower(),
            video_width=int(_env("QAV_VIDEO_WIDTH", "512")),
            video_height=int(_env("QAV_VIDEO_HEIGHT", "512")),
            video_fps=float(_env("QAV_VIDEO_FPS", "25")),
        )


CONFIG = EngineConfig.from_env()
