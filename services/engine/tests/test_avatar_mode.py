"""Where the face renders: in-process (CPU placeholder) or a dispatched GPU worker."""

from __future__ import annotations

from qav_engine.avatar.session import choose_mode
from qav_engine.persona import Persona
from qav_face.avatars import AvatarSpec


def _spec(renderer: str) -> AvatarSpec:
    return AvatarSpec.from_catalog({"id": "a", "renderer": renderer}, width=256, height=256, fps=25)


def test_cpu_renderer_runs_locally_gpu_renderers_are_dispatched() -> None:
    assert choose_mode(_spec("procedural"), "auto") == "local"
    assert choose_mode(_spec("musetalk"), "auto") == "remote"
    assert choose_mode(_spec("liveavatar"), "auto") == "remote"


def test_a_gpu_renderer_is_never_forced_into_the_engine_process() -> None:
    # QAV_FACE_MODE=local is an operator preference, not a licence to load a 14B model
    # inside the voice pipeline; it degrades to remote with a warning.
    assert choose_mode(_spec("musetalk"), "local") == "remote"
    assert choose_mode(_spec("procedural"), "local") == "local"
    assert choose_mode(_spec("procedural"), "remote") == "remote"


def test_persona_carries_the_catalog_avatar_through_to_the_face() -> None:
    import json

    meta = {
        "sessionId": "s1",
        "personaConfig": {"name": "N", "avatarId": "mt-yongen", "voiceId": "voice-mock"},
        "avatar": {"id": "mt-yongen", "renderer": "musetalk", "assets": {"avatarDir": "yongen"}},
        "sessionOptions": {"videoWidth": 512, "videoHeight": 512},
    }
    persona = Persona.from_job_metadata(json.dumps(meta))
    assert persona.avatar["renderer"] == "musetalk"
    spec = AvatarSpec.from_catalog(persona.avatar, avatar_id=persona.avatar_id, width=512, height=512, fps=25)
    assert spec.renderer == "musetalk" and spec.assets["avatarDir"] == "yongen"


def test_persona_without_a_catalog_avatar_falls_back_to_the_placeholder() -> None:
    import json

    persona = Persona.from_job_metadata(
        json.dumps({"sessionId": "s", "personaConfig": {"name": "N", "avatarId": "x", "voiceId": "voice-mock"}})
    )
    assert persona.avatar["renderer"] == "procedural"
