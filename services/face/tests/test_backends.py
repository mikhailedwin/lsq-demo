"""Checks the GPU backends that CI cannot execute: registry wiring, the frame
contract they promise the stream, and that a missing GPU / missing checkout
fails with an actionable message instead of an ImportError deep in a vendor repo."""

from __future__ import annotations

import os

import pytest

from qav_face.avatars import AvatarSpec
from qav_face.backends import available_backends, create_backend


def _spec(renderer: str, **assets) -> AvatarSpec:
    return AvatarSpec(id="t", name="t", renderer=renderer, width=384, height=384, fps=25, assets=assets)


def test_all_three_backends_are_registered() -> None:
    assert available_backends() == ["liveavatar", "musetalk", "procedural"]


@pytest.mark.parametrize("renderer", ["musetalk", "liveavatar"])
def test_gpu_backends_import_without_torch_installed(renderer: str) -> None:
    """Constructing a backend must not import torch/vendor code — that happens in warmup(),
    so the worker can start, log, and report a clear error per session."""
    backend = create_backend(_spec(renderer))
    assert backend.width == 384 and backend.fps == 25
    assert backend.model_sample_rate == 16_000
    assert backend.run_in_thread is True, "GPU inference must not block the event loop"


def test_musetalk_declares_lookahead_matching_its_whisper_window() -> None:
    from qav_face.backends.musetalk import FEATURE_SLOTS_PER_FRAME, LEFT_CONTEXT_FRAMES, RIGHT_CONTEXT_FRAMES

    backend = create_backend(_spec("musetalk"))
    assert backend.lookahead_frames == RIGHT_CONTEXT_FRAMES
    # MuseTalk's UNet is conditioned on 10 whisper slots per frame (2 past + current + 2 future, ×2)
    assert FEATURE_SLOTS_PER_FRAME == 2 * (LEFT_CONTEXT_FRAMES + RIGHT_CONTEXT_FRAMES + 1)
    assert backend.batch_frames >= 1


def test_liveavatar_clip_size_is_a_whole_number_of_latent_blocks() -> None:
    from qav_face.backends.liveavatar import INFER_FRAMES, MODEL_FPS

    # the pipeline works in 4-frame latents, 3 latent frames per denoising block
    assert INFER_FRAMES % 4 == 0
    assert (INFER_FRAMES // 4) % 3 == 0, "clip must divide into whole num_frames_per_block blocks"
    backend = create_backend(_spec("liveavatar"))
    assert backend.batch_frames == INFER_FRAMES
    assert MODEL_FPS == 25


def test_musetalk_without_checkout_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MUSETALK_ROOT", raising=False)
    backend = create_backend(_spec("musetalk"))
    with pytest.raises(RuntimeError, match="MUSETALK_ROOT"):
        backend.warmup()


def test_liveavatar_without_reference_image_says_what_to_do(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("QAV_AVATAR_DIR", str(tmp_path))
    monkeypatch.setenv("LIVEAVATAR_ROOT", str(tmp_path))
    backend = create_backend(_spec("liveavatar"))
    with pytest.raises(RuntimeError, match="reference image"):
        backend.warmup()


def test_avatar_assets_resolve_under_the_avatar_dir(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("QAV_AVATAR_DIR", str(tmp_path))
    spec = _spec("musetalk", avatarDir="yongen")
    assert spec.asset_path("avatarDir") == os.path.join(str(tmp_path), "yongen")
    absolute = _spec("liveavatar", image="/data/face.jpg")
    assert absolute.asset_path("image") == "/data/face.jpg"
