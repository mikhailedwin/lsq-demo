"""Renders frames from synthetic speech and checks the face actually moves."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest
from PIL import Image

from qav_face.avatars import AvatarStyle
from qav_face.lipsync import LipSyncAnalyzer
from qav_face.renderers import get_renderer
from qav_face.testing import synth_speech

SR = 24_000
FPS = 25
SPF = SR // FPS


def _pcm(text: str) -> np.ndarray:
    return synth_speech(text, sample_rate=SR)


@pytest.mark.parametrize("hair", ["short", "long", "bun", "bald"])
def test_renders_rgba_at_requested_size(hair: str) -> None:
    r = get_renderer("procedural", width=256, height=256, style=AvatarStyle(hair_style=hair))
    r.warmup()
    frame = r.render(LipSyncAnalyzer(SR, FPS).update_idle())
    assert frame.shape == (256, 256, 4)
    assert frame.dtype == np.uint8
    assert frame[..., 3].min() == 255, "output must be opaque"


def test_mouth_opens_on_speech_and_closes_in_silence() -> None:
    an = LipSyncAnalyzer(SR, FPS, seed=1)
    audio = _pcm("Hello there, how are you doing today?")
    opens = []
    for i in range(0, len(audio) - SPF, SPF):
        opens.append(an.update_audio(audio[i : i + SPF]).mouth_open)
    assert max(opens) > 0.6, "loud syllables should open the mouth"
    assert min(opens) < 0.15, "gaps between words should close it"

    for _ in range(FPS):
        st = an.update_idle()
    assert st.mouth_open < 0.05
    assert not st.speaking


def test_blinks_happen_while_idle() -> None:
    an = LipSyncAnalyzer(SR, FPS, seed=3)
    blinks = [an.update_idle().blink for _ in range(FPS * 8)]
    assert max(blinks) > 0.8


def test_speaking_frames_differ_from_idle_frames() -> None:
    r = get_renderer("procedural", width=320, height=320, style=AvatarStyle())
    an = LipSyncAnalyzer(SR, FPS, seed=2)
    idle = r.render(an.update_idle()).astype(int)
    audio = _pcm("Wow")
    loud = None
    for i in range(0, len(audio) - SPF, SPF):
        st = an.update_audio(audio[i : i + SPF])
        if st.mouth_open > 0.5:
            loud = r.render(st).astype(int)
            break
    assert loud is not None
    diff = np.abs(loud - idle).sum(axis=2)
    assert (diff > 60).sum() > 200, "an open mouth should change a visible patch of pixels"


def test_render_is_realtime_capable() -> None:
    r = get_renderer("procedural", width=512, height=512, style=AvatarStyle())
    r.warmup()
    an = LipSyncAnalyzer(SR, FPS)
    audio = _pcm("Quick brown fox jumps over the lazy dog again and again")
    n = 0
    t0 = time.perf_counter()
    for i in range(0, len(audio) - SPF, SPF):
        r.render(an.update_audio(audio[i : i + SPF]))
        n += 1
    ms = (time.perf_counter() - t0) * 1000 / n
    # generous bound for CI boxes; a frame budget at 25 fps is 40 ms
    assert ms < 35, f"{ms:.1f} ms/frame is too slow for 25 fps"


def test_writes_preview_when_requested(tmp_path) -> None:
    """QAV_PREVIEW_DIR=/some/dir pytest -k preview  → dumps a contact sheet to eyeball."""
    out = os.environ.get("QAV_PREVIEW_DIR")
    if not out:
        pytest.skip("set QAV_PREVIEW_DIR to write preview images")
    os.makedirs(out, exist_ok=True)
    styles = {
        "nova": AvatarStyle(hair_style="long"),
        "atlas": AvatarStyle(skin="#c98f63", hair="#1f1a17", eyes="#4a3b2a", lips="#9c5a52", shirt="#1f2937", background="#f3f0ea"),
        "sage": AvatarStyle(skin="#e8b89a", hair="#8a6f4e", eyes="#4f7c5a", lips="#b86b7a", shirt="#3b5d50", background="#eef4ee", hair_style="bun"),
    }
    audio = _pcm("Hello, I am your avatar. What can I do for you?")
    tiles = []
    for name, style in styles.items():
        r = get_renderer("procedural", width=256, height=256, style=style)
        an = LipSyncAnalyzer(SR, FPS, seed=7)
        picks = [8, 22, 40, 61]
        k = 0
        for i in range(0, len(audio) - SPF, SPF):
            st = an.update_audio(audio[i : i + SPF])
            if k in picks:
                tiles.append(Image.fromarray(r.render(st)))
            k += 1
        tiles.append(Image.fromarray(r.render(an.update_idle())))
    cols = 5
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGBA", (cols * 256, rows * 256))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 256, (i // cols) * 256))
    sheet.save(os.path.join(out, "qav-faces.png"))
