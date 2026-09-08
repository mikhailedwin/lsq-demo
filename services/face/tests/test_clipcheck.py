"""The clip gate has to be right: a false pass wastes a GPU rental and a demo."""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from qav_face.clipcheck import check_clip  # noqa: E402


def _write_clip(
    path: str,
    *,
    seconds: float = 12.0,
    fps: int = 25,
    size: tuple[int, int] = (640, 640),
    face_px: int = 340,
    sharp: bool = True,
    brightness: int = 130,
) -> str:
    """Synthesize a clip with a crude but detectable frontal face."""
    w, h = size
    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(3)
    for i in range(int(seconds * fps)):
        frame = np.full((h, w, 3), brightness // 2, dtype=np.uint8)
        cx, cy = w // 2, h // 2
        r = face_px // 2
        # face oval + eyes + mouth: enough structure for the haar cascade
        cv2.ellipse(frame, (cx, cy), (int(r * 0.78), r), 0, 0, 360, (brightness + 60,) * 3, -1)
        for sx in (-1, 1):
            cv2.circle(frame, (cx + sx * r // 3, cy - r // 4), max(2, r // 12), (25, 25, 25), -1)
        mouth_h = max(2, r // 12 + (r // 20 if i % 6 < 3 else 0))
        cv2.ellipse(frame, (cx, cy + r // 2), (r // 3, mouth_h), 0, 0, 360, (40, 30, 30), -1)
        cv2.ellipse(frame, (cx, cy + r // 6), (r // 14, r // 8), 0, 0, 360, (brightness + 30,) * 3, -1)
        if sharp:
            noise = rng.integers(0, 45, (h, w, 3), dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise - 22, 0, 255).astype(np.uint8)
        else:
            frame = cv2.GaussianBlur(frame, (31, 31), 0)
        writer.write(frame)
    writer.release()
    return path


def test_a_good_clip_passes(tmp_path) -> None:
    rep = check_clip(_write_clip(str(tmp_path / "good.mp4")))
    assert not rep.failed, rep.render()
    assert rep.score >= 60, rep.render()
    assert rep.duration == pytest.approx(12.0, abs=0.5)
    assert rep.fps == pytest.approx(25.0, abs=0.5)


def test_passing_checks_do_not_cost_points(tmp_path) -> None:
    rep = check_clip(_write_clip(str(tmp_path / "clean.mp4")))
    assert not [f for f in rep.findings if f.level in ("fail", "warn")], rep.render()
    assert rep.score == 100


def test_a_too_short_clip_fails_with_a_fix(tmp_path) -> None:
    rep = check_clip(_write_clip(str(tmp_path / "short.mp4"), seconds=3.0))
    assert rep.failed
    fail = [f for f in rep.findings if f.level == "fail"]
    assert any("footage" in f.message for f in fail)
    assert all(f.fix for f in fail), "every failure must tell the user what to do"


def test_a_tiny_face_fails(tmp_path) -> None:
    rep = check_clip(_write_clip(str(tmp_path / "tiny.mp4"), size=(640, 640), face_px=120))
    assert rep.failed
    assert any("px across" in f.message for f in rep.findings if f.level == "fail")


def test_a_soft_clip_is_caught(tmp_path) -> None:
    rep = check_clip(_write_clip(str(tmp_path / "soft.mp4"), sharp=False))
    assert any("soft" in f.message or "sharpness" in f.message for f in rep.findings if f.level in ("fail", "warn"))


def test_a_dark_clip_warns(tmp_path) -> None:
    rep = check_clip(_write_clip(str(tmp_path / "dark.mp4"), brightness=30))
    assert any("dark" in f.message for f in rep.findings)


def test_a_missing_file_fails_cleanly(tmp_path) -> None:
    rep = check_clip(str(tmp_path / "nope.mp4"))
    assert rep.failed and rep.score < 100
    assert "could not open" in rep.findings[0].message
    rep.render()  # must not raise on an empty report


def test_report_renders_every_finding(tmp_path) -> None:
    rep = check_clip(_write_clip(str(tmp_path / "r.mp4"), seconds=8.0))
    text = rep.render()
    for f in rep.findings:
        assert f.message in text
    assert "score:" in text
