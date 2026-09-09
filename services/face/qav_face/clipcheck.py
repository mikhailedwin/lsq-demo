"""Source-clip quality gate.

Lip-sync output is bounded by the footage you feed it: a soft, dim, or
side-on clip produces a soft, dim, side-on avatar no matter how good the
model is. This checks a candidate clip *before* you rent a GPU and tells you
exactly what to fix, rather than letting you discover it in a demo.

Runs on CPU with opencv, so it works on a laptop:

    qav-face check --video alice.mp4
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .quality import sharpness

# What MuseTalk needs to look its best. The model trains at 25 fps on a 256²
# face crop, so a face smaller than ~256 px in the source is upscaled garbage.
MIN_FACE_PX = 220
GOOD_FACE_PX = 320
MIN_SECONDS = 6.0
GOOD_SECONDS = 10.0
MAX_SECONDS = 40.0
MIN_SHARPNESS = 60.0
GOOD_SHARPNESS = 150.0


@dataclass
class Finding:
    level: str  # "fail" | "warn" | "ok"
    message: str
    fix: str = ""


@dataclass
class ClipReport:
    path: str
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int = 0
    duration: float = 0.0
    face_px: float = 0.0
    face_frames: int = 0
    sharpness: float = 0.0
    exposure: float = 0.0
    exposure_drift: float = 0.0
    motion: float = 0.0
    yaw_spread: float = 0.0
    findings: list[Finding] = field(default_factory=list)

    @property
    def failed(self) -> bool:
        return any(f.level == "fail" for f in self.findings)

    @property
    def score(self) -> int:
        """0-100, for a quick "is this good enough" call. Only problems deduct."""
        s = 100
        for f in self.findings:
            if f.level == "fail":
                s -= 35
            elif f.level == "warn":
                s -= 12
        return max(0, s)

    def render(self) -> str:
        lines = [
            f"clip:      {self.path}",
            f"video:     {self.width}x{self.height} @ {self.fps:.2f} fps, {self.duration:.1f}s ({self.frames} frames)",
            f"face:      {self.face_px:.0f} px across, detected in {self.face_frames}/{max(1, self.frames)} sampled frames",
            f"sharpness: {self.sharpness:.0f}   exposure: {self.exposure:.0f} (drift {self.exposure_drift:.1f})",
            f"motion:    {self.motion:.2f}   head-turn spread: {self.yaw_spread:.2f}",
            "",
        ]
        for f in self.findings:
            mark = {"fail": "FAIL", "warn": "warn", "ok": "ok  "}[f.level]
            lines.append(f"  [{mark}] {f.message}")
            if f.fix:
                lines.append(f"         → {f.fix}")
        lines.append("")
        lines.append(f"score: {self.score}/100 — {'NOT usable as-is' if self.failed else 'usable'}")
        return "\n".join(lines)


def check_clip(path: str, *, sample_frames: int = 90) -> ClipReport:
    import cv2

    rep = ClipReport(path=path)
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        rep.findings.append(Finding("fail", "could not open the file", "check the path and that ffmpeg can read it"))
        return rep

    rep.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    rep.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rep.fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    rep.duration = total / rep.fps if rep.fps > 0 else 0.0

    idx = np.linspace(0, max(0, total - 1), min(sample_frames, max(1, total))).astype(int)
    detector = _face_detector(cv2)
    if detector is None:
        rep.findings.append(
            Finding("warn", "no face detector available in this OpenCV build", "face size and framing were not checked")
        )

    faces, sharps, means, prev_gray, motions, centers = [], [], [], None, [], []
    for i in idx:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        rep.frames += 1
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        means.append(float(gray.mean()))
        det = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80)) if detector else []
        if len(det):
            x, y, w, h = max(det, key=lambda d: d[2] * d[3])
            faces.append(float(max(w, h)))
            centers.append(((x + w / 2) / frame.shape[1], (y + h / 2) / frame.shape[0]))
            sharps.append(sharpness(frame[y : y + h, x : x + w]))
        if prev_gray is not None and prev_gray.shape == gray.shape:
            motions.append(float(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)).mean()))
        prev_gray = gray
    cap.release()

    rep.face_frames = len(faces)
    rep.face_px = float(np.mean(faces)) if faces else 0.0
    rep.sharpness = float(np.mean(sharps)) if sharps else 0.0
    rep.exposure = float(np.mean(means)) if means else 0.0
    rep.exposure_drift = float(np.std(means)) if means else 0.0
    rep.motion = float(np.mean(motions)) if motions else 0.0
    rep.yaw_spread = float(np.std([c[0] for c in centers])) if len(centers) > 1 else 0.0

    _judge(rep, checked_faces=detector is not None)
    return rep


def _face_detector(cv2):  # type: ignore[no-untyped-def]
    """Haar frontal-face detector, or None if this OpenCV build has neither the
    class nor the bundled cascade (OpenCV 5 removed both)."""
    try:
        path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        clf = cv2.CascadeClassifier(path)
        return None if clf.empty() else clf
    except Exception:  # noqa: BLE001
        return None


def _judge(rep: ClipReport, *, checked_faces: bool = True) -> None:
    f = rep.findings

    if rep.frames == 0:
        f.append(Finding("fail", "no readable frames", "re-encode with: ffmpeg -i in.mp4 -c:v libx264 out.mp4"))
        return

    if not checked_faces:
        pass
    elif rep.face_frames < max(1, rep.frames * 0.8):
        f.append(
            Finding(
                "fail",
                f"a face was found in only {rep.face_frames}/{rep.frames} sampled frames",
                "use a clip where the person stays in frame, facing the camera, the whole time",
            )
        )
    else:
        f.append(Finding("ok", "face present throughout"))

    if not checked_faces:
        pass
    elif rep.face_px and rep.face_px < MIN_FACE_PX:
        f.append(
            Finding(
                "fail",
                f"the face is only ~{rep.face_px:.0f} px across; the model works at 256 px",
                "shoot or crop closer — head and shoulders, face filling ~1/3 of the frame height",
            )
        )
    elif rep.face_px < GOOD_FACE_PX:
        f.append(Finding("warn", f"face is ~{rep.face_px:.0f} px; {GOOD_FACE_PX}+ looks noticeably better", "crop tighter or use a higher-resolution source"))
    else:
        f.append(Finding("ok", f"face resolution is good ({rep.face_px:.0f} px)"))

    if rep.duration < MIN_SECONDS:
        f.append(Finding("fail", f"only {rep.duration:.1f}s of footage", f"{GOOD_SECONDS:.0f}-20s gives the idle loop enough variety"))
    elif rep.duration < GOOD_SECONDS:
        f.append(Finding("warn", f"{rep.duration:.1f}s is short; the idle loop will feel repetitive", "10-20s is the sweet spot"))
    elif rep.duration > MAX_SECONDS:
        f.append(Finding("warn", f"{rep.duration:.1f}s is longer than needed", "preparation time and memory scale with length; 10-20s is plenty"))
    else:
        f.append(Finding("ok", f"duration is good ({rep.duration:.1f}s)"))

    if rep.fps and abs(rep.fps - 25.0) > 0.6:
        f.append(
            Finding(
                "warn",
                f"{rep.fps:.2f} fps — the model is trained at 25",
                "preparation resamples to 25 fps automatically; shooting at 25 or 50 avoids judder",
            )
        )

    if rep.sharpness and rep.sharpness < MIN_SHARPNESS:
        f.append(Finding("fail", f"the face is soft (sharpness {rep.sharpness:.0f})", "use a sharper source; soft input cannot be recovered"))
    elif rep.sharpness < GOOD_SHARPNESS:
        f.append(Finding("warn", f"sharpness {rep.sharpness:.0f} is mediocre", "better focus/lighting, or a higher-bitrate source"))
    else:
        f.append(Finding("ok", f"sharp footage ({rep.sharpness:.0f})"))

    if rep.exposure < 55:
        f.append(Finding("warn", "the clip is dark", "brighter, even, front-facing light — dark footage exaggerates model artifacts"))
    elif rep.exposure > 205:
        f.append(Finding("warn", "the clip is over-exposed", "reduce highlights; blown skin loses the detail the model needs"))
    if rep.exposure_drift > 12:
        f.append(Finding("warn", "the exposure shifts during the clip", "lock exposure/white balance; drift makes the idle loop pulse"))

    if rep.motion > 12:
        f.append(Finding("warn", "a lot of movement in frame", "a mostly still head and background loops far more cleanly"))
    if rep.yaw_spread > 0.06:
        f.append(Finding("warn", "the head turns a lot", "keep the head roughly front-on; big turns break the mouth paste"))
