"""Face restoration for generated mouths.

MuseTalk regenerates the mouth at 256² inside a higher-resolution frame, and
its own README says facial detail "is not always well preserved" and suggests
pairing it with a super-resolution model. This wraps GFPGAN (Apache-2.0) as
that step, with two adaptations for live use:

* it restores only the **generated crop**, not the whole frame — the rest of
  the picture is untouched real footage and restoring it would both cost time
  and subtly change the person's identity between talking and idle;
* it **measures itself at warmup** and refuses to run if it can't keep up with
  the frame budget, because a sharp avatar that stutters is worse than a
  slightly soft one that doesn't. ``QAV_FACE_RESTORE=1`` forces it on anyway.

Weights are not bundled; the Docker image fetches GFPGANv1.4.pth.
"""

from __future__ import annotations

import logging
import os
import time

import numpy as np

logger = logging.getLogger("qav.face.restore")

DEFAULT_WEIGHT_URL = "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth"


class FaceRestorer:
    """Sharpens generated face crops. Falls back to a no-op on any failure —
    never let post-processing take down a live session."""

    def __init__(
        self,
        *,
        mode: str = "auto",
        weight_path: str | None = None,
        upscale: int = 1,
        blend: float = 0.75,
        frame_budget_ms: float = 40.0,
    ) -> None:
        self.mode = mode  # auto | on | off
        self.weight_path = weight_path or os.getenv("QAV_GFPGAN_WEIGHTS", "/models/gfpgan/GFPGANv1.4.pth")
        self.upscale = upscale
        #: how much of the restored image to mix back; < 1 keeps some of the
        #: original texture so the mouth doesn't look pasted on
        self.blend = float(np.clip(blend, 0.0, 1.0))
        self.frame_budget_ms = frame_budget_ms
        self.enabled = False
        self.measured_ms: float | None = None
        self._restorer = None

    @classmethod
    def from_env(cls, *, fps: float) -> FaceRestorer:
        mode = os.getenv("QAV_FACE_RESTORE", "auto").lower()
        if mode in ("0", "false", "no"):
            mode = "off"
        elif mode in ("1", "true", "yes"):
            mode = "on"
        return cls(
            mode=mode,
            blend=float(os.getenv("QAV_FACE_RESTORE_BLEND", "0.75")),
            # leave half the frame budget for the model itself
            frame_budget_ms=float(os.getenv("QAV_FACE_RESTORE_BUDGET_MS", str(1000.0 / fps * 0.5))),
        )

    # ------------------------------------------------------------------ setup
    def warmup(self, sample: np.ndarray) -> None:
        if self.mode == "off":
            logger.info("face restoration disabled")
            return
        try:
            from gfpgan import GFPGANer  # type: ignore
        except Exception as e:  # noqa: BLE001
            logger.warning("face restoration unavailable (%s); mouths will be softer", e)
            return
        if not os.path.exists(self.weight_path):
            logger.warning("GFPGAN weights not found at %s; skipping restoration", self.weight_path)
            return
        try:
            self._restorer = GFPGANer(model_path=self.weight_path, upscale=self.upscale, arch="clean", channel_multiplier=2)
        except Exception as e:  # noqa: BLE001
            logger.warning("could not load GFPGAN (%s); skipping restoration", e)
            return

        # measure on the real crop size, after a throwaway pass for CUDA warmup
        self._restore_one(sample)
        t0 = time.perf_counter()
        runs = 3
        for _ in range(runs):
            self._restore_one(sample)
        self.measured_ms = (time.perf_counter() - t0) * 1000.0 / runs

        if self.mode == "on":
            self.enabled = True
        else:
            self.enabled = self.measured_ms <= self.frame_budget_ms
        logger.info(
            "face restoration %s (%.1f ms/frame, budget %.1f ms)%s",
            "ON" if self.enabled else "OFF",
            self.measured_ms,
            self.frame_budget_ms,
            "" if self.enabled else " — set QAV_FACE_RESTORE=1 to force",
        )

    # ---------------------------------------------------------------- restore
    def __call__(self, crops: list[np.ndarray]) -> list[np.ndarray]:
        """`crops` are BGR uint8 face crops; returns same shape, sharpened."""
        if not self.enabled or self._restorer is None:
            return crops
        out = []
        for c in crops:
            try:
                r = self._restore_one(c)
            except Exception as e:  # noqa: BLE001
                logger.warning("restoration failed mid-stream (%s); disabling", e)
                self.enabled = False
                return crops
            out.append(r if r is not None else c)
        return out

    def _restore_one(self, bgr: np.ndarray) -> np.ndarray | None:
        assert self._restorer is not None
        _, _, restored = self._restorer.enhance(bgr, has_aligned=False, only_center_face=True, paste_back=True)
        if restored is None:
            return None
        if restored.shape != bgr.shape:
            import cv2

            restored = cv2.resize(restored, (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_AREA)
        if self.blend >= 1.0:
            return restored
        return (restored.astype(np.float32) * self.blend + bgr.astype(np.float32) * (1.0 - self.blend)).astype(np.uint8)
