"""MuseTalk 1.5 backend — photoreal lip-sync on a real video clip of a person.

Port of MuseTalk's `scripts/realtime_inference.py` (TMElyralab/MuseTalk,
MIT, pinned in Dockerfile.musetalk) to a *streaming* contract:

* preparation (once per avatar, `qav-face prepare --renderer musetalk`):
  frames → face landmarks/bboxes → 256² crops → VAE latents (masked + ref)
  and per-frame blending masks, all cached on disk;
* runtime (per 8-frame batch, ~320 ms): whisper-tiny features over a rolling
  window → per-frame 10×5×384 context → UNet(t=0) → VAE decode → paste the
  new mouth into the original frame with the cached mask.

The original computes whisper features over a whole file with 2 frames of
zero padding on each side of every frame's window; we keep 2 frames of real
past audio and ask the stream for 2 frames of lookahead, which is why
``lookahead_frames = 2``. While silent the original footage plays, so the
person breathes and blinks naturally.

Env: MUSETALK_ROOT (repo checkout), MUSETALK_MODELS (default $ROOT/models),
QAV_MUSETALK_BATCH (frames per batch, default 8), QAV_AVATAR_DIR.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from ..avatars import AvatarSpec, avatar_root
from .base import FaceBackend

logger = logging.getLogger("qav.face.musetalk")

FACE_SIZE = 256
WHISPER_FPS = 50
LEFT_CONTEXT_FRAMES = 2
RIGHT_CONTEXT_FRAMES = 2
FEATURE_SLOTS_PER_FRAME = 2 * (LEFT_CONTEXT_FRAMES + RIGHT_CONTEXT_FRAMES + 1)  # 10


def musetalk_root() -> str:
    root = os.getenv("MUSETALK_ROOT")
    if not root or not os.path.isdir(root):
        raise RuntimeError(
            "MUSETALK_ROOT must point at a MuseTalk checkout "
            "(git clone https://github.com/TMElyralab/MuseTalk; see services/face/Dockerfile.musetalk)"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def models_dir() -> str:
    return os.getenv("MUSETALK_MODELS") or os.path.join(musetalk_root(), "models")


def _fit_box(src_w: int, src_h: int, dst_w: int, dst_h: int) -> tuple[int, int, int, int]:
    """Largest centred crop of the source with the destination aspect ratio."""
    src_ar = src_w / src_h
    dst_ar = dst_w / dst_h
    if src_ar > dst_ar:
        w = int(round(src_h * dst_ar))
        x0 = (src_w - w) // 2
        return x0, 0, x0 + w, src_h
    h = int(round(src_w / dst_ar))
    y0 = max(0, int((src_h - h) * 0.35))  # keep more headroom than chin
    return 0, y0, src_w, y0 + h


class MuseTalkBackend(FaceBackend):
    batch_frames = int(os.getenv("QAV_MUSETALK_BATCH", "8"))
    lookahead_frames = RIGHT_CONTEXT_FRAMES
    model_sample_rate = 16_000
    run_in_thread = True

    def __init__(self, spec: AvatarSpec) -> None:
        super().__init__(spec)
        self._avatar_dir = spec.asset_path("avatarDir") or os.path.join(avatar_root(), spec.id)
        self._spf = int(self.model_sample_rate / spec.fps)
        self._tail = np.zeros(LEFT_CONTEXT_FRAMES * self._spf, dtype=np.int16)
        self._idx = 0
        self._loaded = False
        # populated by warmup()
        self._frames: list[np.ndarray] = []
        self._masks: list[np.ndarray] = []
        self._coords: list[list[int]] = []
        self._mask_coords: list[list[int]] = []
        self._latents: list[Any] = []
        self._crop: tuple[int, int, int, int] | None = None

    # ---------------------------------------------------------------- setup
    def warmup(self) -> None:
        if self._loaded:
            return
        # check the checkout first: a missing MUSETALK_ROOT is the common operator
        # mistake and must not surface as "No module named cv2"
        musetalk_root()

        import cv2
        import torch
        from transformers import AutoFeatureExtractor, WhisperModel

        from musetalk.models.unet import PositionalEncoding, UNet
        from musetalk.models.vae import VAE
        from musetalk.utils.blending import get_image_blending

        if not torch.cuda.is_available():
            raise RuntimeError("MuseTalk needs a CUDA GPU")
        self._torch = torch
        self._cv2 = cv2
        self._blend = get_image_blending
        self._device = torch.device("cuda:0")
        mdir = models_dir()

        info_path = os.path.join(self._avatar_dir, "info.json")
        if not os.path.exists(info_path):
            raise RuntimeError(
                f"avatar {self.spec.id!r} is not prepared ({info_path} missing); run "
                f"`qav-face prepare --renderer musetalk --avatar {self.spec.id} --video <clip.mp4>`"
            )
        info = json.load(open(info_path))
        logger.info("loading MuseTalk avatar %s (%d frames)", self.spec.id, info["frames"])

        self._vae = VAE(model_path=os.path.join(mdir, "sd-vae"))
        self._unet = UNet(
            unet_config=os.path.join(mdir, "musetalkV15", "musetalk.json"),
            model_path=os.path.join(mdir, "musetalkV15", "unet.pth"),
            device=self._device,
        )
        self._pe = PositionalEncoding(d_model=384).half().to(self._device)
        self._vae.vae = self._vae.vae.half().to(self._device)
        self._unet.model = self._unet.model.half().to(self._device)
        self._timesteps = torch.tensor([0], device=self._device)
        self._dtype = self._unet.model.dtype

        self._feature_extractor = AutoFeatureExtractor.from_pretrained(os.path.join(mdir, "whisper"))
        self._whisper = WhisperModel.from_pretrained(os.path.join(mdir, "whisper")).to(self._device, dtype=self._dtype).eval()
        self._whisper.requires_grad_(False)

        frames_dir = Path(self._avatar_dir) / "frames"
        masks_dir = Path(self._avatar_dir) / "masks"
        names = sorted(p.name for p in frames_dir.glob("*.png"))
        self._frames = [cv2.imread(str(frames_dir / n)) for n in names]
        self._masks = [cv2.imread(str(masks_dir / n), cv2.IMREAD_GRAYSCALE) for n in names]
        self._coords = json.load(open(os.path.join(self._avatar_dir, "coords.json")))
        self._mask_coords = json.load(open(os.path.join(self._avatar_dir, "mask_coords.json")))
        self._latents = torch.load(os.path.join(self._avatar_dir, "latents.pt"), map_location="cpu")
        n = min(len(self._frames), len(self._masks), len(self._coords), len(self._mask_coords), len(self._latents))
        if n == 0:
            raise RuntimeError(f"avatar {self.spec.id!r} has no usable frames")
        self._frames, self._masks = self._frames[:n], self._masks[:n]
        self._coords, self._mask_coords, self._latents = self._coords[:n], self._mask_coords[:n], self._latents[:n]

        h, w = self._frames[0].shape[:2]
        self._crop = _fit_box(w, h, self.width, self.height)
        self._loaded = True
        # one dummy pass so CUDA kernels are compiled before the first real batch
        self.speech_frames(np.zeros((self.batch_frames + self.lookahead_frames) * self._spf, dtype=np.int16), self.batch_frames)
        self._idx = 0
        self._tail[:] = 0
        logger.info("MuseTalk ready: %d cycle frames, output %dx%d", n, self.width, self.height)

    # --------------------------------------------------------------- frames
    def idle_frame(self) -> np.ndarray:
        self.warmup()
        frame = self._frames[self._idx % len(self._frames)]
        self._idx += 1
        return self._fit(frame)

    def speech_frames(self, audio: np.ndarray, n_frames: int) -> list[np.ndarray]:
        self.warmup()
        torch = self._torch
        n = n_frames
        # window = 2 frames of history + this batch + lookahead the stream already appended
        window = np.concatenate([self._tail, audio])
        keep = LEFT_CONTEXT_FRAMES * self._spf
        body = audio[: n * self._spf]
        self._tail = body[-keep:] if len(body) >= keep else np.concatenate([self._tail, body])[-keep:]

        feats = self._whisper_features(window)  # [T, 5, 384] at 50 Hz
        need = 2 * n + FEATURE_SLOTS_PER_FRAME - 2
        if feats.shape[0] < need:
            feats = torch.cat([feats, torch.zeros(need - feats.shape[0], *feats.shape[1:], device=feats.device, dtype=feats.dtype)])
        clips = torch.stack([feats[2 * j : 2 * j + FEATURE_SLOTS_PER_FRAME] for j in range(n)])  # [n, 10, 5, 384]
        clips = clips.reshape(n, FEATURE_SLOTS_PER_FRAME * clips.shape[2], clips.shape[3])  # [n, 50, 384]

        with torch.no_grad():
            audio_feature = self._pe(clips.to(self._device, dtype=self._dtype))
            latent_batch = torch.cat([self._latents[(self._idx + j) % len(self._latents)] for j in range(n)], dim=0)
            latent_batch = latent_batch.to(device=self._device, dtype=self._dtype)
            pred = self._unet.model(latent_batch, self._timesteps, encoder_hidden_states=audio_feature).sample
            recon = self._vae.decode_latents(pred.to(dtype=self._vae.vae.dtype))  # n × 256×256×3 BGR

        out = []
        for j, face in enumerate(recon):
            k = (self._idx + j) % len(self._frames)
            x1, y1, x2, y2 = self._coords[k]
            face = self._cv2.resize(face.astype(np.uint8), (x2 - x1, y2 - y1))
            composed = self._blend(self._frames[k].copy(), face, [x1, y1, x2, y2], self._masks[k], self._mask_coords[k])
            out.append(self._fit(composed))
        self._idx += n
        return out

    def segment_end(self) -> None:
        self._tail[:] = 0

    def reset(self) -> None:
        self._tail[:] = 0

    def close(self) -> None:
        if self._loaded:
            self._frames.clear()
            self._masks.clear()
            self._latents.clear()
            try:
                self._torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass

    # ------------------------------------------------------------- helpers
    def _whisper_features(self, audio16k: np.ndarray):
        torch = self._torch
        x = audio16k.astype(np.float32) / 32768.0
        feats = self._feature_extractor(x, return_tensors="pt", sampling_rate=self.model_sample_rate).input_features
        with torch.no_grad():
            hidden = self._whisper.encoder(feats.to(self._device, dtype=self._dtype), output_hidden_states=True).hidden_states
        stacked = torch.stack(hidden, dim=2)[0]  # [1500, layers, 384]
        actual = int(len(x) / self.model_sample_rate * WHISPER_FPS)
        return stacked[:actual]

    def _fit(self, bgr: np.ndarray) -> np.ndarray:
        assert self._crop is not None
        x0, y0, x1, y1 = self._crop
        crop = bgr[y0:y1, x0:x1]
        if crop.shape[1] != self.width or crop.shape[0] != self.height:
            crop = self._cv2.resize(crop, (self.width, self.height), interpolation=self._cv2.INTER_AREA)
        rgba = self._cv2.cvtColor(crop, self._cv2.COLOR_BGR2RGBA)
        return np.ascontiguousarray(rgba)


# --------------------------------------------------------------------------
# Preparation (offline, GPU): port of Avatar.prepare_material with v1.5 settings
# --------------------------------------------------------------------------

def prepare_avatar(
    *,
    avatar_id: str,
    video_path: str,
    bbox_shift: int = 0,
    fps: int = 25,
    max_seconds: float = float(os.getenv("QAV_MUSETALK_MAX_SECONDS", "8")),
    max_dim: int = int(os.getenv("QAV_MUSETALK_MAX_DIM", "720")),
    extra_margin: int = 10,
    parsing_mode: str = "jaw",
    left_cheek_width: int = 90,
    right_cheek_width: int = 90,
) -> str:
    """Turn a clip of a person into a MuseTalk avatar directory under QAV_AVATAR_DIR."""
    import cv2
    import torch

    root = musetalk_root()
    out_dir = Path(avatar_root()) / avatar_id
    if out_dir.exists():
        shutil.rmtree(out_dir)
    frames_dir, masks_dir, tmp_dir = out_dir / "frames", out_dir / "masks", out_dir / "_src"
    for d in (frames_dir, masks_dir, tmp_dir):
        d.mkdir(parents=True, exist_ok=True)

    # 1. decode at the model's frame rate, bounded in length and size
    subprocess.run(
        [
            "ffmpeg", "-y", "-loglevel", "error", "-i", os.path.abspath(video_path), "-t", str(max_seconds),
            "-vf", f"fps={fps},scale='min({max_dim},iw)':-2", str(tmp_dir / "%08d.png"),
        ],
        check=True,
    )
    src_imgs = sorted(str(p) for p in tmp_dir.glob("*.png"))
    if not src_imgs:
        raise RuntimeError("ffmpeg produced no frames")

    # 2. MuseTalk's preprocessing modules load their weights via paths relative to the repo root
    cwd = os.getcwd()
    os.chdir(root)
    try:
        from musetalk.utils.blending import get_image_prepare_material
        from musetalk.utils.face_parsing import FaceParsing
        from musetalk.utils.preprocessing import get_landmark_and_bbox
        from musetalk.utils.utils import load_all_model

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vae, _unet, _pe = load_all_model(
            unet_model_path=os.path.join(models_dir(), "musetalkV15", "unet.pth"),
            vae_type="sd-vae",
            unet_config=os.path.join(models_dir(), "musetalkV15", "musetalk.json"),
            device=device,
        )
        vae.vae = vae.vae.half().to(device)
        fp = FaceParsing(left_cheek_width=left_cheek_width, right_cheek_width=right_cheek_width)

        logger.info("extracting landmarks for %d frames", len(src_imgs))
        coord_list, frame_list = get_landmark_and_bbox(src_imgs, bbox_shift)
        placeholder = (0.0, 0.0, 0.0, 0.0)
        latents, keep_frames, keep_coords = [], [], []
        for bbox, frame in zip(coord_list, frame_list):
            if bbox == placeholder:
                continue  # no face in this frame: drop it from the loop
            x1, y1, x2, y2 = bbox
            y2 = min(y2 + extra_margin, frame.shape[0])
            crop = cv2.resize(frame[y1:y2, x1:x2], (FACE_SIZE, FACE_SIZE), interpolation=cv2.INTER_LANCZOS4)
            latents.append(vae.get_latents_for_unet(crop).cpu())
            keep_frames.append(frame)
            keep_coords.append([int(x1), int(y1), int(x2), int(y2)])
        if not latents:
            raise RuntimeError("no face detected in the clip")

        # ping-pong cycle so the idle loop never jumps
        frames_cycle = keep_frames + keep_frames[::-1]
        coords_cycle = keep_coords + keep_coords[::-1]
        latents_cycle = latents + latents[::-1]

        mask_coords = []
        for i, frame in enumerate(frames_cycle):
            cv2.imwrite(str(frames_dir / f"{i:08d}.png"), frame)
            mask, crop_box = get_image_prepare_material(frame, coords_cycle[i], fp=fp, mode=parsing_mode)
            cv2.imwrite(str(masks_dir / f"{i:08d}.png"), mask)
            mask_coords.append([int(v) for v in crop_box])
    finally:
        os.chdir(cwd)

    torch.save(latents_cycle, out_dir / "latents.pt")
    json.dump(coords_cycle, open(out_dir / "coords.json", "w"))
    json.dump(mask_coords, open(out_dir / "mask_coords.json", "w"))
    h, w = frames_cycle[0].shape[:2]
    json.dump(
        {
            "avatar_id": avatar_id,
            "renderer": "musetalk",
            "version": "v15",
            "source": os.path.abspath(video_path),
            "frames": len(frames_cycle),
            "fps": fps,
            "width": w,
            "height": h,
            "bbox_shift": bbox_shift,
            "extra_margin": extra_margin,
            "parsing_mode": parsing_mode,
        },
        open(out_dir / "info.json", "w"),
        indent=2,
    )
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return str(out_dir)
