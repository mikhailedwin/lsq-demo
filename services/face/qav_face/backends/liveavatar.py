"""Live Avatar backend — Alibaba's Wan2.2-S2V-14B + Live-Avatar LoRA (Apache-2.0).

EXPERIMENTAL. This is the highest-quality option: a 14B video-diffusion model
that generates the whole face from one reference image. Hardware reality:

* the authors' 45 fps figure is a 5×H800 pipeline; single-GPU mode needs an
  80 GB card (48 GB with ``QAV_LIVEAVATAR_FP8=1``) and generates a 48-frame clip
  (1.92 s at 25 fps) per step — the clip must generate faster than it plays
  or audio will stall. Measure with ``qav-face selftest --renderer liveavatar``
  on the target GPU before putting it in front of users.
* the upstream code only exposes an offline ``generate(audio_path)``; the
  streaming below re-implements its per-clip loop (kv-cache prefill, 4-step
  Euler sampling per 3-latent-frame block, online VAE decode with 73 motion
  frames of overlap) as a stateful object fed one clip of audio at a time.
  Cross-checked against LiveAvatar@c3c47d0 `causal_s2v_pipeline.py`.

Env: LIVEAVATAR_ROOT (repo checkout), LIVEAVATAR_CKPT (Wan2.2-S2V-14B dir),
LIVEAVATAR_LORA (default HF id Quark-Vision/Live-Avatar), QAV_LIVEAVATAR_SIZE
(default "704*384"), QAV_LIVEAVATAR_FP8, QAV_LIVEAVATAR_COMPILE.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
import threading
from collections import deque
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np

from ..avatars import AvatarSpec, avatar_root
from .base import FaceBackend

logger = logging.getLogger("qav.face.liveavatar")

MODEL_FPS = 25
INFER_FRAMES = 48  # per clip; must be a multiple of 4 (authors use 48 for real time)
AUDIO_HISTORY_SECONDS = 1.0

DEMO_PROMPTS: dict[str, str] = {
    "anchor": "A professional news anchor in a modern studio, looking at the camera, speaking clearly and calmly with natural, subtle head movements. Soft key light, shallow depth of field, broadcast quality.",
    "fashion_blogger": "A friendly fashion blogger in a bright apartment, talking directly to the camera with warm, expressive gestures. Natural daylight, crisp detail.",
    "kitchen_grandmother": "A warm elderly woman in a cosy kitchen, chatting to the camera with gentle smiles and natural head motion. Soft window light, homely atmosphere.",
}


def liveavatar_root() -> str:
    root = os.getenv("LIVEAVATAR_ROOT")
    if not root or not os.path.isdir(root):
        raise RuntimeError(
            "LIVEAVATAR_ROOT must point at a LiveAvatar checkout "
            "(git clone https://github.com/Alibaba-Quark/LiveAvatar; see services/face/Dockerfile.liveavatar)"
        )
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def prepare_avatar(*, avatar_id: str, image_path: str, prompt: str = "") -> str:
    """A Live Avatar 'avatar' is just a reference image + scene prompt."""
    out = Path(avatar_root()) / avatar_id
    out.mkdir(parents=True, exist_ok=True)
    ext = Path(image_path).suffix.lower() or ".jpg"
    shutil.copyfile(image_path, out / f"reference{ext}")
    json.dump(
        {"avatar_id": avatar_id, "renderer": "liveavatar", "image": f"reference{ext}", "prompt": prompt, "source": os.path.abspath(image_path)},
        open(out / "info.json", "w"),
        indent=2,
    )
    return str(out)


class LiveAvatarBackend(FaceBackend):
    batch_frames = INFER_FRAMES
    lookahead_frames = 0
    model_sample_rate = 16_000
    run_in_thread = True

    def __init__(self, spec: AvatarSpec) -> None:
        super().__init__(spec)
        self._streamer: _LiveAvatarStreamer | None = None
        self._idle_frames: deque[np.ndarray] = deque()
        self._prefetch: threading.Thread | None = None
        self._lock = threading.Lock()
        self._history = np.zeros(int(AUDIO_HISTORY_SECONDS * self.model_sample_rate), dtype=np.int16)
        self._last_frame: np.ndarray | None = None

    # ---------------------------------------------------------------- setup
    def _resolve_reference(self) -> tuple[str, str]:
        img = self.spec.asset_path("image")
        prompt = str(self.spec.assets.get("prompt") or "")
        if not img:
            info_path = os.path.join(avatar_root(), self.spec.id, "info.json")
            if os.path.exists(info_path):
                info = json.load(open(info_path))
                img = os.path.join(avatar_root(), self.spec.id, info["image"])
                prompt = prompt or info.get("prompt", "")
        if not img or not os.path.exists(img):
            raise RuntimeError(
                f"Live Avatar {self.spec.id!r} has no reference image; run "
                f"`qav-face prepare --renderer liveavatar --avatar {self.spec.id} --video <portrait.jpg> --prompt '...'`"
            )
        return img, prompt or "A person looking at the camera, speaking naturally with subtle head movements."

    def warmup(self) -> None:
        if self._streamer is not None:
            return
        img, prompt = self._resolve_reference()
        self._streamer = _LiveAvatarStreamer(
            size=os.getenv("QAV_LIVEAVATAR_SIZE", "704*384"),
            fp8=os.getenv("QAV_LIVEAVATAR_FP8", "0") == "1",
            compile_model=os.getenv("QAV_LIVEAVATAR_COMPILE", "0") == "1",
            seed=int(os.getenv("QAV_LIVEAVATAR_SEED", "420")),
        )
        self._streamer.load()
        self._streamer.begin(img, prompt)
        logger.info("Live Avatar ready for %s", self.spec.id)
        self._ensure_idle_clip()

    # --------------------------------------------------------------- frames
    def idle_frame(self) -> np.ndarray:
        self.warmup()
        with self._lock:
            if self._idle_frames:
                frame = self._idle_frames.popleft()
                low = len(self._idle_frames) < INFER_FRAMES // 2
            else:
                frame = None
                low = True
        if low:
            self._ensure_idle_clip()
        if frame is None:
            # generation is behind real time: hold the last picture rather than go black
            if self._last_frame is None:
                self._prefetch.join() if self._prefetch else None
                with self._lock:
                    frame = self._idle_frames.popleft() if self._idle_frames else None
            frame = frame if frame is not None else self._last_frame
        self._last_frame = frame
        return frame

    def speech_frames(self, audio: np.ndarray, n_frames: int) -> list[np.ndarray]:
        self.warmup()
        assert self._streamer is not None
        if self._prefetch is not None:
            self._prefetch.join()  # never interleave two generations
        with self._lock:
            self._idle_frames.clear()  # silence frames generated ahead are now stale
        clip_audio = audio[: INFER_FRAMES * (self.model_sample_rate // MODEL_FPS)]
        frames = self._generate(clip_audio)
        # the model always yields whole clips; return exactly what the stream asked for
        if len(frames) < n_frames:
            frames = frames + [frames[-1]] * (n_frames - len(frames))
        self._last_frame = frames[n_frames - 1]
        return frames[:n_frames]

    def segment_end(self) -> None:
        self._ensure_idle_clip()

    def reset(self) -> None:
        self._history[:] = 0

    def close(self) -> None:
        if self._prefetch is not None:
            self._prefetch.join(timeout=30)
        if self._streamer is not None:
            self._streamer.close()
            self._streamer = None

    # ------------------------------------------------------------- helpers
    def _generate(self, clip_audio16k: np.ndarray) -> list[np.ndarray]:
        assert self._streamer is not None
        n = INFER_FRAMES * (self.model_sample_rate // MODEL_FPS)
        if len(clip_audio16k) < n:
            clip_audio16k = np.pad(clip_audio16k, (0, n - len(clip_audio16k)))
        window = np.concatenate([self._history, clip_audio16k])
        self._history = clip_audio16k[-len(self._history) :]
        frames = self._streamer.next_clip(window, clip_frames=INFER_FRAMES)
        return [self._fit(f) for f in frames]

    def _ensure_idle_clip(self) -> None:
        if self._prefetch is not None and self._prefetch.is_alive():
            return

        def _run() -> None:
            silence = np.zeros(INFER_FRAMES * (self.model_sample_rate // MODEL_FPS), dtype=np.int16)
            frames = self._generate(silence)
            with self._lock:
                self._idle_frames.extend(frames)

        self._prefetch = threading.Thread(target=_run, name="liveavatar-idle", daemon=True)
        self._prefetch.start()

    def _fit(self, rgb: np.ndarray) -> np.ndarray:
        import cv2

        h, w = rgb.shape[:2]
        src_ar, dst_ar = w / h, self.width / self.height
        if src_ar > dst_ar:
            nw = int(round(h * dst_ar))
            x0 = (w - nw) // 2
            rgb = rgb[:, x0 : x0 + nw]
        elif src_ar < dst_ar:
            nh = int(round(w / dst_ar))
            y0 = max(0, int((h - nh) * 0.35))
            rgb = rgb[y0 : y0 + nh]
        if rgb.shape[1] != self.width or rgb.shape[0] != self.height:
            rgb = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return np.ascontiguousarray(cv2.cvtColor(rgb, cv2.COLOR_RGB2RGBA))


class _LiveAvatarStreamer:
    """Stateful port of ``WanS2V.generate`` (single GPU, online decode) driven one clip at a time."""

    def __init__(self, *, size: str, fp8: bool, compile_model: bool, seed: int) -> None:
        self.size = size
        self.fp8 = fp8
        self.compile_model = compile_model
        self.seed = seed
        self.s2v: Any = None
        self.r = 0

    # ------------------------------------------------------------- loading
    def load(self) -> None:
        import torch
        import torch.distributed as dist

        root = liveavatar_root()
        from liveavatar.models.wan.causal_s2v_pipeline import WanS2V
        from liveavatar.models.wan.wan_2_2.configs import WAN_CONFIGS
        from liveavatar.utils.args_config import parse_args_for_training_config

        if not torch.cuda.is_available():
            raise RuntimeError("Live Avatar needs a CUDA GPU (80 GB, or 48 GB with FP8)")
        torch.cuda.set_device(0)
        # the pipeline reads dist.get_rank() even on one GPU
        if not dist.is_initialized():
            os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
            os.environ.setdefault("MASTER_PORT", os.getenv("QAV_LIVEAVATAR_DIST_PORT", "29511"))
            os.environ.setdefault("RANK", "0")
            os.environ.setdefault("WORLD_SIZE", "1")
            dist.init_process_group(backend="nccl", init_method="env://", rank=0, world_size=1)

        os.environ["ENABLE_COMPILE"] = "true" if self.compile_model else "false"
        os.environ["ENABLE_FP8"] = "true" if self.fp8 else "false"
        ckpt_dir = os.getenv("LIVEAVATAR_CKPT", os.path.join(root, "ckpt", "Wan2.2-S2V-14B"))
        lora = os.getenv("LIVEAVATAR_LORA", "Quark-Vision/Live-Avatar")
        training = parse_args_for_training_config(os.path.join(root, "liveavatar", "configs", "s2v_causal_sft.yaml"))
        self.cfg = WAN_CONFIGS["s2v-14B"]

        logger.info("loading Wan2.2-S2V-14B from %s (fp8=%s, compile=%s)", ckpt_dir, self.fp8, self.compile_model)
        s2v = WanS2V(
            config=self.cfg,
            checkpoint_dir=ckpt_dir,
            device_id=0,
            rank=0,
            convert_model_dtype=True,
            single_gpu=True,
            offload_kv_cache=os.getenv("QAV_LIVEAVATAR_OFFLOAD_KV", "0") == "1",
        )
        s2v.noise_model = s2v.add_lora_to_model(
            s2v.noise_model,
            lora_rank=training["lora_rank"],
            lora_alpha=training["lora_alpha"],
            lora_target_modules=training["lora_target_modules"],
            init_lora_weights=training["init_lora_weights"],
            pretrained_lora_path=lora,
            load_lora_weight_only=False,
        )
        if self.fp8 and hasattr(torch, "_scaled_mm"):
            from liveavatar.utils.fp8_linear import replace_linear_with_scaled_fp8

            replace_linear_with_scaled_fp8(
                s2v.noise_model,
                ignore_keys=["text_embedding", "time_embedding", "time_projection", "head.head", "casual_audio_encoder.encoder.final_linear"],
            )
        self.s2v = s2v
        self.torch = torch
        self.dist = dist

    # --------------------------------------------------------------- begin
    def begin(self, ref_image_path: str, prompt: str) -> None:
        """Step 1 of generate(): everything that does not depend on audio."""
        import torch
        from PIL import Image
        from torchvision import transforms

        from liveavatar.models.wan.wan_2_2.configs import MAX_AREA_CONFIGS
        from liveavatar.models.wan.wan_2_2.utils.fm_solvers import FlowMatchEulerDiscreteScheduler  # type: ignore

        s2v = self.s2v
        size = s2v.get_gen_size(size=None, max_area=MAX_AREA_CONFIGS[self.size], ref_image_path=ref_image_path, pre_video_path=None)
        self.H, self.W = size
        resize_op = transforms.Resize(min(self.H, self.W))
        crop_op = transforms.CenterCrop((self.H, self.W))
        to_tensor = transforms.ToTensor()

        s2v.audio_encoder.model.to(device=s2v.device, dtype=s2v.param_dtype)
        s2v.audio_encoder.model.requires_grad_(False)
        s2v.audio_encoder.model.eval()
        s2v.vae.model.to(s2v.device)

        ref_image = np.array(Image.open(ref_image_path).convert("RGB"))
        self.lat_motion_frames = (s2v.motion_frames + 3) // 4
        model_pic = crop_op(resize_op(Image.fromarray(ref_image)))
        ref_pixels = to_tensor(model_pic).unsqueeze(1).unsqueeze(0) * 2 - 1.0
        ref_pixels = ref_pixels.to(dtype=s2v.vae.dtype, device=s2v.vae.device).repeat(1, 1, 5, 1, 1)
        self.ref_latents = torch.stack(s2v.vae.encode(ref_pixels))[:, :, 1:]

        motion = ref_pixels.repeat(1, 1, s2v.motion_frames, 1, 1)
        self.videos_last_frames = motion.detach()
        self.motion_latents = torch.stack(s2v.vae.encode(motion))

        self.COND = s2v.load_pose_cond(pose_video=None, num_repeat=1, infer_frames=INFER_FRAMES, size=size)
        self.context, _ = s2v.encode_prompt(prompt, s2v.sample_neg_prompt, True)
        self.scheduler = FlowMatchEulerDiscreteScheduler(num_train_timesteps=s2v.num_train_timesteps, shift=3)
        self._timesteps = None
        s2v.kv_cache1 = None
        s2v.shared_cond_cache = None
        self.r = 0
        logger.info("Live Avatar canvas %dx%d, motion frames %d", self.W, self.H, s2v.motion_frames)

    # ------------------------------------------------------------ audio → emb
    def _audio_embed(self, window16k: np.ndarray, clip_frames: int):
        """wav2vec over the window; keep the last `clip_frames` frames' embeddings → [1, L, D, clip_frames]."""
        s2v = self.s2v
        enc = s2v.audio_encoder
        enc.model.to(device=s2v.device)
        z = enc.extract_audio_feat_from_array(window16k.astype(np.float32) / 32768.0, sample_rate=16_000, return_all_layers=True)
        bucket, _ = enc.get_audio_embed_bucket_fps(z, fps=MODEL_FPS, batch_frames=clip_frames, m=s2v.audio_sample_m)
        bucket = bucket.to(s2v.device, dtype=s2v.param_dtype).unsqueeze(0)
        bucket = bucket.permute(0, 2, 3, 1) if bucket.dim() == 4 else bucket.permute(0, 2, 1)
        n_window = int(len(window16k) / 16_000 * MODEL_FPS)
        end = min(n_window, bucket.shape[-1])
        emb = bucket[..., max(0, end - clip_frames) : end]
        if emb.shape[-1] < clip_frames:
            pad = self.torch.zeros(*emb.shape[:-1], clip_frames - emb.shape[-1], device=emb.device, dtype=emb.dtype)
            emb = self.torch.cat([emb, pad], dim=-1)
        return emb

    # ------------------------------------------------------------- one clip
    def next_clip(self, window16k: np.ndarray, *, clip_frames: int = INFER_FRAMES) -> list[np.ndarray]:
        """Steps 2.1-2.3 of generate() for one clip, online-decoded. Returns RGB uint8 frames."""
        torch = self.torch
        s2v = self.s2v
        r = self.r
        audio_input = self._audio_embed(window16k, clip_frames)

        with torch.amp.autocast("cuda", dtype=s2v.param_dtype), torch.no_grad():
            seed_g = torch.Generator(device=s2v.device)
            seed_g.manual_seed(self.seed + r)
            lat_target_frames = (clip_frames + 3 + s2v.motion_frames) // 4 - self.lat_motion_frames
            target_shape = [lat_target_frames, self.H // 8, self.W // 8]
            frame_seq_length = self.H // 8 * self.W // 8 // 2 // 2
            clip_noise = [torch.randn(16, *target_shape, dtype=s2v.param_dtype, device=s2v.device, generator=seed_g)]
            clip_output = torch.zeros_like(clip_noise[0])
            max_seq_len = int(np.prod(target_shape)) // 4

            if s2v.kv_cache1 is None:
                s2v.noise_model.to(s2v.device)
                s2v.vae.model.cpu()
                s2v.text_encoder.model.cpu()
                s2v.audio_encoder.model.cpu()
                torch.cuda.empty_cache()
                s2v.kv_cache1 = {}
                if not s2v.offload_kv_cache:
                    s2v.shared_cond_cache = []
                    for _ in range(s2v.noise_model.num_layers):
                        s2v.shared_cond_cache.append(
                            {
                                "cond_k": torch.zeros([1, 2800, 40, 128], dtype=s2v.param_dtype, device="cuda:0"),
                                "cond_v": torch.zeros([1, 2800, 40, 128], dtype=s2v.param_dtype, device="cuda:0"),
                                "cond_end": torch.tensor([0], dtype=torch.long, device="cuda:0"),
                            }
                        )
                else:
                    s2v.shared_cond_cache = None
                for gpu_id in range(4):
                    s2v._initialize_kv_cache(batch_size=1, dtype=s2v.param_dtype, device=f"cuda:{gpu_id + 1}", gpu_id=gpu_id + 1, kv_cache_size=max_seq_len)
                s2v._initialize_crossattn_cache(batch_size=1, dtype=s2v.param_dtype, device=s2v.device)

            clip_latents = deepcopy(clip_noise)
            cond_latents = (self.COND[0] * 0).to(dtype=s2v.param_dtype, device=s2v.device)
            input_motion_latents = self.motion_latents.clone()
            s2v.noise_model.to(s2v.device)
            s2v.vae.model.cpu()
            torch.cuda.empty_cache()

            nfb = s2v.num_frames_per_block
            common = {
                "context": self.context[0:1],
                "seq_len": None,
                "motion_latents": input_motion_latents,
                "ref_latents": self.ref_latents,
                "motion_frames": [s2v.motion_frames, self.lat_motion_frames],
                "drop_motion_frames": False,
            }

            # 2.2.0 prefill the clean caches (first two clips, as upstream does with online decode)
            if r in (0, 1):
                for gpu_id in range(4):
                    s2v._move_kv_cache_to_working_gpu(gpu_id + 1)
                    block_latents = clip_latents[0][:, 0:nfb]
                    arg = {
                        **common,
                        "cond_states": cond_latents[:, :, 0:nfb],
                        "audio_input": audio_input[..., 0 : nfb * 4],
                        "sink_flag": True,
                    }
                    timestep = torch.ones([1, nfb], device=s2v.device, dtype=s2v.param_dtype) * 0
                    s2v.noise_model(
                        [block_latents], t=timestep * 0, **arg,
                        kv_cache=s2v.kv_cache1[str(gpu_id + 1)], crossattn_cache=s2v.crossattn_cache,
                        current_start=0, current_end=nfb * frame_seq_length,
                    )
                    s2v._move_kv_cache_to_working_gpu(gpu_id + 1, gpu_id + 1)

            num_blocks = target_shape[0] // nfb
            for block_index in range(num_blocks):
                if self._timesteps is None:
                    self.scheduler.set_timesteps(4, device=s2v.device)
                    self._timesteps = self.scheduler.timesteps
                    self._sigmas = self.scheduler.sigmas
                self.scheduler.timesteps = self._timesteps
                self.scheduler.sigmas = self._sigmas
                self.scheduler._step_index = self.dist.get_rank()
                self.scheduler._begin_index = 0

                block_latents = clip_latents[0][:, block_index * nfb : (block_index + 1) * nfb]
                arg = {
                    **common,
                    "cond_states": cond_latents[:, :, block_index * nfb : (block_index + 1) * nfb],
                    "audio_input": audio_input[..., block_index * nfb * 4 : (block_index + 1) * nfb * 4],
                }
                for i, t in enumerate(self._timesteps):
                    timestep = torch.tensor([t] * nfb).to(s2v.device).unsqueeze(0)
                    s2v._move_kv_cache_to_working_gpu(i + 1)
                    pred = s2v.noise_model(
                        [block_latents], t=timestep, **arg,
                        kv_cache=s2v.kv_cache1[str(i + 1)], crossattn_cache=s2v.crossattn_cache,
                        current_start=(block_index + r * num_blocks) * nfb * frame_seq_length,
                        current_end=(block_index + 1 + r * num_blocks) * nfb * frame_seq_length,
                        mask=None,
                    )
                    noise_pred = torch.cat(pred, dim=0)
                    s2v._move_kv_cache_to_working_gpu(i + 1, i + 1)
                    block_latents = self.scheduler.step(
                        noise_pred.unsqueeze(0), t, block_latents.unsqueeze(0), return_dict=False, generator=seed_g
                    )[0].squeeze(0)
                clip_output[:, block_index * nfb : (block_index + 1) * nfb] = block_latents

            # 2.3 online decode: motion latents roll forward by this clip
            s2v.noise_model.cpu() if os.getenv("QAV_LIVEAVATAR_OFFLOAD_DIT", "0") == "1" else None
            s2v.vae.model.to(s2v.device)
            if r == 0:
                self.ref_latents = clip_output.unsqueeze(0)[:, :, 0:1]
            decode_latents = torch.cat([self.motion_latents, clip_output.unsqueeze(0)], dim=2)
            image = torch.stack(s2v.vae.decode(decode_latents))
            image = image[:, :, -clip_frames:]
            if r == 0:
                image = image[:, :, 3:]
            overlap = min(s2v.motion_frames, image.shape[2])
            self.videos_last_frames = torch.cat([self.videos_last_frames[:, :, overlap:], image[:, :, -overlap:]], dim=2)
            self.videos_last_frames = self.videos_last_frames.to(dtype=self.motion_latents.dtype, device=self.motion_latents.device)
            self.motion_latents = torch.stack(s2v.vae.encode(self.videos_last_frames)).type_as(clip_latents[0])
            s2v.vae.model.cpu()
            s2v.noise_model.to(s2v.device)

        self.r += 1
        frames = ((image[0].float().clamp(-1, 1) + 1) * 127.5).permute(1, 2, 3, 0).to(torch.uint8).cpu().numpy()  # [F, H, W, 3] RGB
        return list(frames)

    def close(self) -> None:
        if self.s2v is not None:
            self.s2v.kv_cache1 = None
            self.s2v.shared_cond_cache = None
            self.s2v.crossattn_cache = None
            self.s2v = None
            try:
                self.torch.cuda.empty_cache()
            except Exception:  # noqa: BLE001
                pass
