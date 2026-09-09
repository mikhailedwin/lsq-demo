# qav-face — the face renderer

Everything that turns a persona's speech audio into a talking face. Three
backends behind one interface; the engine picks per avatar.

| Backend | Look | Source material | Hardware | Upstream (license) |
|---|---|---|---|---|
| `procedural` | Stylised 2D placeholder | a palette | CPU, ~8 ms/frame at 512² | built in |
| `musetalk` | **Photoreal** — a real person's clip with a synthesized mouth | 8–30 s video of a person | NVIDIA ≥ 12 GB, 25 fps+ | [TMElyralab/MuseTalk](https://github.com/TMElyralab/MuseTalk) 1.5 (MIT) |
| `liveavatar` | **Generative** — whole face from one photo | one portrait + scene prompt | NVIDIA 80 GB (48 GB w/ FP8) | [Alibaba-Quark/LiveAvatar](https://github.com/Alibaba-Quark/LiveAvatar) (Apache-2.0) |

## How it fits together

```
engine (TTS audio) ──data stream──▶ qav-face worker ──▶ FaceBackend ──▶ AvatarRunner ──▶ WebRTC video+audio
                                    (joins the room as `qav-avatar`, publishing on behalf of the engine)
```

`BackendVideoGenerator` owns everything a backend shouldn't care about:
buffering audio into whole video frames, batching, lookahead, resampling to the
model's rate, idle frames while silent, interruptions, and AV sync. A backend
implements three methods:

```python
class FaceBackend:
    batch_frames = 1        # frames per inference call
    lookahead_frames = 0    # future frames the model needs for the last frame of a batch
    model_sample_rate = 16_000
    run_in_thread = False   # True for GPU work, so inference never blocks the event loop

    def idle_frame(self) -> np.ndarray: ...                              # (H, W, 4) RGBA
    def speech_frames(self, audio: np.ndarray, n: int) -> list[np.ndarray]: ...
```

Register a new one with `register_backend("my-face", "my_pkg:MyBackend")` and
set `QAV_RENDERER=my-face` (or `renderer` on the avatar's catalog entry).

## Getting the face right

Output quality is decided long before inference runs. Work in this order — each
step is a command, not a guess.

### 1. Gate the footage (CPU, before you rent a GPU)

```bash
qav-face check --video alice.mp4
```

Scores resolution, face size, duration, sharpness, exposure drift, motion and
head rotation, and exits non-zero with a specific fix for anything that fails:

```
  [FAIL] the face is only ~112 px across; the model works at 256 px
         → shoot or crop closer — head and shoulders, face filling ~1/3 of the frame height
  [warn] the clip is dark
         → brighter, even, front-facing light — dark footage exaggerates model artifacts
score: 0/100 — NOT usable as-is
```

What good footage looks like: **10–20 s, 25 fps, face ≥ 320 px across**, front-on,
evenly lit, locked exposure and white balance, still head, plain background,
mouth closed at start and end. Soft or dim input cannot be recovered downstream.

### 2. Prepare

```bash
qav-face prepare --renderer musetalk --avatar alice --video alice.mp4
```

Extracts face crops, VAE latents and blend masks, and picks a **seamless idle
loop** — the frame pair whose appearance *and* motion match best, so the loop
plays forward forever. (The reference implementation appends the clip reversed,
which plays the person's head motion backwards; obvious on a talking head.)

### 3. Tune the mouth

```bash
qav-face tune --avatar alice --video alice.mp4          # renders -7,-3,0,3,7
```

Writes one contact-sheet row per `bbox_shift` value. Positive values open the
mouth more, negative less; pick the row whose mouth closes fully on consonants
and opens naturally on vowels, then re-run `prepare --bbox-shift <value>`.

### 4. Verify

```bash
qav-face selftest --renderer musetalk --avatar alice --out ./out
```

Drives the real frame loop — same code path as a live session — and writes
`out/selftest.mp4` (video **with** the audio it lip-synced), a contact sheet,
PNG frames, and the numbers that matter:

```
  mouth sharpness       1127
  face sharpness        1010
  ratio                 1.12   (1.0 = mouth as sharp as the rest of the face)
  temporal jitter      0.034   (still head ~0.03, visible flicker > 0.25)
  verdict           good
  speed                  2.3 ms/frame vs 40 ms budget at 25 fps  OK
```

Watch the MP4 first; the numbers tell you *what* to change when it looks wrong.

## What runs automatically

MuseTalk's own README lists two known weaknesses. Both are handled in-pipeline:

| Weakness | Fix | Control |
|---|---|---|
| Mouth detail "not always well preserved" (256² generation) | **GFPGAN restoration** on the generated crop only — never on the real footage around it, so identity doesn't shift between talking and idle. It measures itself at warmup and stays off if it can't fit the frame budget, because a sharp avatar that stutters is worse than a slightly soft one that doesn't. | `QAV_FACE_RESTORE=auto\|1\|0`, `QAV_FACE_RESTORE_BLEND` |
| Jitter from single-frame generation | **Adaptive temporal filter.** Amplitude can't separate flicker from articulation — flicker is often larger. Frequency can: per-frame noise alternates every frame (12.5 Hz at 25 fps) while speech articulates at 4–8 Hz. A 3-tap [1,2,1] kernel nulls the former and passes the latter, gated per pixel so genuine motion is untouched. | `QAV_FACE_SMOOTH` (0–1, default 0.6) |

Upscaling also uses Lanczos rather than area interpolation — resizing a face
crop up with `INTER_AREA` is exactly how photoreal turns soft.

**Use footage you have the right to use.** The seeded demo avatars point at
sample assets from the upstream repos; anything else needs the subject's
consent. Don't build a likeness of a real person without it.

## Run it

See `infra/gpu/setup.sh` (one-shot provisioning for an Ubuntu GPU box) and
`docker-compose.gpu.yml`. Environment reference is in `.env.example`.
