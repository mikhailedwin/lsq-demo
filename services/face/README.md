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

## Verify a renderer without a session

```bash
qav-face selftest --renderer musetalk --avatar yongen --out ./out
# → out/00000.png…, out/contact-sheet.png, out/selftest.mp4 (video + the audio it lip-synced)
```

This drives the real frame loop with synthetic speech (or `--wav yours.wav`),
so it exercises exactly what a live session does — and prints ms/frame, which
is what decides whether the backend keeps up with real time.

## Prepare avatars

```bash
# MuseTalk: a clip of a person → cropped faces, VAE latents and blend masks
qav-face prepare --renderer musetalk --avatar alice --video alice.mp4

# Live Avatar: a portrait + the scene you want
qav-face prepare --renderer liveavatar --avatar alice --video alice.jpg \
  --prompt "A friendly presenter in a bright studio, talking to camera."

# or just the demo avatars bundled with the upstream repos
qav-face demo-avatars
```

Source-material guidance for MuseTalk: 8–30 s, 25 fps, one face, front-facing,
even lighting, mouth closed at the start and end, minimal head rotation. The
clip loops (forward then reversed) while the persona is idle, so pick footage
whose first and last frames match.

**Use footage you have the right to use.** The seeded demo avatars point at
sample assets from the upstream repos; anything else needs the subject's
consent. Don't build a likeness of a real person without it.

## Run it

See `infra/gpu/setup.sh` (one-shot provisioning for an Ubuntu GPU box) and
`docker-compose.gpu.yml`. Environment reference is in `.env.example`.
