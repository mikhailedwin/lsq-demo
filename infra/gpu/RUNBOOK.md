# Running the photoreal face on a rented GPU

Step by step, from nothing to a photoreal avatar in the demo. Budget about an
hour the first time, most of it waiting on downloads.

---

## 0. Which provider

**Use [RunPod](https://runpod.io) Secure Cloud.** It is Docker-native (our image
drops straight in), bills per second, and its Secure Cloud tier carries a 99%
uptime SLA. For a client demo that last part is the whole argument.

| Provider | RTX 4090 | SLA | Verdict |
|---|---|---|---|
| **RunPod Secure Cloud** | ~$0.69/hr | 99% | **Use this for a demo.** Reliable, Docker-first |
| RunPod Community Cloud | ~$0.34/hr | none | Fine for your own testing, not for a live demo |
| [Vast.ai](https://vast.ai) | ~$0.29–0.39/hr | none | Cheapest, but spot instances can be reclaimed on 15 seconds' notice — do not present from one |
| [Lambda](https://lambdalabs.com) | — (A100 $2.06/hr) | 99.9% | Most reliable, priciest, US-only. Overkill for MuseTalk |

A **24 GB RTX 4090 or a 24 GB L4/A10** is right for `musetalk`. You do not need
an A100 or H100 unless you are running `liveavatar`, which wants 80 GB (48 GB
with FP8) and is still experimental.

Rough demo cost: **under $1/hr**, and RunPod bills per second, so stop the pod
between rehearsals.

---

## 1. Where the rest of the stack lives

The GPU pod only needs **outbound** network access — the face worker dials your
LiveKit server, nothing dials in. So you never expose a port on the pod.

The cheapest reliable topology for a demo:

| Piece | Where | Why |
|---|---|---|
| LiveKit | [LiveKit Cloud](https://cloud.livekit.io) free tier | Gives you a `wss://` URL and TURN with no infra. Self-hosting means TLS + UDP + a public IP |
| `qav-api` + `apps/web` | Your laptop, or any small host | Only your browser needs to reach these |
| `qav-face` | The GPU pod | Dials out to LiveKit |
| `qav-engine` | Your laptop or the pod | Also only dials out |

Take the LiveKit URL, API key and secret from LiveKit Cloud and put them in your
`.env`. Every service reads the same three values.

---

## 2. Create a network volume first

Do this **before** the pod. MuseTalk's weights (~1.5 GB), GFPGAN (~350 MB) and
your prepared avatars all live here, so they survive stopping and restarting the
pod — otherwise you re-download every time.

RunPod → **Storage → Network Volume** → 50 GB, in the region you'll deploy in.

---

## 3. Deploy the pod

RunPod → **Pods → Deploy** →

- **GPU**: RTX 4090 (24 GB), Secure Cloud
- **Template**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
- **Network volume**: the one from step 2, mounted at `/workspace`
- **Environment variables**:

```
LIVEKIT_URL=wss://<your-project>.livekit.cloud
LIVEKIT_API_KEY=<key>
LIVEKIT_API_SECRET=<secret>
QAV_RENDERER=musetalk
QAV_AVATAR_DIR=/workspace/avatars
MUSETALK_ROOT=/workspace/musetalk
MUSETALK_MODELS=/workspace/musetalk/models
QAV_GFPGAN_WEIGHTS=/workspace/models/gfpgan/GFPGANv1.4.pth
```

Deploy, then open the pod's web terminal.

---

## 4. One-time setup on the pod

```bash
cd /workspace
git clone https://github.com/mikhailedwin/lsq-demo.git
cd lsq-demo

# MuseTalk itself (pinned) + its weights, onto the network volume
git clone https://github.com/TMElyralab/MuseTalk.git /workspace/musetalk
cd /workspace/musetalk && git checkout 0a89dec45a0192b824e3cf4daf96c239440c5ed8

pip install -U pip uv
uv pip install --system torch==2.3.1 torchvision==0.18.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install --system -r /workspace/musetalk/requirements.txt
uv pip install --system openmim && mim install "mmengine" "mmcv==2.0.1" "mmdet==3.1.0" "mmpose==1.1.0"

cd /workspace/musetalk && chmod +x download_weights.sh && ./download_weights.sh   # ~1.5 GB

# face restoration weights
mkdir -p /workspace/models/gfpgan
curl -fsSL https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth \
  -o /workspace/models/gfpgan/GFPGANv1.4.pth

# QAV itself
cd /workspace/lsq-demo && uv pip install --system "services/face[musetalk]"
```

> Prefer a prebuilt image? `docker build -f services/face/Dockerfile.musetalk -t
> <you>/qav-face:musetalk .` locally, push it to a registry, and point the pod at
> it instead — the image already does everything above. Faster to start, slower
> the first time.

---

## 5. Check your footage *before* preparing it

```bash
qav-face check --video /workspace/alice.mp4
```

This is CPU-only — you can run it on your laptop before renting anything. It
scores resolution, face size, duration, sharpness, exposure drift and head
rotation, and tells you exactly what to fix. Aim for **10–20 s, 25 fps, face
≥ 320 px across**, front-on, evenly lit, locked exposure, still head, plain
background. A soft or dim source cannot be recovered later.

No footage of your own yet? Use the bundled demo avatars:

```bash
qav-face demo-avatars --only musetalk
```

---

## 6. Prepare, tune, verify

```bash
qav-face prepare --renderer musetalk --avatar alice --video /workspace/alice.mp4

# compare mouth openness across bbox_shift values, pick the best row
qav-face tune --avatar alice --video /workspace/alice.mp4
qav-face prepare --renderer musetalk --avatar alice --video /workspace/alice.mp4 --bbox-shift <value>

# render a real sample through the live frame loop
qav-face selftest --renderer musetalk --avatar alice --out /workspace/out
```

`selftest` writes `out/selftest.mp4` — video with the audio it lip-synced —
plus the numbers that matter:

```
  ratio             1.12   (mouth vs face sharpness; < 0.7 means enable restoration)
  temporal jitter  0.034   (still head ~0.03; > 0.25 means raise QAV_FACE_SMOOTH)
  speed              2.3 ms/frame vs 40 ms budget at 25 fps  OK
```

**Watch the MP4 before you go further.** If `speed` exceeds the budget, drop
`QAV_VIDEO_WIDTH/HEIGHT` or `QAV_MUSETALK_BATCH`.

---

## 7. Run the worker

```bash
cd /workspace/lsq-demo && python3 -m qav_face.worker start
```

It registers with LiveKit and waits. Leave it running.

Locally, point your `.env` at the same LiveKit Cloud project and start the API,
engine and web app as in the main README. Pick a **photoreal** avatar in the
demo — the engine dispatches this worker automatically for any avatar whose
renderer is `musetalk` or `liveavatar`. No restart, no config change.

---

## 8. Costs and housekeeping

- **Stop the pod** when you're not rehearsing. RunPod bills per second for
  compute; the network volume is charged separately and is a few dollars a month.
- One session per GPU. Two concurrent demo callers need two pods.
- Weights and prepared avatars persist on `/workspace`, so a restart is quick.

---

## Troubleshooting

| Symptom | Cause |
|---|---|
| `MUSETALK_ROOT must point at a MuseTalk checkout` | Env var missing or the clone didn't land on the volume |
| Worker starts but never gets a job | `LIVEKIT_URL/KEY/SECRET` differ from the engine's. All services must share one project |
| `avatar ... is not prepared` | Run `qav-face prepare` for that avatar id, or check `QAV_AVATAR_DIR` |
| Mouth looks soft | `selftest` ratio < 0.7 → `QAV_FACE_RESTORE=1`. If it's still soft, the source footage is the limit |
| Visible flicker | Raise `QAV_FACE_SMOOTH` (0–1, default 0.6) |
| Audio stalls / video stutters | `selftest` speed over budget: lower the resolution or batch size |
