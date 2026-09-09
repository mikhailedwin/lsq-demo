# QAV — real-time AI avatars over WebRTC

QAV is a self-hosted stack for real-time conversational avatars: a **persona**
(face + voice + LLM + prompt) that holds a live, face-to-face conversation in
the browser. Every piece runs on infrastructure you control, and the face is a
pluggable open model rather than a hosted black box.

```
Browser  ──WebRTC──▶  LiveKit SFU  ◀──WebRTC──  QAV engine (Python)
   │                    ▲     ▲                  ├─ STT  (Deepgram / OpenAI)
   │  @qav/js-sdk       │     │                  ├─ LLM  (Claude / OpenAI)
   │                    │     │  TTS audio       └─ TTS  (ElevenLabs / Cartesia / OpenAI)
   ▼                    │     │  (data stream)          │
QAV API (Node) ─────────┘     └──  qav-face worker ◀────┘
   ▲   rooms, dispatch,           MuseTalk / Live Avatar / procedural
   │   session tokens             → lip-synced video published on behalf of the engine
Your backend
```

## The face

The hard part of a stack like this is the neural face model. QAV puts three
open ones behind a single interface, chosen per avatar:

| Renderer | Look | Needs | Upstream (license) |
|---|---|---|---|
| **`musetalk`** | **Photoreal.** It *is* a real person's footage; the model synthesizes the mouth. | 8–30 s clip of a person; NVIDIA ≥ 12 GB (25 fps+) | [MuseTalk 1.5](https://github.com/TMElyralab/MuseTalk) (MIT) |
| **`liveavatar`** | **Generative.** Whole face and head motion from one photo — today's top open-source quality. | one portrait; NVIDIA **80 GB** (48 GB with FP8) | [Live Avatar](https://github.com/Alibaba-Quark/LiveAvatar), ECCV 2026 (Apache-2.0) |
| `procedural` | Stylised placeholder | nothing — CPU, ~8 ms/frame | built in |

`procedural` exists so the whole stack runs and tests on a laptop with no GPU
and no API keys. It is a placeholder, not a product face — pick a photoreal
avatar in the demo once you have a GPU. Details, the backend interface and the
`qav-face selftest` workflow: [`services/face/README.md`](services/face/README.md).

### Choosing a GPU

| Renderer | Instance (indicative) | ~$/hr | Notes |
|---|---|---|---|
| `musetalk` | 1× L4 / A10G (24 GB) — AWS `g6.xlarge`, GCP `g2-standard-8` | ~$0.8–1.1 | comfortably real time; the sweet spot for a demo |
| `musetalk` | 1× RTX 4090 (community clouds) | ~$0.4–0.7 | fastest per dollar |
| `liveavatar` | 1× H100 80 GB — AWS `p5.48xlarge` slice, or a single-GPU cloud | ~$3–4 | single-GPU mode; measure before trusting it in real time |
| `liveavatar` | 1× A100/L40S 48 GB with `QAV_LIVEAVATAR_FP8=1` | ~$1.5–2.5 | slight quality loss |

One session per GPU. Provision with:

```bash
./infra/gpu/setup.sh musetalk     # driver check → docker + nvidia toolkit → build → run command
```

The face quality workflow is four commands, in order — **check** your footage
before you rent anything, then prepare, tune and verify:

```bash
qav-face check    --video alice.mp4                       # CPU: is this footage usable at all?
qav-face prepare  --renderer musetalk --avatar alice --video alice.mp4
qav-face tune     --avatar alice --video alice.mp4        # compare bbox_shift values side by side
qav-face selftest --renderer musetalk --avatar alice --out ./out
```

`selftest` drives the real frame loop and writes an MP4 with the audio it
lip-synced, plus mouth-sharpness, jitter and ms/frame numbers. GFPGAN mouth
restoration and an adaptive de-flicker filter run automatically — see
[`services/face/README.md`](services/face/README.md) for what they do and how
to tune them.

## How a session works

| Step | What happens |
|---|---|
| 1. Your server mints a token | `POST /v1/auth/session-token` with your API key and a `personaConfig` (or a saved `personaId`) → a short-lived session token |
| 2. Browser starts the session | `createClient(token).streamToVideoElement(id)`; the SDK calls `POST /v1/engine/session` for you |
| 3. Engine joins | `qav-engine` joins the room, then dispatches `qav-face` for GPU avatars or renders the CPU face itself |
| 4. Media | The face publishes video and audio *on behalf of* the engine, so clients see one participant, and subscribes to your mic |
| 5. Transcripts / state | `MESSAGE_HISTORY_UPDATED` and `PERSONA_STATE_CHANGED` events; `talk()` and `sendUserMessage()` to drive it |

The face running as a second participant that publishes on behalf of the agent
— fed the agent's TTS audio over a LiveKit data stream — is the standard
LiveKit Agents avatar pattern, so `AvatarSession` slots in wherever that shape
is expected.

## Repository layout

```
apps/api/            QAV control plane (Hono, TypeScript)   — personas, tokens, sessions, dispatch
apps/web/            Demo (Next.js)                          — persona picker, video, transcript
packages/js-sdk/     @qav/js-sdk (browser)                   — browser client on livekit-client
services/engine/     qav-engine (Python, LiveKit Agents)     — STT→LLM→TTS pipeline, RPC, avatar session
services/face/       qav-face (Python)                       — face backends + the GPU worker
infra/livekit/       livekit-server config for local dev
infra/gpu/setup.sh   one-shot GPU box provisioning
docker-compose.yml   livekit + api + engine   (+ docker-compose.gpu.yml for the face worker)
```

## Quickstart (local, no GPU, no API keys)

Prereqs: Node 20+, pnpm, Python 3.10+, [uv](https://docs.astral.sh/uv/), Docker.

```bash
cp .env.example .env            # set QAV_STT/QAV_LLM/QAV_TTS to `mock` to skip vendor keys
pnpm install && pnpm build

docker compose up livekit       # 1. WebRTC server
pnpm dev:api                    # 2. control plane  → http://localhost:8787

cd services/face  && uv venv && uv pip install -e ".[dev]"   # 3. face package
cd ../engine      && uv venv && uv pip install -e . -e ../face
python -m qav_engine.worker download-files                   #    VAD weights (skip if QAV_STT=mock)
python -m qav_engine.worker dev                              # 4. engine

pnpm dev:web                    # 5. demo → http://localhost:3000 → Start
```

Pick a "placeholder (CPU)" avatar for this mode; photoreal avatars need the GPU
worker below.

## Adding the photoreal face

```bash
# on the GPU box
./infra/gpu/setup.sh musetalk
docker run -d --gpus all --network host \
  -e LIVEKIT_URL=wss://your-livekit -e LIVEKIT_API_KEY=... -e LIVEKIT_API_SECRET=... \
  -v qav-avatars:/avatars qav-face:musetalk
```

The engine dispatches this worker automatically whenever a session picks an
avatar whose `renderer` is `musetalk` or `liveavatar` — no engine restart, no
config change. `docker compose -f docker-compose.yml -f docker-compose.gpu.yml
--profile musetalk up` does the same locally.

## Using the SDK

```ts
import { createClient, QavEvent } from "@qav/js-sdk";

const { sessionToken } = await fetch("/api/session-token", { method: "POST" }).then((r) => r.json());
const qav = createClient(sessionToken);

qav.addListener(QavEvent.MESSAGE_HISTORY_UPDATED, (messages) => render(messages));
qav.addListener(QavEvent.PERSONA_STATE_CHANGED, (s) => console.log(s)); // listening | thinking | speaking

await qav.streamToVideoElement("persona-video");    // mic on, avatar in your <video>
await qav.talk("Welcome back!");                     // persona says this verbatim
await qav.sendUserMessage("What's on my calendar?"); // persona answers as if spoken
await qav.stopStreaming();
```

## API

All admin routes take `Authorization: Bearer $QAV_API_KEY`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/auth/session-token` | `{ personaConfig }` or `{ personaId }` (+ `sessionOptions`, `expiresIn`, `environment`) → `{ sessionToken }` |
| `GET/POST/PUT/DELETE` | `/v1/personas[/:id]` | Saved persona configs |
| `GET/POST` | `/v1/avatars`, `/v1/voices`, `/v1/llms` | Catalogs (an avatar carries its `renderer` + `assets`) |
| `GET` | `/v1/sessions[/:id]`, `/v1/sessions/:id/transcript` | Session records + transcripts |
| `POST` | `/v1/sessions/:id/stop` | End a session |

Session-token routes (what the SDK calls): `POST /v1/engine/session`,
`POST /v1/engine/session/stop`.

`environment: { livekitUrl, livekitToken }` in the token request is the
"bring your own LiveKit" mode: your own voice agent runs the conversation and
QAV only supplies the face, joining the room you nominate.

## Tests

```bash
pnpm --filter @qav/api test                  # tokens, auth tiers, catalog
cd services/face   && python -m pytest       # frame loop, backends, lip-sync, renderer perf
cd services/engine && python -m pytest       # persona parsing, local vs remote face
QAV_PREVIEW_DIR=/tmp/qav python -m pytest -k preview   # contact sheet of the CPU faces
```

The GPU backends are unit-tested for contract and error handling without a GPU;
their visual output is verified with `qav-face selftest` on the GPU box.

## Production notes

- Real `QAV_API_KEY` / `QAV_SESSION_SECRET`, TLS in front of the API and LiveKit (`wss://`).
- LiveKit needs UDP (or TURN) reachable from browsers — see `infra/livekit/livekit.yaml`.
- The API store is in-memory with JSON persistence (`QAV_DATA_FILE`); swap `apps/api/src/store.ts` for Postgres when you outgrow it.
- Scale horizontally: every engine registers as `qav-engine` and every face worker as `qav-face`; LiveKit load-balances dispatches. Budget one GPU per concurrent face session.
- Only build a likeness of a real person with their consent; the seeded demo avatars use sample assets from the upstream model repos.
