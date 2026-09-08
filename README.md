# QAV — real-time AI avatars over WebRTC

QAV is a self-hosted re-implementation of the [Anam AI](https://anam.ai) stack:
a **persona** (face + voice + LLM + prompt) that holds a live, face-to-face
conversation in the browser. Same architecture, same API shapes, same SDK
surface — but every piece runs on infrastructure you control, and the face
renderer is a pluggable component instead of a proprietary black box.

```
Browser  ──WebRTC──▶  LiveKit SFU  ◀──WebRTC──  QAV engine (Python)
   │                      ▲                        ├─ STT  (Deepgram / OpenAI)
   │  @qav/js-sdk         │                        ├─ LLM  (Claude / OpenAI)
   │                      │                        ├─ TTS  (ElevenLabs / Cartesia / OpenAI)
   ▼                      │                        └─ QAV face  ─ audio ─▶ lip-synced video track
QAV API (Node) ───────────┘  creates room, dispatches engine, mints tokens
   ▲
   │  session tokens (never the API key)
Your backend
```

## How a session works (mirrors Anam 1:1)

| Step | Anam | QAV |
|---|---|---|
| 1. Your server mints a short-lived token | `POST /v1/auth/session-token` with API key | same path, same body (`personaConfig` / `personaId`) |
| 2. Browser starts the session | `createClient(token).streamToVideoElement(id)` → `POST /v1/engine/session` | identical — the SDK does it for you |
| 3. Engine joins | Anam cloud joins the room with a face worker | `qav-engine` (LiveKit Agents job) joins, brings the face with it |
| 4. Media | Avatar publishes video+audio, subscribes to your mic | identical; browser SDK attaches tracks to your `<video>` |
| 5. Transcripts / state | `MESSAGE_HISTORY_UPDATED`, talk commands | same events, same `talk()` / `sendUserMessage()` |

Anam's own LiveKit plugin (`livekit-plugins-anam`) uses exactly this shape —
a second participant that publishes *on behalf of* the agent and receives the
agent's TTS audio — so QAV's `AvatarSession` is a drop-in for it.

## Repository layout

```
apps/api/            QAV control plane (Hono, TypeScript)  — personas, tokens, engine sessions
apps/web/            Demo (Next.js)                         — persona picker, video, transcript
packages/js-sdk/     @qav/js-sdk (browser)                   — Anam-shaped client on livekit-client
services/engine/     qav-engine (Python, LiveKit Agents)     — STT→LLM→TTS pipeline + face renderer
infra/livekit/       livekit-server config for local dev
docker-compose.yml   livekit + api + engine
```

## Quickstart (local, ~5 minutes)

Prereqs: Node 20+, pnpm, Python 3.10+, [uv](https://docs.astral.sh/uv/), Docker (for LiveKit).

```bash
cp .env.example .env            # fill in vendor keys, or leave the mock providers (see below)
pnpm install
pnpm build                      # builds the SDK, API and web app

# 1. WebRTC server
docker compose up livekit       # or: livekit-server --dev

# 2. Control plane
pnpm dev:api                    # http://localhost:8787

# 3. Engine (conversation + face)
cd services/engine
uv venv && uv pip install -e ".[dev]"
python -m qav_engine.worker download-files   # VAD weights (skip when QAV_STT=mock)
python -m qav_engine.worker dev

# 4. Demo
pnpm dev:web                    # http://localhost:3000 → Start
```

### Running with zero vendor keys

Set `QAV_STT=mock QAV_LLM=mock QAV_TTS=mock` in `.env`. The persona then speaks
a synthesised tone envelope (so you can see lip-sync and hear audio), echoes
whatever you type, and skips speech recognition. This is what the end-to-end
test uses.

### Real pipeline

| Role | Env | Default | Alternatives |
|---|---|---|---|
| Ears | `QAV_STT` | `deepgram` (`DEEPGRAM_API_KEY`) | `openai` |
| Brain | `QAV_LLM` | `anthropic` → `claude-opus-5` (`ANTHROPIC_API_KEY`) | `openai` |
| Voice | `QAV_TTS` | `elevenlabs` (`ELEVEN_API_KEY`) | `cartesia`, `openai` |
| Face | `QAV_RENDERER` | `procedural` (CPU, built in) | your own — see below |

Per-persona choices (voice, LLM model, avatar) come from the catalog in the
API (`/v1/voices`, `/v1/llms`, `/v1/avatars`); the env picks which providers
the deployment can serve.

## Using the SDK

```ts
import { createClient, QavEvent } from "@qav/js-sdk";

const { sessionToken } = await fetch("/api/session-token", { method: "POST" }).then((r) => r.json());
const qav = createClient(sessionToken);

qav.addListener(QavEvent.MESSAGE_HISTORY_UPDATED, (messages) => render(messages));
qav.addListener(QavEvent.PERSONA_STATE_CHANGED, (state) => console.log(state)); // listening | thinking | speaking

await qav.streamToVideoElement("persona-video");   // mic on, avatar in your <video>
await qav.talk("Welcome back!");                    // persona says this verbatim
await qav.sendUserMessage("What's on my calendar?"); // persona answers as if spoken
await qav.stopStreaming();
```

## API

All admin routes take `Authorization: Bearer $QAV_API_KEY`.

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/v1/auth/session-token` | `{ personaConfig }` or `{ personaId }` (+ `sessionOptions`, `expiresIn`, `environment`) → `{ sessionToken }` |
| `GET/POST/PUT/DELETE` | `/v1/personas[/:id]` | Saved persona configs |
| `GET/POST` | `/v1/avatars`, `/v1/voices`, `/v1/llms` | Catalogs |
| `GET` | `/v1/sessions[/:id]`, `/v1/sessions/:id/transcript` | Session records + transcripts |
| `POST` | `/v1/sessions/:id/stop` | End a session |

Session-token routes (what the SDK calls): `POST /v1/engine/session`,
`POST /v1/engine/session/stop`.

`environment: { livekitUrl, livekitToken }` in the token request is the
"bring your own LiveKit" mode: your own voice agent runs the conversation and
QAV only supplies the face — the same shape Anam's LiveKit plugin sends.

## Building your own face

The only truly proprietary part of Anam is the neural face model. QAV isolates
it behind one interface:

```python
# services/engine/qav_engine/avatar/renderers/base.py
class FaceRenderer(ABC):
    def warmup(self) -> None: ...
    def render(self, state: FaceState) -> np.ndarray:   # (H, W, 4) uint8 RGBA, once per video frame
        ...
```

`FaceState` carries per-frame animation parameters derived from the TTS audio
(`mouth_open`, `mouth_width`, `teeth`, `blink`, `head_yaw/pitch`, `gaze_*`,
`energy`) by `LipSyncAnalyzer`. The built-in `ProceduralFaceRenderer` draws a
stylised character from an `AvatarStyle` palette at ~8 ms/frame on one CPU
core. To plug in a neural renderer (MuseTalk, Wav2Lip, a diffusion talking
head, ...), implement `FaceRenderer`, call `register_renderer("my-face", MyRenderer)`,
and set `QAV_RENDERER=my-face`. Renderers that want raw audio instead of
parameters can subclass `FaceVideoGenerator`.

## Tests

```bash
pnpm --filter @qav/api test                      # token / auth / catalog tests
cd services/engine && python -m pytest           # lip-sync + renderer tests
QAV_PREVIEW_DIR=/tmp/qav python -m pytest -k preview   # writes a contact sheet of faces
```

## Production notes

- Put a real `QAV_API_KEY` / `QAV_SESSION_SECRET` in place and terminate TLS in front of the API and LiveKit (`wss://`).
- LiveKit needs UDP (or TURN) reachable from browsers; see `infra/livekit/livekit.yaml` and LiveKit's deployment docs.
- The API's store is in-memory with JSON persistence (`QAV_DATA_FILE`); swap `apps/api/src/store.ts` for Postgres when you outgrow it.
- Scale the engine horizontally: every worker registers as `qav-engine` and LiveKit load-balances dispatches across them.
