# Giving a client a working link

Four services. Only the first is public.

| Piece | Where | Cost |
|---|---|---|
| Web app (`apps/web`) | **Vercel** | free |
| Control plane (`apps/api`) + engine | **Railway** | ~$5/mo |
| LiveKit (signalling, TURN, UDP) | **LiveKit Cloud** free tier | free |
| Face worker (`qav-face`) | **RunPod**, only while demoing | $0.69/hr |

Standing cost is about **$5/month**; the GPU is roughly **$1 per demo** if you
start the pod beforehand and stop it after. See
[`gpu/RUNBOOK.md`](gpu/RUNBOOK.md) for the pod itself.

---

## 1. LiveKit Cloud

[cloud.livekit.io](https://cloud.livekit.io) → new project. Copy the **WebSocket
URL**, **API key** and **API secret**; every other service needs all three.

The free tier gives 5,000 WebRTC participant-minutes and 5 concurrent sessions,
which is ample for demos. No card required.

---

## 2. Railway — control plane + engine

[railway.app](https://railway.app) → New Project → Deploy from GitHub → this repo.

Create **two services** from the same repo.

**Service A — `qav-api`**

| Setting | Value |
|---|---|
| Root directory | `/` |
| Build | `pnpm install --frozen-lockfile && pnpm --filter @qav/api build` |
| Start | `node apps/api/dist/index.js` |

Variables:

```
LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET   # from step 1
QAV_API_KEY=<long random string>                   # server-to-server only
QAV_SESSION_SECRET=<long random string>
QAV_DATA_FILE=/data/qav.json                       # attach a volume to persist
QAV_STT=deepgram  QAV_LLM=anthropic  QAV_TTS=elevenlabs
ANTHROPIC_API_KEY, DEEPGRAM_API_KEY, ELEVEN_API_KEY
```

Generate the secrets with `openssl rand -hex 32`. Note the public URL Railway
gives this service — the web app needs it.

**Service B — `qav-engine`**

| Setting | Value |
|---|---|
| Build | `pip install uv && uv pip install --system ./services/face ./services/engine` |
| Start | `python -m qav_engine.worker start` |

Same `LIVEKIT_*` and provider keys. Add `QAV_FACE_MODE=remote` so it always
dispatches the face worker rather than rendering in-process.

---

## 3. Vercel — the web app

[vercel.com](https://vercel.com) → Add New → Project → this repo.

- **Root Directory**: **`apps/web`** — this is not optional. Vercel's Next.js
  builder looks for `next` in the package.json *at the Root Directory*, and the
  repo root has no dependencies at all, so pointing it at `./` fails with
  "No Next.js version detected". `apps/web/vercel.json` is read from there too.
- **Framework**: Next.js (detected).

`buildCommand` builds `@qav/js-sdk` before the web app, because the web app
imports it from `dist/` and a fresh clone has no `dist/`. `pnpm --filter` walks
up to find the workspace root on its own, so nothing needs to `cd`.

Environment variables:

```
QAV_API_URL=https://<your-railway-api>.up.railway.app
QAV_API_KEY=<same value as Railway's QAV_API_KEY>
QAV_ACCESS_CODE=<your 4-character code>
QAV_ACCESS_SECRET=<openssl rand -hex 32>
```

`QAV_API_KEY` and `QAV_ACCESS_CODE` are read server-side only — they are never
sent to the browser. Do **not** prefix anything secret with `NEXT_PUBLIC_`,
which would inline it into the client bundle.

Deploy. That URL is what the client gets.

---

## 4. The access gate

With `QAV_ACCESS_CODE` set, visitors see a code screen before anything else —
the app's markup isn't even sent until they're through. A correct code sets a
signed, httpOnly cookie good for 8 hours.

Leave `QAV_ACCESS_CODE` empty to disable the gate (that's the local default).

**What the gate is and isn't.** It stops a stray visitor spending your GPU, LLM
and TTS budget. It is not authentication. Four alphanumeric characters is ~1.7M
combinations, so the rate limiting does the real work: 5 attempts per IP, then
a 15-minute lockout that a correct code will not bypass.

Two caveats worth knowing:

- The attempt counter is **in-memory**, so it is per-instance. On Railway or a
  VPS that's one process and it works. On Vercel, traffic fans out across
  instances and an attacker gets more attempts than the numbers suggest. For a
  short-lived demo that's an acceptable trade; if the link will live for weeks,
  either use a longer code or move the counter to Vercel KV / Upstash.
- Anyone with the code can share it. Rotate it between clients by changing
  `QAV_ACCESS_CODE` and redeploying — existing cookies stay valid until they
  expire.

---

## 5. Before the demo

1. Start the RunPod pod (see [`gpu/RUNBOOK.md`](gpu/RUNBOOK.md)) and wait for
   `registered worker` in its logs.
2. Run `qav-face selftest` once on the pod and **watch the MP4**. Don't present
   a face you haven't seen.
3. Open the Vercel URL yourself, enter the code, run a full call.
4. Stop the pod afterwards — it bills per second.

If the pod isn't running, a photoreal avatar fails with a clear message about
no face worker. The CPU placeholder avatars still work without it.
