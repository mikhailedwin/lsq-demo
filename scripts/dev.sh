#!/usr/bin/env bash
# Start the whole stack locally and open the demo.
#
#   ./scripts/dev.sh
#
# Brings up LiveKit, the control-plane API, the engine, the face worker and the
# web app, waits until each is actually answering, then prints the URL. Ctrl-C
# stops everything.
#
# No GPU needed: the default .env runs the offline mocks and the CPU face.
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/.."
ROOT=$(pwd)
LOGS="$ROOT/.dev-logs"
mkdir -p "$LOGS"

say() { printf "\033[1;34m▸\033[0m %s\n" "$*"; }
warn() { printf "\033[1;33m!\033[0m %s\n" "$*"; }
die() { printf "\033[1;31m✗\033[0m %s\n" "$*" >&2; exit 1; }

# ---------------------------------------------------------------- prerequisites
for cmd in node pnpm python3; do
  command -v "$cmd" >/dev/null || die "$cmd is not installed"
done
command -v uv >/dev/null || die "uv is not installed — https://docs.astral.sh/uv/"

if [ ! -f .env ]; then
  say "creating .env from .env.example (offline mocks, no vendor keys needed)"
  cp .env.example .env
  # make the out-of-the-box run work with no API keys at all
  sed -i.bak -E 's/^QAV_STT=.*/QAV_STT=mock/; s/^QAV_LLM=.*/QAV_LLM=mock/; s/^QAV_TTS=.*/QAV_TTS=mock/' .env
  rm -f .env.bak
fi
set -a; . ./.env; set +a
: "${LIVEKIT_URL:=ws://localhost:7880}"
: "${QAV_API_KEY:=qav_dev_key_change_me}"

PIDS=()
cleanup() {
  echo
  say "stopping…"
  for pid in "${PIDS[@]:-}"; do kill "$pid" 2>/dev/null || true; done
  [ -n "${LK_DOCKER:-}" ] && docker compose stop livekit >/dev/null 2>&1 || true
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

wait_for() { # wait_for <name> <url> [header]
  local name=$1 url=$2 hdr=${3:-} i
  for i in $(seq 1 60); do
    if [ -n "$hdr" ]; then
      curl -fsS -o /dev/null -H "$hdr" "$url" 2>/dev/null && return 0
    else
      curl -fsS -o /dev/null "$url" 2>/dev/null && return 0
    fi
    sleep 1
  done
  die "$name did not come up — see $LOGS/${name}.log"
}

# ----------------------------------------------------------------- dependencies
if [ ! -d node_modules ]; then
  say "installing node dependencies"
  pnpm install
fi
say "building workspace packages"
pnpm build >"$LOGS/build.log" 2>&1 || die "build failed — see $LOGS/build.log"

for svc in face engine; do
  if [ ! -d "services/$svc/.venv" ]; then
    say "creating the $svc virtualenv"
    (cd "services/$svc" && uv venv >/dev/null)
  fi
done
say "installing python dependencies"
(cd services/face && VIRTUAL_ENV="$ROOT/services/face/.venv" uv pip install -q -e ".[dev]")
(cd services/engine && VIRTUAL_ENV="$ROOT/services/engine/.venv" uv pip install -q -e . -e ../face)

# --------------------------------------------------------------------- livekit
if curl -fsS -o /dev/null http://localhost:7880 2>/dev/null; then
  say "LiveKit already running on :7880"
elif command -v livekit-server >/dev/null; then
  say "starting LiveKit (native)"
  livekit-server --config infra/livekit/livekit.yaml --node-ip 127.0.0.1 >"$LOGS/livekit.log" 2>&1 &
  PIDS+=($!)
elif command -v docker >/dev/null && docker info >/dev/null 2>&1; then
  say "starting LiveKit (docker)"
  LK_DOCKER=1
  docker compose up -d livekit >"$LOGS/livekit.log" 2>&1
else
  die "LiveKit needs Docker or the livekit-server binary:
    macOS:  brew install livekit
    other:  curl -sSL https://get.livekit.io | bash
    or start Docker Desktop and re-run"
fi
wait_for livekit http://localhost:7880

# ------------------------------------------------------------------------- api
say "starting the control-plane API"
(cd apps/api && node dist/index.js) >"$LOGS/api.log" 2>&1 &
PIDS+=($!)
wait_for api "http://localhost:${QAV_API_PORT:-8787}/v1/avatars" "authorization: Bearer $QAV_API_KEY"

# --------------------------------------------------------------------- workers
if [ "${QAV_STT:-mock}" != "mock" ]; then
  say "fetching VAD weights (first run only)"
  (cd services/engine && .venv/bin/python -m qav_engine.worker download-files) >>"$LOGS/engine.log" 2>&1 || true
fi

say "starting the engine"
(cd services/engine && .venv/bin/python -m qav_engine.worker start) >"$LOGS/engine.log" 2>&1 &
PIDS+=($!)

say "starting the face worker (${QAV_RENDERER:-procedural})"
(cd services/face && .venv/bin/python -m qav_face.worker start) >"$LOGS/face.log" 2>&1 &
PIDS+=($!)

for svc in engine face; do
  for i in $(seq 1 45); do
    grep -q "registered worker" "$LOGS/$svc.log" 2>/dev/null && break
    [ "$i" = 45 ] && die "$svc worker never registered — see $LOGS/$svc.log"
    sleep 1
  done
done

# ------------------------------------------------------------------------- web
say "starting the web app"
(cd apps/web && pnpm dev) >"$LOGS/web.log" 2>&1 &
PIDS+=($!)
wait_for web http://localhost:3000

printf '
  \033[1;32m●\033[0m QAV is running — \033[1mhttp://localhost:3000\033[0m

    Allow the microphone when the browser asks, then press Start call.

    Without a GPU, pick an avatar marked \033[1m(placeholder, CPU)\033[0m in
    Settings (the gear, top right). The photoreal ones need the GPU face
    worker — see infra/gpu/RUNBOOK.md.

    Logs: %s/    Ctrl-C to stop everything.

' "$LOGS"

wait
