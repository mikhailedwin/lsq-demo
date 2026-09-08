#!/usr/bin/env bash
# Provision a fresh Ubuntu 22.04 GPU box to run the QAV face worker.
#
#   curl -fsSL https://raw.githubusercontent.com/<you>/lsq-demo/main/infra/gpu/setup.sh | bash -s -- musetalk
#   ./infra/gpu/setup.sh liveavatar
#
# Installs the NVIDIA container toolkit + Docker, builds the face image, and
# prints the exact `docker run` line. Weights download on first start into a
# named volume, so rebuilding the image doesn't re-fetch tens of GB.
set -euo pipefail

BACKEND="${1:-musetalk}"
REPO_DIR="${REPO_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"

case "$BACKEND" in
  musetalk)   MIN_VRAM_GB=12 ;;
  liveavatar) MIN_VRAM_GB=48 ;;
  *) echo "usage: $0 [musetalk|liveavatar]" >&2; exit 2 ;;
esac

echo "==> checking GPU"
if ! command -v nvidia-smi >/dev/null; then
  echo "no nvidia-smi: install the NVIDIA driver first (ubuntu-drivers autoinstall && reboot)" >&2
  exit 1
fi
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
VRAM_MB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
if [ "$VRAM_MB" -lt $((MIN_VRAM_GB * 1000)) ]; then
  echo "WARNING: $BACKEND wants >= ${MIN_VRAM_GB} GB VRAM, this GPU has $((VRAM_MB / 1000)) GB." >&2
  [ "$BACKEND" = "liveavatar" ] && echo "         Set QAV_LIVEAVATAR_FP8=1 to fit in 48 GB (slight quality loss)." >&2
fi

echo "==> installing docker + nvidia-container-toolkit"
if ! command -v docker >/dev/null; then
  curl -fsSL https://get.docker.com | sh
fi
if ! docker info 2>/dev/null | grep -q nvidia; then
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
    | sudo gpg --yes --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
  curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list >/dev/null
  sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
  sudo nvidia-ctk runtime configure --runtime=docker
  sudo systemctl restart docker
fi
docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi -L

echo "==> building qav-face:$BACKEND (this takes a while on first run)"
cd "$REPO_DIR"
docker build -f "services/face/Dockerfile.$BACKEND" -t "qav-face:$BACKEND" .

cat <<EOF

==> done.

Point the face worker at your LiveKit server and start it:

  docker run -d --name qav-face --gpus all --network host --restart unless-stopped \\
    -e LIVEKIT_URL=wss://your-livekit-host \\
    -e LIVEKIT_API_KEY=... -e LIVEKIT_API_SECRET=... \\
    -v qav-avatars:/avatars $( [ "$BACKEND" = liveavatar ] && echo '-v qav-ckpt:/ckpt \\' ) \\
    qav-face:$BACKEND

Verify the renderer on this box before wiring up a session (writes PNGs + an MP4):

  docker run --rm --gpus all -v qav-avatars:/avatars -v \$PWD/out:/out qav-face:$BACKEND \\
    qav-face selftest --renderer $BACKEND $( [ "$BACKEND" = musetalk ] && echo '--avatar yongen' || echo '--avatar anchor' ) --out /out

Then set QAV_RENDERER=$BACKEND in the engine's environment (or just pick a
$BACKEND avatar in the demo — the engine dispatches this worker per session).
EOF
