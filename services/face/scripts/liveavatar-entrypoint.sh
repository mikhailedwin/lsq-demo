#!/usr/bin/env bash
# Download Live Avatar weights into the /ckpt volume once, prepare demo avatars, run the worker.
set -euo pipefail

if [ ! -f "$LIVEAVATAR_CKPT/config.json" ]; then
  echo "[qav-face] downloading Wan2.2-S2V-14B (~30 GB) to $LIVEAVATAR_CKPT ..."
  huggingface-cli download Wan-AI/Wan2.2-S2V-14B --local-dir "$LIVEAVATAR_CKPT"
fi
if [ ! -f "$LIVEAVATAR_LORA/liveavatar.safetensors" ] && [ ! -d "$LIVEAVATAR_LORA" ]; then
  echo "[qav-face] downloading Live-Avatar LoRA to $LIVEAVATAR_LORA ..."
  huggingface-cli download Quark-Vision/Live-Avatar --local-dir "$LIVEAVATAR_LORA"
fi
if [ ! -d "$QAV_AVATAR_DIR/anchor" ]; then
  qav-face demo-avatars --only liveavatar
fi
exec python3 -m qav_face.worker start
