#!/bin/bash
set -e
echo "=== Mount check ==="
ls -lh /runpod-volume/ || echo "No volume!"
mkdir -p /comfyui/models/checkpoints /comfyui/models/loras
for f in /runpod-volume/*.safetensors; do
  [ -e "$f" ] || continue
  echo "Linking $f"
  ln -sf "$f" /comfyui/models/checkpoints/$(basename "$f")
  ln -sf "$f" /comfyui/models/loras/$(basename "$f")
done
ls -lh /comfyui/models/loras/
if [ -f "/runpod-volume/Uiharu_Railgun.safetensors" ]; then
  echo "Loading LoRA: /runpod-volume/Uiharu_Railgun.safetensors at weight 0.8 - FOUND"
fi
exec python -u /handler.py "$@"
