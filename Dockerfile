FROM runpod/worker-comfyui:5.1.0-base
RUN mkdir -p /comfyui/models/loras && ln -s /runpod-volume /comfyui/models/loras/volumeloras