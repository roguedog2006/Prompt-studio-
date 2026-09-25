FROM runpod/worker-comfyui:5.4.0-base
COPY start.sh /start.sh
RUN chmod +x /start.sh
ENTRYPOINT ["/start.sh"]
