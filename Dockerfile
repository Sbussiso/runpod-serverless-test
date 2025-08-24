FROM unsloth/unsloth:latest

# Speed up HF pulls; give cold starts more room by default.
ENV HF_HUB_ENABLE_HF_TRANSFER=1
ENV RUNPOD_INIT_TIMEOUT=900

WORKDIR /app

# Serverless runtime + training stack
RUN pip install --no-cache-dir \
    runpod \
    datasets \
    trl \
    transformers \
    huggingface_hub \
    accelerate

# Your worker
COPY rp_handler.py /app/rp_handler.py

CMD ["python", "-u", "/app/rp_handler.py"]
