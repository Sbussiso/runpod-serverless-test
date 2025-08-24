FROM unsloth/unsloth:latest

# Faster HF transfers
ENV HF_HUB_ENABLE_HF_TRANSFER=1
WORKDIR /app

# Install deps used by your script
RUN pip install --no-cache-dir runpod datasets trl transformers huggingface_hub requests accelerate

# Copy your handler and (optionally) your local dataset for quick tests
COPY rp_handler.py /app/rp_handler.py
# COPY scraped_training_data.json /app/scraped_training_data.json   # uncomment if you want it baked in

CMD ["python", "-u", "/app/rp_handler.py"]
