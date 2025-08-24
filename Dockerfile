FROM unsloth/unsloth:latest
WORKDIR /app
COPY rp_handler.py /app/rp_handler.py
RUN pip install --no-cache-dir runpod
CMD ["python", "-u", "/app/rp_handler.py"]
