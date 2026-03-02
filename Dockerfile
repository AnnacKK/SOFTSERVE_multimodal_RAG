# --- STAGE 1: Builder ---

FROM python:3.11-slim as builder
LABEL authors="AnnacKK"
WORKDIR /app
ENV PATH="/root/.local/bin:${PATH}"
# Install build essentials for heavy ML libraries like sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --user --no-cache-dir \
    --default-timeout=100 \
    --retries 5 \
    -r requirements.txt

# --- STAGE 2: Final Image ---
# 🟢 Use NVIDIA's CUDA base image if you want the container to handle the GPU tasks
# (SentenceTransformers/Reranker) instead of just the CPU.
FROM python:3.11-slim
LABEL authors="AnnacKK"
WORKDIR /app

# Install system dependencies for OpenCV and Vision tasks (required by Moondream/Llama logic)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libmagic1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed libraries
COPY --from=builder /root/.local /root/.local
# Copy project structure
COPY ./src ./src
COPY ./api.py .
COPY ./templates ./templates

# Environment setup
ENV PATH=/root/.local/bin:$PATH
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app
# 🟢 CRITICAL: Force local Ollama connection
ENV OLLAMA_BASE_URL="http://host.docker.internal:11434"

EXPOSE 8000

# uvicorn with more workers for local performance
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]