# ==============================================================================
# Vision Sentinel — AWS ECR / ECS Graviton & x86_64 Container Definition
# ==============================================================================

FROM python:3.10-slim-bullseye

# Install runtime codecs and libraries required for OpenCV 5 headless video decoding
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Ensure latest pip
RUN python -m pip install --upgrade pip setuptools wheel

# Install pinned dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Security: Non-root user execution
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Copy application code
COPY app/ app/
COPY scripts/ scripts/
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY THIRD_PARTY_NOTICES.md .

RUN mkdir -p data/uploads data/outputs data/generated models && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
