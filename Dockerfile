# ==============================================================================
# Vision Sentinel — Production Multi-Stage Dockerfile
# OpenCV AI Competition 2026, powered by AWS
# Supports Linux AMD64 and Linux ARM64 (AWS Graviton)
# ==============================================================================

FROM python:3.10-slim-bullseye AS base

# System dependencies for OpenCV video and headless image codecs
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Upgrade pip and install wheel
RUN python -m pip install --upgrade pip setuptools wheel

# Install pinned application dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Security: Create unprivileged system user
RUN groupadd -g 10001 appuser && \
    useradd -u 10001 -g appuser -s /bin/bash -m appuser

# Copy application source code and configurations
COPY app/ app/
COPY scripts/ scripts/
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .
COPY THIRD_PARTY_NOTICES.md .

# Create and grant permissions on data and models directories
RUN mkdir -p data/uploads data/outputs data/generated models && \
    chown -R appuser:appuser /app

USER appuser

# Expose HTTP service port
EXPOSE 8000

# Container healthcheck querying /health endpoint
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Launch production server via Uvicorn
ENTRYPOINT ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
