# ==============================================================================
# IMDb Recommender - Dockerfile
# ==============================================================================
# Multi-stage build for production-grade Python ML service.

FROM python:3.12-slim AS base

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Dependencies layer (cached) ──────────────────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Application code ─────────────────────────────────────────────────────────
COPY src/ src/
COPY api/ api/
COPY ui/ ui/
COPY configs/ configs/
COPY Makefile .

# ── Data volume (model artifacts mounted at runtime) ─────────────────────────
RUN mkdir -p data/raw data/processed

# ── Default: API server ──────────────────────────────────────────────────────
EXPOSE 8000
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
