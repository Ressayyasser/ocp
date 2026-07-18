# ──────────────────────────────────────────────────────────────────────────────
# OCP Cogeneration AI Platform — Dockerfile
#
# Single image containing both packages:
#   backend/  — FastAPI REST API + WebSockets (port 8000)
#   frontend/ — Dash multi-page dashboard      (port 8050)
#
# Build:
#   docker build -t ocp-cogeneration-ai .
#
# Run (both services via docker-compose — recommended):
#   docker compose up
#
# Run backend only:
#   docker run -p 8000:8000 ocp-cogeneration-ai
#
# Run frontend only (pointing at a running backend):
#   docker run -p 8050:8050 -e API_BASE_URL=http://backend:8000 \
#     ocp-cogeneration-ai python /app/frontend/app.py
# ──────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# Build tools needed by scientific wheels (tigramite, shap) on slim images
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install dependencies first for better layer caching
COPY backend/requirements.txt  /app/backend/requirements.txt
COPY frontend/requirements.txt /app/frontend/requirements.txt
# CPU-only PyTorch keeps the image small (no CUDA needed for inference)
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r /app/backend/requirements.txt \
    && pip install -r /app/frontend/requirements.txt

# Copy application code
COPY backend/  /app/backend/
COPY frontend/ /app/frontend/

EXPOSE 8000 8050

# Default command: the FastAPI backend (imports resolve from backend/)
WORKDIR /app/backend
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
