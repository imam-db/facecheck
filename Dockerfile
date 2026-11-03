# syntax=docker/dockerfile:1

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# opencv-contrib-python-headless still needs libglib; it does not need libGL.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE NOTICE ./
COPY third_party ./third_party
COPY src ./src
RUN pip install ".[postgres]"

COPY scripts ./scripts
RUN python scripts/download_models.py --models-dir /app/models

RUN useradd --create-home --uid 10001 facecheck \
    && mkdir -p /data \
    && chown -R facecheck:facecheck /app /data

ENV HOME=/home/facecheck \
    FACECHECK_MODELS_DIR=/app/models \
    FACECHECK_DB_URL=sqlite:////data/facecheck.db

USER facecheck

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -fsS http://localhost:8000/healthz || exit 1

CMD ["uvicorn", "facecheck.api:app", "--host", "0.0.0.0", "--port", "8000"]
