# syntax=docker/dockerfile:1

# ---------------------------------------------------------------------------
# CivicAgent backend — multi-stage production image.
#   builder: installs all runtime deps into /deps (isolated prefix).
#   runner : slim runtime, migrations handled by docker/entrypoint.sh.
#   test   : runner + dev/test tooling (pytest, ruff) for compose/GHA.
# ---------------------------------------------------------------------------

# ---- Stage 1: builder -----------------------------------------------------
FROM python:3.11-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

# Build tools for any dependency that falls back to a source build.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install the project (metadata only) and its full dependency closure into a
# dedicated prefix so no build tooling leaks into the runtime image.
COPY pyproject.toml alembic.ini ./
COPY app ./app
COPY alembic ./alembic
COPY main.py seed.py ./

RUN python -m pip install --upgrade pip && \
    python -m pip install --prefix=/deps .

# ---- Stage 2: runner (production) -----------------------------------------
FROM python:3.11-slim AS runner

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH="/deps/lib/python3.11/site-packages:/app" \
    PATH="/deps/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/* \
    && addgroup --system --gid 10001 app \
    && adduser --system --uid 10001 --ingroup app --home /app --disabled-password app \
    && mkdir -p /app/uploads \
    && chown -R app:app /app

WORKDIR /app

COPY --from=builder /deps /deps
COPY --from=builder /build/app ./app
COPY --from=builder /build/alembic ./alembic
COPY --from=builder /build/main.py ./main.py
COPY --from=builder /build/seed.py ./seed.py
COPY --from=builder /build/alembic.ini ./alembic.ini
COPY --from=builder /build/pyproject.toml ./pyproject.toml

# Entrypoint: waits for the DB, applies alembic migrations, then exec's CMD.
COPY docker/entrypoint.sh docker/log_config.json ./docker/
RUN chmod +x docker/entrypoint.sh && chown -R app:app /app

USER app

EXPOSE 8000

# $PORT is the container port Render provisions (unknown at build time); the
# 8000 fallback keeps docker-compose and other local handlers unchanged.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=5 \
    CMD sh -c 'curl -fsS http://127.0.0.1:${PORT:-8000}/api/v1/health || exit 1'

# Production default: structured (JSON) logs. Scale with `--scale`/orchestrator.
ENTRYPOINT ["/app/docker/entrypoint.sh"]
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port \"${PORT:-8000}\" --log-config docker/log_config.json"]

# ---- Stage 3: test ---------------------------------------------------------
FROM runner AS test

# Dev/test tooling only. Adapted at image build time when base envs are set,
# then consumed by `docker compose -f docker-compose.yml -f docker-compose.test.yml run backend`.
RUN python -m pip install --no-cache-dir ".[dev]"

ENV SEED_DEMO_DATA=false