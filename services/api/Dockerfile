# Recommendation API: FastAPI + ONNX Runtime + FAISS + Feast online reads.
FROM python:3.12.13-slim-trixie AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
# Pick up Debian security fixes released after the base image was built.
RUN apt-get update && apt-get upgrade -y --no-install-recommends && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group serve --no-install-project

FROM deps AS runtime
COPY src ./src
COPY feature_repo ./feature_repo
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group serve --no-editable \
    && useradd --create-home --uid 10001 app
# One worker per container; each native library gets one thread so they do not compete.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    SERVING_DIR=/artifacts/serving \
    FEATURE_REPO=/app/feature_repo
USER app
EXPOSE 8000
CMD ["uvicorn", "streamrank.serving.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--no-access-log"]
