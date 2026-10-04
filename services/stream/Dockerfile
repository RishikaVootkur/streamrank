# Streaming session-feature job: Quix Streams consumer writing to Redis.
FROM python:3.12.13-slim-trixie AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group stream --no-install-project

FROM deps AS runtime
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group stream --no-editable \
    && useradd --create-home --uid 10001 app \
    && mkdir -p /state && chown app /state
USER app
ENTRYPOINT ["python", "-m", "streamrank.streaming.app", "--state-dir", "/state", "--serving-dir", "/artifacts/serving"]
