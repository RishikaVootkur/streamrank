# Streamlit demo UI.
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
    uv sync --locked --no-default-groups --group demo --no-install-project

FROM deps AS runtime
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group demo --no-editable \
    && useradd --create-home --uid 10001 app
ENV SERVING_DIR=/artifacts/serving \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_CLIENT_TOOLBAR_MODE=minimal
USER app
EXPOSE 8501
CMD ["streamlit", "run", "/app/src/streamrank/demo/app.py", "--server.port", "8501", "--server.address", "0.0.0.0"]
