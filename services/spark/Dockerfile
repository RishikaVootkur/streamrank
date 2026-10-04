# Spark batch jobs: Java 21 runtime, Python 3.12, PySpark, and the streamrank package.
FROM eclipse-temurin:21.0.12.1_1-jre-noble AS base
COPY --from=ghcr.io/astral-sh/uv:0.11.33 /uv /bin/uv
ENV UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3.12 python3.12-venv \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app

FROM base AS deps
COPY pyproject.toml uv.lock README.md ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group spark --no-install-project

FROM deps AS test
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group spark --group dev --no-install-project
COPY src ./src
COPY pipelines ./pipelines
COPY tests ./tests
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group spark --group dev
CMD ["pytest", "tests/spark", "-q"]

FROM deps AS runtime
COPY src ./src
COPY pipelines ./pipelines
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-default-groups --group spark --no-editable \
    && useradd --create-home --uid 10001 spark \
    && mkdir -p /data && chown spark /data
USER spark
ENTRYPOINT ["python", "-m", "pipelines.spark.offline_features"]
