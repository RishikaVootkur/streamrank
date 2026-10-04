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
# Trim jars this job never loads (Hive's Derby metastore, the Spark Connect REPL) and
# replace the bundled netty-handler with the patched release (CVE-2026-75595).
ARG NETTY_VERSION=4.2.17.Final
ARG NETTY_SHA1=ed53dae90d09306b42bc886fa292510abdf5c5dc
RUN set -eu; \
    jars=/opt/venv/lib/python3.12/site-packages/pyspark/jars; \
    rm -rf "$jars/connect-repl" "$jars"/derby*.jar "$jars"/netty-handler-4.*.Final.jar; \
    python3 -c "import hashlib, sys, urllib.request; \
data = urllib.request.urlopen(sys.argv[1]).read(); \
assert hashlib.sha1(data).hexdigest() == sys.argv[2], 'checksum mismatch'; \
open(sys.argv[3], 'wb').write(data)" \
    "https://repo1.maven.org/maven2/io/netty/netty-handler/${NETTY_VERSION}/netty-handler-${NETTY_VERSION}.jar" \
    "$NETTY_SHA1" "$jars/netty-handler-${NETTY_VERSION}.jar"

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
