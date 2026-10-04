"""Recommendation API (FastAPI).

Run: `uvicorn streamrank.serving.app:app`. Models, the index, item features, and user
profiles load once at startup; `/readyz` turns ready after a warm-up request.
"""

import asyncio
import logging
import os
import sys
import time
from collections.abc import AsyncIterator, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio
import numpy as np
import polars as pl
import redis
from fastapi import FastAPI, HTTPException, Query, Response
from prometheus_client import Counter, Histogram, make_asgi_app

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.ranking.features import ITEM_STATS, USER_STATS
from streamrank.ranking.online import ItemTable
from streamrank.serving.engine import Artifacts, Engine

log = logging.getLogger(__name__)
BUCKETS = (
    0.001,
    0.0025,
    0.005,
    0.0075,
    0.01,
    0.015,
    0.02,
    0.025,
    0.03,
    0.04,
    0.05,
    0.075,
    0.1,
    0.25,
    0.5,
    1.0,
)
LATENCY = Histogram(
    "recommend_latency_seconds", "Recommendation latency", ["stage"], buckets=BUCKETS
)
REQUESTS = Counter("recommend_requests_total", "Recommendation requests", ["source", "status"])


def _feast_store() -> Any:
    # Feast imports PyTorch when it is installed; this process never needs it, and on macOS
    # a second OpenMP runtime next to FAISS crashes (see dev notes). The serving image has
    # no PyTorch, so this only matters when running the API on a development machine.
    sys.modules.setdefault("torch", None)  # type: ignore[arg-type]
    from streamrank.features.store import open_store  # noqa: PLC0415

    return open_store()


def load_item_table(store: Any, art: Artifacts, batch: int = 10_000) -> ItemTable:
    """Latest item statistics from the online store, aligned to the catalog."""
    refs = [f"item_stats:{c}" for c in ITEM_STATS]
    frames = []
    for start in range(0, art.item_ids.size, batch):
        ids = [int(x) for x in art.item_ids[start : start + batch]]
        resp = store.get_online_features(features=refs, entity_rows=[{"item_id": i} for i in ids])
        frames.append(
            pl.DataFrame(resp.to_dict(), schema_overrides={c: pl.Float64 for c in ITEM_STATS})
        )
    stats = pl.concat(frames).select("item_id", *ITEM_STATS)
    return ItemTable.build(art.item_ids, stats, art.year, art.genres, art.has_movie)


def check_clock(items: ItemTable, now_ts: int) -> None:
    """Refuse to serve statistics from after the serving clock (they would leak the future
    and produce values the ranker never saw in training)."""
    latest = float(np.nanmax(items.stats["item_last_ts"]))
    if latest >= now_ts:
        raise RuntimeError(
            f"online item statistics include events at {latest:.0f}, at or after the serving "
            f"clock {now_ts}; materialize the online store with --end-at-cutoff"
        )


def user_feature_fn(store: Any) -> Any:
    refs = [f"user_stats:{c}" for c in USER_STATS]

    def fetch(user_id: int) -> Mapping[str, float | None]:
        resp = store.get_online_features(features=refs, entity_rows=[{"user_id": user_id}])
        row = resp.to_dict()
        return {c: (None if row[c][0] is None else float(row[c][0])) for c in USER_STATS}

    return fetch


def _setup_tracing(app: FastAPI) -> None:
    """Export traces over OTLP when OTEL_EXPORTER_OTLP_ENDPOINT is set."""
    if not os.environ.get("OTEL_EXPORTER_OTLP_ENDPOINT"):
        return
    from opentelemetry import trace  # noqa: PLC0415
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (  # noqa: PLC0415
        OTLPSpanExporter,
    )
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor  # noqa: PLC0415
    from opentelemetry.sdk.resources import Resource  # noqa: PLC0415
    from opentelemetry.sdk.trace import TracerProvider  # noqa: PLC0415
    from opentelemetry.sdk.trace.export import BatchSpanProcessor  # noqa: PLC0415

    provider = TracerProvider(resource=Resource.create({"service.name": "streamrank-api"}))
    provider.add_span_processor(BatchSpanProcessor(OTLPSpanExporter()))
    trace.set_tracer_provider(provider)
    FastAPIInstrumentor.instrument_app(app, excluded_urls="metrics,livez,readyz")


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    root = Path(os.environ.get("SERVING_DIR", str(settings.artifacts_dir / "serving")))
    t0 = time.perf_counter()
    art = Artifacts.load(root)
    client = redis.Redis(host=settings.redis_host, port=settings.redis_port)
    store = _feast_store()
    items = await anyio.to_thread.run_sync(load_item_table, store, art)
    check_clock(items, art.cutoff_ts)
    engine = Engine(art, client, user_feature_fn(store), items)
    warm = int(art.profile_users[0]) if art.profile_users.size else 1
    await anyio.to_thread.run_sync(engine.recommend, warm, 10)
    app.state.engine = engine
    app.state.redis = client
    # The engine holds the GIL for most of a request, so extra threads add no throughput.
    # Under overload, the default 40 threads contend for the GIL and each request costs
    # several times more CPU; a small cap keeps the backlog in the event loop instead.
    app.state.limiter = anyio.CapacityLimiter(int(os.environ.get("ENGINE_THREADS", "8")))
    app.state.ready = True
    log.info(
        "ready",
        extra={"seconds": round(time.perf_counter() - t0, 1), "items": int(art.item_ids.size)},
    )
    yield
    app.state.ready = False
    client.close()


app = FastAPI(title="StreamRank", lifespan=lifespan)
app.state.ready = False
app.mount("/metrics", make_asgi_app())
_setup_tracing(app)


@app.get("/livez")
async def livez() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/readyz")
async def readyz(response: Response) -> dict[str, str]:
    ready = bool(app.state.ready)
    if ready:
        try:
            await asyncio.to_thread(app.state.redis.ping)
        except redis.RedisError:
            ready = False
    if not ready:
        response.status_code = 503
        return {"status": "starting"}
    return {"status": "ready"}


@app.get("/recommendations/{user_id}")
async def recommendations(user_id: int, k: int = Query(10, ge=1, le=100)) -> dict[str, Any]:
    if not app.state.ready:
        raise HTTPException(status_code=503, detail="not ready")
    t0 = time.perf_counter()
    try:
        rec = await anyio.to_thread.run_sync(
            app.state.engine.recommend, user_id, k, limiter=app.state.limiter
        )
    except Exception:
        REQUESTS.labels(source="error", status="500").inc()
        log.exception("recommendation failed", extra={"user_id": user_id})
        raise
    total = time.perf_counter() - t0
    LATENCY.labels(stage="total").observe(total)
    for stage, ms in rec.timings_ms.items():
        LATENCY.labels(stage=stage).observe(ms / 1000)
    REQUESTS.labels(source=rec.source, status="200").inc()
    return {
        "user_id": user_id,
        "source": rec.source,
        "items": [
            {"item_id": i, "title": t, "score": s}
            for i, t, s in zip(rec.item_ids, rec.titles, rec.scores, strict=True)
        ],
        "timings_ms": rec.timings_ms | {"total": round(total * 1000, 3)},
    }
