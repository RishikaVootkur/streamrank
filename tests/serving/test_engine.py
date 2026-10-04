"""Serving pipeline and API over tiny artifacts (Redis database 15)."""

from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import pytest
import redis
from fastapi.testclient import TestClient

from streamrank.ranking.features import ITEM_STATS
from streamrank.ranking.online import ItemTable
from streamrank.serving import app as app_module
from streamrank.serving.engine import Artifacts, Engine, left_pad, retrieve
from streamrank.serving.state import read

pytestmark = pytest.mark.integration


def _items(art: Artifacts) -> ItemTable:
    stats = pl.DataFrame(
        {"item_id": art.item_ids, **{c: np.ones(art.item_ids.size) for c in ITEM_STATS}}
    )
    return ItemTable.build(art.item_ids, stats, art.year, art.genres, art.has_movie)


def test_left_pad_matches_training_layout(serving_dir: Path, client: redis.Redis) -> None:
    padded = left_pad(read(client, 1), 6)
    assert padded["tokens"].tolist() == [[0, 0, 0, 1, 2, 3]]
    assert padded["positive"].tolist() == [[0, 0, 0, 1, 1, 0]]


def test_retrieve_skips_seen_items() -> None:
    from streamrank.retrieval.index import IndexSpec, build  # noqa: PLC0415

    vecs = np.eye(4, dtype=np.float32)
    rows, scores = retrieve(
        build(vecs, IndexSpec("flat")).index, vecs[[0]], np.array([0, 2]), 4, k=3
    )
    assert rows.tolist()[0] not in (0, 2) and set(rows.tolist()) <= {1, 3}
    assert np.all(np.diff(scores) <= 0)


def test_engine_personalized_and_fallback(serving_dir: Path, client: redis.Redis) -> None:
    art = Artifacts.load(serving_dir)
    engine = Engine(art, client, lambda _u: {"user_n_ratings": 3.0}, _items(art))
    rec = engine.recommend(1, k=10)
    assert rec.source == "personalized"
    assert len(rec.item_ids) == 10 and len(set(rec.item_ids)) == 10
    assert not {1, 2, 3} & set(rec.item_ids)  # seen items are excluded
    assert rec.scores == sorted(rec.scores, reverse=True)
    assert {"state", "user_tower", "retrieval", "user_features", "ranking"} <= set(rec.timings_ms)
    unknown = engine.recommend(999, k=5)
    assert unknown.source == "popular" and unknown.item_ids == [21, 22, 23, 24, 25]


class _FakeResponse:
    def __init__(self, data: dict[str, list[Any]]) -> None:
        self.data = data

    def to_dict(self) -> dict[str, list[Any]]:
        return self.data


class _FakeStore:
    def get_online_features(
        self, features: list[str], entity_rows: list[dict[str, int]]
    ) -> _FakeResponse:
        key = next(iter(entity_rows[0]))
        names = [f.split(":")[1] for f in features]
        out: dict[str, list[Any]] = {key: [r[key] for r in entity_rows]}
        for n in names:
            out[n] = [1.0] * len(entity_rows)
        return _FakeResponse(out)


def test_api_endpoints(
    serving_dir: Path, client: redis.Redis, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("SERVING_DIR", str(serving_dir))
    monkeypatch.setenv("REDIS_PORT", str(client.connection_pool.connection_kwargs["port"]))
    monkeypatch.setattr(app_module, "_feast_store", _FakeStore)
    real_redis = app_module.redis.Redis
    monkeypatch.setattr(app_module.redis, "Redis", lambda **kw: real_redis(**kw, db=15))
    from streamrank.common.config import get_settings  # noqa: PLC0415

    get_settings.cache_clear()
    with TestClient(app_module.app) as http:
        assert http.get("/livez").json() == {"status": "ok"}
        assert http.get("/readyz").status_code == 200
        body = http.get("/recommendations/1?k=5").json()
        assert body["source"] == "personalized" and len(body["items"]) == 5
        assert body["items"][0]["title"].startswith("Movie ")
        assert http.get("/recommendations/12345").json()["source"] == "popular"
        assert http.get("/recommendations/1?k=0").status_code == 422
        metrics = http.get("/metrics/").text
        assert 'recommend_latency_seconds_bucket{le="0.05",stage="total"}' in metrics
        assert 'recommend_requests_total{source="popular",status="200"} 1.0' in metrics
    get_settings.cache_clear()
