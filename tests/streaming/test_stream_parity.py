"""Streaming job parity on 10,000 replayed events (needs Redpanda and Redis from Compose)."""

from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import redis

from streamrank.common.config import get_settings
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.serving.state import read
from streamrank.streaming.events import replay_log
from streamrank.streaming.parity import run

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> Iterator[redis.Redis]:
    s = get_settings()
    c = redis.Redis(host=s.redis_host, port=s.redis_port, db=13)
    c.flushdb()
    yield c
    c.flushdb()


def test_online_matches_offline_on_10000_events(client: redis.Redis, tmp_path: Path) -> None:
    raw = generate(SyntheticConfig(n_users=1500, n_items=400, mean_interactions=30, seed=8)).ratings
    ratings = raw.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    log = replay_log(ratings, start_ts=int(ratings["ts"].quantile(0.4)), limit=10_000)
    assert log.height == 10_000
    catalog = np.unique(ratings["item_id"].to_numpy())
    token_of = {int(item): i + 1 for i, item in enumerate(catalog)}
    result = run(log, get_settings().kafka_broker, client, tmp_path / "state", catalog)
    assert result["events"] == 10_000
    assert result["mismatches"] == 0, result["examples"]
    assert result["latency"]["p95_ms"] < 5_000
    # The serving state also advanced: each user's latest replayed item is now in their history.
    last = log.group_by("user_id").last()
    for row in last.head(50).iter_rows(named=True):
        state = read(client, row["user_id"])
        assert state.tokens[-1] == token_of[row["item_id"]]
