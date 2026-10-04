"""Per-user serving state in Redis (uses database 15 so real keys are untouched)."""

from collections.abc import Iterator

import numpy as np
import polars as pl
import pytest
import redis

from streamrank.common.config import get_settings
from streamrank.eval.dataset import build_train
from streamrank.models.sequences import build_sequences, gap_bucket
from streamrank.serving.state import append_event, bulk_load, read

pytestmark = pytest.mark.integration


@pytest.fixture
def client() -> Iterator[redis.Redis]:
    s = get_settings()
    c = redis.Redis(host=s.redis_host, port=s.redis_port, db=15)
    c.flushdb()
    yield c
    c.flushdb()


def test_bulk_load_read_and_append(client: redis.Redis) -> None:
    history = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2],
            "item_id": [10, 20, 30, 20],
            "rating": [5.0, 2.0, 4.0, 4.5],
            "ts": [100, 200, 400, 50],
        }
    )
    train = build_train(history, cutoff_ts=1000)
    seqs = build_sequences(train)
    assert bulk_load(client, train.user_ids, seqs, max_len=2) == 2

    u1 = read(client, 1)
    assert u1.known
    # Only the last two events are kept; seen keeps all three items (catalog indices).
    assert u1.tokens.tolist() == [2, 3]
    assert u1.positive.tolist() == [0, 1]
    assert u1.last_ts == 400
    assert u1.seen.tolist() == [0, 1, 2]
    assert not read(client, 99).known

    append_event(client, 1, token=1, positive=True, ts=460, max_len=2)
    after = read(client, 1)
    assert after.tokens.tolist() == [3, 1]
    assert after.gaps[-1] == gap_bucket(np.array([60]))[0]
    assert after.last_ts == 460

    append_event(client, 7, token=2, positive=False, ts=5, max_len=2)
    new = read(client, 7)
    assert new.tokens.tolist() == [2] and new.gaps.tolist() == [1] and new.seen.tolist() == [1]
