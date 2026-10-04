import math

import numpy as np
import polars as pl

from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.features.definitions import (
    SESSION_FEATURES,
    SESSION_WINDOW_SECONDS,
    Event,
    SessionState,
    session_snapshot,
)
from streamrank.streaming.events import replay_log
from streamrank.streaming.offline import session_features_at_last_event


def _same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)


def test_session_state_matches_reference_at_every_event() -> None:
    rng = np.random.default_rng(0)
    ts = np.cumsum(rng.integers(0, 900, 300))  # gaps up to 15 minutes, many same-second
    events = [
        Event(1, int(rng.integers(1, 50)), float(rng.choice([1.0, 3.5, 4.0, 5.0])), int(t))
        for t in ts
    ]
    state = SessionState(events=[])
    for i, e in enumerate(events):
        online = state.update(e)
        offline = session_snapshot(events[: i + 1], e.ts)
        assert all(_same(online[f], offline[f]) for f in SESSION_FEATURES), i


def test_window_boundary_is_exclusive_on_the_left() -> None:
    a = Event(1, 10, 5.0, 0)
    b = Event(1, 11, 2.0, SESSION_WINDOW_SECONDS)  # exactly 30 minutes later: a drops out
    snap = session_snapshot([a, b], b.ts)
    assert snap["session_n_events"] == 1.0 and snap["session_last_item_id"] == 11.0
    empty = session_snapshot([a], 10**6)
    assert empty["session_n_events"] == 0.0 and math.isnan(empty["session_mean_rating"])


def test_polars_batch_matches_reference_on_a_replay_log() -> None:
    raw = generate(SyntheticConfig(n_users=200, n_items=100, seed=3)).ratings
    ratings = raw.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    log = replay_log(ratings, start_ts=int(ratings["ts"].quantile(0.5)), limit=5000)
    batch = session_features_at_last_event(log)
    by_user: dict[int, list[Event]] = {}
    for row in log.iter_rows():
        by_user.setdefault(row[0], []).append(Event(*row))
    for row in batch.iter_rows(named=True):
        evs = by_user[row["user_id"]]
        ref = session_snapshot(evs, evs[-1].ts)
        assert all(_same(float(row[f]), ref[f]) for f in SESSION_FEATURES), row["user_id"]
    assert batch.height == len(by_user)


def test_replay_log_order_is_time_then_hash() -> None:
    df = pl.DataFrame(
        {"user_id": [1, 2, 3, 4], "item_id": [9, 8, 7, 6], "rating": [4.0] * 4, "ts": [5, 5, 3, 9]}
    )
    log = replay_log(df, start_ts=4)
    assert log["ts"].to_list() == [5, 5, 9]
    assert sorted(log.head(2)["user_id"].to_list()) == [1, 2]
