"""Feast over reference feature snapshots: online values and point-in-time joins."""

import os
import shutil
import subprocess
import sys
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import pytest

from streamrank.common.config import get_settings
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.features.definitions import (
    ITEM_FEATURES,
    SECONDS_PER_DAY,
    USER_FEATURES,
    Event,
    FeatureSet,
    day_of,
    reference_snapshots,
    snapshot,
)
from streamrank.features.store import (
    DEFAULT_REPO,
    ITEM_REFS,
    USER_REFS,
    historical_features,
    materialize,
    online_features,
    open_store,
)

pytestmark = pytest.mark.integration


def _snapshot_frame(fs: FeatureSet, events: list[Event]) -> pl.DataFrame:
    rows = reference_snapshots(fs, events)
    counts = {a.name for a in fs.aggregates if a.kind == "count" or a.column == "ts"}
    schema = {fs.key: pl.Int64, "snapshot_day": pl.Int64}
    schema |= {n: pl.Int64 if n in counts else pl.Float64 for n in fs.names}
    df = pl.DataFrame(rows, schema=schema)
    return df.with_columns(
        event_timestamp=pl.from_epoch(pl.col("snapshot_day") * SECONDS_PER_DAY, "s")
        .dt.cast_time_unit("us")
        .dt.replace_time_zone("UTC")
    )


@pytest.fixture(scope="module")
def events() -> list[Event]:
    data = generate(SyntheticConfig(n_users=60, n_items=40, mean_interactions=15, seed=21))
    r = data.ratings.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    return [Event(*row) for row in r.select("user_id", "item_id", "rating", "ts").iter_rows()]


@pytest.fixture(scope="module")
def repo(tmp_path_factory: pytest.TempPathFactory, events: list[Event]) -> Iterator[Path]:
    root = tmp_path_factory.mktemp("feast")
    features = root / "features"
    features.mkdir()
    for fs in (USER_FEATURES, ITEM_FEATURES):
        _snapshot_frame(fs, events).write_parquet(features / f"{fs.entity}_features.parquet")
    items = sorted({e.item_id for e in events})
    pl.DataFrame(
        {
            "item_id": items,
            "item_year": [1990 + i % 30 for i in items],
            "item_genres": [["Drama"] if i % 2 else ["Comedy", "Romance"] for i in items],
            "event_timestamp": [datetime(1970, 1, 1, tzinfo=UTC)] * len(items),
        }
    ).write_parquet(features / "item_content.parquet")

    repo = root / "repo"
    repo.mkdir()
    shutil.copy(DEFAULT_REPO / "features.py", repo / "features.py")
    settings = get_settings()
    (repo / "feature_store.yaml").write_text(
        "project: streamrank_test\n"
        "provider: local\n"
        f"registry: {root / 'registry.db'}\n"
        "offline_store:\n  type: duckdb\n"
        "online_store:\n  type: redis\n"
        f"  connection_string: {settings.redis_host}:{settings.redis_port}\n"
        "entity_key_serialization_version: 3\n"
    )
    env = {**os.environ, "STREAMRANK_FEATURES_DIR": str(features)}
    feast = Path(sys.executable).parent / "feast"
    subprocess.run([str(feast), "apply"], cwd=repo, env=env, check=True, capture_output=True)
    os.environ["STREAMRANK_FEATURES_DIR"] = str(features)
    yield repo
    subprocess.run([str(feast), "teardown"], cwd=repo, env=env, check=False, capture_output=True)


def _close(a: object, b: object) -> bool:
    if a is None or (isinstance(a, float) and np.isnan(a)):
        return b is None
    return b is not None and abs(float(a) - float(b)) < 1e-9  # type: ignore[arg-type]


def test_online_store_serves_latest_snapshot(repo: Path, events: list[Event]) -> None:
    store = open_store(repo)
    materialize(store, datetime(1990, 1, 1, tzinfo=UTC), datetime(2030, 1, 1, tzinfo=UTC))
    users = sorted({e.user_id for e in events})[:15]
    got = online_features(store, USER_REFS, "user_id", users)
    for i, user in enumerate(users):
        evs = [e for e in events if e.user_id == user]
        # After the last expiry every count window is empty, the snapshot is "all history".
        expected = snapshot(USER_FEATURES, evs, 10**6)
        for name in USER_FEATURES.names:
            assert _close(got[name][i], expected[name]), (user, name)
    items = sorted({e.item_id for e in events})[:5]
    content = online_features(store, ITEM_REFS, "item_id", items)
    assert content["item_year"] == [1990 + i % 30 for i in items]
    assert content["item_genres"][1] in (["Drama"], ["Comedy", "Romance"])


def test_historical_join_is_point_in_time(repo: Path, events: list[Event]) -> None:
    store = open_store(repo)
    rng = np.random.default_rng(0)
    sample = [events[i] for i in rng.choice(len(events), size=40, replace=False)]
    # Query at each event's own time and a few days later; neither may see that day's events.
    times = [e.ts for e in sample] + [e.ts + 3 * SECONDS_PER_DAY + 7 for e in sample]
    users = [e.user_id for e in sample] * 2
    entity_df = pd.DataFrame(
        {"user_id": users, "event_timestamp": pd.to_datetime(times, unit="s", utc=True)}
    )
    out = historical_features(store, entity_df, USER_REFS)
    assert len(out) == len(entity_df)
    for row in out.itertuples(index=False):
        t = int(row.event_timestamp.timestamp())
        evs = [e for e in events if e.user_id == row.user_id]
        expected = snapshot(USER_FEATURES, evs, day_of(t))
        if expected["user_n_ratings"] == 0:
            # No earlier activity means no snapshot row exists yet.
            assert pd.isna(row.user_n_ratings)
            continue
        for name in USER_FEATURES.names:
            assert _close(getattr(row, name), expected[name]), (row.user_id, t, name)
