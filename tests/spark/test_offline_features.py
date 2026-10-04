"""Spark job parity with the reference feature definitions (runs in the Spark image)."""

import math
from collections.abc import Iterator
from pathlib import Path

import polars as pl
import pytest
from pipelines.spark.offline_features import compute, run, spark_session
from pyspark.sql import SparkSession

from streamrank.common.provenance import verify_outputs
from streamrank.data.ingest import ingest
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.features.definitions import (
    FEATURE_SETS,
    SECONDS_PER_DAY,
    Event,
    FeatureSet,
    reference_snapshots,
)


@pytest.fixture(scope="module")
def spark() -> Iterator[SparkSession]:
    session = spark_session(driver_memory="1g", partitions=4)
    yield session
    session.stop()


@pytest.fixture(scope="module")
def ratings() -> pl.DataFrame:
    raw = generate(SyntheticConfig(n_users=150, n_items=80, mean_interactions=25, seed=4)).ratings
    return raw.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})


def _close(a: object, b: object) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)  # type: ignore[arg-type]


def _check_parity(
    spark: SparkSession, ratings: pl.DataFrame, fs: FeatureSet, from_day: int | None, tmp: Path
) -> None:
    events = [
        Event(*row) for row in ratings.select("user_id", "item_id", "rating", "ts").iter_rows()
    ]
    expected = {
        (r[fs.key], r["snapshot_day"]): r for r in reference_snapshots(fs, events, from_day)
    }
    path = tmp / "ratings.parquet"
    ratings.write_parquet(path)
    got_df = compute(spark.read.parquet(str(path)), fs, from_day)
    got = {(r[fs.key], r["snapshot_day"]): r.asDict() for r in got_df.collect()}
    assert got.keys() == expected.keys()
    for key, exp in expected.items():
        row = got[key]
        assert row["event_timestamp"].timestamp() == key[1] * SECONDS_PER_DAY
        bad = {n: (row[n], exp[n]) for n in fs.names if not _close(row[n], exp[n])}
        assert not bad, (key, bad)


@pytest.mark.parametrize("fs", FEATURE_SETS, ids=lambda f: f.entity)
def test_matches_reference(
    spark: SparkSession, ratings: pl.DataFrame, fs: FeatureSet, tmp_path: Path
) -> None:
    _check_parity(spark, ratings, fs, None, tmp_path)


@pytest.mark.parametrize("fs", FEATURE_SETS, ids=lambda f: f.entity)
def test_matches_reference_from_day(
    spark: SparkSession, ratings: pl.DataFrame, fs: FeatureSet, tmp_path: Path
) -> None:
    mid = int(ratings["ts"].median()) // SECONDS_PER_DAY  # type: ignore[arg-type]
    _check_parity(spark, ratings, fs, mid, tmp_path)


def test_run_writes_tables_and_manifest(spark: SparkSession, tmp_path: Path) -> None:
    data = generate(SyntheticConfig(n_users=60, n_items=40, seed=8))
    raw = tmp_path / "raw"
    data.write_csv(raw)
    processed = tmp_path / "processed"
    ingest(raw, processed)
    out = tmp_path / "features"
    rows = run(spark, processed, out, from_date="2020-01-01")
    manifest = verify_outputs(out)
    assert manifest["stage"] == "offline_features"
    assert set(manifest["outputs"]) == {
        "user_features.parquet",
        "item_features.parquet",
        "item_content.parquet",
    }
    users = pl.read_parquet(out / "user_features.parquet")
    assert users.height == rows["user_features"]
    assert users.schema["event_timestamp"] == pl.Datetime("us", "UTC") or users.schema[
        "event_timestamp"
    ] == pl.Datetime("us")
    assert users["snapshot_day"].min() >= 18262  # 2020-01-01
    content = pl.read_parquet(out / "item_content.parquet")
    assert content.height == 40
