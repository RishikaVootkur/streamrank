"""Compute point-in-time correct user and item feature snapshots with Spark.

The feature specs come from `streamrank.features.definitions`, the same module the
streaming job uses. For each entity, daily partial aggregates are merged with the days on
which a snapshot is due, then every aggregate becomes a window over day numbers that ends
the day before the snapshot (`rangeBetween(..., -1)`), so a snapshot never sees events from
its own day.

Run inside the Spark image: `python -m pipelines.spark.offline_features --help`.
"""

import argparse
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pyspark.sql import Column, DataFrame, SparkSession, Window
from pyspark.sql import functions as F

from streamrank.common.logging import configure_logging
from streamrank.common.provenance import begin_stage, verify_outputs, write_manifest
from streamrank.features.definitions import (
    FEATURE_SETS,
    POSITIVE_RATING,
    SECONDS_PER_DAY,
    Aggregate,
    FeatureSet,
    value_type,
)

log = logging.getLogger(__name__)
DAY = "snapshot_day"
EVENT_TIMESTAMP = "event_timestamp"
_MERGE = {"count": F.sum, "sum": F.sum, "min": F.min, "max": F.max}


def spark_session(driver_memory: str = "4g", partitions: int = 64) -> SparkSession:
    """A local session that writes UTC microsecond timestamps."""
    return (
        SparkSession.builder.master("local[*]")
        .appName("streamrank-offline-features")
        .config("spark.driver.memory", driver_memory)
        .config("spark.sql.shuffle.partitions", str(partitions))
        .config("spark.sql.session.timeZone", "UTC")
        .config("spark.sql.parquet.outputTimestampType", "TIMESTAMP_MICROS")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )


def _partial(spec: Aggregate) -> Column:
    """The per-day partial aggregate for one spec."""
    keep = F.col("rating") >= POSITIVE_RATING if spec.positives_only else F.lit(True)
    if spec.kind == "count":
        return F.count(F.when(keep, 1)).alias(spec.name)
    value = F.when(keep, F.col(str(spec.column)))
    agg = {"sum": F.sum, "min": F.min, "max": F.max}[spec.kind]
    return agg(value).alias(spec.name)


def _windowed(spec: Aggregate, key: str) -> Column:
    """Merge partials over the days before each snapshot, within the spec's window."""
    start = -spec.window_days if spec.window_days else Window.unboundedPreceding
    w = Window.partitionBy(key).orderBy(DAY).rangeBetween(start, -1)
    merged = _MERGE[spec.kind](F.col(spec.name)).over(w)
    if spec.kind in ("count", "sum"):
        merged = F.coalesce(merged, F.lit(0))
    cast = {"int64": "bigint", "float64": "double"}[value_type(spec)]
    return merged.cast(cast).alias(spec.name)


def compute(ratings: DataFrame, fs: FeatureSet, from_day: int | None = None) -> DataFrame:
    """Snapshot rows for one feature set: key, snapshot_day, event_timestamp, features."""
    key = fs.key
    events = ratings.withColumn(DAY, F.floor(F.col("ts") / SECONDS_PER_DAY).cast("bigint"))
    daily = events.groupBy(key, DAY).agg(*(_partial(a) for a in fs.aggregates))

    offsets = [1, *(1 + w for w in fs.windows)]
    due = daily.select(
        key, F.explode(F.array(*(F.col(DAY) + o for o in offsets))).alias(DAY)
    ).distinct()
    if from_day is not None:
        # Same rule as `snapshot_days`: trim early days, add a boundary row at `from_day`.
        boundary = (
            daily.where(F.col(DAY) < from_day)
            .select(key)
            .distinct()
            .withColumn(DAY, F.lit(from_day).cast("bigint"))
        )
        due = due.where(F.col(DAY) >= from_day).unionByName(boundary).distinct()
    due = due.withColumn("_due", F.lit(True))
    timeline = daily.join(due, [key, DAY], "full_outer").fillna(False, subset=["_due"])

    snap = timeline.select(key, DAY, "_due", *(_windowed(a, key) for a in fs.aggregates)).where(
        F.col("_due")
    )
    if from_day is not None:
        snap = snap.where(F.col(DAY) >= from_day)
    for r in fs.ratios:
        num = F.coalesce(F.col(r.numerator).cast("double"), F.lit(0.0))
        den = F.coalesce(F.col(r.denominator).cast("double"), F.lit(0.0))
        snap = snap.withColumn(r.name, (num + r.prior * r.weight) / (den + r.weight))
    return snap.select(
        key,
        DAY,
        F.timestamp_seconds(F.col(DAY) * SECONDS_PER_DAY).alias(EVENT_TIMESTAMP),
        *fs.names,
    )


def item_content(movies: DataFrame) -> DataFrame:
    """Static item attributes, stamped at the epoch so they are valid at any query time."""
    return movies.select(
        "item_id",
        F.col("year").cast("bigint").alias("item_year"),
        F.col("genres").alias("item_genres"),
        F.timestamp_seconds(F.lit(0)).alias(EVENT_TIMESTAMP),
    )


def write_single(df: DataFrame, path: Path) -> None:
    """Write `df` as one Parquet file at `path` (via a temporary Spark output directory)."""
    tmp = path.with_suffix(".spark")
    shutil.rmtree(tmp, ignore_errors=True)
    df.coalesce(1).write.mode("overwrite").parquet(str(tmp))
    (part,) = tmp.glob("part-*.parquet")
    part.replace(path)
    shutil.rmtree(tmp)


def run(
    spark: SparkSession, processed_dir: Path, out_dir: Path, from_date: str | None = None
) -> dict[str, int]:
    """Compute every feature table from processed ratings and movies."""
    data_hash = str(verify_outputs(processed_dir)["data_hash"])
    from_day = None
    if from_date:
        start = datetime.fromisoformat(from_date).replace(tzinfo=UTC)
        from_day = int(start.timestamp()) // SECONDS_PER_DAY
    begin_stage(out_dir)
    ratings = spark.read.parquet(str(processed_dir / "ratings.parquet"))
    outputs: list[Path] = []
    rows: dict[str, int] = {}
    for fs in FEATURE_SETS:
        path = out_dir / f"{fs.entity}_features.parquet"
        write_single(compute(ratings, fs, from_day).orderBy(fs.key, DAY), path)
        outputs.append(path)
        rows[path.stem] = spark.read.parquet(str(path)).count()
        log.info("features written", extra={"table": path.name, "rows": rows[path.stem]})
    content = out_dir / "item_content.parquet"
    write_single(item_content(spark.read.parquet(str(processed_dir / "movies.parquet"))), content)
    outputs.append(content)
    rows[content.stem] = spark.read.parquet(str(content)).count()
    write_manifest(
        out_dir,
        stage="offline_features",
        config={
            "from_date": from_date,
            "feature_sets": {fs.entity: list(fs.names) for fs in FEATURE_SETS},
        },
        data_hash=data_hash,
        seed=None,
        outputs=outputs,
        extra={"rows": rows},
    )
    return rows


def main(argv: list[str] | None = None) -> None:
    """Compute offline feature snapshots."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--out-dir", type=Path, default=Path("data/features"))
    parser.add_argument("--from-date", default=None, help="drop snapshots before this UTC date")
    parser.add_argument("--driver-memory", default="4g")
    args = parser.parse_args(argv)
    configure_logging()
    spark = spark_session(args.driver_memory)
    try:
        run(spark, args.processed_dir, args.out_dir, args.from_date)
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
