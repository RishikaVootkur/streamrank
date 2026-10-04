# 4. Point-in-time feature snapshots computed with Spark

Date: 2026-10-04

## Status

Accepted

## Context

The ranker needs user and item statistics (activity counts, recent counts, average rating, positive rate, first and last activity) that are correct as of each training example's time, and the same values must be served online. Computing them twice in different code (batch and streaming) invites training and serving skew. Training examples span months, so a lookup must return the value as of any query time, not just "now".

## Options considered

1. Compute features inside the training script with Polars. Simple, but nothing is shared with serving and nothing enforces point-in-time correctness.
2. One row per event with running aggregates ("as of this event"). Exact, but a rating then sits in its own features unless every consumer remembers to shift by one, and the table has one row per event (32M+ per entity type).
3. Daily snapshots at midnight UTC covering events from earlier days only, written the day after each active day and again when each windowed count expires. A feature store's "latest row at or before t" lookup then returns exact values for all days before t's day.

## Decision

Option 3, with declarative specs in `streamrank.features.definitions` used by both the Spark job and the streaming job.

- Each aggregate becomes a Spark window over day numbers, `rangeBetween(-W, -1)` for a W-day window or `rangeBetween(unboundedPreceding, -1)` for all history, applied to per-day partial aggregates merged with the days a snapshot is due. The `-1` upper bound keeps every event out of snapshots on its own day.
- Expiry rows (day d + 1 + W) make windowed counts drop to their true value when activity stops, so no stale value is ever visible.
- `--from-date` trims old snapshot rows to keep tables small, and adds a boundary row on that date for every entity active earlier, so lookups after the trim date still find full history.
- Smoothed ratios shrink toward a prior: mean rating toward 3.5 and positive rate toward 0.5, each with weight 5, so items with two ratings do not get extreme values.
- Static item attributes (year, genres) go to a separate table stamped at the epoch.
- Trade-off: same-day activity is not in these features (up to one day stale). Session features from the streaming job cover recent activity.
- The job runs in Docker (`eclipse-temurin:21.0.12.1_1-jre-noble`, Ubuntu 24.04 with Python 3.12, PySpark 4.2.0 installed with uv) so Java stays off the host. The official `spark` image ships Python 3.10, which this project does not support.
- Spark writes UTC microsecond timestamps (`spark.sql.session.timeZone=UTC`, `spark.sql.parquet.outputTimestampType=TIMESTAMP_MICROS`) so Feast and Polars read them without conversion.
- Parity tests run the Spark job in its image on synthetic data and require exact equality with the reference implementation for every snapshot row, with and without `--from-date`.

## Consequences

- Any new feature is a spec change, and parity tests check both pipelines automatically.
- Snapshot tables have a few rows per active entity day instead of one per event.
- CI builds the Spark image to run parity tests, which adds a few minutes per run (cached layers keep it short).

## Sources

- PySpark installation and supported versions: https://spark.apache.org/docs/latest/api/python/getting_started/install.html
- PySpark 4.2.0 release metadata: https://pypi.org/pypi/pyspark/json
- Spark Parquet options: https://spark.apache.org/docs/latest/sql-data-sources-parquet.html
- Spark official image (Python 3.10 base): https://hub.docker.com/_/spark
- Eclipse Temurin images: https://hub.docker.com/_/eclipse-temurin
- Feast point-in-time joins: https://docs.feast.dev/getting-started/concepts/point-in-time-joins
