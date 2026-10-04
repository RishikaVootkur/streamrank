# 5. Feast with a DuckDB offline store and Redis online store

Date: 2026-10-04

## Status

Accepted

## Context

Ranker training needs point-in-time joins of snapshot features onto labeled examples, and serving needs the latest values per user and item in milliseconds. Both must read the same snapshot tables written by the Spark job (ADR 0004).

## Options considered

1. Hand-written joins in Polars and direct Redis writes. Few dependencies, but it re-implements what a feature store already does, and the online key layout would be custom.
2. Feast with the `file` (Dask) offline store and Redis online store. The `file` store is an alias for Dask, whose Python-side joins are documented as slow on large tables.
3. Feast with the DuckDB offline store and Redis online store.

## Decision

Option 3, with Feast 0.66.

- Feature views are generated from the shared specs (`feature_repo/features.py`), so a new feature needs no Feast edits: `user_stats`, `item_stats`, and `item_content` (year and genres, stamped at the epoch).
- TTL is unlimited. Each snapshot is valid until the next one, and Feast's "latest row at or before t" lookup is exactly the snapshot semantics.
- Materialization loads snapshot views from the trim date and static views from the epoch (`python -m streamrank.features.store`), because a static row stamped in 1970 falls outside any later start date.
- The Redis address comes from `REDIS_CONNECTION_STRING` (set by the Makefile from `.env`).
- Feast caps pandas below 3, redis-py below 8, and prometheus-client below 0.25. MLflow is the only other pandas user and accepts 2.x, so the whole project shares one lock with those caps.
- An integration test writes reference snapshots for synthetic data, runs `feast apply`, materializes into the Compose Redis, and checks that online values equal the latest snapshot and that historical joins at 80 query times (each event's own time, and three days later) equal a direct computation. A query at an event's own time never sees that day's events.

## Consequences

- `make features` runs Spark, `feast apply`, and materialization: about 2 minutes on MovieLens 32M, loading 288,533 entity keys into Redis.
- A first measurement shows a 200-item online lookup through the Feast SDK takes about 11 ms. Serving keeps item features in an in-process cache refreshed from Redis and fetches only user features per request (M7).

## Sources

- Feast DuckDB offline store: https://docs.feast.dev/reference/offline-stores/duckdb
- Feast Dask (file) offline store: https://docs.feast.dev/reference/offline-stores/dask
- Feast Redis online store: https://docs.feast.dev/reference/online-stores/redis
- Feast point-in-time joins: https://docs.feast.dev/getting-started/concepts/point-in-time-joins
- Feast CLI (apply, materialize): https://docs.feast.dev/reference/feast-cli-commands
- Feast 0.66.0 dependency constraints: https://raw.githubusercontent.com/feast-dev/feast/master/pyproject.toml
