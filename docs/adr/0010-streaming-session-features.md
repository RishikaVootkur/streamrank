# 10. Streaming session features with Redpanda and Quix Streams

Date: 2026-10-04

## Status

Accepted

## Context

Batch features (ADR 0004) are up to a day stale by design. A user who just rated three thrillers should see that reflected in the next request, both in session features and in the user tower's input sequence. The job must be deterministic enough that online values can be checked against an offline computation.

## Options considered

1. Built-in Quix Streams windows (sliding or session). Event-time, but outputs are keyed to window boundaries and depend on grace periods and closing rules, which makes exact parity awkward.
2. A custom stateful step: a per-user list of events inside the trailing 30 minutes, kept in the Quix state store, updated on each event.
3. Feast push sources for the online write. They write through `write_to_online_store`, and the Redis store skips writes whose timestamp is not newer than the stored one, which breaks on same-second MovieLens events.

## Decision

Option 2, writing directly to Redis.

- Definitions live in `streamrank.features.definitions` next to the batch features: `session_snapshot` (the reference) and `SessionState` (incremental). The window is `(t - 30 min, t]` as of the latest event, and features are count, positives, mean rating, last item, and last timestamp.
- The replay producer sends events after the training cutoff in a canonical order (time, then a hash of user and item, never item ID), keyed by user so each user's events stay in order on one partition. Message timestamps are event times.
- Each event updates `session:<user_id>` (a Redis hash) and appends to the user's serving sequence and seen set (`serving.state.append_event`). The next recommendation request therefore uses the new event in the user tower input and in the seen filter. Redelivered and out-of-order events are skipped, so at-least-once delivery cannot corrupt the sequence.
- The sink flushes at each commit, and the 0.25-second commit interval bounds click-to-Redis latency.
- Parity: an independent Polars rolling-window computation over the same replay log gives each user's features at their last event, which must equal Redis exactly.

## Results

10,000 MovieLens events after the training cutoff (631 users): 0 mismatches between online and offline session features. The job processed the replay in 6.5 s. With the job running, click-to-Redis latency over 20 probes: p50 529 ms, p95 540 ms, max 548 ms (each probe includes creating a producer).

## Consequences

- `make stream` runs the job against Compose Redpanda; `make stream-parity` reproduces the check. CI runs the same parity test on 10,000 synthetic events.
- Session features are served alongside batch statistics; the ranker was trained on batch features only, so session features influence recommendations through the updated user sequence rather than as ranker inputs.

## Sources

- Redpanda quick start and `rpk redpanda start` flags: https://docs.redpanda.com/current/get-started/quick-start/
- Quix Streams processing and stateful functions: https://quix.io/docs/quix-streams/processing.html
- Quix Streams windowing (event time, closing): https://quix.io/docs/quix-streams/windowing.html
- Quix Streams custom sinks: https://quix.io/docs/quix-streams/connectors/sinks/custom-sinks.html
- Feast push sources and online write deduplication: https://docs.feast.dev/reference/data-sources/push
