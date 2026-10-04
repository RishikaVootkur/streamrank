"""End-to-end streaming check: replay events, run the job, compare Redis with batch features.

1. Replay N events to a fresh topic.
2. Run the streaming job until it has processed all N.
3. For every replayed user, compare `session:<user>` in Redis with the offline Polars
   computation (`offline.session_features_at_last_event`).
4. Measure click-to-Redis latency: with the job running, produce single events and time
   how long until each user's session hash shows them.
"""

import argparse
import json
import logging
import math
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import redis

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import verify_outputs
from streamrank.features.definitions import SESSION_FEATURES
from streamrank.streaming.app import RedisSink, build_app, session_key
from streamrank.streaming.events import replay_log
from streamrank.streaming.offline import session_features_at_last_event
from streamrank.streaming.producer import produce

log = logging.getLogger(__name__)
PROBE_TIMEOUT_S = 30


def _redis_rows(client: redis.Redis, users: list[int]) -> dict[int, dict[str, float]]:
    pipe = client.pipeline(transaction=False)
    for u in users:
        pipe.hgetall(session_key(u))
    out = {}
    for u, raw in zip(users, pipe.execute(), strict=True):
        out[u] = {k.decode(): float(v) for k, v in raw.items()}
    return out


def compare(log_df: pl.DataFrame, client: redis.Redis) -> dict[str, Any]:
    """Count users whose online session features differ from the batch computation."""
    expected = session_features_at_last_event(log_df)
    users = expected["user_id"].to_list()
    online = _redis_rows(client, users)
    mismatches = []
    for row in expected.iter_rows(named=True):
        got = online.get(row["user_id"], {})
        for f in SESSION_FEATURES:
            a, b = got.get(f, math.inf), float(row[f])
            if not (
                math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9) or (math.isnan(a) and math.isnan(b))
            ):
                mismatches.append(
                    {"user_id": row["user_id"], "feature": f, "online": a, "offline": b}
                )
    return {
        "events": log_df.height,
        "users": len(users),
        "mismatches": len(mismatches),
        "examples": mismatches[:5],
    }


def measure_latency(
    broker: str, topic: str, client: redis.Redis, events: pl.DataFrame, n: int = 20
) -> dict[str, float]:
    """Produce single events while the job runs; time until each appears in Redis."""
    times: list[float] = []
    for row in events.head(n).iter_rows(named=True):
        # Fresh user IDs so each probe starts a new session the hash can only reflect once.
        probe = dict(row, user_id=10**9 + len(times))
        start = time.perf_counter()
        produce(pl.DataFrame([probe]), broker, topic)
        key = session_key(probe["user_id"])
        while time.perf_counter() - start < PROBE_TIMEOUT_S:
            if client.hget(key, "session_last_ts") is not None:
                break
            time.sleep(0.01)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "max_ms": float(max(times)),
        "probes": len(times),
    }


def run(
    log_df: pl.DataFrame,
    broker: str,
    client: redis.Redis,
    state_dir: Path,
    catalog: np.ndarray | None = None,
) -> dict[str, Any]:
    """Replay `log_df`, process it, compare, then measure latency with the job running."""
    topic = f"ratings-check-{uuid.uuid4().hex[:8]}"
    db = client.connection_pool.connection_kwargs.get("db", 0)
    host = client.connection_pool.connection_kwargs.get("host", "localhost")
    port = client.connection_pool.connection_kwargs.get("port", 6379)
    produce(log_df, broker, topic)
    sink = RedisSink(host, port, db, catalog, max_len=200)
    app = build_app(broker, sink, topic=topic, consumer_group=f"check-{topic}", state_dir=state_dir)
    t0 = time.perf_counter()
    app.run(count=log_df.height, timeout=60)
    processing_s = time.perf_counter() - t0
    result = compare(log_df, client)
    result["processing_seconds"] = round(processing_s, 2)

    # Latency: run the job as a separate process (Quix Streams needs the main thread) and
    # probe it with single events once it has caught up on the replayed ones.
    cmd = [
        sys.executable,
        "-m",
        "streamrank.streaming.app",
        "--broker",
        broker,
        "--topic",
        topic,
        "--consumer-group",
        f"live-{topic}",
        "--redis-db",
        str(db),
        "--state-dir",
        str(state_dir / "live"),
        "--timeout",
        "30",
    ]
    # Arguments are this interpreter and fixed module flags, not external input.
    with subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL) as live:  # noqa: S603
        try:
            _wait_until_caught_up(client, log_df)
            result["latency"] = measure_latency(broker, topic, client, log_df)
        finally:
            live.terminate()
    return result


def _wait_until_caught_up(client: redis.Redis, log_df: pl.DataFrame, seconds: float = 60) -> None:
    """Readiness probe: the live job has re-applied the last replayed event."""
    marker = log_df.tail(1).row(0, named=True)
    key = session_key(int(marker["user_id"]))
    client.delete(key)
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        if client.hget(key, "session_last_ts") is not None:
            return
        time.sleep(0.1)
    raise TimeoutError("the live streaming job did not catch up")


def main(argv: list[str] | None = None) -> None:
    """Run the streaming parity check on MovieLens events after the training cutoff."""
    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--split-dir", type=Path, default=s.data_dir / "split" / "full")
    p.add_argument("--events", type=int, default=10_000)
    p.add_argument("--out", type=Path, default=s.artifacts_dir / "streaming" / "parity.json")
    args = p.parse_args(argv)
    configure_logging(s.log_level)
    start_ts = int(verify_outputs(args.split_dir)["t1"])
    log_df = replay_log(
        pl.read_parquet(s.data_dir / "processed" / "ratings.parquet"), start_ts, args.events
    )
    client = redis.Redis(host=s.redis_host, port=s.redis_port, db=14)
    client.flushdb()
    result = run(log_df, s.kafka_broker, client, s.artifacts_dir / "stream-state-check")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if result["mismatches"]:
        raise SystemExit("online and offline session features differ")


if __name__ == "__main__":
    main()
