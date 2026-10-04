"""Replay rating events to Redpanda in time order (`make simulate` uses this)."""

import argparse
import json
import logging
import time
from pathlib import Path

import polars as pl
from confluent_kafka import Producer

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import verify_outputs
from streamrank.streaming.events import TOPIC, replay_log

log = logging.getLogger(__name__)


def produce(log_df: pl.DataFrame, broker: str, topic: str = TOPIC, rate: float = 0.0) -> int:
    """Send events keyed by user (one partition per user keeps order); `rate` events/s, 0 = max."""
    producer = Producer({"bootstrap.servers": broker, "linger.ms": 5, "enable.idempotence": True})
    start = time.perf_counter()
    for n, row in enumerate(log_df.iter_rows(named=True), start=1):
        producer.produce(
            topic,
            key=str(row["user_id"]),
            value=json.dumps(row).encode(),
            timestamp=int(row["ts"]) * 1000,
        )
        if n % 10_000 == 0:
            producer.poll(0)
        if rate > 0:
            lag = n / rate - (time.perf_counter() - start)
            if lag > 0:
                time.sleep(lag)
    producer.flush(30)
    return log_df.height


def main(argv: list[str] | None = None) -> None:
    """Replay events after the training cutoff to the ratings topic."""
    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--split-dir", type=Path, default=s.data_dir / "split" / "full")
    p.add_argument("--broker", default=s.kafka_broker)
    p.add_argument("--limit", type=int, default=10_000)
    p.add_argument(
        "--rate", type=float, default=0.0, help="events per second, 0 = as fast as possible"
    )
    args = p.parse_args(argv)
    configure_logging(s.log_level)
    start_ts = int(verify_outputs(args.split_dir)["t1"])
    ratings = pl.read_parquet(s.data_dir / "processed" / "ratings.parquet")
    n = produce(replay_log(ratings, start_ts, args.limit), args.broker, rate=args.rate)
    log.info("replayed", extra={"events": n, "start_ts": start_ts})


if __name__ == "__main__":
    main()
