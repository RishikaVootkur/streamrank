"""Streaming job: per-user session features and live serving state from replayed events.

For each rating event (keyed by user, in time order per partition):
- session features over the last 30 minutes are updated with `SessionState` (state kept in
  Quix Streams' per-key store) and written to the Redis hash `session:<user_id>`;
- the user's serving sequence and seen set are extended (`serving.state.append_event`), so
  the next recommendation request already reflects the event.
"""

import argparse
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np
import redis
from quixstreams import Application, State
from quixstreams.sinks.base import BatchingSink, SinkBatch

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.features.definitions import POSITIVE_RATING, Event, SessionState
from streamrank.serving.state import append_event
from streamrank.streaming.events import TOPIC

log = logging.getLogger(__name__)


def session_key(user_id: int) -> str:
    return f"session:{user_id}"


def update_session(value: dict[str, Any], state: State[Any, Any]) -> dict[str, Any]:
    """Advance the user's session state with one event and attach the new features."""
    stored = state.get("events", []) or []
    session = SessionState(events=[(int(t), int(i), float(r)) for t, i, r in stored])
    event = Event(
        int(value["user_id"]), int(value["item_id"]), float(value["rating"]), int(value["ts"])
    )
    features = session.update(event)
    state.set("events", [list(e) for e in session.events])
    return {**value, **features}


class RedisSink(BatchingSink):
    """Writes session hashes and appends events to the serving state, one pipeline per batch."""

    def __init__(self, host: str, port: int, db: int, catalog: np.ndarray | None, max_len: int):
        super().__init__()
        self.host, self.port, self.db = host, port, db
        self.token = (
            {int(item): i + 1 for i, item in enumerate(catalog)} if catalog is not None else {}
        )
        self.max_len = max_len
        self.client: redis.Redis | None = None

    def setup(self) -> None:
        self.client = redis.Redis(host=self.host, port=self.port, db=self.db)
        self.client.ping()

    def write(self, batch: SinkBatch) -> None:
        if self.client is None:
            raise RuntimeError("sink not set up")
        pipe = self.client.pipeline(transaction=False)
        events = []
        for item in batch:
            v = item.value
            fields = {
                k: ("nan" if isinstance(x, float) and math.isnan(x) else x)
                for k, x in v.items()
                if k.startswith("session_")
            }
            pipe.hset(session_key(int(v["user_id"])), mapping=fields)
            events.append(v)
        pipe.execute()
        for v in events:
            token = self.token.get(int(v["item_id"]))
            if token is not None:  # items first seen after the cutoff are not in the index
                append_event(
                    self.client,
                    int(v["user_id"]),
                    token=token,
                    positive=float(v["rating"]) >= POSITIVE_RATING,
                    ts=int(v["ts"]),
                    max_len=self.max_len,
                )


def build_app(
    broker: str,
    sink: RedisSink,
    *,
    topic: str = TOPIC,
    consumer_group: str = "session-features",
    state_dir: Path = Path("artifacts/stream-state"),
) -> Application:
    """The Quix Streams application wired to the ratings topic and the Redis sink."""
    app = Application(
        broker_address=broker,
        consumer_group=consumer_group,
        auto_offset_reset="earliest",
        state_dir=str(state_dir),
        # The sink flushes at each commit, so this bounds click-to-Redis latency.
        commit_interval=0.25,
    )
    source = app.topic(topic, value_deserializer="json", key_deserializer="str")
    sdf = app.dataframe(source)
    sdf = sdf.apply(update_session, stateful=True)
    sdf.sink(sink)
    return app


def main(argv: list[str] | None = None) -> None:
    """Run the streaming session-feature job."""
    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--broker", default=s.kafka_broker)
    p.add_argument("--serving-dir", type=Path, default=s.artifacts_dir / "serving")
    p.add_argument("--consumer-group", default="session-features")
    p.add_argument("--topic", default=TOPIC)
    p.add_argument("--redis-db", type=int, default=0)
    p.add_argument("--state-dir", type=Path, default=s.artifacts_dir / "stream-state")
    p.add_argument("--count", type=int, default=0, help="stop after this many events (0 = run)")
    p.add_argument("--timeout", type=float, default=0.0, help="stop after this many idle seconds")
    args = p.parse_args(argv)
    configure_logging(s.log_level)
    items_path = args.serving_dir / "items.npz"
    catalog = np.load(items_path, allow_pickle=True)["item_ids"] if items_path.exists() else None
    sink = RedisSink(s.redis_host, s.redis_port, args.redis_db, catalog, max_len=200)
    app = build_app(
        args.broker,
        sink,
        topic=args.topic,
        consumer_group=args.consumer_group,
        state_dir=args.state_dir,
    )
    app.run(count=args.count, timeout=args.timeout)  # 0 and 0 mean run until stopped


if __name__ == "__main__":
    main()
