"""Per-user serving state in Redis: the recent event sequence and the set of seen items.

`seq:<user_id>` holds the user's last `max_len` events as packed int32 rows
(token, positive, gap bucket, timestamp). `seen:<user_id>` holds the sorted catalog
indices of every item the user has rated. The batch loader fills both from training
history; the streaming job appends new events with `append_event`.
"""

from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
import redis

from streamrank.models.sequences import Sequences, gap_bucket

IntArray = npt.NDArray[np.int64]
SEQ_COLUMNS = 4  # token, positive, gap, ts


def seq_key(user_id: int) -> str:
    return f"seq:{user_id}"


def seen_key(user_id: int) -> str:
    return f"seen:{user_id}"


def pack(rows: npt.NDArray[np.int64]) -> bytes:
    return np.ascontiguousarray(rows, dtype=np.int32).tobytes()


def unpack(blob: bytes | None, columns: int) -> npt.NDArray[np.int32]:
    if not blob:
        return np.zeros((0, columns), dtype=np.int32)
    return np.frombuffer(blob, dtype=np.int32).reshape(-1, columns)


@dataclass(frozen=True)
class UserState:
    """What the API needs about a user to build the user tower input and filter items."""

    tokens: IntArray
    positive: IntArray
    gaps: IntArray
    last_ts: int | None
    seen: IntArray  # catalog indices

    @property
    def known(self) -> bool:
        return self.tokens.size > 0


def bulk_load(
    client: redis.Redis,
    user_ids: IntArray,
    seqs: Sequences,
    *,
    max_len: int,
    batch: int = 2000,
) -> int:
    """Write every user's last `max_len` events and seen set; returns users written."""
    pipe = client.pipeline(transaction=False)
    lengths = seqs.lengths()
    for u, user_id in enumerate(user_ids):
        lo, hi = int(seqs.indptr[u]), int(seqs.indptr[u + 1])
        if hi == lo:
            continue
        start = max(lo, hi - max_len)
        rows = np.stack(
            [
                seqs.tokens[start:hi],
                seqs.positive[start:hi].astype(np.int64),
                seqs.gaps[start:hi],
                seqs.ts[start:hi],
            ],
            axis=1,
        )
        pipe.set(seq_key(int(user_id)), pack(rows))
        pipe.set(seen_key(int(user_id)), pack(np.sort(seqs.tokens[lo:hi] - 1)))
        if (u + 1) % batch == 0:
            pipe.execute()
    pipe.execute()
    return int((lengths > 0).sum())


def read(client: redis.Redis, user_id: int) -> UserState:
    """Fetch a user's sequence and seen set in one round trip."""
    seq_blob, seen_blob = cast(
        list[bytes | None], client.mget([seq_key(user_id), seen_key(user_id)])
    )
    seq = unpack(seq_blob, SEQ_COLUMNS).astype(np.int64)
    seen = unpack(seen_blob, 1).ravel().astype(np.int64)
    return UserState(
        tokens=seq[:, 0],
        positive=seq[:, 1],
        gaps=seq[:, 2],
        last_ts=int(seq[-1, 3]) if seq.size else None,
        seen=seen,
    )


def append_event(
    client: redis.Redis, user_id: int, *, token: int, positive: bool, ts: int, max_len: int
) -> bool:
    """Add one event to a user's sequence (keeping the last `max_len`) and seen set.

    Returns False without writing when the event is a redelivery (the item is already in
    the seen set: MovieLens has one rating per user and item) or arrives out of order (older
    than the user's latest stored event), so the stored sequence always matches what the
    offline pipeline would build from the same events.
    """
    seq_blob, seen_blob = cast(
        list[bytes | None], client.mget([seq_key(user_id), seen_key(user_id)])
    )
    old = unpack(seq_blob, SEQ_COLUMNS).astype(np.int64)
    seen = unpack(seen_blob, 1).ravel().astype(np.int64)
    if np.isin(token - 1, seen) or (old.size and ts < old[-1, 3]):
        return False
    gap = 1 if old.size == 0 else int(gap_bucket(np.array([ts - old[-1, 3]]))[0])
    keep = old[max(0, len(old) - (max_len - 1)) :] if max_len > 1 else old[:0]
    rows = np.concatenate([keep, np.array([[token, int(positive), gap, ts]], dtype=np.int64)])
    pipe = client.pipeline(transaction=True)
    pipe.set(seq_key(user_id), pack(rows))
    pipe.set(seen_key(user_id), pack(np.union1d(seen, np.array([token - 1]))))
    pipe.execute()
    return True
