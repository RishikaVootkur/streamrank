"""The replay event log: canonical order shared by the producer and offline checks."""

import numpy as np
import numpy.typing as npt
import polars as pl

TOPIC = "ratings"


def _id_hash(users: npt.NDArray[np.int64], items: npt.NDArray[np.int64]) -> npt.NDArray[np.uint64]:
    """Deterministic pseudo-random key for ordering same-second events (not by item ID)."""
    with np.errstate(over="ignore"):
        x = users.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + items.astype(np.uint64)
        x ^= x >> np.uint64(31)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(29)
    return x


def replay_log(ratings: pl.DataFrame, start_ts: int, limit: int | None = None) -> pl.DataFrame:
    """Events at or after `start_ts` in replay order: time, then a hash of (user, item)."""
    df = ratings.filter(pl.col("ts") >= start_ts).select("user_id", "item_id", "rating", "ts")
    key = _id_hash(df["user_id"].to_numpy(), df["item_id"].to_numpy())
    df = df.with_columns(_tie=pl.Series(key)).sort(["ts", "_tie"]).drop("_tie")
    return df.head(limit) if limit is not None else df
