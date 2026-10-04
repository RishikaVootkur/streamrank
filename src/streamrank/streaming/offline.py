"""Batch computation of session features over a replay log (the offline side of parity).

Implemented with Polars rolling windows, independently of the streaming code: for each
user, the features as of their last replayed event.
"""

import polars as pl

from streamrank.features.definitions import POSITIVE_RATING, SESSION_WINDOW_SECONDS


def session_features_at_last_event(log: pl.DataFrame) -> pl.DataFrame:
    """One row per user: session features over (last_ts - window, last_ts]."""
    df = log.with_row_index("order").with_columns(
        positive=(pl.col("rating") >= POSITIVE_RATING).cast(pl.Float64)
    )
    rolled = (
        df.sort("user_id", "ts", "order")
        .rolling(
            index_column="ts",
            period=f"{SESSION_WINDOW_SECONDS}i",
            closed="right",
            group_by="user_id",
        )
        .agg(
            session_n_events=pl.len().cast(pl.Float64),
            session_n_positive=pl.col("positive").sum(),
            session_mean_rating=pl.col("rating").mean(),
            session_last_item_id=pl.col("item_id").last().cast(pl.Float64),
            session_last_ts=pl.col("ts").last().cast(pl.Float64),
        )
    )
    return rolled.group_by("user_id", maintain_order=True).last().sort("user_id")
