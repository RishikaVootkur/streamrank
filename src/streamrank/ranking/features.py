"""Ranker features for (user, candidate item) pairs.

User and item statistics come from the Feast offline store, joined point-in-time at the
evaluation cutoff, so they use only data from days before it. Content features come from
the movie table, and user-item affinity features from the user's training history.

Feast is imported only inside `stats_at_cutoff`: Feast loads PyTorch when it is installed,
and on macOS PyTorch cannot share a process with LightGBM (see dev notes), so the ranker
training process must be able to import this module without pulling Feast in.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import polars as pl

from streamrank.data.columns import GENRES
from streamrank.features.definitions import POSITIVE_RATING, SECONDS_PER_DAY

if TYPE_CHECKING:
    from feast import FeatureStore

USER_STATS = (
    "user_n_ratings",
    "user_n_positive",
    "user_n_ratings_7d",
    "user_n_ratings_30d",
    "user_mean_rating",
    "user_positive_rate",
    "user_first_ts",
    "user_last_ts",
)
ITEM_STATS = (
    "item_n_ratings",
    "item_n_positive",
    "item_n_ratings_7d",
    "item_n_ratings_30d",
    "item_mean_rating",
    "item_positive_rate",
    "item_first_ts",
    "item_last_ts",
)
FEATURES: tuple[str, ...] = (
    "retrieval_score",
    "retrieval_rank",
    "user_log_n_ratings",
    "user_positive_rate",
    "user_mean_rating",
    "user_n_ratings_30d",
    "user_days_since_last",
    "user_tenure_days",
    "item_log_n_ratings",
    "item_positive_rate",
    "item_mean_rating",
    "item_log_n_ratings_7d",
    "item_log_n_ratings_30d",
    "item_age_days",
    "item_days_since_last",
    "item_year",
    "item_n_genres",
    "genre_affinity",
    "year_gap",
)


def _cutoff_frame(key: str, ids: Sequence[int], cutoff_ts: int) -> pd.DataFrame:
    ts = datetime.fromtimestamp(cutoff_ts, tz=UTC)
    return pd.DataFrame({key: np.asarray(ids, dtype=np.int64), "event_timestamp": ts})


def stats_at_cutoff(
    store: "FeatureStore", cutoff_ts: int, users: Sequence[int], items: Sequence[int]
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """User and item statistics from Feast as of `cutoff_ts` (point-in-time join)."""
    from streamrank.features.store import historical_features  # noqa: PLC0415

    u = historical_features(
        store,
        _cutoff_frame("user_id", users, cutoff_ts),
        [f"user_stats:{n}" for n in USER_STATS],
    )
    i = historical_features(
        store,
        _cutoff_frame("item_id", items, cutoff_ts),
        [f"item_stats:{n}" for n in ITEM_STATS],
    )
    return (
        pl.from_pandas(u.drop(columns=["event_timestamp"])),
        pl.from_pandas(i.drop(columns=["event_timestamp"])),
    )


def user_profiles(history: pl.DataFrame, movies: pl.DataFrame) -> pl.DataFrame:
    """Per-user genre distribution over positives and mean release year of rated movies."""
    m = movies.select("item_id", "year", "genres")
    h = history.join(m, on="item_id", how="left")
    year = h.group_by("user_id").agg(user_mean_year=pl.col("year").mean())
    pos = h.filter(pl.col("rating") >= POSITIVE_RATING).explode("genres")
    counts = pos.filter(pl.col("genres").is_in(list(GENRES))).group_by("user_id", "genres").len()
    wide = counts.pivot(on="genres", index="user_id", values="len").fill_null(0)
    for g in GENRES:
        if g not in wide.columns:
            wide = wide.with_columns(pl.lit(0, pl.UInt32).alias(g))
    total = pl.sum_horizontal([pl.col(g) for g in GENRES])
    norm = pl.sum_horizontal([pl.col(g).cast(pl.Float64) ** 2 for g in GENRES]).sqrt()
    wide = wide.with_columns(
        [(pl.col(g) / pl.when(norm > 0).then(norm).otherwise(1.0)).alias(f"ug_{g}") for g in GENRES]
    ).with_columns(user_n_pos_genres=total)
    return year.join(wide.select("user_id", *[f"ug_{g}" for g in GENRES]), on="user_id", how="left")


def build_features(
    candidates: pl.DataFrame,
    user_stats: pl.DataFrame,
    item_stats: pl.DataFrame,
    profiles: pl.DataFrame,
    movies: pl.DataFrame,
    *,
    cutoff_ts: int,
) -> pl.DataFrame:
    """Join every feature onto the candidates; returns candidates plus FEATURES columns."""
    day = float(SECONDS_PER_DAY)
    # Entities without a snapshot before the cutoff get nulls (filled below where needed).
    user_stats = user_stats.with_columns(
        [pl.lit(None, pl.Float64).alias(c) for c in USER_STATS if c not in user_stats.columns]
    )
    item_stats = item_stats.with_columns(
        [pl.lit(None, pl.Float64).alias(c) for c in ITEM_STATS if c not in item_stats.columns]
    )
    genre_cols = [
        pl.col("genres").list.contains(g).cast(pl.Float64).alias(f"ig_{g}") for g in GENRES
    ]
    items = (
        movies.select("item_id", "year", "genres")
        .with_columns(genre_cols)
        .with_columns(item_n_genres=pl.sum_horizontal([pl.col(f"ig_{g}") for g in GENRES]))
    )
    df = (
        candidates.join(user_stats, on="user_id", how="left")
        .join(item_stats, on="item_id", how="left")
        .join(profiles, on="user_id", how="left")
        .join(items.drop("genres"), on="item_id", how="left")
    )
    dot = pl.sum_horizontal(
        [pl.col(f"ug_{g}").fill_null(0) * pl.col(f"ig_{g}").fill_null(0) for g in GENRES]
    )
    item_norm = pl.col("item_n_genres").sqrt()
    df = df.with_columns(
        user_log_n_ratings=pl.col("user_n_ratings").fill_null(0).log1p(),
        user_days_since_last=(cutoff_ts - pl.col("user_last_ts")) / day,
        user_tenure_days=(cutoff_ts - pl.col("user_first_ts")) / day,
        item_log_n_ratings=pl.col("item_n_ratings").fill_null(0).log1p(),
        item_log_n_ratings_7d=pl.col("item_n_ratings_7d").fill_null(0).log1p(),
        item_log_n_ratings_30d=pl.col("item_n_ratings_30d").fill_null(0).log1p(),
        item_age_days=(cutoff_ts - pl.col("item_first_ts")) / day,
        item_days_since_last=(cutoff_ts - pl.col("item_last_ts")) / day,
        item_year=pl.col("year").cast(pl.Float64),
        genre_affinity=pl.when(item_norm > 0).then(dot / item_norm).otherwise(0.0),
        year_gap=(pl.col("year") - pl.col("user_mean_year")).abs(),
    )
    keep = list(candidates.columns) + [f for f in FEATURES if f not in candidates.columns]
    return df.select(keep).with_columns([pl.col(f).cast(pl.Float32) for f in FEATURES])
