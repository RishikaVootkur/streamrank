"""Per-request ranker features computed with NumPy.

The same formulas as `ranking.features.build_features` (the offline Polars version used for
training), vectorized over one user's candidates so a request takes well under a
millisecond. A parity test checks both versions agree on every feature.
"""

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl

from streamrank.features.definitions import SECONDS_PER_DAY
from streamrank.ranking.features import FEATURES, ITEM_STATS

FloatArray = npt.NDArray[np.float64]
_INDEX = {name: i for i, name in enumerate(FEATURES)}


@dataclass(frozen=True)
class ItemTable:
    """Item statistics and content in catalog order (NaN where unknown)."""

    stats: dict[str, FloatArray]  # ITEM_STATS name -> values
    year: FloatArray
    genres: npt.NDArray[np.float32]  # (n_items, n_genres) multi-hot
    n_genres: FloatArray

    @classmethod
    def build(
        cls,
        item_ids: npt.NDArray[np.int64],
        stats: pl.DataFrame,
        year: FloatArray,
        genres: npt.NDArray[np.float32],
        has_movie: npt.NDArray[np.bool_] | None = None,
    ) -> "ItemTable":
        """Align a stats frame (item_id + ITEM_STATS columns) to the catalog order.

        `year`, `genres`, and `has_movie` must already be in catalog order. Items without a
        movie row get a NaN genre count, as the offline left join produces.
        """
        n = item_ids.size
        if (
            year.size != n
            or genres.shape[0] != n
            or (has_movie is not None and has_movie.size != n)
        ):
            raise ValueError("item content arrays must be in catalog order")
        aligned = pl.DataFrame({"item_id": item_ids}).join(
            stats, on="item_id", how="left", maintain_order="left", validate="1:1"
        )
        if aligned.height != n:
            raise ValueError("stats must have one row per item")
        cols = {
            c: (
                aligned[c].cast(pl.Float64).fill_null(np.nan).to_numpy()
                if c in aligned.columns
                else np.full(item_ids.size, np.nan)
            )
            for c in ITEM_STATS
        }
        n_genres = genres.sum(axis=1).astype(np.float64)
        if has_movie is not None:
            n_genres[~has_movie] = np.nan
        return cls(cols, year.astype(np.float64), genres, n_genres)


def _log1p0(x: FloatArray | float) -> FloatArray:
    return np.log1p(np.nan_to_num(np.asarray(x, dtype=np.float64), nan=0.0))


def _num(value: float | None) -> float:
    return np.nan if value is None else float(value)


def request_features(
    rows: npt.NDArray[np.int64],
    scores: npt.NDArray[np.floating],
    user: Mapping[str, float | None],
    profile: tuple[float, npt.NDArray[np.float32]] | None,
    *,
    items: ItemTable,
    now_ts: int,
) -> npt.NDArray[np.float32]:
    """Feature matrix (len(rows), len(FEATURES)) for one user's candidates, in rank order.

    `rows` must be the retrieval candidates in score order (as exported for training), so
    position i has retrieval rank i + 1.
    """
    if np.any(np.diff(np.asarray(scores, dtype=np.float64)) > 0):
        raise ValueError("candidates must be in non-increasing score order")
    day = float(SECONDS_PER_DAY)
    n = rows.size
    out = np.empty((n, len(FEATURES)), dtype=np.float64)
    out[:, _INDEX["retrieval_score"]] = scores
    out[:, _INDEX["retrieval_rank"]] = np.arange(1, n + 1)
    out[:, _INDEX["user_log_n_ratings"]] = _log1p0(_num(user.get("user_n_ratings")))
    out[:, _INDEX["user_positive_rate"]] = _num(user.get("user_positive_rate"))
    out[:, _INDEX["user_mean_rating"]] = _num(user.get("user_mean_rating"))
    out[:, _INDEX["user_n_ratings_30d"]] = _num(user.get("user_n_ratings_30d"))
    out[:, _INDEX["user_days_since_last"]] = (now_ts - _num(user.get("user_last_ts"))) / day
    out[:, _INDEX["user_tenure_days"]] = (now_ts - _num(user.get("user_first_ts"))) / day
    s = {k: v[rows] for k, v in items.stats.items()}
    out[:, _INDEX["item_log_n_ratings"]] = _log1p0(s["item_n_ratings"])
    out[:, _INDEX["item_positive_rate"]] = s["item_positive_rate"]
    out[:, _INDEX["item_mean_rating"]] = s["item_mean_rating"]
    out[:, _INDEX["item_log_n_ratings_7d"]] = _log1p0(s["item_n_ratings_7d"])
    out[:, _INDEX["item_log_n_ratings_30d"]] = _log1p0(s["item_n_ratings_30d"])
    out[:, _INDEX["item_age_days"]] = (now_ts - s["item_first_ts"]) / day
    out[:, _INDEX["item_days_since_last"]] = (now_ts - s["item_last_ts"]) / day
    year = items.year[rows]
    out[:, _INDEX["item_year"]] = year
    n_genres = items.n_genres[rows]
    out[:, _INDEX["item_n_genres"]] = n_genres
    if profile is None:
        mean_year, user_genres = np.nan, np.zeros(items.genres.shape[1], dtype=np.float32)
    else:
        mean_year, user_genres = profile
    dot = items.genres[rows].astype(np.float64) @ np.nan_to_num(user_genres.astype(np.float64))
    norm = np.sqrt(n_genres)
    out[:, _INDEX["genre_affinity"]] = np.divide(dot, norm, out=np.zeros(n), where=norm > 0)
    out[np.isnan(n_genres), _INDEX["genre_affinity"]] = 0.0
    out[:, _INDEX["year_gap"]] = np.abs(year - mean_year)
    return out.astype(np.float32)
