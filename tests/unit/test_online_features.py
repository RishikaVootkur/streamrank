"""The per-request NumPy features must equal the offline Polars features used in training."""

import numpy as np
import polars as pl
import pytest

from streamrank.data.columns import GENRES
from streamrank.ranking.features import FEATURES, ITEM_STATS, USER_STATS, build_features
from streamrank.ranking.online import ItemTable, request_features

DAY = 86_400
NOW = 20_000 * DAY


def _movies(n: int, rng: np.random.Generator) -> pl.DataFrame:
    genres = [[g for g in GENRES if rng.random() < 0.15] for _ in range(n)]
    years = [None if rng.random() < 0.1 else int(rng.integers(1930, 2020)) for _ in range(n)]
    return pl.DataFrame(
        {
            "item_id": np.arange(1, n + 1),
            "title": [str(i) for i in range(n)],
            "year": years,
            "genres": genres,
        },
        schema_overrides={"year": pl.Int64},
    )


def _item_stats(n: int, rng: np.random.Generator) -> pl.DataFrame:
    data: dict[str, object] = {"item_id": np.arange(1, n + 1)}
    for c in ITEM_STATS:
        vals = rng.integers(1, 5000, n).astype(float)
        if c.endswith("_ts"):
            vals = NOW - rng.integers(1, 3000, n) * DAY
        if c in ("item_mean_rating", "item_positive_rate"):
            vals = rng.random(n) * 5
        vals = [None if rng.random() < 0.05 else v for v in vals]
        data[c] = vals
    return pl.DataFrame(data, schema={"item_id": pl.Int64} | {c: pl.Float64 for c in ITEM_STATS})


@pytest.mark.parametrize("with_profile,with_stats", [(True, True), (False, True), (True, False)])
def test_numpy_features_match_polars(with_profile: bool, with_stats: bool) -> None:
    rng = np.random.default_rng(0)
    n_items, k, user_id = 300, 50, 7
    movies = _movies(n_items, rng)
    item_stats = _item_stats(n_items, rng)
    rows = rng.choice(n_items, size=k, replace=False)  # catalog index = item_id - 1
    scores = np.sort(rng.random(k))[::-1].astype(np.float32)
    user = {c: float(rng.integers(1, 100)) for c in USER_STATS}
    user["user_first_ts"], user["user_last_ts"] = float(NOW - 900 * DAY), float(NOW - 3 * DAY)
    user["user_mean_rating"], user["user_positive_rate"] = 3.7, 0.6
    if not with_stats:
        user = dict.fromkeys(USER_STATS)
    genre_vec = rng.random(len(GENRES)).astype(np.float32)
    profile = (1995.5, genre_vec / np.linalg.norm(genre_vec)) if with_profile else None

    candidates = pl.DataFrame(
        {
            "user_id": [user_id] * k,
            "item_id": rows + 1,
            "retrieval_score": scores,
            "retrieval_rank": np.arange(1, k + 1),
            "label": [0] * k,
            "n_relevant": [1] * k,
        }
    )
    user_df = pl.DataFrame(
        {"user_id": [user_id], **{c: [user[c]] for c in USER_STATS}},
        schema={"user_id": pl.Int64} | {c: pl.Float64 for c in USER_STATS},
    )
    prof_df = pl.DataFrame(
        {
            "user_id": [user_id],
            "user_mean_year": [profile[0] if profile else None],
            **{
                f"ug_{g}": [float(profile[1][i]) if profile else None] for i, g in enumerate(GENRES)
            },
        },
        schema={"user_id": pl.Int64, "user_mean_year": pl.Float64}
        | {f"ug_{g}": pl.Float64 for g in GENRES},
    )
    offline = build_features(candidates, user_df, item_stats, prof_df, movies, cutoff_ts=NOW)
    expected = offline.select(FEATURES).to_numpy()

    m = movies.sort("item_id")
    genres = np.array(
        [[1.0 if g in gs else 0.0 for g in GENRES] for gs in m["genres"].to_list()],
        dtype=np.float32,
    )
    table = ItemTable.build(
        np.arange(1, n_items + 1),
        item_stats.sample(fraction=1.0, shuffle=True, seed=1),  # row order must not matter
        m["year"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        genres,
    )
    got = request_features(rows, scores, user, profile, items=table, now_ts=NOW)
    assert got.shape == expected.shape
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=1e-6, equal_nan=True)


def test_rejects_unsorted_candidates_and_misaligned_tables() -> None:
    rng = np.random.default_rng(1)
    stats = _item_stats(10, rng)
    genres = np.zeros((10, len(GENRES)), dtype=np.float32)
    table = ItemTable.build(np.arange(1, 11), stats, np.full(10, 2000.0), genres)
    with pytest.raises(ValueError, match="score order"):
        request_features(np.array([0, 1]), np.array([0.1, 0.9]), {}, None, items=table, now_ts=NOW)
    with pytest.raises(ValueError, match="catalog order"):
        ItemTable.build(np.arange(1, 11), stats, np.full(9, 2000.0), genres)
    with pytest.raises(pl.exceptions.ComputeError):
        ItemTable.build(
            np.arange(1, 11), pl.concat([stats, stats.head(1)]), np.full(10, 2000.0), genres
        )


def test_items_without_movie_rows_match_offline_nulls() -> None:
    rng = np.random.default_rng(2)
    movies = _movies(20, rng)
    stats = _item_stats(20, rng)
    has_movie = np.ones(20, dtype=bool)
    has_movie[[3, 7]] = False
    kept = movies.filter(~pl.col("item_id").is_in([4, 8]))
    rows = np.arange(20)
    scores = np.linspace(1, 0, 20)
    cand = pl.DataFrame(
        {
            "user_id": [1] * 20,
            "item_id": rows + 1,
            "retrieval_score": scores,
            "retrieval_rank": rows + 1,
            "label": [0] * 20,
            "n_relevant": [1] * 20,
        }
    )
    empty_user = pl.DataFrame({"user_id": [1]}, schema={"user_id": pl.Int64})
    empty_prof = pl.DataFrame({"user_id": [1]}, schema={"user_id": pl.Int64}).with_columns(
        user_mean_year=pl.lit(None, pl.Float64),
        **{f"ug_{g}": pl.lit(None, pl.Float64) for g in GENRES},
    )
    expected = build_features(cand, empty_user, stats, empty_prof, kept, cutoff_ts=NOW)
    m = movies.sort("item_id")
    genres = np.array(
        [[1.0 if g in gs else 0.0 for g in GENRES] for gs in m["genres"].to_list()],
        dtype=np.float32,
    )
    genres[~has_movie] = 0
    year = m["year"].cast(pl.Float64).fill_null(np.nan).to_numpy().copy()
    year[~has_movie] = np.nan
    table = ItemTable.build(np.arange(1, 21), stats, year, genres, has_movie)
    got = request_features(rows, scores, {}, None, items=table, now_ts=NOW)
    np.testing.assert_allclose(
        got, expected.select(FEATURES).to_numpy(), rtol=1e-6, atol=1e-6, equal_nan=True
    )
