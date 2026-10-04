"""Synthetic data with the MovieLens schema, used by tests and CI.

The generator plants learnable structure so that models behave meaningfully on it:

- Item popularity follows a Zipf-like law and user activity is log-normal (long tails).
- Users and items have latent taste vectors; users pick items with probability
  proportional to popularity times exp(affinity), so personalization beats popularity.
- Each user's taste drifts from a start vector to an end vector, and items are consumed
  in an order that follows that drift, so recent history predicts the next item.
- Genres and tags are derived from the item latent vectors, so content features carry signal.
"""

import argparse
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl

from streamrank.data.columns import (
    GENRES,
    LINKS_FILE,
    MOVIES_FILE,
    NO_GENRES,
    RATINGS_FILE,
    TAGS_FILE,
)

# Rates of quirks that mirror the real data.
_NO_GENRE_RATE = 0.02
_SECOND_GENRE_MIN_SIM = 0.3
_ARTICLE_TITLE_RATE = 0.05
_NO_YEAR_RATE = 0.01
_MISSING_TMDB_RATE = 0.01
_WHOLE_STAR_USER_RATE = 0.2

_TAG_WORDS = (
    "classic",
    "funny",
    "dark",
    "visually stunning",
    "twist ending",
    "atmospheric",
    "based on a book",
    "quirky",
    "thought-provoking",
    "violent",
    "romantic",
    "slow",
    "great soundtrack",
    "cult film",
    "feel-good",
    "space",
)


@dataclass(frozen=True)
class SyntheticConfig:
    """Size and shape of the synthetic dataset."""

    n_users: int = 2_000
    n_items: int = 1_000
    mean_interactions: float = 40.0
    min_interactions: int = 5
    latent_dim: int = 8
    affinity_strength: float = 2.0
    popularity_exponent: float = 1.0
    tag_rate: float = 0.03
    start: datetime = datetime(2015, 1, 1, tzinfo=UTC)
    end: datetime = datetime(2023, 10, 1, tzinfo=UTC)
    seed: int = 42

    def __post_init__(self) -> None:
        if min(self.n_users, self.n_items, self.min_interactions, self.latent_dim) < 1:
            raise ValueError("sizes must be positive")
        if self.n_items < 2 * self.min_interactions:
            raise ValueError("n_items must be at least twice min_interactions")
        if not self.start < self.end:
            raise ValueError("start must be before end")


@dataclass(frozen=True)
class SyntheticData:
    """Synthetic tables with the same columns as the MovieLens CSV files."""

    ratings: pl.DataFrame
    movies: pl.DataFrame
    tags: pl.DataFrame
    links: pl.DataFrame

    def write_csv(self, out_dir: Path) -> None:
        """Write the four tables as MovieLens-style CSV files into `out_dir`."""
        out_dir.mkdir(parents=True, exist_ok=True)
        self.ratings.write_csv(out_dir / RATINGS_FILE)
        self.movies.write_csv(out_dir / MOVIES_FILE)
        self.tags.write_csv(out_dir / TAGS_FILE)
        self.links.write_csv(out_dir / LINKS_FILE, null_value="")


FloatArray = npt.NDArray[np.float64]


def _unit_rows(x: FloatArray) -> FloatArray:
    out: FloatArray = x / np.linalg.norm(x, axis=1, keepdims=True)
    return out


def _round_half(x: FloatArray) -> FloatArray:
    out: FloatArray = np.clip(np.round(x * 2.0) / 2.0, 0.5, 5.0)
    return out


def _movies(
    cfg: SyntheticConfig,
    rng: np.random.Generator,
    movie_ids: np.ndarray,
    item_vecs: FloatArray,
    first_rating_year: np.ndarray,
) -> pl.DataFrame:
    genre_vecs = _unit_rows(rng.normal(size=(len(GENRES), cfg.latent_dim)))
    sims = item_vecs @ genre_vecs.T
    # A movie is never released after its first rating.
    years = np.minimum(rng.integers(1930, cfg.end.year + 1, size=cfg.n_items), first_rating_year)
    genre_strs: list[str] = []
    titles: list[str] = []
    for i in range(cfg.n_items):
        if rng.random() < _NO_GENRE_RATE:
            genre_strs.append(NO_GENRES)
        else:
            order = np.argsort(-sims[i])
            chosen = [order[0], *[g for g in order[1:3] if sims[i, g] > _SECOND_GENRE_MIN_SIM]]
            genre_strs.append("|".join(GENRES[g] for g in sorted(chosen)))
        name = f"Movie {movie_ids[i]}"
        if rng.random() < _ARTICLE_TITLE_RATE:
            name = f"{name}, The"
        # A few titles lack a year, as in the real data.
        titles.append(name if rng.random() < _NO_YEAR_RATE else f"{name} ({years[i]})")
    return pl.DataFrame(
        {"movieId": movie_ids, "title": titles, "genres": genre_strs},
        schema={"movieId": pl.Int64, "title": pl.String, "genres": pl.String},
    )


def _links(rng: np.random.Generator, movie_ids: np.ndarray) -> pl.DataFrame:
    n = len(movie_ids)
    imdb = rng.choice(np.arange(1, 9_999_999), size=n, replace=False)
    tmdb = rng.choice(np.arange(1, 1_000_000), size=n, replace=False)
    tmdb_missing = rng.random(n) < _MISSING_TMDB_RATE
    return pl.DataFrame(
        {
            "movieId": movie_ids,
            # The real file stores IMDb IDs as 7-digit zero-padded strings.
            "imdbId": [f"{i:07d}" for i in imdb],
            "tmdbId": pl.Series(tmdb, dtype=pl.Int64).scatter(np.flatnonzero(tmdb_missing), None),
        },
        schema={"movieId": pl.Int64, "imdbId": pl.String, "tmdbId": pl.Int64},
    )


def _first_rating_year(ratings: pl.DataFrame, movie_ids: np.ndarray) -> np.ndarray:
    first = ratings.group_by("movieId").agg(pl.col("timestamp").min())
    years = dict(
        zip(
            first["movieId"].to_list(),
            first.select(pl.from_epoch("timestamp").dt.year())["timestamp"].to_list(),
            strict=True,
        )
    )
    return np.array([years.get(int(m), 9999) for m in movie_ids])


def generate(cfg: SyntheticConfig | None = None) -> SyntheticData:
    """Generate a synthetic dataset. The same config always yields the same data."""
    cfg = cfg or SyntheticConfig()
    rng = np.random.default_rng(cfg.seed)
    movie_ids = np.sort(rng.choice(np.arange(1, cfg.n_items * 3 + 1), cfg.n_items, replace=False))
    item_vecs = _unit_rows(rng.normal(size=(cfg.n_items, cfg.latent_dim)))
    ranks = rng.permutation(cfg.n_items) + 1
    log_pop = -cfg.popularity_exponent * np.log(ranks)

    user_start = _unit_rows(rng.normal(size=(cfg.n_users, cfg.latent_dim)))
    drift = rng.normal(scale=0.6, size=(cfg.n_users, cfg.latent_dim))
    user_end = _unit_rows(user_start + drift)
    user_mid = _unit_rows(user_start + user_end)

    sigma = 1.0
    mu = np.log(cfg.mean_interactions) - sigma**2 / 2
    counts = np.clip(
        np.round(rng.lognormal(mu, sigma, cfg.n_users)).astype(int),
        cfg.min_interactions,
        cfg.n_items // 2,
    )

    t0, t1 = cfg.start.timestamp(), cfg.end.timestamp()
    span = t1 - t0
    whole_star_users = rng.random(cfg.n_users) < _WHOLE_STAR_USER_RATE

    user_col: list[np.ndarray] = []
    item_col: list[np.ndarray] = []
    rating_col: list[np.ndarray] = []
    ts_col: list[np.ndarray] = []
    for u in range(cfg.n_users):
        n = counts[u]
        affinity = item_vecs @ user_mid[u]
        keys = log_pop + cfg.affinity_strength * affinity + rng.gumbel(size=cfg.n_items)
        chosen = np.argpartition(-keys, n - 1)[:n]
        # Order consumption along the user's taste drift so sequence order carries signal.
        progress = item_vecs[chosen] @ (user_end[u] - user_start[u])
        chosen = chosen[np.argsort(progress + rng.normal(scale=0.3, size=n))]

        start = t0 + rng.random() * span * 0.9
        duration = min(rng.exponential(span * 0.1) + 3600.0, t1 - start)
        times = np.sort(start + rng.random(n) * duration).astype(np.int64)

        taste = item_vecs[chosen] @ (user_start[u] + user_end[u]) / 2
        raw = 3.5 + 1.2 * taste + rng.normal(scale=0.7, size=n)
        ratings = np.clip(np.round(raw), 1.0, 5.0) if whole_star_users[u] else _round_half(raw)

        user_col.append(np.full(n, u + 1))
        item_col.append(movie_ids[chosen])
        rating_col.append(ratings)
        ts_col.append(times)

    ratings_df = pl.DataFrame(
        {
            "userId": np.concatenate(user_col),
            "movieId": np.concatenate(item_col),
            "rating": np.concatenate(rating_col),
            "timestamp": np.concatenate(ts_col),
        },
        schema={
            "userId": pl.Int64,
            "movieId": pl.Int64,
            "rating": pl.Float64,
            "timestamp": pl.Int64,
        },
    ).sort(["userId", "timestamp"])

    tag_rows = ratings_df.filter(
        pl.Series(rng.random(ratings_df.height) < cfg.tag_rate, dtype=pl.Boolean)
    )
    id_to_index = {int(m): i for i, m in enumerate(movie_ids)}
    tag_dirs = _unit_rows(rng.normal(size=(len(_TAG_WORDS), cfg.latent_dim)))
    tag_words = [
        _TAG_WORDS[int(np.argmax(tag_dirs @ item_vecs[id_to_index[m]] + rng.normal(scale=0.3)))]
        for m in tag_rows["movieId"].to_list()
    ]
    tags_df = tag_rows.select(
        "userId",
        "movieId",
        pl.Series("tag", tag_words, dtype=pl.String),
        (pl.col("timestamp") + 60).alias("timestamp"),
    )

    return SyntheticData(
        ratings=ratings_df,
        movies=_movies(cfg, rng, movie_ids, item_vecs, _first_rating_year(ratings_df, movie_ids)),
        tags=tags_df,
        links=_links(rng, movie_ids),
    )


def main(argv: list[str] | None = None) -> None:
    """Write a synthetic MovieLens-style dataset to a directory."""
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--out", type=Path, required=True, help="output directory")
    parser.add_argument("--users", type=int, default=SyntheticConfig.n_users)
    parser.add_argument("--items", type=int, default=SyntheticConfig.n_items)
    parser.add_argument("--seed", type=int, default=SyntheticConfig.seed)
    args = parser.parse_args(argv)
    cfg = SyntheticConfig(n_users=args.users, n_items=args.items, seed=args.seed)
    generate(cfg).write_csv(args.out)


if __name__ == "__main__":
    main()
