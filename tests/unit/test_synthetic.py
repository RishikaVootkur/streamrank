from pathlib import Path

import polars as pl
import pytest

from streamrank.data.columns import GENRES, NO_GENRES, RAW_COLUMNS
from streamrank.data.synthetic import SyntheticConfig, SyntheticData, generate, main

SMALL = SyntheticConfig(n_users=300, n_items=200, seed=3)


@pytest.fixture(scope="module")
def data() -> SyntheticData:
    return generate(SMALL)


def test_same_seed_is_deterministic(data: SyntheticData) -> None:
    again = generate(SMALL)
    assert data.ratings.equals(again.ratings)
    assert data.movies.equals(again.movies)
    assert data.tags.equals(again.tags)


def test_different_seed_differs(data: SyntheticData) -> None:
    other = generate(SyntheticConfig(n_users=300, n_items=200, seed=4))
    assert not data.ratings.equals(other.ratings)


def test_ratings_are_valid(data: SyntheticData) -> None:
    r = data.ratings
    assert r["rating"].min() >= 0.5
    assert r["rating"].max() <= 5.0
    assert ((r["rating"] * 2) % 1 == 0).all()
    assert r.select(pl.struct("userId", "movieId").is_unique().all()).item()
    assert r["movieId"].is_in(data.movies["movieId"].implode()).all()
    assert r["userId"].n_unique() == SMALL.n_users
    per_user = r.group_by("userId").len()["len"]
    assert per_user.min() >= SMALL.min_interactions
    assert r["timestamp"].min() >= int(SMALL.start.timestamp())
    assert r["timestamp"].max() <= int(SMALL.end.timestamp())


def test_popularity_is_skewed(data: SyntheticData) -> None:
    counts = data.ratings.group_by("movieId").len().sort("len", descending=True)["len"]
    top_decile = counts.head(len(counts) // 10).sum()
    assert top_decile / counts.sum() > 0.25


def test_genres_and_titles(data: SyntheticData) -> None:
    allowed = {*GENRES, NO_GENRES}
    for g in data.movies["genres"].to_list():
        assert set(g.split("|")) <= allowed
    assert data.movies["movieId"].is_unique().all()
    assert data.movies["title"].str.contains(r"\(\d{4}\)$").mean() > 0.9  # type: ignore[operator]


def test_tags_reference_rated_pairs(data: SyntheticData) -> None:
    assert data.tags.height > 0
    joined = data.tags.join(data.ratings, on=["userId", "movieId"], how="anti")
    assert joined.height == 0


def test_cli_writes_movielens_headers(tmp_path: Path) -> None:
    main(["--out", str(tmp_path), "--users", "50", "--items", "40", "--seed", "1"])
    for name, cols in RAW_COLUMNS.items():
        header = (tmp_path / name).read_text().splitlines()[0]
        assert tuple(header.split(",")) == cols
    links = pl.read_csv(tmp_path / "links.csv")
    assert links.height == 40


def test_release_year_not_after_first_rating(data: SyntheticData) -> None:
    years = data.movies.select(
        "movieId", pl.col("title").str.extract(r"\((\d{4})\)$", 1).cast(pl.Int32).alias("year")
    )
    first = data.ratings.group_by("movieId").agg(pl.from_epoch("timestamp").min().dt.year())
    joined = years.join(first, on="movieId").drop_nulls()
    assert (joined["year"] <= joined["timestamp"]).all()


def test_imdb_ids_are_zero_padded(data: SyntheticData) -> None:
    assert data.links["imdbId"].str.contains(r"^\d{7}$").all()


@pytest.mark.parametrize(
    "kwargs",
    [{"n_items": 8, "min_interactions": 5}, {"n_users": 0}, {"latent_dim": 0}],
)
def test_invalid_config_is_rejected(kwargs: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="must"):
        SyntheticConfig(**kwargs)  # type: ignore[arg-type]
