"""Convert raw MovieLens-style CSV files into validated, typed Parquet tables."""

import argparse
import logging
from pathlib import Path

import polars as pl

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import combined_hash, write_manifest
from streamrank.data.columns import LINKS_FILE, MOVIES_FILE, NO_GENRES, RATINGS_FILE, TAGS_FILE
from streamrank.data.schemas import Links, Movies, Ratings, Tags

log = logging.getLogger(__name__)

_YEAR_PATTERN = r"\((\d{4})\)\s*$"


def read_ratings(raw_dir: Path) -> pl.DataFrame:
    """Read and validate ratings."""
    df = pl.read_csv(
        raw_dir / RATINGS_FILE,
        schema={
            "userId": pl.Int64,
            "movieId": pl.Int64,
            "rating": pl.Float64,
            "timestamp": pl.Int64,
        },
    ).rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    return Ratings.validate(df)


def read_movies(raw_dir: Path) -> pl.DataFrame:
    """Read movies, parse the release year from the title, and split genres into a list."""
    df = pl.read_csv(
        raw_dir / MOVIES_FILE,
        schema={"movieId": pl.Int64, "title": pl.String, "genres": pl.String},
    )
    df = df.select(
        pl.col("movieId").alias("item_id"),
        pl.col("title").str.strip_chars(),
        pl.col("title").str.extract(_YEAR_PATTERN, 1).cast(pl.Int64).alias("year"),
        pl.when(pl.col("genres") == NO_GENRES)
        .then(pl.lit([], dtype=pl.List(pl.String)))
        .otherwise(pl.col("genres").str.split("|"))
        .alias("genres"),
    )
    return Movies.validate(df)


def read_tags(raw_dir: Path) -> pl.DataFrame:
    """Read tags, normalize case and whitespace, and drop empty ones."""
    df = pl.read_csv(
        raw_dir / TAGS_FILE,
        schema={"userId": pl.Int64, "movieId": pl.Int64, "tag": pl.String, "timestamp": pl.Int64},
        quote_char='"',
    )
    df = df.select(
        pl.col("userId").alias("user_id"),
        pl.col("movieId").alias("item_id"),
        pl.col("tag").str.strip_chars().str.to_lowercase().alias("tag"),
        pl.col("timestamp").alias("ts"),
    ).filter(pl.col("tag").is_not_null() & (pl.col("tag").str.len_chars() > 0))
    return Tags.validate(df)


def read_links(raw_dir: Path) -> pl.DataFrame:
    """Read external identifiers."""
    df = (
        pl.read_csv(
            raw_dir / LINKS_FILE,
            schema={"movieId": pl.Int64, "imdbId": pl.String, "tmdbId": pl.Int64},
        )
        .with_columns(pl.col("imdbId").cast(pl.Int64))
        .rename({"movieId": "item_id", "imdbId": "imdb_id", "tmdbId": "tmdb_id"})
    )
    return Links.validate(df)


def ingest(raw_dir: Path, out_dir: Path) -> dict[str, int]:
    """Validate the raw CSV files and write Parquet tables plus a manifest to `out_dir`.

    Ratings that reference unknown movies are dropped, and the count is reported.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    movies = read_movies(raw_dir)
    ratings = read_ratings(raw_dir)
    tags = read_tags(raw_dir)
    links = read_links(raw_dir)

    known = movies["item_id"].implode()
    orphan_ratings = ratings.filter(~pl.col("item_id").is_in(known)).height
    ratings = ratings.filter(pl.col("item_id").is_in(known))
    tags = tags.filter(pl.col("item_id").is_in(known))

    tables = {"ratings": ratings, "movies": movies, "tags": tags, "links": links}
    for name, df in tables.items():
        df.write_parquet(out_dir / f"{name}.parquet")
    counts = {name: df.height for name, df in tables.items()}
    raw_files = [raw_dir / f for f in (RATINGS_FILE, MOVIES_FILE, TAGS_FILE, LINKS_FILE)]
    write_manifest(
        out_dir,
        stage="ingest",
        config={"raw_dir": str(raw_dir)},
        data_hash=combined_hash(raw_files),
        seed=None,
        extra={"row_counts": counts, "dropped_orphan_ratings": orphan_ratings},
    )
    log.info("ingested", extra={"row_counts": counts, "dropped_orphan_ratings": orphan_ratings})
    return counts


def main(argv: list[str] | None = None) -> None:
    """Ingest raw CSV files into validated Parquet tables."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--raw-dir", type=Path, default=settings.data_dir / "raw")
    parser.add_argument("--out-dir", type=Path, default=settings.data_dir / "processed")
    args = parser.parse_args(argv)
    configure_logging(settings.log_level)
    ingest(args.raw_dir, args.out_dir)


if __name__ == "__main__":
    main()
