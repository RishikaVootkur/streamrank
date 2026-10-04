"""Convert raw MovieLens-style CSV files into validated, typed Parquet tables."""

import argparse
import logging
from pathlib import Path

import polars as pl

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import (
    begin_stage,
    combined_hash,
    write_manifest,
    write_parquet_atomic,
)
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

    Ratings and tags that reference unknown movies, and empty tags, are dropped; every
    drop is counted in the manifest. The old manifest is removed first and written last,
    so a crashed run never leaves a manifest describing mixed outputs.
    """
    begin_stage(out_dir)
    movies = read_movies(raw_dir)
    ratings = read_ratings(raw_dir)
    raw_tag_rows = pl.scan_csv(raw_dir / TAGS_FILE, quote_char='"').select(pl.len()).collect()
    tags = read_tags(raw_dir)
    links = read_links(raw_dir)

    known = movies["item_id"].implode()
    drops = {
        "orphan_ratings": ratings.filter(~pl.col("item_id").is_in(known)).height,
        "empty_tags": int(raw_tag_rows.item()) - tags.height,
        "orphan_tags": tags.filter(~pl.col("item_id").is_in(known)).height,
    }
    ratings = ratings.filter(pl.col("item_id").is_in(known))
    tags = tags.filter(pl.col("item_id").is_in(known))

    tables = {"ratings": ratings, "movies": movies, "tags": tags, "links": links}
    outputs = []
    for name, df in tables.items():
        path = out_dir / f"{name}.parquet"
        write_parquet_atomic(df, path)
        outputs.append(path)
    counts = {name: df.height for name, df in tables.items()}
    raw_files = [raw_dir / f for f in (RATINGS_FILE, MOVIES_FILE, TAGS_FILE, LINKS_FILE)]
    write_manifest(
        out_dir,
        stage="ingest",
        config={
            "raw_dir": str(raw_dir),
            "out_dir": str(out_dir),
            "drop_rules": "ratings/tags of unknown movies; empty tags after trimming",
        },
        data_hash=combined_hash(raw_files),
        seed=None,
        outputs=outputs,
        extra={"row_counts": counts, "dropped": drops},
    )
    log.info("ingested", extra={"row_counts": counts, "dropped": drops})
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
