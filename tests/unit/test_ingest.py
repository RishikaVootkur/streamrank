from pathlib import Path

import pandera.errors
import polars as pl
import pytest

from streamrank.common.provenance import read_manifest
from streamrank.data.ingest import ingest, main, read_movies, read_tags
from streamrank.data.synthetic import SyntheticConfig, generate


@pytest.fixture
def raw_dir(tmp_path: Path) -> Path:
    out = tmp_path / "raw"
    generate(SyntheticConfig(n_users=100, n_items=80, seed=5)).write_csv(out)
    return out


def test_ingest_writes_typed_tables_and_manifest(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    counts = ingest(raw_dir, out)
    ratings = pl.read_parquet(out / "ratings.parquet")
    assert ratings.columns == ["user_id", "item_id", "rating", "ts"]
    assert counts["ratings"] == ratings.height
    movies = pl.read_parquet(out / "movies.parquet")
    assert movies.schema["genres"] == pl.List(pl.String)
    manifest = read_manifest(out)
    assert manifest["stage"] == "ingest"
    assert len(manifest["data_hash"]) == 64
    assert manifest["row_counts"] == counts


def test_ingest_is_idempotent(raw_dir: Path, tmp_path: Path) -> None:
    out = tmp_path / "processed"
    ingest(raw_dir, out)
    first = pl.read_parquet(out / "ratings.parquet")
    hash_one = read_manifest(out)["data_hash"]
    ingest(raw_dir, out)
    assert pl.read_parquet(out / "ratings.parquet").equals(first)
    assert read_manifest(out)["data_hash"] == hash_one


def test_orphan_ratings_are_dropped(raw_dir: Path, tmp_path: Path) -> None:
    with (raw_dir / "ratings.csv").open("a") as f:
        f.write("1,999999,4.0,1500000000\n")
    out = tmp_path / "processed"
    ingest(raw_dir, out)
    assert read_manifest(out)["dropped_orphan_ratings"] == 1


@pytest.mark.parametrize(
    "bad_row",
    ["1,{item},4.3,1500000000\n", "1,{item},4.0,-5\n", "0,{item},4.0,1500000000\n"],
)
def test_invalid_ratings_fail(raw_dir: Path, tmp_path: Path, bad_row: str) -> None:
    item = pl.read_csv(raw_dir / "movies.csv")["movieId"][0]
    with (raw_dir / "ratings.csv").open("a") as f:
        f.write(bad_row.format(item=item))
    with pytest.raises(pandera.errors.SchemaError):
        ingest(raw_dir, tmp_path / "processed")


def test_duplicate_rating_fails(raw_dir: Path, tmp_path: Path) -> None:
    first = (raw_dir / "ratings.csv").read_text().splitlines()[1]
    with (raw_dir / "ratings.csv").open("a") as f:
        f.write(first + "\n")
    with pytest.raises(pandera.errors.SchemaError):
        ingest(raw_dir, tmp_path / "processed")


def test_movies_parse_year_and_genres(tmp_path: Path) -> None:
    (tmp_path / "movies.csv").write_text(
        "movieId,title,genres\n"
        '1,"American President, The (1995)",Comedy|Drama|Romance\n'
        "2,Untitled Project,(no genres listed)\n"
        "3, Spaced Title (2001) ,Horror\n"
    )
    movies = read_movies(tmp_path)
    assert movies["year"].to_list() == [1995, None, 2001]
    assert movies["genres"].to_list() == [["Comedy", "Drama", "Romance"], [], ["Horror"]]
    assert movies["title"][2] == "Spaced Title (2001)"


def test_tags_are_normalized(tmp_path: Path) -> None:
    (tmp_path / "tags.csv").write_text(
        'userId,movieId,tag,timestamp\n1,2,"  Dark Comedy ",10\n1,3,"   ",11\n1,4,,12\n'
    )
    tags = read_tags(tmp_path)
    assert tags["tag"].to_list() == ["dark comedy"]


def test_cli(raw_dir: Path, tmp_path: Path) -> None:
    main(["--raw-dir", str(raw_dir), "--out-dir", str(tmp_path / "p")])
    assert (tmp_path / "p" / "manifest.json").exists()
