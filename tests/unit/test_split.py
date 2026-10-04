from pathlib import Path

import polars as pl
import pytest

from streamrank.common.provenance import ProvenanceError, read_manifest, verify_outputs
from streamrank.data.ingest import ingest
from streamrank.data.split import (
    LeakageError,
    Split,
    SplitConfig,
    check_no_leakage,
    cut_points,
    main,
    sample_users,
    temporal_split,
)
from streamrank.data.synthetic import SyntheticConfig, generate


@pytest.fixture(scope="module")
def ratings() -> pl.DataFrame:
    raw = generate(SyntheticConfig(n_users=400, n_items=200, seed=11)).ratings
    return raw.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})


@pytest.fixture(scope="module")
def split(ratings: pl.DataFrame) -> Split:
    return temporal_split(ratings, SplitConfig())


def test_partitions_are_ordered_in_time(split: Split) -> None:
    assert split.train["ts"].max() < split.t1  # type: ignore[operator]
    assert split.t1 <= split.val["ts"].min()  # type: ignore[operator]
    assert split.val["ts"].max() < split.t2  # type: ignore[operator]
    assert split.t2 <= split.test["ts"].min()  # type: ignore[operator]


def test_partitions_cover_all_rows_once(ratings: pl.DataFrame, split: Split) -> None:
    total = split.train.height + split.val.height + split.test.height
    assert total == ratings.height
    combined = pl.concat([split.train, split.val, split.test]).select("user_id", "item_id")
    assert combined.unique().height == ratings.height


def test_fractions_are_close_to_config(ratings: pl.DataFrame, split: Split) -> None:
    assert split.val.height / ratings.height == pytest.approx(0.05, abs=0.01)
    assert split.test.height / ratings.height == pytest.approx(0.05, abs=0.01)


def test_label_is_rating_at_least_four(split: Split) -> None:
    df = pl.concat([split.train, split.val, split.test])
    assert (df["label"] == (df["rating"] >= 4.0).cast(pl.Int8)).all()


def test_leakage_check_catches_future_rows_in_train(split: Split) -> None:
    leaked = Split(
        train=pl.concat([split.train, split.test.head(1)]),
        val=split.val,
        test=split.test,
        t1=split.t1,
        t2=split.t2,
    )
    with pytest.raises(LeakageError, match="train"):
        check_no_leakage(leaked)


def test_leakage_check_catches_validation_overlap(split: Split) -> None:
    leaked = Split(
        train=split.train,
        val=pl.concat([split.val, split.test.head(1)]),
        test=split.test,
        t1=split.t1,
        t2=split.t2,
    )
    with pytest.raises(LeakageError, match="validation"):
        check_no_leakage(leaked)


def test_leakage_check_catches_duplicated_interaction(split: Split) -> None:
    row = split.val.head(1)
    leaked = Split(
        train=split.train,
        val=split.val,
        test=pl.concat(
            [split.test, row.with_columns(pl.lit(split.t2, dtype=pl.Int64).alias("ts"))]
        ),
        t1=split.t1,
        t2=split.t2,
    )
    with pytest.raises(LeakageError, match="more than once"):
        check_no_leakage(leaked)  # the same pair copied into test with a new timestamp
    dup = Split(
        train=split.train,
        val=pl.concat([split.val, row]),
        test=split.test,
        t1=split.t1,
        t2=split.t2,
    )
    with pytest.raises(LeakageError, match="more than once"):
        check_no_leakage(dup)


def test_empty_partition_is_rejected(split: Split) -> None:
    empty = Split(
        train=split.train, val=split.val.head(0), test=split.test, t1=split.t1, t2=split.t2
    )
    with pytest.raises(LeakageError, match="empty"):
        check_no_leakage(empty)


def test_sample_is_deterministic_and_user_level(split: Split) -> None:
    a = sample_users(split.train, 10, 42)
    b = sample_users(split.train, 10, 42)
    assert a.equals(b)
    users = a["user_id"].unique()
    assert 0.05 < users.len() / split.train["user_id"].n_unique() < 0.15
    # The same users are kept in every partition.
    val_users = sample_users(split.val, 10, 42)["user_id"].unique()
    all_val = split.val["user_id"].unique()
    expected = sample_users(pl.DataFrame({"user_id": all_val}), 10, 42)["user_id"]
    assert set(val_users.to_list()) == set(expected.to_list())
    assert sample_users(split.train, 100, 42).height == split.train.height


def test_different_seeds_give_different_samples() -> None:
    users = pl.DataFrame({"user_id": pl.arange(1, 20_001, eager=True)})
    picks = {s: set(sample_users(users, 10, s)["user_id"].to_list()) for s in (42, 43, 142)}
    for s, chosen in picks.items():
        assert 0.09 < len(chosen) / users.height < 0.11, s
    for a, b in ((42, 43), (42, 142)):
        overlap = len(picks[a] & picks[b]) / len(picks[a])
        assert overlap < 0.2, (a, b, overlap)  # independent samples overlap about 10%


@pytest.mark.parametrize(
    "kwargs",
    [
        {"val_fraction": 0.0},
        {"val_fraction": 0.6, "test_fraction": 0.5},
        {"sample_pct": 0},
    ],
)
def test_bad_config_is_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="must"):
        SplitConfig(**kwargs)  # type: ignore[arg-type]


def test_degenerate_cut_points_rejected() -> None:
    same_time = pl.DataFrame({"user_id": [1, 2, 3], "item_id": [1, 2, 3], "ts": [5, 5, 5]})
    with pytest.raises(ValueError, match="degenerate"):
        cut_points(same_time, SplitConfig())


def test_cli_end_to_end(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    generate(SyntheticConfig(n_users=300, n_items=150, seed=2)).write_csv(raw)
    ingest(raw, tmp_path / "processed")
    doc = tmp_path / "split.md"
    main(
        [
            "--processed-dir",
            str(tmp_path / "processed"),
            "--split-dir",
            str(tmp_path / "split"),
            "--stats-doc",
            str(doc),
            "--source",
            "synthetic data",
        ]
    )
    for sub in ("full", "sample10"):
        for name in ("train", "val", "test"):
            assert (tmp_path / "split" / sub / f"{name}.parquet").exists()
        manifest = verify_outputs(tmp_path / "split" / sub)
        assert manifest["stage"] == "split"
        assert manifest["seed"] == 42
        assert manifest["data_hash"] == read_manifest(tmp_path / "processed")["data_hash"]
    text = doc.read_text()
    assert "| train |" in text
    assert "Users with train history" in text
    assert "synthetic data" in text
    assert read_manifest(tmp_path / "processed")["data_hash"] in text


def test_split_refuses_unverified_input(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    generate(SyntheticConfig(n_users=100, n_items=60, seed=2)).write_csv(raw)
    ingest(raw, tmp_path / "processed")
    pl.DataFrame({"user_id": [1]}).write_parquet(tmp_path / "processed" / "ratings.parquet")
    with pytest.raises(ProvenanceError):
        main(["--processed-dir", str(tmp_path / "processed"), "--split-dir", str(tmp_path / "s")])
