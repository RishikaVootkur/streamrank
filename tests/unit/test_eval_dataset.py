from pathlib import Path

import numpy as np
import polars as pl
import pytest

from streamrank.data.split import SplitConfig, temporal_split, write_split
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.eval.dataset import build_targets, build_train, load_setup


def _frame(rows: list[tuple[int, int, float, int]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=["user_id", "item_id", "rating", "ts"], orient="row")


@pytest.fixture
def history() -> pl.DataFrame:
    return _frame([(10, 100, 5.0, 1), (10, 200, 2.0, 2), (20, 200, 4.0, 3), (30, 300, 4.5, 4)])


def test_build_train_indexes_and_labels(history: pl.DataFrame) -> None:
    train = build_train(history, cutoff_ts=10)
    assert train.user_ids.tolist() == [10, 20, 30]
    assert train.item_ids.tolist() == [100, 200, 300]
    assert train.interactions.toarray().tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]
    assert train.positives.toarray().tolist() == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert train.timestamps[0, 1] == 2
    np.testing.assert_allclose(train.item_popularity(), [0.25, 0.5, 0.25])


def test_build_train_rejects_future_rows(history: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="cutoff"):
        build_train(history, cutoff_ts=4)


def test_build_train_rejects_empty(history: pl.DataFrame) -> None:
    with pytest.raises(ValueError, match="empty"):
        build_train(history.clear(), cutoff_ts=10)


def test_build_targets_handles_cold_users_and_new_items(history: pl.DataFrame) -> None:
    train = build_train(history, cutoff_ts=10)
    eval_df = _frame(
        [(10, 300, 5.0, 11), (10, 999, 4.0, 12), (20, 100, 1.0, 13), (40, 100, 5.0, 14)]
    ).with_columns(label=(pl.col("rating") >= 4.0).cast(pl.Int8))
    targets = build_targets(train, eval_df)
    # User 20 has only a negative, user 40 is cold: only user 10 is evaluated.
    assert targets.user_rows.tolist() == [0]
    assert targets.n_cold_users == 1
    # Item 300 is in the catalog; item 999 is new and gets a column past the catalog.
    assert targets.relevant.shape == (1, 4)
    assert targets.relevant.toarray().tolist() == [[0, 0, 1, 1]]
    assert targets.exclude.toarray().tolist() == [[1, 1, 0]]


def test_load_setup_reads_a_written_split(tmp_path: Path) -> None:
    raw = generate(SyntheticConfig(n_users=300, n_items=150, seed=3)).ratings
    ratings = raw.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    cfg = SplitConfig()
    split = temporal_split(ratings, cfg)
    write_split(split, tmp_path, cfg, data_hash="test")

    val = load_setup(tmp_path, "val")
    test = load_setup(tmp_path, "test")
    assert val.train.cutoff_ts == split.t1
    assert test.train.cutoff_ts == split.t2
    assert test.train.interactions.nnz == split.train.height + split.val.height
    assert val.targets.user_rows.size > 0
    with pytest.raises(ValueError, match="unknown partition"):
        load_setup(tmp_path, "train")
