from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from streamrank.common.config import get_settings
from streamrank.common.provenance import verify_outputs
from streamrank.data.split import SplitConfig, temporal_split, write_split
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.eval.dataset import load_setup
from streamrank.models.candidates import export_candidates
from streamrank.models.train_retrieval import TrainConfig, run
from streamrank.models.two_tower import TwoTowerConfig


def test_export_candidates(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synth = generate(SyntheticConfig(n_users=400, n_items=120, seed=3))
    ratings = synth.ratings.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    cfg = SplitConfig(val_fraction=0.1, test_fraction=0.1)
    split_dir = tmp_path / "split"
    write_split(temporal_split(ratings, cfg), split_dir, cfg, data_hash="synthetic")
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    synth.movies.rename({"movieId": "item_id"}).with_columns(
        year=pl.col("title").str.extract(r"\((\d{4})\)").cast(pl.Int64),
        genres=pl.col("genres").str.split("|"),
    ).write_parquet(processed / "movies.parquet")
    synth.tags.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"}).write_parquet(
        processed / "tags.parquet"
    )
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "streamrank.models.train_retrieval.pick_device", lambda: torch.device("cpu")
    )
    model_dir = tmp_path / "model"
    small = TwoTowerConfig(dim=16, max_len=10, n_layers=1, n_heads=2, n_random_negatives=32)
    run(
        split_dir,
        model_dir,
        small,
        TrainConfig(epochs=1, batch_size=64),
        compare_ease=False,
        n_resamples=10,
    )
    path = export_candidates(model_dir, split_dir, tmp_path / "cand", k=20)
    get_settings.cache_clear()

    cand = pl.read_parquet(path)
    setup = load_setup(split_dir, "val")
    assert cand["user_id"].n_unique() == setup.targets.user_rows.size
    assert cand.group_by("user_id").len()["len"].max() <= 20
    # Never a training item, ranks start at 1 and follow the score order.
    hist = pl.read_parquet(split_dir / "train.parquet").select("user_id", "item_id")
    assert cand.join(hist, on=["user_id", "item_id"]).is_empty()
    first = cand.filter(pl.col("user_id") == cand["user_id"][0]).sort("retrieval_rank")
    assert first["retrieval_rank"][0] == 1
    assert np.all(np.diff(first["retrieval_score"].to_numpy()) <= 1e-6)
    # Labels agree with validation positives; n_relevant counts all of them.
    pos = pl.read_parquet(split_dir / "val.parquet").filter(pl.col("label") == 1)
    joined = cand.join(
        pos.select("user_id", "item_id", pl.lit(1).alias("p")),
        on=["user_id", "item_id"],
        how="left",
    )
    assert (joined["label"] == joined["p"].fill_null(0)).all()
    assert (
        cand.group_by("user_id")
        .agg(pl.col("label").sum(), pl.col("n_relevant").first())
        .filter(pl.col("label") > pl.col("n_relevant"))
        .is_empty()
    )
    assert verify_outputs(tmp_path / "cand")["stage"] == "candidates"
    np.testing.assert_array_equal(
        np.load(tmp_path / "cand" / "early_stop_users.npy"),
        np.load(model_dir / "early_stop_users.npy"),
    )
