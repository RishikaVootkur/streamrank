import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest
import torch

from streamrank.common.provenance import verify_outputs
from streamrank.data.split import SplitConfig, temporal_split, write_split
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.eval.dataset import load_setup
from streamrank.eval.evaluate import fit_and_evaluate
from streamrank.models.baselines import Popularity
from streamrank.models.train_retrieval import (
    TrainConfig,
    TwoTowerRecommender,
    run,
    subset_targets,
)
from streamrank.models.two_tower import TwoTowerConfig

SMALL = TwoTowerConfig(dim=32, max_len=20, n_layers=1, n_heads=2, n_random_negatives=64)


@pytest.fixture(scope="module")
def data(tmp_path_factory: pytest.TempPathFactory) -> tuple[Path, pl.DataFrame, pl.DataFrame]:
    root = tmp_path_factory.mktemp("tt")
    synth = generate(
        SyntheticConfig(
            n_users=1500, n_items=300, affinity_strength=4.0, popularity_exponent=0.5, seed=5
        )
    )
    ratings = synth.ratings.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    cfg = SplitConfig(val_fraction=0.1, test_fraction=0.1)
    split_dir = root / "split"
    write_split(temporal_split(ratings, cfg), split_dir, cfg, data_hash="synthetic")
    movies = synth.movies.rename({"movieId": "item_id"}).with_columns(
        year=pl.col("title").str.extract(r"\((\d{4})\)").cast(pl.Int64),
        genres=pl.col("genres").str.split("|"),
    )
    tags = synth.tags.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    return split_dir, movies.select("item_id", "title", "year", "genres"), tags


def test_recommender_trains_and_beats_popularity(
    data: tuple[Path, pl.DataFrame, pl.DataFrame],
) -> None:
    split_dir, movies, tags = data
    setup = load_setup(split_dir, "val")
    model_cfg = TwoTowerConfig(
        dim=32, max_len=50, n_layers=1, n_heads=2, n_random_negatives=128, temperature=0.1
    )
    rec = TwoTowerRecommender(
        model_cfg,
        TrainConfig(epochs=20, batch_size=32, warmup_steps=10, lr=3e-3, patience=5),
        movies,
        tags,
        early_stop_targets=setup.targets,
        device=torch.device("cpu"),
    )
    result = fit_and_evaluate(rec, setup, n_resamples=50)
    pop = fit_and_evaluate(Popularity(signal="positive"), setup, n_resamples=50)
    assert len(rec.history) >= 2
    assert rec.history[-1]["loss"] < rec.history[0]["loss"]
    assert result.metrics["recall@50"].mean > pop.metrics["recall@50"].mean
    scores = rec.score(setup.targets.user_rows[:3])
    assert scores.shape == (3, setup.train.n_items)
    assert np.isfinite(scores).all()


def test_subset_targets_keeps_requested_users(
    data: tuple[Path, pl.DataFrame, pl.DataFrame],
) -> None:
    setup = load_setup(data[0], "val")
    ids = setup.train.user_ids[setup.targets.user_rows]
    sub = subset_targets(setup.targets, ids[:5], setup.train)
    assert sub.user_rows.tolist() == setup.targets.user_rows[:5].tolist()
    assert sub.relevant.shape[0] == 5


def test_run_writes_artifacts(
    data: tuple[Path, pl.DataFrame, pl.DataFrame], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    split_dir, movies, tags = data
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    movies.write_parquet(processed / "movies.parquet")
    tags.write_parquet(processed / "tags.parquet")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    from streamrank.common.config import get_settings  # noqa: PLC0415

    get_settings.cache_clear()
    monkeypatch.setattr(
        "streamrank.models.train_retrieval.pick_device", lambda: torch.device("cpu")
    )
    out = tmp_path / "out"
    summary = run(
        split_dir,
        out,
        SMALL,
        TrainConfig(epochs=2, batch_size=64, warmup_steps=2),
        n_resamples=20,
    )
    get_settings.cache_clear()
    manifest = verify_outputs(out)
    assert manifest["stage"] == "train_retrieval"
    saved = json.loads((out / "summary.json").read_text())
    assert saved["result"]["model"] == "two_tower"
    assert "recall@100_minus_ease" in summary
    assert np.load(out / "item_vectors.npy").shape[1] == SMALL.dim
