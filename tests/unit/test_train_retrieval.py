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
    early_stop_users,
    load_recommender,
    run,
    split_targets,
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


def test_split_targets_partitions_users(data: tuple[Path, pl.DataFrame, pl.DataFrame]) -> None:
    setup = load_setup(data[0], "val")
    ids = setup.train.user_ids[setup.targets.user_rows]
    inside, outside = split_targets(setup.targets, ids[:5], setup.train)
    assert inside.user_rows.tolist() == setup.targets.user_rows[:5].tolist()
    assert inside.relevant.shape[0] == 5
    assert outside.user_rows.size == setup.targets.user_rows.size - 5
    assert not set(inside.user_rows) & set(outside.user_rows)


def test_early_stop_users_default_to_a_seeded_half(
    data: tuple[Path, pl.DataFrame, pl.DataFrame],
) -> None:
    setup = load_setup(data[0], "val")
    a = early_stop_users(setup, None, seed=3)
    assert a.size == setup.targets.user_rows.size // 2
    assert np.array_equal(a, early_stop_users(setup, None, seed=3))


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
    assert "recall@100_minus_ease_heldout" in summary
    assert (
        summary["heldout"]["n_users"] + summary["n_early_stop_users"]
        == (summary["result"]["n_users"])
    )
    assert np.load(out / "item_vectors.npy").shape[1] == SMALL.dim
    early = np.load(out / "early_stop_users.npy")
    assert early.size == summary["n_early_stop_users"]


def test_load_recommender_restores_scores(
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
    run(
        split_dir,
        out,
        SMALL,
        TrainConfig(epochs=1, batch_size=64),
        compare_ease=False,
        n_resamples=10,
    )
    get_settings.cache_clear()
    setup = load_setup(split_dir, "val")
    rec = load_recommender(out, setup.train, movies, tags, device=torch.device("cpu"))
    rows = setup.targets.user_rows[:4]
    np.testing.assert_allclose(
        rec.item_vectors().numpy(), np.load(out / "item_vectors.npy"), atol=1e-5
    )
    assert rec.score(rows).shape == (4, setup.train.n_items)

    from streamrank.models.export_vectors import export  # noqa: PLC0415

    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    path = export(out, split_dir, "val")
    get_settings.cache_clear()
    saved = np.load(path)
    assert saved["items"].shape == (setup.train.n_items, SMALL.dim)
    assert saved["users"].shape == (setup.targets.user_rows.size, SMALL.dim)
    np.testing.assert_array_equal(saved["user_rows"], setup.targets.user_rows)
    np.testing.assert_array_equal(saved["item_ids"], setup.train.item_ids)


def test_test_partition_trains_fixed_epochs(
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
    with pytest.raises(ValueError, match="stop-after"):
        run(split_dir, tmp_path / "bad", SMALL, TrainConfig(epochs=2), partition="test")
    summary = run(
        split_dir,
        tmp_path / "out",
        SMALL,
        TrainConfig(epochs=4, stop_after=2, batch_size=64, warmup_steps=2),
        n_resamples=20,
        partition="test",
        compare_ease=False,
    )
    get_settings.cache_clear()
    assert summary["partition"] == "test"
    assert summary["n_early_stop_users"] == 0
    assert len(summary["history"]) == 2
    assert summary["heldout"]["n_users"] == summary["result"]["n_users"]
