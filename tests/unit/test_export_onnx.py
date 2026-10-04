from pathlib import Path

import numpy as np
import polars as pl
import torch

from streamrank.data.split import SplitConfig, temporal_split, write_split
from streamrank.data.synthetic import SyntheticConfig, generate
from streamrank.eval.dataset import load_setup
from streamrank.models.train_retrieval import TrainConfig, TwoTowerRecommender
from streamrank.models.two_tower import TwoTowerConfig
from streamrank.serving.export_onnx import export, parity, session


def test_onnx_user_tower_matches_pytorch(tmp_path: Path) -> None:
    synth = generate(SyntheticConfig(n_users=300, n_items=100, seed=6))
    ratings = synth.ratings.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    cfg = SplitConfig(val_fraction=0.1, test_fraction=0.1)
    write_split(temporal_split(ratings, cfg), tmp_path / "split", cfg, data_hash="synthetic")
    setup = load_setup(tmp_path / "split", "val")
    movies = synth.movies.rename({"movieId": "item_id"}).with_columns(
        year=pl.col("title").str.extract(r"\((\d{4})\)").cast(pl.Int64),
        genres=pl.col("genres").str.split("|"),
    )
    tags = synth.tags.rename({"userId": "user_id", "movieId": "item_id", "timestamp": "ts"})
    rec = TwoTowerRecommender(
        TwoTowerConfig(dim=16, max_len=12, n_layers=1, n_heads=2, n_random_negatives=16),
        TrainConfig(epochs=1, batch_size=64),
        movies,
        tags,
        device=torch.device("cpu"),
    )
    rec.fit(setup.train)
    path = export(rec, tmp_path / "user_tower.onnx")
    rows = setup.targets.user_rows
    assert parity(rec, path, rows[:7]) < 1e-5
    assert parity(rec, path, rows[:1]) < 1e-5  # dynamic batch size
    out = session(path).run(
        None,
        {
            "tokens": np.zeros((2, 12), dtype=np.int64) + 1,
            "positive": np.ones((2, 12), dtype=np.int64),
            "gaps": np.full((2, 12), 3, dtype=np.int64),
        },
    )[0]
    assert out.shape == (2, 16)
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1.0, atol=1e-5)
