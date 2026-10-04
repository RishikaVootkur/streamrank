"""Ranker ONNX conversion (runs in the LightGBM test process)."""

from pathlib import Path

import lightgbm as lgb
import numpy as np

from streamrank.ranking.export_onnx import convert, max_difference
from streamrank.ranking.features import FEATURES


def test_onnx_ranker_matches_lightgbm(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=(2000, len(FEATURES))).astype(np.float32)
    x[rng.random(x.shape) < 0.05] = np.nan  # missing values must route the same way
    y = (x[:, 0] + rng.normal(size=2000) > 1).astype(int)
    booster = lgb.train(
        {"objective": "lambdarank", "verbosity": -1, "num_leaves": 15, "num_threads": 1},
        lgb.Dataset(x, y, group=[20] * 100),
        num_boost_round=30,
    )
    path = convert(booster, tmp_path / "ranker.onnx")
    assert max_difference(booster, path, x) < 1e-5
