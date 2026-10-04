"""Ranker ONNX conversion (runs in the LightGBM test process)."""

from pathlib import Path

import lightgbm as lgb
import numpy as np

from streamrank.ranking.export_onnx import convert, float32_thresholds, max_difference
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


def test_thresholds_between_adjacent_float32_values(tmp_path: Path) -> None:
    # Feature values one float32 step apart: a threshold halfway between two of them only
    # exists in double precision, so naive rounding to float32 sends one value the wrong way.
    rng = np.random.default_rng(1)
    steps = np.arange(4000, dtype=np.float32)
    col = (np.float32(1.0) + steps * np.finfo(np.float32).eps).astype(np.float32)
    x = rng.normal(size=(4000, len(FEATURES))).astype(np.float32)
    x[:, 0] = col
    y = (np.arange(4000) % 7 < 3).astype(int)  # labels flip between neighboring values
    booster = lgb.train(
        {
            "objective": "lambdarank",
            "verbosity": -1,
            "num_leaves": 63,
            "min_data_in_leaf": 1,
            "num_threads": 1,
            "max_bin": 1023,
        },
        lgb.Dataset(x, y, group=[40] * 100),
        num_boost_round=40,
    )
    assert np.array_equal(
        float32_thresholds(booster).predict(x, num_threads=1), booster.predict(x, num_threads=1)
    )
    path = convert(booster, tmp_path / "ranker.onnx")
    assert max_difference(booster, path, x) < 1e-5
