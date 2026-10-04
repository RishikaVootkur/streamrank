import math
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from streamrank.features.definitions import SECONDS_PER_DAY
from streamrank.monitoring.drift import (
    CATEGORICAL,
    NUMERICAL,
    event_features,
    render,
    run_report,
    summarize,
)

DAY = SECONDS_PER_DAY


def test_event_features_use_the_snapshot_before_the_event_day() -> None:
    ratings = pl.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 10],
            "rating": [4.0, 2.0],
            "ts": [100 * DAY + 3600 * 5, 105 * DAY + 60],
        }
    )
    snaps = pl.DataFrame(
        {
            "item_id": [10, 10, 10],
            "snapshot_day": [100, 101, 106],
            "item_n_ratings_30d": [3, 7, 99],
            "item_first_ts": [50 * DAY] * 3,
        }
    )
    movies = pl.DataFrame({"item_id": [10], "year": [1999], "genres": [["Drama", "War"]]})
    out = event_features(ratings, snaps, movies, 0, 200 * DAY).sort("hour_utc", descending=True)
    first = out.row(0, named=True)  # event on day 100 sees the day-100 snapshot
    assert first["hour_utc"] == 5.0 and first["positive"] == "true"
    assert abs(first["item_log_n_ratings_30d"] - math.log1p(3)) < 1e-12
    assert first["item_genre"] == "Drama" and first["item_year"] == 1999.0
    second = out.row(1, named=True)  # day 105 uses the day-101 snapshot, never day 106
    assert abs(second["item_log_n_ratings_30d"] - math.log1p(7)) < 1e-12
    assert abs(second["item_age_days"] - (55 + 60 / DAY)) < 1e-9


def test_summary_and_markdown() -> None:
    report = {
        "metrics": [
            {
                "config": {"type": "evidently:metric_v2:DriftedColumnsCount"},
                "value": {"count": 1.0, "share": 0.5},
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "a",
                    "method": "Wasserstein distance (normed)",
                    "threshold": 0.1,
                },
                "value": 0.3,
            },
            {
                "config": {
                    "type": "evidently:metric_v2:ValueDrift",
                    "column": "b",
                    "method": "Jensen-Shannon distance",
                    "threshold": 0.1,
                },
                "value": 0.01,
            },
        ]
    }
    s = summarize(report)
    assert s["drifted_columns"] == 1 and s["drift_share"] == 0.5
    assert s["columns"]["a"]["drift"] and not s["columns"]["b"]["drift"]
    s |= {
        "reference": ["2020-01-01", "2020-03-31"],
        "current": ["2021-01-01", "2021-03-31"],
        "rows": {"reference": 10, "current": 10},
    }
    text = render(s)
    assert "| a | Wasserstein distance (normed) | 0.3000 | 0.1 | yes |" in text


def test_run_report_uses_distances_on_small_samples(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    n = 200

    def frame(shift: float) -> pd.DataFrame:
        data: dict[str, object] = {c: rng.normal(shift, 1.0, n) for c in NUMERICAL}
        data |= {c: rng.choice(["a", "b"], n) for c in CATEGORICAL}
        return pd.DataFrame(data)

    summary = run_report(frame(0.0), frame(3.0), tmp_path)
    methods = {info["method"] for info in summary["columns"].values()}
    assert methods == {"wasserstein", "jensenshannon"}
    assert all(summary["columns"][c]["drift"] for c in NUMERICAL)
