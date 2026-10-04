"""Ranker tests (LightGBM runs in its own pytest process; see dev notes)."""

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from streamrank.common.provenance import verify_outputs, write_manifest
from streamrank.data.columns import GENRES
from streamrank.ranking.features import FEATURES, build_features, user_profiles
from streamrank.ranking.train_ranker import RankerConfig, ndcg_by_user, run, split_users

DAY = 86_400


def test_split_users_is_disjoint_and_excludes() -> None:
    users = np.arange(100)
    first, second = split_users(users, np.array([0, 1, 2]), 0.25, seed=1)
    assert not set(first) & set(second)
    assert {0, 1, 2}.isdisjoint(set(first) | set(second))
    assert len(second) == round(97 * 0.25)


def test_ndcg_by_user_matches_hand_computation() -> None:
    df = pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2],
            "label": [0, 1, 0, 1, 0],
            "n_relevant": [2, 2, 2, 1, 1],
            "s": [3.0, 2.0, 1.0, 0.0, 1.0],
            "retrieval_rank": [1, 2, 3, 1, 2],
        }
    )
    got = ndcg_by_user(df, "s", k=10)
    # User 1: one hit at rank 2 out of 2 relevant. User 2: hit at rank 2 of 1 relevant.
    idcg1 = 1 + 1 / np.log2(3)
    assert got[0] == pytest.approx((1 / np.log2(3)) / idcg1)
    assert got[1] == pytest.approx(1 / np.log2(3))


def test_user_profiles_and_features() -> None:
    movies = pl.DataFrame(
        {
            "item_id": [1, 2, 3],
            "title": ["a", "b", "c"],
            "year": [1990, 2000, 2010],
            "genres": [["Drama"], ["Comedy"], ["Drama", "Comedy"]],
        }
    )
    history = pl.DataFrame(
        {"user_id": [7, 7, 7], "item_id": [1, 1, 2], "rating": [5.0, 4.5, 1.0], "ts": [1, 2, 3]}
    ).unique(subset=["user_id", "item_id"], keep="first")
    profiles = user_profiles(history, movies)
    row = profiles.row(0, named=True)
    assert row["ug_Drama"] == pytest.approx(1.0) and row["ug_Comedy"] == 0.0
    assert row["user_mean_year"] == pytest.approx(1995.0)
    candidates = pl.DataFrame(
        {
            "user_id": [7, 7],
            "item_id": [3, 2],
            "retrieval_score": [0.5, 0.4],
            "retrieval_rank": [1, 2],
            "label": [1, 0],
            "n_relevant": [1, 1],
        }
    )
    cutoff = 100 * DAY
    stats_u = pl.DataFrame(
        {
            "user_id": [7],
            "user_n_ratings": [3],
            "user_n_positive": [2],
            "user_n_ratings_7d": [0],
            "user_n_ratings_30d": [1],
            "user_mean_rating": [3.6],
            "user_positive_rate": [0.55],
            "user_first_ts": [10 * DAY],
            "user_last_ts": [90 * DAY],
        }
    )
    stats_i = pl.DataFrame(
        {
            "item_id": [2, 3],
            "item_n_ratings": [9, 99],
            "item_n_positive": [1, 50],
            "item_n_ratings_7d": [0, 5],
            "item_n_ratings_30d": [1, 20],
            "item_mean_rating": [2.5, 4.0],
            "item_positive_rate": [0.2, 0.5],
            "item_first_ts": [0, 50 * DAY],
            "item_last_ts": [80 * DAY, 99 * DAY],
        }
    )
    out = build_features(candidates, stats_u, stats_i, profiles, movies, cutoff_ts=cutoff)
    assert out.height == 2
    assert set(FEATURES) <= set(out.columns)
    first = out.row(0, named=True)
    assert first["user_days_since_last"] == pytest.approx(10.0)
    assert first["item_age_days"] == pytest.approx(50.0)
    assert first["genre_affinity"] == pytest.approx(1 / np.sqrt(2))  # Drama of Drama+Comedy
    assert first["year_gap"] == pytest.approx(15.0)
    assert out.row(1, named=True)["genre_affinity"] == 0.0  # pure Comedy
    assert len(GENRES) == 19


def _synthetic(tmp_path: Path, n_users: int = 300, k: int = 30) -> Path:
    rng = np.random.default_rng(0)
    rows = []
    for u in range(n_users):
        quality = rng.normal(size=k)
        label = (quality + rng.normal(scale=0.5, size=k) > 1.2).astype(np.int8)
        order = np.argsort(-(quality * 0.2 + rng.normal(size=k)))  # weak retrieval order
        for rank, j in enumerate(order, start=1):
            feats = {f: float(rng.normal()) for f in FEATURES}
            feats["item_mean_rating"] = float(quality[j])
            feats["retrieval_rank"] = float(rank)
            rows.append(
                {
                    "user_id": u,
                    "item_id": int(j),
                    "label": int(label[j]),
                    "n_relevant": int(label.sum()) + 1,
                    **feats,
                }
            )
    df = pl.DataFrame(rows)
    path = tmp_path / "ranker_features_val.parquet"
    df.write_parquet(path)
    np.save(tmp_path / "early_stop_users.npy", np.array([0, 1]))
    write_manifest(
        tmp_path, stage="ranker_features", config={}, data_hash="synthetic", seed=0, outputs=[path]
    )
    return path


def test_ranker_learns_and_reports_lift(tmp_path: Path) -> None:
    features = _synthetic(tmp_path)
    out = tmp_path / "ranker"
    cfg = RankerConfig(n_trials=3, timeout_minutes=1.0, max_rounds=200, early_stopping_rounds=20)
    summary = run(
        features,
        out,
        cfg,
        n_resamples=100,
        doc_dir=tmp_path / "docs",
    )
    assert summary["n_excluded_users"] == 2
    assert summary["ndcg@10_lift"]["low"] > 0
    assert next(iter(summary["shap_importance"])) == "item_mean_rating"
    assert verify_outputs(out)["stage"] == "train_ranker"
    eval_users = np.load(out / "eval_users.npy")
    assert eval_users.size == summary["n_users"]["eval"]
    assert not {0, 1} & set(eval_users.tolist())  # retrieval early-stop users are excluded
    assert json.loads((out / "summary.json").read_text())["best_iteration"] >= 1
    assert (tmp_path / "docs" / "ranker-shap.png").stat().st_size > 0


def test_build_features_cli_uses_cutoff_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from streamrank.common.config import get_settings  # noqa: PLC0415
    from streamrank.ranking import build_features as bf  # noqa: PLC0415

    cand_dir = tmp_path / "cand"
    cand_dir.mkdir()
    cand = pl.DataFrame(
        {
            "user_id": [1, 1],
            "item_id": [10, 11],
            "retrieval_score": [0.2, 0.1],
            "retrieval_rank": [1, 2],
            "label": [0, 1],
            "n_relevant": [1, 1],
        }
    )
    cand.write_parquet(cand_dir / "candidates_val.parquet")
    np.save(cand_dir / "early_stop_users.npy", np.array([5]))
    write_manifest(
        cand_dir,
        stage="candidates",
        config={"cutoff_ts": 1000},
        data_hash="x",
        seed=0,
        outputs=[cand_dir / "candidates_val.parquet"],
    )
    split_dir = tmp_path / "split"
    split_dir.mkdir()
    pl.DataFrame({"user_id": [1], "item_id": [10], "rating": [5.0], "ts": [5]}).write_parquet(
        split_dir / "train.parquet"
    )
    processed = tmp_path / "data" / "processed"
    processed.mkdir(parents=True)
    pl.DataFrame(
        {
            "item_id": [10, 11],
            "title": ["a", "b"],
            "year": [2000, 2001],
            "genres": [["Drama"], ["Drama"]],
        }
    ).write_parquet(processed / "movies.parquet")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    get_settings.cache_clear()
    seen: dict[str, object] = {}

    def fake_stats(store: object, cutoff: int, users: list[int], items: list[int]):  # type: ignore[no-untyped-def]
        seen["cutoff"] = cutoff
        return (pl.DataFrame({"user_id": users}), pl.DataFrame({"item_id": items}))

    monkeypatch.setattr(bf, "stats_at_cutoff", fake_stats)
    monkeypatch.setattr(bf, "open_store", lambda: None)
    path = bf.run(cand_dir, split_dir, tmp_path / "out")
    get_settings.cache_clear()
    assert seen["cutoff"] == 1000
    out = pl.read_parquet(path)
    assert out.height == 2 and set(FEATURES) <= set(out.columns)
    assert verify_outputs(tmp_path / "out")["stage"] == "ranker_features"
    assert np.load(tmp_path / "out" / "early_stop_users.npy").tolist() == [5]


def test_ranked_metrics_match_ndcg_definition() -> None:
    import lightgbm as lgb  # noqa: PLC0415

    from streamrank.eval.two_stage import ranked_metrics  # noqa: PLC0415

    rng = np.random.default_rng(5)
    rows = []
    for u in range(20):
        for rank in range(1, 16):
            feats = {f: float(rng.normal()) for f in FEATURES}
            feats["retrieval_rank"] = float(rank)
            rows.append(
                {"user_id": u, "item_id": rank, "label": int(rank == 3), "n_relevant": 2, **feats}
            )
    df = pl.DataFrame(rows)
    x = df.select(FEATURES).to_numpy().astype(np.float32)
    booster = lgb.train(
        {"objective": "regression", "verbosity": -1, "num_threads": 1},
        lgb.Dataset(x, -df["retrieval_rank"].to_numpy()),
        num_boost_round=30,
    )
    out = ranked_metrics(df, booster, np.arange(20))
    # The booster learned retrieval order: the relevant item sits at rank 3 of 2 relevant.
    expected = (1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert np.allclose(out["ndcg@10"].to_numpy(), expected, atol=1e-6)
    assert np.allclose(out["recall@10"].to_numpy(), 0.5)
    # Without a booster the retrieval order is scored directly, giving the same lists.
    plain = ranked_metrics(df, None, np.arange(20))
    assert np.allclose(plain["ndcg@10"].to_numpy(), expected, atol=1e-6)
