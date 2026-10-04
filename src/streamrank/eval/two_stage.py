"""The full two-stage system (two-tower + ranker) against EASE on the same users.

Scores the ranker on its held-out evaluation users (LightGBM; no PyTorch or FAISS in this
process), fits EASE on the same training data, and reports paired NDCG@10 and Recall@10
differences with bootstrap intervals over users.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from streamrank.common.config import get_settings
from streamrank.eval import metrics
from streamrank.eval.dataset import load_setup
from streamrank.eval.evaluate import recommend
from streamrank.models.baselines import EASE
from streamrank.ranking.features import FEATURES


def ranked_metrics(features: pl.DataFrame, booster: lgb.Booster, users: np.ndarray) -> pl.DataFrame:
    """Per-user NDCG@10 and Recall@10 of the ranker order (relative to all relevant items)."""
    df = features.filter(pl.col("user_id").is_in(users.tolist())).sort(
        ["user_id", "retrieval_rank"]
    )
    scores = booster.predict(df.select(FEATURES).to_numpy().astype(np.float32), num_threads=1)
    top = (
        df.with_columns(score=pl.Series(np.asarray(scores)))
        .sort(["user_id", "score", "retrieval_rank"], descending=[False, True, False])
        .group_by("user_id", maintain_order=True)
        .agg(labels=pl.col("label").head(10), n_rel=pl.col("n_relevant").first())
    )
    hits = np.zeros((top.height, 10), dtype=bool)
    for i, labels in enumerate(top["labels"].to_list()):
        hits[i, : len(labels)] = np.asarray(labels, dtype=bool)
    n_rel = top["n_rel"].to_numpy().astype(np.int64)
    return pl.DataFrame(
        {
            "user_id": top["user_id"],
            "ndcg@10": metrics.ndcg_at_k(hits, n_rel, 10),
            "recall@10": metrics.recall_at_k(hits, n_rel, 10),
        }
    )


def ease_metrics(split_dir: Path, users: np.ndarray) -> pl.DataFrame:
    """Per-user NDCG@10 and Recall@10 of EASE (tuned setting) on the full catalog."""
    setup = load_setup(split_dir, "val")
    model = EASE(l2=2000.0, signal="positive")
    model.fit(setup.train)
    t = setup.targets
    keep = np.isin(setup.train.user_ids[t.user_rows], users)
    recs = recommend(model, t.user_rows[keep], t.exclude[keep], 10)
    rel = t.relevant[keep]
    hits = metrics.hit_matrix(recs, rel)
    n_rel = np.diff(rel.indptr).astype(np.int64)
    return pl.DataFrame(
        {
            "user_id": setup.train.user_ids[t.user_rows[keep]],
            "ndcg@10": metrics.ndcg_at_k(hits, n_rel, 10),
            "recall@10": metrics.recall_at_k(hits, n_rel, 10),
        }
    )


def compare(ranker_dir: Path, features_path: Path, split_dir: Path, seed: int) -> dict[str, Any]:
    users = np.load(ranker_dir / "eval_users.npy")
    booster = lgb.Booster(model_file=str(ranker_dir / "ranker.txt"))
    ours = ranked_metrics(pl.read_parquet(features_path), booster, users)
    ease = ease_metrics(split_dir, users)
    both = ours.join(ease, on="user_id", suffix="_ease")
    out: dict[str, Any] = {"users": both.height}
    for m in ("ndcg@10", "recall@10"):
        a, b = both[m].to_numpy(), both[f"{m}_ease"].to_numpy()
        out[m] = {
            "two_stage": float(a.mean()),
            "ease": float(b.mean()),
            "difference": vars(metrics.paired_difference(a, b, n_resamples=1000, seed=seed)),
        }
    return out


def main(argv: list[str] | None = None) -> None:
    """Compare the two-stage system with EASE on the ranker's evaluation users."""
    s = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--ranker-dir", type=Path, default=s.artifacts_dir / "ranker")
    p.add_argument(
        "--features",
        type=Path,
        default=s.artifacts_dir / "ranker_features" / "ranker_features_val.parquet",
    )
    p.add_argument("--split-dir", type=Path, default=s.data_dir / "split" / "full")
    args = p.parse_args(argv)
    result = compare(args.ranker_dir, args.features, args.split_dir, s.seed)
    (args.ranker_dir / "two_stage_vs_ease.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
