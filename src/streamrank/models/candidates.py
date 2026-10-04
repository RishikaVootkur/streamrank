"""Retrieval candidates for ranker training and evaluation.

For every warm user of a split partition, the trained two-tower model's top-K unseen items
(exact scoring over the full catalog) with retrieval score, rank, and whether the item is
relevant. `n_relevant` counts all of the user's relevant items, including ones retrieval
missed or cannot reach, so ranking metrics stay relative to the full relevant set.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from streamrank.common.config import get_settings
from streamrank.common.provenance import (
    begin_stage,
    verify_outputs,
    write_manifest,
    write_parquet_atomic,
)
from streamrank.eval import metrics
from streamrank.eval.dataset import load_setup
from streamrank.eval.evaluate import top_k
from streamrank.models.train_retrieval import load_recommender

K = 200


def export_candidates(
    model_dir: Path, split_dir: Path, out_dir: Path, partition: str = "val", k: int = K
) -> Path:
    """Write `candidates_<partition>.parquet` and a manifest to `out_dir`."""
    settings = get_settings()
    setup = load_setup(split_dir, partition)
    processed = settings.data_dir / "processed"
    rec = load_recommender(
        model_dir,
        setup.train,
        pl.read_parquet(processed / "movies.parquet"),
        pl.read_parquet(processed / "tags.parquet"),
        torch.device("cpu"),
    )
    if rec.model is None or rec.model.n_items != setup.train.n_items:
        raise ValueError(
            f"model catalog does not match the {partition!r} catalog; use a model trained "
            "on that partition's history"
        )
    early_src = model_dir / "early_stop_users.npy"
    if not early_src.exists():
        raise FileNotFoundError(f"{early_src} is missing; retrain with the current trainer")
    early = np.load(early_src)
    targets = setup.targets
    frames = []
    for start in range(0, targets.user_rows.size, 1024):
        rows = targets.user_rows[start : start + 1024]
        scores = rec.score(rows)
        exclude = targets.exclude[start : start + 1024]
        ranked = top_k(scores, exclude, k)
        rel = targets.relevant[start : start + 1024]
        labels = metrics.hit_matrix(ranked, rel)
        picked = np.take_along_axis(scores, np.maximum(ranked, 0), axis=1)
        n_rel = np.diff(rel.indptr)
        b, kk = ranked.shape
        valid = ranked >= 0
        frames.append(
            pl.DataFrame(
                {
                    "user_id": np.repeat(setup.train.user_ids[rows], kk)[valid.ravel()],
                    "item_id": setup.train.item_ids[np.maximum(ranked, 0)].ravel()[valid.ravel()],
                    "retrieval_score": picked.ravel()[valid.ravel()].astype(np.float32),
                    "retrieval_rank": np.tile(np.arange(1, kk + 1), b)[valid.ravel()],
                    "label": labels.astype(np.int8).ravel()[valid.ravel()],
                    "n_relevant": np.repeat(n_rel, kk)[valid.ravel()],
                }
            )
        )
    begin_stage(out_dir)
    path = out_dir / f"candidates_{partition}.parquet"
    write_parquet_atomic(pl.concat(frames), path)
    early_path = out_dir / "early_stop_users.npy"
    np.save(early_path, early)
    manifest = verify_outputs(model_dir)
    write_manifest(
        out_dir,
        stage="candidates",
        config={
            "model_dir": str(model_dir),
            "split": str(split_dir),
            "partition": partition,
            "k": k,
            "cutoff_ts": setup.train.cutoff_ts,
        },
        data_hash=str(manifest["data_hash"]),
        seed=manifest["seed"],
        outputs=[path, early_path],
    )
    return path


def main(argv: list[str] | None = None) -> None:
    """Export top-K retrieval candidates with labels."""
    settings = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, default=settings.data_dir / "split" / "full")
    p.add_argument("--out-dir", type=Path, default=settings.artifacts_dir / "candidates")
    p.add_argument("--partition", default="val", choices=["val", "test"])
    p.add_argument("--k", type=int, default=K)
    args = p.parse_args(argv)
    print(export_candidates(args.model_dir, args.split_dir, args.out_dir, args.partition, args.k))


if __name__ == "__main__":
    main()
