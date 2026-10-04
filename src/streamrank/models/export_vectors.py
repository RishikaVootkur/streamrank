"""Write a trained two-tower model's item vectors and user query vectors to a file.

FAISS and PyTorch must not share a process on macOS (each bundles its own OpenMP runtime
and the process aborts or hangs), so retrieval code reads these vectors from disk.
"""

import argparse
from pathlib import Path

import numpy as np
import polars as pl
import torch

from streamrank.common.config import get_settings
from streamrank.eval.dataset import load_setup
from streamrank.models.train_retrieval import load_recommender


def export(model_dir: Path, split_dir: Path, partition: str = "val") -> Path:
    """Save `vectors_<partition>.npz` (items, users, user_rows, item_ids) in `model_dir`."""
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
    rows = setup.targets.user_rows
    path = model_dir / f"vectors_{partition}.npz"
    np.savez(
        path,
        items=rec.item_vectors().numpy(),
        users=rec.user_vectors(rows).numpy(),
        user_rows=rows,
        item_ids=setup.train.item_ids,
    )
    return path


def main(argv: list[str] | None = None) -> None:
    """Export item and user vectors for a split partition."""
    settings = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--model-dir", type=Path, required=True)
    p.add_argument("--split-dir", type=Path, default=settings.data_dir / "split" / "full")
    p.add_argument("--partition", default="val", choices=["val", "test"])
    args = p.parse_args(argv)
    print(export(args.model_dir, args.split_dir, args.partition))


if __name__ == "__main__":
    main()
