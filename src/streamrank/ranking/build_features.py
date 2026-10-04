"""Build the ranker feature table for exported retrieval candidates."""

import argparse
from pathlib import Path

import polars as pl

from streamrank.common.config import get_settings
from streamrank.common.provenance import (
    begin_stage,
    verify_outputs,
    write_manifest,
    write_parquet_atomic,
)
from streamrank.ranking.features import FEATURES, build_features, stats_at_cutoff, user_profiles


def open_store() -> object:
    """Open the Feast store (imported here: Feast loads PyTorch, see `ranking.features`)."""
    from streamrank.features.store import open_store as _open  # noqa: PLC0415

    return _open()


def run(candidates_dir: Path, split_dir: Path, out_dir: Path, partition: str = "val") -> Path:
    """Join features onto `candidates_<partition>.parquet` and write the result to `out_dir`."""
    settings = get_settings()
    cand_manifest = verify_outputs(candidates_dir)
    cutoff = int(cand_manifest["config"]["cutoff_ts"])
    candidates = pl.read_parquet(candidates_dir / f"candidates_{partition}.parquet")
    history = pl.read_parquet(split_dir / "train.parquet")
    if partition == "test":
        history = pl.concat([history, pl.read_parquet(split_dir / "val.parquet")])
    if int(history["ts"].max()) >= cutoff:  # type: ignore[arg-type]
        raise ValueError("history reaches past the candidate cutoff")
    movies = pl.read_parquet(settings.data_dir / "processed" / "movies.parquet")
    user_stats, item_stats = stats_at_cutoff(
        open_store(),  # type: ignore[arg-type]
        cutoff,
        candidates["user_id"].unique().to_list(),
        candidates["item_id"].unique().to_list(),
    )
    table = build_features(
        candidates,
        user_stats,
        item_stats,
        user_profiles(history, movies),
        movies,
        cutoff_ts=cutoff,
    )
    begin_stage(out_dir)
    path = out_dir / f"ranker_features_{partition}.parquet"
    write_parquet_atomic(table, path)
    write_manifest(
        out_dir,
        stage="ranker_features",
        config={
            "candidates": str(candidates_dir),
            "partition": partition,
            "cutoff_ts": cutoff,
            "features": list(FEATURES),
        },
        data_hash=str(cand_manifest["data_hash"]),
        seed=cand_manifest["seed"],
        outputs=[path],
    )
    return path


def main(argv: list[str] | None = None) -> None:
    """Build ranker features from candidates, Feast statistics, and user history."""
    settings = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--candidates-dir", type=Path, default=settings.artifacts_dir / "candidates")
    p.add_argument("--split-dir", type=Path, default=settings.data_dir / "split" / "full")
    p.add_argument("--out-dir", type=Path, default=settings.artifacts_dir / "ranker_features")
    p.add_argument("--partition", default="val", choices=["val", "test"])
    args = p.parse_args(argv)
    print(run(args.candidates_dir, args.split_dir, args.out_dir, args.partition))


if __name__ == "__main__":
    main()
