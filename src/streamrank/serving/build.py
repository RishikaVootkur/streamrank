"""Assemble everything the API serves from: models, index, item and user arrays, Redis state.

Serving starts from the state at the training cutoff: user sequences and seen sets are
loaded into Redis from training history, item content and user genre profiles become
arrays the API keeps in memory, and a recent-popularity list serves unknown users.
"""

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import polars as pl
import redis

from streamrank.common.config import get_settings
from streamrank.common.provenance import begin_stage, verify_outputs, write_manifest
from streamrank.data.columns import GENRES
from streamrank.eval.dataset import load_setup
from streamrank.models.baselines import RecentPopularity
from streamrank.models.sequences import build_sequences
from streamrank.ranking.features import user_profiles
from streamrank.serving.state import bulk_load

FALLBACK_SIZE = 200
LOADTEST_USERS = 5000


def item_arrays(item_ids: np.ndarray, movies: pl.DataFrame) -> dict[str, np.ndarray]:
    """Year and genre multi-hot per catalog item (catalog order)."""
    m = pl.DataFrame({"item_id": item_ids}).join(
        movies.select("item_id", "title", "year", "genres"), on="item_id", how="left"
    )
    genres = np.zeros((item_ids.size, len(GENRES)), dtype=np.float32)
    index = {g: i for i, g in enumerate(GENRES)}
    for row, gs in enumerate(m["genres"].to_list()):
        for g in gs or ():
            if g in index:
                genres[row, index[g]] = 1.0
    return {
        "item_ids": item_ids.astype(np.int64),
        "has_movie": m["title"].is_not_null().to_numpy(),
        "year": m["year"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        "genres": genres,
        "titles": np.asarray(m["title"].fill_null("").to_list(), dtype=object),
    }


def profile_arrays(history: pl.DataFrame, movies: pl.DataFrame) -> dict[str, np.ndarray]:
    """User genre profiles and mean release year (same code as the offline ranker features)."""
    p = user_profiles(history, movies).sort("user_id")
    return {
        "user_ids": p["user_id"].to_numpy().astype(np.int64),
        "mean_year": p["user_mean_year"].cast(pl.Float64).fill_null(np.nan).to_numpy(),
        "genres": p.select([f"ug_{g}" for g in GENRES])
        .fill_null(0.0)
        .to_numpy()
        .astype(np.float32),
    }


def run(
    split_dir: Path,
    out_dir: Path,
    *,
    onnx: Path,
    model_dir: Path,
    index_dir: Path,
    ranker_dir: Path,
    client: redis.Redis,
    max_len: int,
) -> dict[str, int]:
    """Write serving artifacts to `out_dir` and load user state into Redis."""
    settings = get_settings()
    setup = load_setup(split_dir, "val")
    train = setup.train
    movies = pl.read_parquet(settings.data_dir / "processed" / "movies.parquet")
    history = pl.read_parquet(split_dir / "train.parquet")
    begin_stage(out_dir)
    np.savez(out_dir / "items.npz", **item_arrays(train.item_ids, movies))  # type: ignore[arg-type]
    np.savez(out_dir / "profiles.npz", **profile_arrays(history, movies))  # type: ignore[arg-type]
    pop = RecentPopularity(window_days=30)
    pop.fit(train)
    fallback = np.argsort(-pop.score(np.zeros(1, dtype=np.int64))[0], kind="stable")
    np.save(out_dir / "fallback.npy", fallback[:FALLBACK_SIZE].astype(np.int64))
    shutil.copy(onnx, out_dir / "user_tower.onnx")
    vectors = np.load(model_dir / "item_vectors.npy")
    if vectors.shape[0] != train.n_items:
        raise ValueError("item vectors were trained on a different catalog")
    np.save(out_dir / "item_vectors.npy", vectors.astype(np.float32))
    shutil.copy(index_dir / "items.faiss", out_dir / "items.faiss")
    shutil.copy(index_dir / "items.json", out_dir / "items.json")
    if not np.array_equal(np.load(index_dir / "item_ids.npy"), train.item_ids):
        raise ValueError("the FAISS index was built for a different catalog")
    shutil.copy(ranker_dir / "ranker.onnx", out_dir / "ranker.onnx")
    users = bulk_load(client, train.user_ids, build_sequences(train), max_len=max_len)
    # Request mix for load tests: validation users (those the evaluation scores).
    rng = np.random.default_rng(0)
    pool = train.user_ids[setup.targets.user_rows]
    sample = rng.choice(pool, size=min(LOADTEST_USERS, pool.size), replace=False)
    (out_dir / "loadtest_users.json").write_text(json.dumps([int(u) for u in sample]))
    meta = {
        "cutoff_ts": train.cutoff_ts,
        "max_len": max_len,
        "n_items": train.n_items,
        "users_in_redis": users,
    }
    (out_dir / "serving.json").write_text(json.dumps(meta, indent=2) + "\n")
    outputs = [
        out_dir / n
        for n in (
            "items.npz",
            "profiles.npz",
            "fallback.npy",
            "user_tower.onnx",
            "item_vectors.npy",
            "items.faiss",
            "items.json",
            "ranker.onnx",
            "loadtest_users.json",
            "serving.json",
        )
    ]
    write_manifest(
        out_dir,
        stage="serving",
        config={
            "split": str(split_dir),
            "onnx": str(onnx),
            "index": str(index_dir),
            "ranker": str(ranker_dir),
        },
        data_hash=str(verify_outputs(split_dir)["data_hash"]),
        seed=None,
        outputs=outputs,
    )
    return {"users_in_redis": users, "n_items": train.n_items}


def main(argv: list[str] | None = None) -> None:
    """Build serving artifacts and load user state into Redis."""
    s = get_settings()
    a = s.artifacts_dir
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--split-dir", type=Path, default=s.data_dir / "split" / "full")
    p.add_argument("--out-dir", type=Path, default=a / "serving")
    p.add_argument("--onnx", type=Path, default=a / "onnx" / "user_tower.onnx")
    p.add_argument("--model-dir", type=Path, default=a / "retrieval" / "two_tower_full")
    p.add_argument("--index-dir", type=Path, default=a / "index")
    p.add_argument("--ranker-dir", type=Path, default=a / "ranker")
    p.add_argument("--max-len", type=int, default=200)
    args = p.parse_args(argv)
    client = redis.Redis(host=s.redis_host, port=s.redis_port)
    print(
        run(
            args.split_dir,
            args.out_dir,
            onnx=args.onnx,
            model_dir=args.model_dir,
            index_dir=args.index_dir,
            ranker_dir=args.ranker_dir,
            client=client,
            max_len=args.max_len,
        )
    )


if __name__ == "__main__":
    main()
