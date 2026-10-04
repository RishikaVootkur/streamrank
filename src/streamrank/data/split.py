"""Global temporal split into train, validation, and test, plus a deterministic user sample.

Every interaction before T1 is train, T1 <= t < T2 is validation, and t >= T2 is test.
The cut points are interaction-count quantiles of the timestamp, so the same fractions
work on the real and synthetic data. No user-level split is used: per-user splits leak
other users' future behavior into training.
"""

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import polars as pl

from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.common.provenance import (
    begin_stage,
    verify_outputs,
    write_manifest,
    write_parquet_atomic,
)
from streamrank.data.schemas import POSITIVE_THRESHOLD

log = logging.getLogger(__name__)

PARTITIONS: Final = ("train", "val", "test")
_HASH_MULTIPLIER: Final = 2_654_435_761  # Knuth multiplicative hash, stable across versions
_SEED_MULTIPLIER: Final = 0x9E3779B9  # golden-ratio constant to spread seeds over 32 bits
_HASH_MODULUS: Final = 2**32


class LeakageError(AssertionError):
    """Raised when a split lets later interactions into an earlier partition."""


@dataclass(frozen=True)
class SplitConfig:
    """Fractions of interactions (by time) held out for validation and test."""

    val_fraction: float = 0.05
    test_fraction: float = 0.05
    sample_pct: int = 10
    seed: int = 42

    def __post_init__(self) -> None:
        if not (self.val_fraction > 0 and self.test_fraction > 0):
            raise ValueError("validation and test fractions must be positive")
        if self.val_fraction + self.test_fraction >= 1:
            raise ValueError("validation and test fractions must leave data for training")
        if not 0 < self.sample_pct <= 100:  # noqa: PLR2004 - percent bound
            raise ValueError("sample_pct must be in (0, 100]")


@dataclass(frozen=True)
class Split:
    """The three partitions and their cut points (Unix seconds)."""

    train: pl.DataFrame
    val: pl.DataFrame
    test: pl.DataFrame
    t1: int
    t2: int

    def partitions(self) -> dict[str, pl.DataFrame]:
        """Return the partitions keyed by name."""
        return {"train": self.train, "val": self.val, "test": self.test}


def cut_points(ratings: pl.DataFrame, cfg: SplitConfig) -> tuple[int, int]:
    """Return (T1, T2) as timestamp quantiles leaving the requested fractions after them."""
    ts = ratings["ts"]
    t2 = int(ts.quantile(1 - cfg.test_fraction, interpolation="lower") or 0)
    t1 = int(ts.quantile(1 - cfg.test_fraction - cfg.val_fraction, interpolation="lower") or 0)
    if not t1 < t2:
        raise ValueError(f"degenerate cut points T1={t1}, T2={t2}")
    return t1, t2


def temporal_split(ratings: pl.DataFrame, cfg: SplitConfig) -> Split:
    """Split ratings by global time and add the binary `label` (rating >= 4)."""
    t1, t2 = cut_points(ratings, cfg)
    labeled = ratings.with_columns(
        (pl.col("rating") >= POSITIVE_THRESHOLD).cast(pl.Int8).alias("label")
    ).sort(["ts", "user_id", "item_id"])
    split = Split(
        train=labeled.filter(pl.col("ts") < t1),
        val=labeled.filter((pl.col("ts") >= t1) & (pl.col("ts") < t2)),
        test=labeled.filter(pl.col("ts") >= t2),
        t1=t1,
        t2=t2,
    )
    check_no_leakage(split)
    return split


def check_no_leakage(split: Split) -> None:
    """Raise `LeakageError` if any partition overlaps a later one in time or rows."""
    bounds = {name: (df["ts"].min(), df["ts"].max()) for name, df in split.partitions().items()}
    for name, df in split.partitions().items():
        if df.is_empty():
            raise LeakageError(f"{name} partition is empty")
    if not bounds["train"][1] < split.t1 <= bounds["val"][0]:  # type: ignore[operator]
        raise LeakageError(f"train reaches past T1: {bounds['train']} vs T1={split.t1}")
    if not bounds["val"][1] < split.t2 <= bounds["test"][0]:  # type: ignore[operator]
        raise LeakageError(f"validation reaches past T2: {bounds['val']} vs T2={split.t2}")
    # Ratings are unique per user-item pair, so a pair in two partitions (or twice in one)
    # means rows were copied between partitions.
    keys = [df.select("user_id", "item_id") for df in split.partitions().values()]
    total = sum(k.height for k in keys)
    if pl.concat(keys).unique().height != total:
        raise LeakageError("a user-item pair appears more than once across partitions")


def sample_users(df: pl.DataFrame, pct: int, seed: int) -> pl.DataFrame:
    """Keep a deterministic `pct` percent of users, chosen by a stable hash of (user, seed).

    The seed is mixed in before the multiplicative hash and the bucket comes from the
    high bits, so different seeds give close to independent samples.
    """
    offset = (seed * _SEED_MULTIPLIER) % _HASH_MODULUS
    mixed = ((pl.col("user_id").cast(pl.UInt64) + offset) % _HASH_MODULUS) * _HASH_MULTIPLIER
    bucket = ((mixed % _HASH_MODULUS) // 2**16) % 100
    return df.filter(bucket < pct)


def _iso(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=UTC).isoformat()


def split_stats(split: Split) -> dict[str, Any]:
    """Counts per partition, including users and items not seen in training."""
    train_users = split.train["user_id"].unique().implode()
    train_items = split.train["item_id"].unique().implode()
    stats: dict[str, Any] = {"t1": _iso(split.t1), "t2": _iso(split.t2), "partitions": {}}
    for name, df in split.partitions().items():
        users = df["user_id"].unique()
        items = df["item_id"].unique()
        part = {
            "interactions": df.height,
            "positives": int(df["label"].sum()),
            "users": users.len(),
            "items": items.len(),
            "start": _iso(int(df["ts"].min())),  # type: ignore[arg-type]
            "end": _iso(int(df["ts"].max())),  # type: ignore[arg-type]
        }
        if name != "train":
            warm = df.filter(pl.col("user_id").is_in(train_users))
            part["users_with_train_history"] = warm["user_id"].n_unique()
            part["users_without_train_history"] = users.len() - warm["user_id"].n_unique()
            part["items_unseen_in_train"] = int((~items.is_in(train_items)).sum())
            part["positives_from_warm_users"] = int(warm["label"].sum())
        stats["partitions"][name] = part
    return stats


def write_split(split: Split, out_dir: Path, cfg: SplitConfig, data_hash: str) -> dict[str, Any]:
    """Write partitions as Parquet with a manifest holding the config and statistics."""
    begin_stage(out_dir)
    outputs = []
    for name, df in split.partitions().items():
        path = out_dir / f"{name}.parquet"
        write_parquet_atomic(df, path)
        outputs.append(path)
    stats = split_stats(split)
    write_manifest(
        out_dir,
        stage="split",
        config=asdict(cfg),
        data_hash=data_hash,
        seed=cfg.seed,
        outputs=outputs,
        extra={"t1": split.t1, "t2": split.t2, "stats": stats},
    )
    return stats


def render_stats_markdown(
    full: dict[str, Any], sample: dict[str, Any], cfg: SplitConfig, source: str, data_hash: str
) -> str:
    """Render split statistics as a Markdown document."""
    lines = [
        "# Data split statistics",
        "",
        f"Generated by `make split` from {source}. Global temporal split: train is every",
        "interaction before T1, validation is T1 to T2, test is after T2. A rating of 4.0 or",
        "higher is a positive label. Dates are UTC.",
        "",
        f"- Source data hash (SHA-256 of the raw files): `{data_hash}`",
        f"- Validation fraction: {cfg.val_fraction:.0%} of interactions by time",
        f"- Test fraction: {cfg.test_fraction:.0%} of interactions by time",
        f"- T1: {full['t1']}",
        f"- T2: {full['t2']}",
        "",
    ]
    for title, stats in (("Full data", full), (f"{cfg.sample_pct}% user sample", sample)):
        lines += [
            f"## {title}",
            "",
            "| Partition | Interactions | Positives | Users | Items | Start | End |",
            "| --- | ---: | ---: | ---: | ---: | --- | --- |",
        ]
        for name in PARTITIONS:
            p = stats["partitions"][name]
            lines.append(
                f"| {name} | {p['interactions']:,} | {p['positives']:,} | {p['users']:,} "
                f"| {p['items']:,} | {p['start'][:10]} | {p['end'][:10]} |"
            )
        lines += [
            "",
            "| Partition | Users with train history | Users without | Items unseen in train "
            "| Positives from warm users |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for name in PARTITIONS[1:]:
            p = stats["partitions"][name]
            lines.append(
                f"| {name} | {p['users_with_train_history']:,} "
                f"| {p['users_without_train_history']:,} | {p['items_unseen_in_train']:,} "
                f"| {p['positives_from_warm_users']:,} |"
            )
        lines.append("")
    return "\n".join(lines)


def run(processed_dir: Path, split_dir: Path, cfg: SplitConfig) -> dict[str, Any]:
    """Split the processed ratings and write the full split and the user sample."""
    data_hash = str(verify_outputs(processed_dir)["data_hash"])
    ratings = pl.read_parquet(processed_dir / "ratings.parquet")
    split = temporal_split(ratings, cfg)
    full = write_split(split, split_dir / "full", cfg, data_hash)

    sampled = Split(
        **{
            name: sample_users(df, cfg.sample_pct, cfg.seed)
            for name, df in split.partitions().items()
        },
        t1=split.t1,
        t2=split.t2,
    )
    check_no_leakage(sampled)
    sample = write_split(sampled, split_dir / f"sample{cfg.sample_pct}", cfg, data_hash)
    log.info("split written", extra={"t1": full["t1"], "t2": full["t2"]})
    return {"full": full, "sample": sample, "data_hash": data_hash}


def main(argv: list[str] | None = None) -> None:
    """Create the temporal split from processed ratings."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument("--processed-dir", type=Path, default=settings.data_dir / "processed")
    parser.add_argument("--split-dir", type=Path, default=settings.data_dir / "split")
    parser.add_argument("--stats-doc", type=Path, default=None, help="write Markdown stats here")
    parser.add_argument("--source", default="MovieLens 32M", help="data source name for the doc")
    parser.add_argument("--val-fraction", type=float, default=SplitConfig.val_fraction)
    parser.add_argument("--test-fraction", type=float, default=SplitConfig.test_fraction)
    args = parser.parse_args(argv)
    configure_logging(settings.log_level)
    cfg = SplitConfig(
        val_fraction=args.val_fraction, test_fraction=args.test_fraction, seed=settings.seed
    )
    stats = run(args.processed_dir, args.split_dir, cfg)
    if args.stats_doc:
        args.stats_doc.write_text(
            render_stats_markdown(
                stats["full"], stats["sample"], cfg, args.source, stats["data_hash"]
            )
        )
    log.info("split stats", extra={"stats": json.dumps(stats["full"]["partitions"])})


if __name__ == "__main__":
    main()
