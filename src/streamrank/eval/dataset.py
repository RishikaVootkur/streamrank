"""Index mappings, training matrices, and evaluation targets built from a split.

The candidate catalog is every item with at least one training interaction. Relevant
items a model cannot recommend (first seen after the cut) still count in each user's
relevant set, so recall is honest about them; they get column indices past the catalog.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.sparse as sp

from streamrank.common.provenance import verify_outputs

IntArray = npt.NDArray[np.int64]


@dataclass(frozen=True)
class TrainData:
    """Training interactions as user-by-item matrices over contiguous indices."""

    user_ids: IntArray  # index -> user_id
    item_ids: IntArray  # index -> item_id (the candidate catalog)
    interactions: sp.csr_array  # 1.0 for every rated pair
    positives: sp.csr_array  # 1.0 for pairs rated >= 4
    timestamps: sp.csr_array  # Unix seconds of each rated pair
    cutoff_ts: int  # first timestamp after the training window

    @property
    def n_users(self) -> int:
        return int(self.user_ids.size)

    @property
    def n_items(self) -> int:
        return int(self.item_ids.size)

    def item_popularity(self) -> npt.NDArray[np.float64]:
        """Share of training interactions that each catalog item received."""
        counts = np.asarray(self.interactions.sum(axis=0)).ravel()
        out: npt.NDArray[np.float64] = counts / counts.sum()
        return out


@dataclass(frozen=True)
class EvalTargets:
    """Users to score, the items to exclude for them, and their relevant items."""

    user_rows: IntArray  # row indices into TrainData matrices
    relevant: sp.csr_array  # rows aligned with user_rows; columns >= n_items are unreachable
    exclude: sp.csr_array  # rows aligned with user_rows; items seen before the cut
    n_cold_users: int  # users with eval positives but no history; not scored here


def _csr(
    rows: IntArray, cols: IntArray, values: npt.ArrayLike, shape: tuple[int, int]
) -> sp.csr_array:
    m = sp.csr_array((np.asarray(values), (rows, cols)), shape=shape)
    m.sum_duplicates()
    m.sort_indices()
    return m


def build_train(history: pl.DataFrame, cutoff_ts: int) -> TrainData:
    """Index users and items in `history` and build its interaction matrices."""
    if history.is_empty():
        raise ValueError("history is empty")
    if int(history["ts"].max()) >= cutoff_ts:  # type: ignore[arg-type]
        raise ValueError("history contains interactions at or after the cutoff")
    user_ids = np.sort(history["user_id"].unique().to_numpy()).astype(np.int64)
    item_ids = np.sort(history["item_id"].unique().to_numpy()).astype(np.int64)
    rows = np.searchsorted(user_ids, history["user_id"].to_numpy()).astype(np.int64)
    cols = np.searchsorted(item_ids, history["item_id"].to_numpy()).astype(np.int64)
    shape = (user_ids.size, item_ids.size)
    pos = (history["rating"].to_numpy() >= 4.0).astype(np.float32)  # noqa: PLR2004
    positives = _csr(rows, cols, pos, shape)
    positives.eliminate_zeros()
    return TrainData(
        user_ids=user_ids,
        item_ids=item_ids,
        interactions=_csr(rows, cols, np.ones(rows.size, dtype=np.float32), shape),
        positives=positives,
        timestamps=_csr(rows, cols, history["ts"].to_numpy().astype(np.float64), shape),
        cutoff_ts=cutoff_ts,
    )


def build_targets(train: TrainData, eval_df: pl.DataFrame) -> EvalTargets:
    """Relevant items (eval positives) for warm users, with their history excluded.

    Warm users have at least one training interaction. Users with eval positives but no
    history are counted in `n_cold_users` and left out of these targets: no model here can
    personalize for them, and the serving layer handles them with a popularity fallback.
    """
    positives = eval_df.filter(pl.col("label") == 1)
    users = positives["user_id"].unique()
    warm_mask = users.is_in(pl.Series(train.user_ids).implode())
    warm_users = np.sort(users.filter(warm_mask).to_numpy()).astype(np.int64)
    n_cold = int((~warm_mask).sum())

    warm_pos = positives.filter(pl.col("user_id").is_in(pl.Series(warm_users).implode()))
    target_row = np.searchsorted(warm_users, warm_pos["user_id"].to_numpy())
    item_arr = warm_pos["item_id"].to_numpy()
    in_catalog_idx = np.searchsorted(train.item_ids, item_arr)
    in_catalog = (in_catalog_idx < train.n_items) & (
        train.item_ids[np.clip(in_catalog_idx, 0, train.n_items - 1)] == item_arr
    )
    new_items = np.unique(item_arr[~in_catalog])
    cols = np.where(
        in_catalog,
        in_catalog_idx,
        train.n_items + np.searchsorted(new_items, item_arr),
    ).astype(np.int64)
    relevant = _csr(
        target_row.astype(np.int64),
        cols,
        np.ones(cols.size, dtype=np.float32),
        (warm_users.size, train.n_items + new_items.size),
    )
    relevant.data[:] = 1.0

    user_rows = np.searchsorted(train.user_ids, warm_users).astype(np.int64)
    exclude = train.interactions[user_rows]
    return EvalTargets(
        user_rows=user_rows, relevant=relevant, exclude=sp.csr_array(exclude), n_cold_users=n_cold
    )


@dataclass(frozen=True)
class EvalSetup:
    """Training data and evaluation targets for one evaluation partition."""

    train: TrainData
    targets: EvalTargets
    partition: str


def load_setup(split_dir: Path, partition: str = "val") -> EvalSetup:
    """Build the evaluation setup for `val` (history = train) or `test` (train + val)."""
    manifest = verify_outputs(split_dir)
    if partition == "val":
        history = pl.read_parquet(split_dir / "train.parquet")
        cutoff = int(manifest["t1"])
    elif partition == "test":
        history = pl.concat(
            [
                pl.read_parquet(split_dir / "train.parquet"),
                pl.read_parquet(split_dir / "val.parquet"),
            ]
        )
        cutoff = int(manifest["t2"])
    else:
        raise ValueError(f"unknown partition {partition!r}")
    train = build_train(history, cutoff)
    targets = build_targets(train, pl.read_parquet(split_dir / f"{partition}.parquet"))
    return EvalSetup(train=train, targets=targets, partition=partition)
