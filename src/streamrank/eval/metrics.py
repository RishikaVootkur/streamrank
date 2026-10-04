"""Per-user ranking metrics on full-catalog top-K lists, with bootstrap confidence intervals.

All metrics are computed per user first, so intervals can be bootstrapped over users and
two models can be compared with a paired bootstrap on the same users.

Definitions (binary relevance, `rel` = the user's relevant items, `hits@K` = relevant items
among the top K):

- Recall@K = hits@K / min(K, |rel|), so a perfect list scores 1 even when |rel| > K.
- NDCG@K = DCG@K / IDCG@K with gain 1 and discount 1 / log2(rank + 1); IDCG uses
  min(K, |rel|) relevant items at the top.
- MRR = 1 / rank of the first relevant item within the full list, 0 if none.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def hit_matrix(recs: IntArray, relevant: sp.csr_array) -> BoolArray:
    """Return a boolean matrix marking which recommended items are relevant.

    `recs` has shape (n_users, K) with item indices (or -1 for padding); row `u` of
    `relevant` holds user `u`'s relevant items.
    """
    n_users = recs.shape[0]
    if relevant.shape[0] != n_users:
        raise ValueError("recs and relevant must have the same number of users")
    n_items = relevant.shape[1]
    validate_recs(recs, n_items)
    coo = relevant.tocoo()
    rel_keys = np.sort(coo.row.astype(np.int64) * n_items + coo.col.astype(np.int64))
    rec_keys = np.arange(n_users, dtype=np.int64)[:, None] * n_items + recs
    hits: BoolArray = np.isin(rec_keys, rel_keys) & (recs >= 0)
    return hits


def validate_recs(recs: IntArray, n_items: int) -> None:
    """Reject lists with out-of-range items or an item repeated within one user's list."""
    if recs.size and (recs.max() >= n_items or recs.min() < -1):
        raise ValueError("recommended item index out of range")
    ordered = np.sort(recs, axis=1)
    repeated = (np.diff(ordered, axis=1) == 0) & (ordered[:, 1:] >= 0)
    if repeated.any():
        raise ValueError("a user's list repeats an item")


def check_excluded(recs: IntArray, exclude: sp.csr_array) -> None:
    """Raise if any recommended item is one the user already interacted with."""
    if exclude.shape[0] != recs.shape[0]:
        raise ValueError("recs and exclude must have the same number of users")
    if exclude.nnz and hit_matrix(recs, exclude).any():
        raise ValueError("recommendations include items the user already interacted with")


def recall_at_k(hits: BoolArray, n_relevant: IntArray, k: int) -> FloatArray:
    """Recall@K per user, normalized by min(K, |rel|)."""
    denom = np.minimum(k, n_relevant)
    out: FloatArray = hits[:, :k].sum(axis=1) / np.maximum(denom, 1)
    return np.where(denom > 0, out, 0.0)


def ndcg_at_k(hits: BoolArray, n_relevant: IntArray, k: int) -> FloatArray:
    """NDCG@K per user with binary gains."""
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    dcg = (hits[:, :k] * discounts).sum(axis=1)
    ideal_counts = np.minimum(k, n_relevant)
    cumulative = np.concatenate([[0.0], np.cumsum(discounts)])
    idcg = cumulative[ideal_counts]
    out: FloatArray = np.where(idcg > 0, dcg / np.where(idcg > 0, idcg, 1.0), 0.0)
    return out


def mrr(hits: BoolArray) -> FloatArray:
    """Reciprocal rank of the first hit per user, 0 when there is no hit."""
    any_hit = hits.any(axis=1)
    first = hits.argmax(axis=1)
    out: FloatArray = np.where(any_hit, 1.0 / (first + 1), 0.0)
    return out


def catalog_coverage(recs: IntArray, n_items: int, k: int) -> float:
    """Share of the catalog that appears in at least one user's top K."""
    top = recs[:, :k]
    return float(np.unique(top[top >= 0]).size / n_items)


def average_popularity(recs: IntArray, item_popularity: FloatArray, k: int) -> FloatArray:
    """Per-user mean popularity (share of training interactions) of the top-K items."""
    top = recs[:, :k]
    if (top < 0).all(axis=1).any():
        raise ValueError("a user has no recommendations")
    pop = np.where(top >= 0, item_popularity[np.clip(top, 0, None)], np.nan)
    out: FloatArray = np.nanmean(pop, axis=1)
    return out


@dataclass(frozen=True)
class Interval:
    """A point estimate with a percentile bootstrap confidence interval."""

    mean: float
    low: float
    high: float

    def __str__(self) -> str:
        return f"{self.mean:.4f} [{self.low:.4f}, {self.high:.4f}]"


def gini_index(recs: IntArray, n_items: int, k: int) -> float:
    """Gini coefficient of how often each catalog item is recommended in the top K.

    0 means every item is recommended equally often; values near 1 mean a few items
    take almost all recommendation slots.
    """
    top = recs[:, :k]
    counts = np.sort(np.bincount(top[top >= 0], minlength=n_items)).astype(np.float64)
    total = counts.sum()
    if total == 0:
        return 0.0
    ranks = np.arange(1, n_items + 1)
    return float(((2 * ranks - n_items - 1) * counts).sum() / (n_items * total))


_BOOTSTRAP_CHUNK = 1 << 24  # resampled indices held in memory at once


def bootstrap_mean(
    values: FloatArray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    """Mean of per-user values with a percentile bootstrap interval over users."""
    if values.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    n = values.size
    per_chunk = max(1, _BOOTSTRAP_CHUNK // n)
    means = np.empty(n_resamples)
    for start in range(0, n_resamples, per_chunk):
        stop = min(start + per_chunk, n_resamples)
        idx = rng.integers(0, n, size=(stop - start, n))
        means[start:stop] = values[idx].mean(axis=1)
    alpha = (1 - confidence) / 2
    low, high = np.quantile(means, [alpha, 1 - alpha])
    return Interval(float(values.mean()), float(low), float(high))


def paired_difference(
    a: FloatArray, b: FloatArray, n_resamples: int = 1000, confidence: float = 0.95, seed: int = 0
) -> Interval:
    """Mean of a - b over the same users with a paired bootstrap interval.

    The difference is significant at the given level when the interval excludes 0.
    """
    if a.shape != b.shape:
        raise ValueError("paired arrays must have the same shape")
    return bootstrap_mean(a - b, n_resamples, confidence, seed)


RECALL_KS: tuple[int, ...] = (10, 50, 100, 200)


def per_user_metrics(
    recs: IntArray, relevant: sp.csr_array, ks: tuple[int, ...] = RECALL_KS
) -> dict[str, FloatArray]:
    """Compute every per-user metric for a batch of top-K lists.

    Every user must have at least one relevant item; users without any are not part of
    the evaluation population and would bias the means toward zero.
    """
    needed = max(*ks, 10)
    if recs.shape[1] < needed:
        raise ValueError(f"need at least {needed} recommendations per user")
    n_rel = np.diff(relevant.indptr).astype(np.int64)
    if (n_rel == 0).any():
        raise ValueError("every evaluated user needs at least one relevant item")
    hits = hit_matrix(recs, relevant)
    out = {f"recall@{k}": recall_at_k(hits, n_rel, k) for k in ks}
    out["ndcg@10"] = ndcg_at_k(hits, n_rel, 10)
    out["mrr"] = mrr(hits)
    return out


def summarize(
    per_user: Mapping[str, FloatArray],
    n_resamples: int = 1000,
    seed: int = 0,
    aggregate: Callable[[FloatArray], Interval] | None = None,
) -> dict[str, Interval]:
    """Bootstrap every per-user metric."""
    agg = aggregate or (lambda v: bootstrap_mean(v, n_resamples=n_resamples, seed=seed))
    return {name: agg(values) for name, values in per_user.items()}
