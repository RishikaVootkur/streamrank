"""Turn model scores into top-K lists and evaluate them on a split partition."""

import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
import scipy.sparse as sp

from streamrank.eval import metrics
from streamrank.eval.dataset import EvalSetup, TrainData

IntArray = npt.NDArray[np.int64]
FloatArray = npt.NDArray[np.float32]
K_MAX = 200


class Scorer(Protocol):
    """Anything that scores the full catalog for a batch of training-user rows."""

    def score(self, user_rows: IntArray) -> FloatArray:
        """Return scores of shape (len(user_rows), n_items)."""
        ...


def top_k(scores: FloatArray, exclude: sp.csr_array, k: int) -> IntArray:
    """Top-k item indices per row, best first, never returning excluded items.

    Excluded items are masked to -inf; rows with fewer than k eligible items are padded
    with -1.
    """
    scores = np.array(scores, dtype=np.float32, copy=True)
    coo = exclude.tocoo()
    scores[coo.row, coo.col] = -np.inf
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    part_scores = np.take_along_axis(scores, part, axis=1)
    order = np.argsort(-part_scores, axis=1, kind="stable")
    ranked = np.take_along_axis(part, order, axis=1).astype(np.int64)
    ranked_scores = np.take_along_axis(part_scores, order, axis=1)
    ranked[~np.isfinite(ranked_scores) & (ranked_scores < 0)] = -1
    return ranked


def recommend(
    model: Scorer, user_rows: IntArray, exclude: sp.csr_array, k: int, batch_size: int = 1024
) -> IntArray:
    """Top-k lists for many users, scored in batches to bound memory.

    Lists are padded with -1 when the catalog has fewer than k eligible items.
    """
    out = np.full((user_rows.size, k), -1, dtype=np.int64)
    for start in range(0, user_rows.size, batch_size):
        stop = min(start + batch_size, user_rows.size)
        scores = model.score(user_rows[start:stop])
        ranked = top_k(scores, sp.csr_array(exclude[start:stop]), k)
        out[start:stop, : ranked.shape[1]] = ranked
    return out


@dataclass(frozen=True)
class EvalResult:
    """Metrics with bootstrap intervals plus list-level statistics for one model."""

    name: str
    params: dict[str, Any]
    metrics: dict[str, metrics.Interval]
    coverage_at_10: float
    gini_at_10: float
    n_users: int
    fit_seconds: float
    recommend_seconds: float
    per_user: dict[str, npt.NDArray[np.float64]] = field(repr=False)

    def row(self) -> dict[str, Any]:
        """A flat, JSON-friendly summary."""
        out: dict[str, Any] = {
            "model": self.name,
            "params": self.params,
            "n_users": self.n_users,
            "coverage@10": self.coverage_at_10,
            "gini@10": self.gini_at_10,
            "fit_seconds": round(self.fit_seconds, 2),
            "recommend_seconds": round(self.recommend_seconds, 2),
        }
        for name, iv in self.metrics.items():
            out[name] = {"mean": iv.mean, "low": iv.low, "high": iv.high}
        return out


def evaluate_lists(
    name: str,
    params: dict[str, Any],
    recs: IntArray,
    setup: EvalSetup,
    *,
    n_resamples: int = 1000,
    seed: int = 0,
    fit_seconds: float = 0.0,
    recommend_seconds: float = 0.0,
) -> EvalResult:
    """Score precomputed top-K lists against the setup's relevant items."""
    targets = setup.targets
    metrics.check_excluded(recs, targets.exclude)
    per_user = metrics.per_user_metrics(recs, targets.relevant)
    popularity = setup.train.item_popularity()
    per_user["arp@10"] = metrics.average_popularity(recs, popularity, 10)
    return EvalResult(
        name=name,
        params=params,
        metrics=metrics.summarize(per_user, n_resamples=n_resamples, seed=seed),
        coverage_at_10=metrics.catalog_coverage(recs, setup.train.n_items, 10),
        gini_at_10=metrics.gini_index(recs, setup.train.n_items, 10),
        n_users=int(recs.shape[0]),
        fit_seconds=fit_seconds,
        recommend_seconds=recommend_seconds,
        per_user=per_user,
    )


class Model(Scorer, Protocol):
    """A scorer that can be fitted on training data."""

    name: str

    def fit(self, train: TrainData) -> None:
        """Learn from training interactions."""


def fit_and_evaluate(
    model: Model,
    setup: EvalSetup,
    params: dict[str, Any] | None = None,
    n_resamples: int = 1000,
    seed: int = 0,
) -> EvalResult:
    """Fit a model on the setup's training data and evaluate its top-200 lists."""
    t0 = time.perf_counter()
    model.fit(setup.train)
    t1 = time.perf_counter()
    recs = recommend(model, setup.targets.user_rows, setup.targets.exclude, K_MAX)
    t2 = time.perf_counter()
    return evaluate_lists(
        model.name,
        params or {},
        recs,
        setup,
        n_resamples=n_resamples,
        seed=seed,
        fit_seconds=t1 - t0,
        recommend_seconds=t2 - t1,
    )
