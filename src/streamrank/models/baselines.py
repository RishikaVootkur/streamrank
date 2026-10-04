"""Non-neural baselines: popularity, recent popularity, item KNN, EASE, and ALS.

Every model scores the full candidate catalog for a batch of users. Models can learn
from every rated pair ("all": a rating means the user watched the movie) or only from
positives ("positive": rated >= 4); the choice is tuned on validation.
"""

import os
from dataclasses import dataclass, field
from typing import Literal, Protocol

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.sparse as sp

from streamrank.eval.dataset import TrainData

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
Signal = Literal["all", "positive"]
_SECONDS_PER_DAY = 86_400


class Recommender(Protocol):
    """A model that scores every catalog item for a batch of users."""

    name: str

    def fit(self, train: TrainData) -> None:
        """Learn from training interactions."""

    def score(self, user_rows: IntArray) -> FloatArray:
        """Return scores of shape (len(user_rows), n_items); higher is better."""
        ...


def _signal_matrix(train: TrainData, signal: Signal) -> sp.csr_array:
    return train.interactions if signal == "all" else train.positives


@dataclass
class _Fitted:
    """Holds the training user matrix that score() needs."""

    matrix: sp.csr_array | None = None

    def rows(self, user_rows: IntArray) -> sp.csr_array:
        if self.matrix is None:
            raise RuntimeError("call fit() first")
        return sp.csr_array(self.matrix[user_rows])


@dataclass
class Popularity:
    """Score every item by its number of training interactions."""

    signal: Signal = "all"
    name: str = "popularity"
    _counts: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def fit(self, train: TrainData) -> None:
        x = _signal_matrix(train, self.signal)
        self._counts = np.asarray(x.sum(axis=0), dtype=np.float32).ravel()

    def score(self, user_rows: IntArray) -> FloatArray:
        return np.broadcast_to(self._counts, (user_rows.size, self._counts.size)).copy()


@dataclass
class RecentPopularity:
    """Score items by interactions in the last `window_days` before the cutoff."""

    window_days: int = 90
    signal: Signal = "all"
    name: str = "recent_popularity"
    _counts: FloatArray = field(default_factory=lambda: np.zeros(0, dtype=np.float32))

    def fit(self, train: TrainData) -> None:
        start = train.cutoff_ts - self.window_days * _SECONDS_PER_DAY
        ts = train.timestamps.tocoo()
        x = _signal_matrix(train, self.signal)
        keep = ts.data >= start
        recent = sp.csr_array(
            (np.ones(int(keep.sum()), dtype=np.float32), (ts.row[keep], ts.col[keep])),
            shape=ts.shape,
        ).multiply(x > 0)
        counts = np.asarray(recent.sum(axis=0), dtype=np.float32).ravel()
        # Break ties among items unseen in the window by all-time popularity.
        overall = np.asarray(x.sum(axis=0), dtype=np.float32).ravel()
        self._counts = counts + overall / (overall.max() + 1.0)

    def score(self, user_rows: IntArray) -> FloatArray:
        return np.broadcast_to(self._counts, (user_rows.size, self._counts.size)).copy()


@dataclass
class ItemKNN:
    """Item-based nearest neighbors with shrunk cosine similarity.

    s(i, j) = x_i . x_j / (|x_i| |x_j| + shrink), keeping the top `neighbors` per item.
    """

    neighbors: int = 200
    shrink: float = 10.0
    signal: Signal = "all"
    block_size: int = 1024
    name: str = "item_knn"
    _sim: sp.csr_array | None = None
    _users: _Fitted = field(default_factory=_Fitted)

    def fit(self, train: TrainData) -> None:
        x = sp.csc_array(_signal_matrix(train, self.signal), dtype=np.float32)
        n = x.shape[1]
        norms = np.sqrt(np.asarray(x.multiply(x).sum(axis=0), dtype=np.float32).ravel())
        xt = sp.csr_array(x.T)
        k = min(self.neighbors, n - 1)
        rows: list[npt.NDArray[np.int64]] = []
        cols: list[npt.NDArray[np.int64]] = []
        vals: list[FloatArray] = []
        for start in range(0, n, self.block_size):
            stop = min(start + self.block_size, n)
            gram = (xt[start:stop] @ x).toarray()
            denom = norms[start:stop, None] * norms[None, :] + self.shrink
            sim = np.divide(gram, denom, out=np.zeros_like(gram), where=denom > 0)
            sim[np.arange(stop - start), np.arange(start, stop)] = 0.0
            top = np.argpartition(-sim, k - 1, axis=1)[:, :k]
            top_vals = np.take_along_axis(sim, top, axis=1)
            keep = top_vals > 0
            rows.append(np.repeat(np.arange(start, stop), k)[keep.ravel()])
            cols.append(top[keep])
            vals.append(top_vals[keep])
        self._sim = sp.csr_array(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)
        )
        self._users = _Fitted(sp.csr_array(_signal_matrix(train, self.signal), dtype=np.float32))

    def score(self, user_rows: IntArray) -> FloatArray:
        if self._sim is None:
            raise RuntimeError("call fit() first")
        out: FloatArray = (self._users.rows(user_rows) @ self._sim).toarray().astype(np.float32)
        return out


@dataclass
class EASE:
    """Embarrassingly Shallow Autoencoder (Steck 2019) on the `max_items` most popular items.

    B = I - P / diag(P) with P = (X^T X + lambda I)^-1 and diag(B) = 0. Items outside the
    restricted set score lowest.
    """

    l2: float = 500.0
    max_items: int = 20_000
    signal: Signal = "all"
    name: str = "ease"
    _weights: npt.NDArray[np.float32] | None = None
    _kept: IntArray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    _n_items: int = 0
    _users: _Fitted = field(default_factory=_Fitted)

    def fit(self, train: TrainData) -> None:
        x = sp.csr_array(_signal_matrix(train, self.signal), dtype=np.float64)
        counts = np.asarray(x.sum(axis=0)).ravel()
        kept = np.sort(np.argsort(-counts, kind="stable")[: self.max_items]).astype(np.int64)
        xk = sp.csc_array(x)[:, kept]
        gram = (xk.T @ xk).toarray()
        gram[np.diag_indices_from(gram)] += self.l2
        p = scipy.linalg.inv(gram, overwrite_a=True, check_finite=False)
        b = p / (-np.diag(p))
        b[np.diag_indices_from(b)] = 0.0
        self._weights = b.astype(np.float32)
        self._kept = kept
        self._n_items = x.shape[1]
        self._users = _Fitted(sp.csr_array(sp.csc_array(x, dtype=np.float32)[:, kept]))

    def score(self, user_rows: IntArray) -> FloatArray:
        if self._weights is None:
            raise RuntimeError("call fit() first")
        restricted = self._users.rows(user_rows) @ self._weights
        out = np.full((user_rows.size, self._n_items), -np.inf, dtype=np.float32)
        out[:, self._kept] = restricted
        return out


@dataclass
class ALS:
    """Implicit-feedback matrix factorization (Hu, Koren, Volinsky 2008) via `implicit`."""

    factors: int = 128
    regularization: float = 0.05
    alpha: float = 2.0
    iterations: int = 15
    signal: Signal = "all"
    seed: int = 42
    name: str = "als"
    _user_factors: FloatArray | None = None
    _item_factors: FloatArray | None = None

    def fit(self, train: TrainData) -> None:
        os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
        from implicit.als import AlternatingLeastSquares  # noqa: PLC0415 - heavy optional import

        model = AlternatingLeastSquares(
            factors=self.factors,
            regularization=self.regularization,
            alpha=self.alpha,
            iterations=self.iterations,
            random_state=self.seed,
        )
        x = sp.csr_matrix(_signal_matrix(train, self.signal), dtype=np.float32)
        model.fit(x, show_progress=False)
        self._user_factors = np.asarray(model.user_factors, dtype=np.float32)
        self._item_factors = np.asarray(model.item_factors, dtype=np.float32)

    def score(self, user_rows: IntArray) -> FloatArray:
        if self._user_factors is None or self._item_factors is None:
            raise RuntimeError("call fit() first")
        out: FloatArray = self._user_factors[user_rows] @ self._item_factors.T
        return out
