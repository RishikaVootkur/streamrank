import numpy as np
import pytest
import scipy.sparse as sp

from streamrank.eval.dataset import EvalSetup, EvalTargets, TrainData
from streamrank.eval.evaluate import fit_and_evaluate, recommend, top_k


def test_top_k_orders_and_masks() -> None:
    scores = np.array([[0.1, 0.9, 0.5, 0.7], [1.0, 0.0, 0.2, 0.3]], dtype=np.float32)
    exclude = sp.csr_array(np.array([[0, 1, 0, 0], [1, 0, 0, 0]], dtype=np.float32))
    assert top_k(scores, exclude, 2).tolist() == [[3, 2], [3, 2]]


def test_top_k_pads_when_too_few_items() -> None:
    scores = np.array([[0.3, 0.2, 0.1]], dtype=np.float32)
    exclude = sp.csr_array(np.array([[1, 0, 1]], dtype=np.float32))
    assert top_k(scores, exclude, 3).tolist() == [[1, -1, -1]]


def test_top_k_does_not_modify_input() -> None:
    scores = np.ones((1, 3), dtype=np.float32)
    top_k(scores, sp.csr_array(np.array([[1, 0, 0]], dtype=np.float32)), 1)
    assert np.all(scores == 1)


class _Fixed:
    name = "fixed"

    def __init__(self, scores: np.ndarray) -> None:
        self.scores = scores.astype(np.float32)
        self.fitted = False

    def fit(self, train: object) -> None:
        self.fitted = True

    def score(self, user_rows: np.ndarray) -> np.ndarray:
        return self.scores[user_rows]


def test_recommend_batches_match_single_pass() -> None:
    rng = np.random.default_rng(0)
    scores = rng.random((7, 20))
    exclude = sp.csr_array((rng.random((7, 20)) < 0.2).astype(np.float32))
    rows = np.arange(7, dtype=np.int64)
    full = recommend(_Fixed(scores), rows, exclude, 5, batch_size=100)
    batched = recommend(_Fixed(scores), rows, exclude, 5, batch_size=3)
    assert np.array_equal(full, batched)


def _setup() -> EvalSetup:
    inter = sp.csr_array(np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32))
    train = TrainData(
        user_ids=np.array([1, 2]),
        item_ids=np.array([10, 11, 12, 13]),
        interactions=inter,
        positives=inter,
        timestamps=inter,
        cutoff_ts=100,
    )
    targets = EvalTargets(
        user_rows=np.array([0, 1]),
        relevant=sp.csr_array(np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)),
        exclude=inter,
        n_cold_users=0,
    )
    return EvalSetup(train=train, targets=targets, partition="val")


def test_fit_and_evaluate_perfect_model() -> None:
    model = _Fixed(np.array([[0, 0, 9, 1], [0, 0, 1, 9]]))
    result = fit_and_evaluate(model, _setup(), params={"a": 1}, n_resamples=50)
    assert model.fitted
    assert result.metrics["recall@10"].mean == pytest.approx(1.0)
    assert result.metrics["ndcg@10"].mean == pytest.approx(1.0)
    assert result.metrics["mrr"].mean == pytest.approx(1.0)
    assert result.n_users == 2
    row = result.row()
    assert row["model"] == "fixed"
    assert row["params"] == {"a": 1}
    assert row["recall@10"]["low"] <= row["recall@10"]["mean"] <= row["recall@10"]["high"]
    assert 0 < row["coverage@10"] <= 1


def test_recommend_pads_small_catalogs() -> None:
    scores = np.array([[0.2, 0.1]], dtype=np.float32)
    exclude = sp.csr_array((1, 2), dtype=np.float32)
    recs = recommend(_Fixed(scores), np.array([0]), exclude, 4)
    assert recs.tolist() == [[0, 1, -1, -1]]


def test_row_reports_cold_users() -> None:
    model = _Fixed(np.array([[0, 0, 9, 1], [0, 0, 1, 9]]))
    assert fit_and_evaluate(model, _setup(), n_resamples=10).row()["n_cold_users"] == 0
