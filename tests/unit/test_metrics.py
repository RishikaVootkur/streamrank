import math

import numpy as np
import pytest
import scipy.sparse as sp

from streamrank.eval import metrics as m


def _rel(rows: list[list[int]], n_items: int) -> sp.csr_array:
    data, r, c = [], [], []
    for u, items in enumerate(rows):
        for i in items:
            r.append(u)
            c.append(i)
            data.append(1)
    return sp.csr_array((data, (r, c)), shape=(len(rows), n_items))


def test_hit_matrix_marks_relevant_and_ignores_padding() -> None:
    recs = np.array([[3, 1, -1], [0, 2, 4]])
    rel = _rel([[1, 2], [4]], 5)
    assert m.hit_matrix(recs, rel).tolist() == [[False, True, False], [False, False, True]]


def test_hit_matrix_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="same number"):
        m.hit_matrix(np.zeros((2, 3), dtype=np.int64), _rel([[1]], 5))


def test_recall_ndcg_mrr_hand_computed() -> None:
    hits = np.array([[True, False, True, False], [False, False, False, False]])
    n_rel = np.array([3, 2])
    np.testing.assert_allclose(m.recall_at_k(hits, n_rel, 2), [0.5, 0.0])
    np.testing.assert_allclose(m.recall_at_k(hits, n_rel, 4), [2 / 3, 0.0])
    dcg = 1 + 1 / math.log2(4)
    idcg = 1 + 1 / math.log2(3) + 1 / math.log2(4)
    np.testing.assert_allclose(m.ndcg_at_k(hits, n_rel, 4), [dcg / idcg, 0.0])
    np.testing.assert_allclose(m.mrr(hits), [1.0, 0.0])


def test_recall_normalizes_by_min_k_and_relevant() -> None:
    hits = np.ones((1, 10), dtype=bool)
    assert m.recall_at_k(hits, np.array([50]), 10)[0] == 1.0
    assert m.ndcg_at_k(hits, np.array([50]), 10)[0] == pytest.approx(1.0)


def test_users_without_relevant_items_score_zero() -> None:
    hits = np.zeros((1, 5), dtype=bool)
    assert m.recall_at_k(hits, np.array([0]), 5)[0] == 0.0
    assert m.ndcg_at_k(hits, np.array([0]), 5)[0] == 0.0


def test_mrr_uses_first_hit() -> None:
    hits = np.array([[False, False, True, True]])
    assert m.mrr(hits)[0] == pytest.approx(1 / 3)


def test_coverage_popularity_and_gini() -> None:
    recs = np.array([[0, 1], [0, 2]])
    assert m.catalog_coverage(recs, 4, 2) == 0.75
    pop = np.array([0.5, 0.2, 0.1, 0.0])
    np.testing.assert_allclose(m.average_popularity(recs, pop, 2), [0.35, 0.3])
    uniform = np.array([[0, 1], [2, 3]])
    assert m.gini_index(uniform, 4, 2) == pytest.approx(0.0)
    concentrated = np.array([[0, -1], [0, -1]])
    assert m.gini_index(concentrated, 4, 2) == pytest.approx(0.75)
    assert m.gini_index(np.full((2, 2), -1), 4, 2) == 0.0


def test_bootstrap_interval_contains_mean_and_is_reproducible() -> None:
    rng = np.random.default_rng(0)
    values = rng.random(500)
    a = m.bootstrap_mean(values, n_resamples=300, seed=1)
    b = m.bootstrap_mean(values, n_resamples=300, seed=1)
    assert a == b
    assert a.low < a.mean < a.high
    assert a.high - a.low < 0.1
    assert "[" in str(a)
    with pytest.raises(ValueError, match="empty"):
        m.bootstrap_mean(np.array([]))


def test_bootstrap_chunking_matches_single_pass(monkeypatch: pytest.MonkeyPatch) -> None:
    values = np.arange(100, dtype=np.float64)
    full = m.bootstrap_mean(values, n_resamples=50, seed=3)
    monkeypatch.setattr(m, "_BOOTSTRAP_CHUNK", 250)
    assert m.bootstrap_mean(values, n_resamples=50, seed=3) == full


def test_paired_difference_detects_real_gap() -> None:
    rng = np.random.default_rng(0)
    base = rng.random(1000)
    better = base + 0.05
    diff = m.paired_difference(better, base, n_resamples=200)
    assert diff.low > 0
    same = m.paired_difference(base, base.copy(), n_resamples=200)
    assert same.low <= 0 <= same.high
    with pytest.raises(ValueError, match="same shape"):
        m.paired_difference(base, base[:10])


def test_duplicate_and_out_of_range_items_rejected() -> None:
    rel = _rel([[0], [0]], 5)
    with pytest.raises(ValueError, match="repeats"):
        m.hit_matrix(np.array([[0, 0, 1], [1, 2, 3]]), rel)
    with pytest.raises(ValueError, match="out of range"):
        m.hit_matrix(np.array([[5], [0]]), rel)
    with pytest.raises(ValueError, match="out of range"):
        m.hit_matrix(np.array([[-2], [0]]), rel)
    m.hit_matrix(np.array([[1, -1, -1], [2, 3, -1]]), rel)  # repeated padding is fine


def test_check_excluded() -> None:
    exclude = _rel([[1], []], 5)
    m.check_excluded(np.array([[0, 2], [1, 3]]), exclude)
    with pytest.raises(ValueError, match="already interacted"):
        m.check_excluded(np.array([[1, 2], [0, 3]]), exclude)
    with pytest.raises(ValueError, match="same number"):
        m.check_excluded(np.array([[1, 2]]), exclude)


def test_users_without_recommendations_rejected_for_popularity() -> None:
    with pytest.raises(ValueError, match="no recommendations"):
        m.average_popularity(np.array([[-1, -1]]), np.array([0.5]), 2)


def test_per_user_metrics_and_summary() -> None:
    n_items = 300
    recs = np.tile(np.arange(200), (2, 1))
    rel = _rel([[0, 150, 299], [250]], n_items)
    out = m.per_user_metrics(recs, rel)
    assert set(out) == {"recall@10", "recall@50", "recall@100", "recall@200", "ndcg@10", "mrr"}
    np.testing.assert_allclose(out["recall@200"], [2 / 3, 0.0])
    np.testing.assert_allclose(out["mrr"], [1.0, 0.0])
    summary = m.summarize(out, n_resamples=50)
    assert summary["mrr"].mean == pytest.approx(0.5)
    with pytest.raises(ValueError, match="at least 200"):
        m.per_user_metrics(recs[:, :100], rel)
    with pytest.raises(ValueError, match="at least 10"):
        m.per_user_metrics(recs[:, :5], rel, ks=(5,))
    with pytest.raises(ValueError, match="relevant item"):
        m.per_user_metrics(recs, _rel([[0], []], n_items))
