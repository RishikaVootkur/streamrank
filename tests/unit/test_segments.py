import numpy as np

from streamrank.eval.segments import (
    HISTORY_EDGES,
    HISTORY_LABELS,
    VectorScorer,
    bucket,
    segment_table,
)


def test_bucket_edges_are_inclusive() -> None:
    labels = bucket(np.array([1, 20, 21, 100, 101, 500, 501.0]), HISTORY_EDGES, HISTORY_LABELS)
    assert labels.tolist() == ["1-20", "1-20", "21-100", "21-100", "101-500", "101-500", ">500"]


def test_vector_scorer_and_segment_table() -> None:
    users = np.eye(2, 3, dtype=np.float32)
    items = np.eye(3, dtype=np.float32)
    scorer = VectorScorer(users, items, np.array([10, 20]))
    assert scorer.score(np.array([20, 10])).tolist() == [[0, 1, 0], [1, 0, 0]]
    rng = np.random.default_rng(0)
    tt, ease = rng.random(100), rng.random(100)
    labels = np.array(["a"] * 60 + ["b"] * 40)
    rows = segment_table(
        diff=tt - ease, tt=tt, ease=ease, labels=labels, order=["a", "b", "c"], seed=0
    )
    assert [r["segment"] for r in rows] == ["a", "b"]  # "c" has no users
    assert rows[0]["users"] == 60
    assert abs(rows[0]["diff"].mean - (tt[:60] - ease[:60]).mean()) < 1e-12
