import numpy as np
import polars as pl
import pytest

from streamrank.data.columns import GENRES
from streamrank.eval.dataset import build_train
from streamrank.models.sequences import (
    MAX_GAP_BUCKET,
    build_item_content,
    build_sequences,
    gap_bucket,
    query_batch,
    training_batch,
)


@pytest.fixture
def history() -> pl.DataFrame:
    # User 1: items 10 (t=100, pos), 30 (t=50, neg), 20 (t=400, pos). User 2: 20 then 30.
    return pl.DataFrame(
        {
            "user_id": [1, 1, 1, 2, 2, 3],
            "item_id": [10, 30, 20, 20, 30, 10],
            "rating": [5.0, 2.0, 4.0, 4.5, 3.0, 4.0],
            "ts": [100, 50, 400, 10, 20, 5],
        }
    )


def test_gap_buckets() -> None:
    assert gap_bucket(np.array([0, 1, 3, 2**40])).tolist() == [2, 3, 4, MAX_GAP_BUCKET]


def test_sequences_are_time_ordered(history: pl.DataFrame) -> None:
    seqs = build_sequences(build_train(history, cutoff_ts=1000))
    assert seqs.n_users == 3
    assert seqs.lengths().tolist() == [3, 2, 1]
    # Tokens are catalog index + 1 (items 10, 20, 30 -> 1, 2, 3).
    u1 = slice(seqs.indptr[0], seqs.indptr[1])
    assert seqs.tokens[u1].tolist() == [3, 1, 2]
    assert seqs.positive[u1].tolist() == [False, True, True]
    assert seqs.gaps[u1].tolist() == [1, *gap_bucket(np.array([50, 300])).tolist()]


def test_training_batch_targets_are_next_tokens(history: pl.DataFrame) -> None:
    seqs = build_sequences(build_train(history, cutoff_ts=1000))
    batch = training_batch(seqs, np.array([0, 1, 2]), max_len=4, rng=np.random.default_rng(0))
    assert batch.inputs.shape == (3, 4)
    # User 1 fits entirely: inputs [3, 1], targets [1, 2], both targets positive.
    assert batch.inputs[0].tolist() == [0, 0, 3, 1]
    assert batch.targets[0].tolist() == [0, 0, 1, 2]
    assert batch.target_mask[0].tolist() == [False, False, True, True]
    # User 2's only target (item 30) is negative, so nothing is trained on.
    assert batch.targets[1].tolist() == [0, 0, 0, 3]
    assert not batch.target_mask[1].any()
    # User 3 has a single event: an all-padding row.
    assert not batch.inputs[2].any()


def test_training_windows_cover_long_histories() -> None:
    n = 50
    history = pl.DataFrame(
        {
            "user_id": [1] * n,
            "item_id": list(range(100, 100 + n)),
            "rating": [5.0] * n,
            "ts": list(range(n)),
        }
    )
    seqs = build_sequences(build_train(history, cutoff_ts=1000))
    rng = np.random.default_rng(1)
    firsts = set()
    for _ in range(200):
        batch = training_batch(seqs, np.array([0]), max_len=8, rng=rng)
        row = batch.inputs[0]
        assert (row > 0).all()  # full windows for long users
        assert (np.diff(row) == 1).all()  # contiguous, in time order
        assert batch.targets[0, -1] == row[-1] + 1
        firsts.add(int(row[0]))
    assert min(firsts) == 1 and max(firsts) == n - 8


def test_query_batch_uses_latest_events(history: pl.DataFrame) -> None:
    seqs = build_sequences(build_train(history, cutoff_ts=1000))
    batch = query_batch(seqs, np.array([0, 2]), max_len=2)
    assert batch.inputs.tolist() == [[1, 2], [0, 1]]
    assert batch.input_positive.tolist() == [[True, True], [False, True]]


def test_item_content_uses_only_past_tags() -> None:
    movies = pl.DataFrame(
        {
            "item_id": [10, 20, 30],
            "title": ["a", "b", "c"],
            "year": [1995, None, 1871],
            "genres": [["Drama", "Comedy"], [], ["Unknown"]],
        }
    )
    tags = pl.DataFrame(
        {
            "user_id": [1, 1, 2, 2, 3],
            "item_id": [10, 10, 20, 10, 30],
            "tag": ["Funny ", "funny", "dark", "future", "old"],
            "ts": [1, 2, 3, 99, 4],
        }
    )
    content = build_item_content(np.array([10, 20, 30]), movies, tags, cutoff_ts=50, max_tags=2)
    assert content.genres.shape == (4, len(GENRES))
    assert content.genres[1, GENRES.index("Drama")] == 1.0
    assert content.genres[1].sum() == 2
    assert content.genres[3].sum() == 0  # unknown genre ignored
    assert content.year.tolist() == [0, 1 + (1995 - 1870) // 5, 0, 1]
    # "future" (ts 99) is after the cutoff; "funny" is normalized and most frequent.
    assert content.n_tags == 3
    assert content.tags[1].tolist() == [1, 0]
    assert content.tags[2, 0] > 0 and content.tags[3, 0] > 0
    assert content.n_year_buckets == content.year.max() + 1


def test_same_second_ties_are_not_ordered_by_item() -> None:
    n = 40
    history = pl.DataFrame(
        {"user_id": [1] * n, "item_id": list(range(n)), "rating": [5.0] * n, "ts": [7] * n}
    )
    seqs = build_sequences(build_train(history, cutoff_ts=100))
    tokens = seqs.tokens.tolist()
    assert sorted(tokens) == list(range(1, n + 1))
    assert tokens != sorted(tokens)
    # Deterministic across builds.
    assert build_sequences(build_train(history, cutoff_ts=100)).tokens.tolist() == tokens
