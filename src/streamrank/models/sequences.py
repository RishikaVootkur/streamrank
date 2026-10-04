"""Per-user interaction sequences and item content features for the two-tower model.

Item indices are shifted by one so 0 can mark padding: catalog item `j` (column `j` of the
training matrices) is token `j + 1`.
"""

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import polars as pl
import scipy.sparse as sp

from streamrank.data.columns import GENRES
from streamrank.eval.dataset import TrainData

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float32]
PAD = 0
MAX_GAP_BUCKET = 32  # log2 buckets of seconds since the previous event
YEAR_BUCKET = 5  # years per bucket
FIRST_YEAR = 1870


@dataclass(frozen=True)
class Sequences:
    """Every training user's events in time order, stored as flat arrays with offsets."""

    indptr: IntArray  # user u's events are [indptr[u], indptr[u + 1])
    tokens: IntArray  # item index + 1
    positive: BoolArray  # rating >= 4
    gaps: IntArray  # bucket of seconds since the user's previous event (1 for the first)

    @property
    def n_users(self) -> int:
        return int(self.indptr.size - 1)

    def lengths(self) -> IntArray:
        out: IntArray = np.diff(self.indptr)
        return out


def gap_bucket(seconds: IntArray) -> IntArray:
    """Map non-negative gaps in seconds to buckets 2..MAX_GAP_BUCKET (1 is "first event")."""
    out: IntArray = np.minimum(
        2 + np.floor(np.log2(np.maximum(seconds, 0) + 1)).astype(np.int64), MAX_GAP_BUCKET
    )
    return out


def _tie_key(rows: IntArray, cols: IntArray) -> npt.NDArray[np.uint64]:
    """Deterministic pseudo-random key per (user, item) for ordering same-second events.

    MovieLens has many bulk ratings within one second; ordering them by item index would
    teach the model that the next item usually has a larger index.
    """
    with np.errstate(over="ignore"):
        x = rows.astype(np.uint64) * np.uint64(0x9E3779B97F4A7C15) + cols.astype(np.uint64)
        x ^= x >> np.uint64(31)
        x *= np.uint64(0xBF58476D1CE4E5B9)
        x ^= x >> np.uint64(29)
    return x


def build_sequences(train: TrainData) -> Sequences:
    """Sort every user's training interactions by time (same-second ties in hashed order)."""
    marked = sp.csr_array(train.interactions + 2 * train.positives)
    marked.sort_indices()
    ts = sp.csr_array(train.timestamps)
    ts.sort_indices()
    if not (
        np.array_equal(marked.indptr, ts.indptr) and np.array_equal(marked.indices, ts.indices)
    ):
        raise ValueError("timestamps and interactions must share a sparsity pattern")
    rows = np.repeat(np.arange(train.n_users, dtype=np.int64), np.diff(marked.indptr))
    cols = marked.indices.astype(np.int64)
    times = ts.data.astype(np.int64)
    order = np.lexsort((_tie_key(rows, cols), times, rows))
    rows, cols, times = rows[order], cols[order], times[order]
    positive = marked.data[order] >= 3  # noqa: PLR2004 - 1 (rated) + 2 (positive)
    indptr = np.zeros(train.n_users + 1, dtype=np.int64)
    np.add.at(indptr, rows + 1, 1)
    indptr = np.cumsum(indptr)
    first = np.zeros(rows.size, dtype=bool)
    first[indptr[:-1][np.diff(indptr) > 0]] = True
    prev = np.concatenate(([0], times[:-1]))
    gaps = np.where(first, 1, gap_bucket(times - prev))
    return Sequences(indptr=indptr, tokens=cols + 1, positive=positive, gaps=gaps)


@dataclass(frozen=True)
class Batch:
    """Left-padded model inputs; `targets[:, t]` is the token after `inputs[:, t]`."""

    inputs: IntArray  # (B, L)
    input_positive: BoolArray  # (B, L)
    input_gaps: IntArray  # (B, L)
    targets: IntArray  # (B, L), 0 where there is no next event
    target_mask: BoolArray  # (B, L), next event exists and is positive


def training_batch(
    seqs: Sequences, users: IntArray, max_len: int, rng: np.random.Generator
) -> Batch:
    """One random window of up to `max_len + 1` events per user.

    The window end is drawn uniformly from the positions that allow a full window (or the
    sequence end for short users), so long histories are covered across epochs. Users with
    fewer than two events get an all-padding row.
    """
    b = users.size
    inputs = np.zeros((b, max_len), dtype=np.int64)
    in_pos = np.zeros((b, max_len), dtype=bool)
    in_gap = np.zeros((b, max_len), dtype=np.int64)
    targets = np.zeros((b, max_len), dtype=np.int64)
    mask = np.zeros((b, max_len), dtype=bool)
    lengths = seqs.lengths()
    for i, user in enumerate(users):
        n = int(lengths[user])
        if n < 2:  # noqa: PLR2004 - need one input and one target
            continue
        lo = min(n, max_len + 1)
        stop = int(rng.integers(lo, n + 1))
        start = max(0, stop - max_len - 1)
        base = int(seqs.indptr[user])
        window = slice(base + start, base + stop)
        tok, pos, gap = seqs.tokens[window], seqs.positive[window], seqs.gaps[window]
        k = tok.size - 1
        off = max_len - k
        inputs[i, off:] = tok[:-1]
        in_pos[i, off:] = pos[:-1]
        in_gap[i, off:] = gap[:-1]
        targets[i, off:] = tok[1:]
        mask[i, off:] = pos[1:]
    return Batch(inputs, in_pos, in_gap, targets, mask)


def query_batch(seqs: Sequences, users: IntArray, max_len: int) -> Batch:
    """The last `max_len` events of each user, for scoring the next item."""
    b = users.size
    inputs = np.zeros((b, max_len), dtype=np.int64)
    in_pos = np.zeros((b, max_len), dtype=bool)
    in_gap = np.zeros((b, max_len), dtype=np.int64)
    lengths = seqs.lengths()
    for i, user in enumerate(users):
        n = int(lengths[user])
        k = min(n, max_len)
        window = slice(int(seqs.indptr[user]) + n - k, int(seqs.indptr[user]) + n)
        inputs[i, max_len - k :] = seqs.tokens[window]
        in_pos[i, max_len - k :] = seqs.positive[window]
        in_gap[i, max_len - k :] = seqs.gaps[window]
    empty = np.zeros((b, max_len), dtype=np.int64)
    return Batch(inputs, in_pos, in_gap, empty, empty.astype(bool))


@dataclass(frozen=True)
class ItemContent:
    """Content features per token (row 0 is padding)."""

    genres: FloatArray  # (n_items + 1, n_genres) multi-hot
    year: IntArray  # (n_items + 1,) year bucket, 0 when unknown
    tags: IntArray  # (n_items + 1, max_tags) tag vocabulary index + 1, 0 = none
    n_tags: int  # tag vocabulary size

    @property
    def n_year_buckets(self) -> int:
        return int(self.year.max()) + 1


def build_item_content(
    item_ids: IntArray,
    movies: pl.DataFrame,
    tags: pl.DataFrame,
    *,
    cutoff_ts: int,
    vocab_size: int = 1000,
    max_tags: int = 8,
) -> ItemContent:
    """Genres, release-year bucket, and top tags for each catalog item.

    Only tags applied before `cutoff_ts` are used, so content never reflects the future.
    """
    n = item_ids.size
    catalog = pl.DataFrame({"item_id": item_ids, "token": np.arange(1, n + 1, dtype=np.int64)})
    m = catalog.join(movies.select("item_id", "year", "genres"), on="item_id", how="left")
    genre_index = {g: i for i, g in enumerate(GENRES)}
    genres = np.zeros((n + 1, len(GENRES)), dtype=np.float32)
    year = np.zeros(n + 1, dtype=np.int64)
    for token, yr, gs in m.select("token", "year", "genres").iter_rows():
        for g in gs or ():
            if g in genre_index:
                genres[token, genre_index[g]] = 1.0
        if yr is not None:
            year[token] = 1 + max(0, int(yr) - FIRST_YEAR) // YEAR_BUCKET

    recent = tags.filter(pl.col("ts") < cutoff_ts).with_columns(
        tag=pl.col("tag").str.to_lowercase().str.strip_chars()
    )
    vocab = (
        recent.group_by("tag").len().sort(["len", "tag"], descending=[True, False]).head(vocab_size)
    )
    vocab = vocab.with_row_index("tag_index", offset=1).select("tag", "tag_index")
    per_item = (
        recent.join(vocab, on="tag")
        .join(catalog, on="item_id")
        .group_by("token", "tag_index")
        .len()
        .sort(["token", "len", "tag_index"], descending=[False, True, False])
        .group_by("token", maintain_order=True)
        .head(max_tags)
    )
    tag_arr = np.zeros((n + 1, max_tags), dtype=np.int64)
    slot = per_item.with_columns(slot=pl.int_range(pl.len()).over("token"))
    tag_arr[slot["token"].to_numpy(), slot["slot"].to_numpy()] = slot["tag_index"].to_numpy()
    return ItemContent(genres=genres, year=year, tags=tag_arr, n_tags=vocab.height)
