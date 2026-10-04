from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

from streamrank.eval.evaluate import top_k
from streamrank.retrieval.benchmark import (
    evaluate_grid,
    filter_seen,
    plot,
    render_markdown,
    search_unseen,
)
from streamrank.retrieval.index import IndexSpec, build, load, recall_vs_exact, save, search


@pytest.fixture(scope="module")
def vectors() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(40, 32))
    items = (centers[rng.integers(0, 40, 3000)] + 0.4 * rng.normal(size=(3000, 32))).astype(
        np.float32
    )
    items /= np.linalg.norm(items, axis=1, keepdims=True)
    users = rng.normal(size=(50, 32)).astype(np.float32)
    users /= np.linalg.norm(users, axis=1, keepdims=True)
    return items, users


def test_flat_is_exact(vectors: tuple[np.ndarray, np.ndarray]) -> None:
    items, users = vectors
    _, rows = search(build(items, IndexSpec("flat")).index, users, 10)
    expected = np.argsort(-(users @ items.T), axis=1)[:, :10]
    assert recall_vs_exact(rows, expected) == 1.0


@pytest.mark.parametrize(
    "spec",
    [
        IndexSpec("hnsw", m=16, ef_construction=100, ef_search=200),
        IndexSpec("ivfpq", nlist=32, pq_m=8, nprobe=16, refine=True, train_size=3000),
    ],
    ids=lambda s: s.kind,
)
def test_approximate_indexes_find_most_neighbors(
    vectors: tuple[np.ndarray, np.ndarray], spec: IndexSpec
) -> None:
    items, users = vectors
    _, exact = search(build(items, IndexSpec("flat")).index, users, 50)
    built = build(items, spec)
    _, rows = search(built.index, users, 50)
    assert recall_vs_exact(rows, exact) > 0.8
    assert built.size_bytes > 0
    assert spec.label().startswith(spec.kind)


def test_pq_m_must_divide_dimension(vectors: tuple[np.ndarray, np.ndarray]) -> None:
    with pytest.raises(ValueError, match="divide"):
        build(vectors[0], IndexSpec("ivfpq", pq_m=7))


def test_save_and_load_keep_search_params(
    vectors: tuple[np.ndarray, np.ndarray], tmp_path: Path
) -> None:
    items, users = vectors
    built = build(items, IndexSpec("hnsw", m=8, ef_search=123))
    save(built, tmp_path / "x.faiss")
    loaded = load(tmp_path / "x.faiss")
    assert loaded.spec == built.spec
    assert loaded.index.hnsw.efSearch == 123
    np.testing.assert_array_equal(
        search(loaded.index, users, 5)[1], search(built.index, users, 5)[1]
    )


def test_filter_seen_drops_excluded_and_pads() -> None:
    rows = np.array([[3, 1, 2, -1], [0, 1, 2, 3]])
    exclude = sp.csr_array(np.array([[0, 1, 0, 1], [0, 0, 0, 0]], dtype=np.float32))
    assert filter_seen(rows, exclude, 3).tolist() == [[2, -1, -1], [0, 1, 2]]


def test_grid_report(vectors: tuple[np.ndarray, np.ndarray], tmp_path: Path) -> None:
    items, users = vectors
    rng = np.random.default_rng(1)
    relevant = sp.csr_array((rng.random((50, 3000)) < 0.01).astype(np.float32))
    relevant[0, 0] = 1.0
    relevant = sp.csr_array(relevant)
    keep = np.diff(relevant.indptr) > 0
    relevant = sp.csr_array(relevant[keep])
    exclude = sp.csr_array((rng.random((int(keep.sum()), 3000)) < 0.005).astype(np.float32))
    grid = [IndexSpec("flat"), IndexSpec("hnsw", m=8, ef_search=400)]
    rows = evaluate_grid(items, users[keep], relevant, exclude, grid, n_latency=20)
    assert rows[0]["recall@200_vs_exact"] == 1.0
    assert rows[1]["recall@200_vs_exact"] > 0.8
    plot(rows, tmp_path / "p.png", 3000)
    assert (tmp_path / "p.png").stat().st_size > 0
    text = render_markdown(
        rows, {"model_dir": "m", "n_items": 3000, "dim": 32, "n_queries": 10, "n_latency": 20}
    )
    assert "| flat | 1.0000 |" in text
    with pytest.raises(ValueError, match="exact"):
        evaluate_grid(items, users[keep], relevant, exclude, grid[1:], n_latency=5)


def test_search_unseen_matches_full_catalog_ranking(vectors: tuple[np.ndarray, np.ndarray]) -> None:
    items, users = vectors
    rng = np.random.default_rng(2)
    # Some users have seen most of the catalog, more than any fixed overfetch would cover.
    seen = rng.random((users.shape[0], items.shape[0])) < rng.random((users.shape[0], 1)) * 0.9
    exclude = sp.csr_array(seen.astype(np.float32))
    got = search_unseen(build(items, IndexSpec("flat")).index, users, exclude, 50, items.shape[0])
    expected = top_k(users @ items.T, exclude, 50)
    np.testing.assert_array_equal(got, expected)
