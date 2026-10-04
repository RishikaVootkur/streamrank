"""Recall versus latency for FAISS indexes over the trained item vectors.

For every index setting this measures, on the validation users' query vectors:
- recall@200 against exact top-200 (how much of the true candidate set the index finds),
- downstream Recall@100 of relevant items after removing each user's training items. Each
  user fetches K plus their number of seen items, so K unseen items remain and the metric
  equals the full-catalog evaluation for exact search,
- single-thread latency per query (p50 and p99), build time, and index size.

Vectors come from `streamrank.models.export_vectors` (this module never imports PyTorch:
on macOS the two libraries' OpenMP runtimes cannot share a process).
"""

import argparse
import json
import logging
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

import faiss
from streamrank.common.config import get_settings
from streamrank.common.logging import configure_logging
from streamrank.eval import metrics
from streamrank.eval.dataset import load_setup
from streamrank.retrieval.index import (
    FloatArray,
    IndexSpec,
    IntArray,
    build,
    recall_vs_exact,
    save,
    search,
    set_search_params,
)

log = logging.getLogger(__name__)
K = 200
OVERFETCH = 2  # latency is measured for a 2K fetch, the serving default


def default_grid(dim: int = 128) -> list[IndexSpec]:
    """HNSW and IVF-PQ settings spanning fast-and-rough to slow-and-exact.

    Search fetches 2K = 400 items, so HNSW's ef_search starts at 400.
    """
    grid = [IndexSpec("flat")]
    grid += [
        IndexSpec("hnsw", m=m, ef_construction=200, ef_search=ef)
        for m in (16, 32, 48)
        for ef in (400, 800, 1600)
    ]
    pq_m = dim // 4
    grid += [
        IndexSpec("ivfpq", nlist=nlist, pq_m=pq_m, nprobe=nprobe, refine=refine)
        for nlist in (256, 1024)
        for refine in (False, True)
        for nprobe in (8, 32, 64)
    ]
    return grid


def _build_key(spec: IndexSpec) -> IndexSpec:
    """The spec with search-time parameters reset, so one build serves several searches."""
    return replace(spec, ef_search=0, nprobe=0, refine_factor=0)


def filter_seen(rows: IntArray, exclude: sp.csr_array, k: int) -> IntArray:
    """Drop each user's excluded items from ranked rows and keep the first k (pad with -1)."""
    out = np.full((rows.shape[0], k), -1, dtype=np.int64)
    for i in range(rows.shape[0]):
        seen = exclude.indices[exclude.indptr[i] : exclude.indptr[i + 1]]
        keep = rows[i][(rows[i] >= 0) & ~np.isin(rows[i], seen)][:k]
        out[i, : keep.size] = keep
    return out


def search_unseen(
    index: Any, users: FloatArray, exclude: sp.csr_array, k: int, n_items: int
) -> IntArray:
    """Top-k unseen items per user: fetch k + seen count (in buckets), drop seen, keep k."""
    seen = np.diff(exclude.indptr)
    need = np.minimum(k + seen, n_items)
    bucket = np.minimum(2 ** np.ceil(np.log2(np.maximum(need, 1))).astype(np.int64), n_items)
    out = np.full((users.shape[0], k), -1, dtype=np.int64)
    for fetch in np.unique(bucket):
        idx = np.flatnonzero(bucket == fetch)
        _, raw = search(index, users[idx], int(fetch))
        out[idx] = filter_seen(raw, sp.csr_array(exclude[idx]), k)
    return out


def latency_ms(index: Any, queries: FloatArray, n: int, k: int) -> tuple[float, float]:
    """p50 and p99 single-query latency in milliseconds, one thread."""
    times = []
    for q in queries[:n]:
        t0 = time.perf_counter()
        index.search(q[None, :], k)
        times.append((time.perf_counter() - t0) * 1000)
    return float(np.percentile(times, 50)), float(np.percentile(times, 99))


def evaluate_grid(
    items: FloatArray,
    users: FloatArray,
    relevant: sp.csr_array,
    exclude: sp.csr_array,
    grid: list[IndexSpec],
    *,
    n_latency: int = 1000,
) -> list[dict[str, Any]]:
    """Benchmark every index setting against exact search."""
    n_rel = np.diff(relevant.indptr).astype(np.int64)
    exact_rows: IntArray | None = None
    rows_out = []
    cache: dict[IndexSpec, Any] = {}
    threads = faiss.omp_get_max_threads()
    for spec in grid:
        key = _build_key(spec)
        if key not in cache:
            faiss.omp_set_num_threads(threads)
            cache = {key: build(items, spec)}  # keep one build in memory at a time
        built = cache[key]
        set_search_params(built.index, spec)
        built.spec = spec
        faiss.omp_set_num_threads(1)
        _, raw = search(built.index, users, K * OVERFETCH)
        if exact_rows is None:
            if spec.kind != "flat":
                raise ValueError("the grid must start with the exact index")
            exact_rows = raw
        top = raw[:, :K]
        recs = search_unseen(built.index, users, exclude, K, items.shape[0])
        hits = metrics.hit_matrix(recs, relevant)
        p50, p99 = latency_ms(built.index, users, n_latency, K * OVERFETCH)
        row = {
            "label": spec.label(),
            "spec": asdict(spec),
            "recall@200_vs_exact": recall_vs_exact(top, exact_rows[:, :K]),
            "recall@100": float(metrics.recall_at_k(hits, n_rel, 100).mean()),
            "p50_ms": p50,
            "p99_ms": p99,
            "build_s": built.build_seconds,
            "size_mb": built.size_bytes / 1e6,
        }
        log.info("index", extra=row)
        rows_out.append(row)
    return rows_out


def plot(rows: list[dict[str, Any]], path: Path, n_items: int) -> None:
    """Recall@200 versus p99 latency, one line per index family."""
    import matplotlib as mpl  # noqa: PLC0415

    mpl.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(7, 4.5))
    families: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        s = r["spec"]
        key = {
            "flat": "exact (flat)",
            "hnsw": f"HNSW M={s['m']}",
            "ivfpq": f"IVF{s['nlist']}-PQ{s['pq_m']}" + (" + refine" if s["refine"] else ""),
        }[s["kind"]]
        families.setdefault(key, []).append(r)
    for key, family in families.items():
        rs = sorted(family, key=lambda r: r["p99_ms"])
        marker = "*" if key.startswith("exact") else "o"
        ax.plot(
            [r["p99_ms"] for r in rs],
            [r["recall@200_vs_exact"] for r in rs],
            marker=marker,
            label=key,
        )
    ax.set_xscale("log")
    ax.set_xlabel("p99 latency per query (ms, one thread, log scale)")
    ax.set_ylabel("recall@200 vs exact")
    ax.set_title(f"FAISS recall versus latency ({n_items:,} items)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def render_markdown(rows: list[dict[str, Any]], context: dict[str, Any]) -> str:
    """Results table for the docs."""
    lines = [
        "# FAISS index benchmark",
        "",
        f"Item vectors from `{context['model_dir']}` ({context['n_items']:,} items, "
        f"dimension {context['dim']}); {context['n_queries']:,} validation users as queries. "
        f"Each query fetches {K * OVERFETCH} items so that {K} remain after removing the "
        "user's training items. Latency is single-threaded, one query at a time, over "
        f"{context['n_latency']:,} queries.",
        "",
        "![Recall versus latency](faiss-recall-latency.png)",
        "",
        "| Index | recall@200 vs exact | Recall@100 | p50 ms | p99 ms | build s | size MB |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['label']} | {r['recall@200_vs_exact']:.4f} | {r['recall@100']:.4f} | "
            f"{r['p50_ms']:.2f} | {r['p99_ms']:.2f} | {r['build_s']:.1f} | {r['size_mb']:.1f} |"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    """Benchmark FAISS indexes and save the chosen one."""
    settings = get_settings()
    p = argparse.ArgumentParser(description=main.__doc__)
    p.add_argument("--model-dir", type=Path, required=True, help="has vectors_val.npz")
    p.add_argument("--split-dir", type=Path, default=settings.data_dir / "split" / "full")
    p.add_argument("--doc-dir", type=Path, default=Path("docs"))
    p.add_argument("--out-dir", type=Path, default=settings.artifacts_dir / "index")
    p.add_argument("--choose", default=None, help="label of the index to save")
    p.add_argument("--n-latency", type=int, default=1000)
    args = p.parse_args(argv)
    configure_logging(settings.log_level)

    setup = load_setup(args.split_dir, "val")
    vectors = np.load(args.model_dir / "vectors_val.npz")
    if not np.array_equal(vectors["user_rows"], setup.targets.user_rows):
        raise ValueError("vectors were exported for a different split")
    items, users = vectors["items"], vectors["users"]
    rows = evaluate_grid(
        items,
        users,
        setup.targets.relevant,
        setup.targets.exclude,
        default_grid(items.shape[1]),
        n_latency=args.n_latency,
    )
    context = {
        "model_dir": str(args.model_dir),
        "n_items": items.shape[0],
        "dim": items.shape[1],
        "n_queries": users.shape[0],
        "n_latency": args.n_latency,
    }
    args.doc_dir.mkdir(parents=True, exist_ok=True)
    plot(rows, args.doc_dir / "faiss-recall-latency.png", items.shape[0])
    (args.doc_dir / "faiss-benchmark.md").write_text(render_markdown(rows, context))
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "benchmark.json").write_text(
        json.dumps({"context": context, "rows": rows}, indent=2) + "\n"
    )
    if args.choose:
        labels = [r["label"] for r in rows]
        if args.choose not in labels:
            raise ValueError(f"unknown index {args.choose!r}; choose one of {labels}")
        chosen = rows[labels.index(args.choose)]
        faiss.omp_set_num_threads(faiss.omp_get_max_threads())
        built = build(items, IndexSpec(**chosen["spec"]))
        faiss.omp_set_num_threads(1)
        _, exact = search(build(items, IndexSpec("flat")).index, users, K)
        _, got = search(built.index, users, K)
        chosen = {
            **chosen,
            "saved_recall@200_vs_exact": recall_vs_exact(got, exact),
            "saved_build_s": built.build_seconds,
        }
        built.meta = {"model_dir": str(args.model_dir), "benchmark": chosen}
        save(built, args.out_dir / "items.faiss")
        np.save(args.out_dir / "item_ids.npy", vectors["item_ids"])


if __name__ == "__main__":
    main()
