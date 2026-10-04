"""Build, query, and store FAISS indexes over L2-normalized item vectors (inner product).

Three index kinds are supported:
- `flat`: exact brute force (`IndexFlatIP`), the ground truth.
- `hnsw`: graph index (`IndexHNSWFlat`, inner product), tuned by `m`, `ef_construction`,
  and the search-time `ef_search`.
- `ivfpq`: inverted lists with product-quantized codes (`IVF{nlist},PQ{m}x8`), tuned by
  `nlist`, `pq_m`, and the search-time `nprobe`, optionally re-ranked exactly (`refine`).
"""

import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import numpy.typing as npt

import faiss

FloatArray = npt.NDArray[np.float32]
IntArray = npt.NDArray[np.int64]
Kind = Literal["flat", "hnsw", "ivfpq"]


@dataclass(frozen=True)
class IndexSpec:
    """Index type with build-time and search-time parameters."""

    kind: Kind
    m: int = 32  # HNSW graph degree
    ef_construction: int = 200
    ef_search: int = 256
    nlist: int = 1024  # IVF lists
    pq_m: int = 32  # PQ sub-quantizers (must divide the dimension)
    nprobe: int = 32
    refine: bool = False  # re-rank IVF-PQ candidates with exact distances
    refine_factor: int = 4  # with refine: re-rank refine_factor * k PQ candidates
    train_size: int = 100_000

    def label(self) -> str:
        if self.kind == "flat":
            return "flat"
        if self.kind == "hnsw":
            return f"hnsw(M={self.m},efC={self.ef_construction},ef={self.ef_search})"
        tail = f",refine x{self.refine_factor}" if self.refine else ""
        return f"ivfpq(nlist={self.nlist},m={self.pq_m},nprobe={self.nprobe}{tail})"


@dataclass
class BuiltIndex:
    """A FAISS index plus how it was built."""

    spec: IndexSpec
    index: Any
    build_seconds: float
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def size_bytes(self) -> int:
        return int(faiss.serialize_index(self.index).size)


def build(vectors: FloatArray, spec: IndexSpec, seed: int = 0) -> BuiltIndex:
    """Build an index over `vectors` (rows are item vectors in catalog order)."""
    x = np.ascontiguousarray(vectors, dtype=np.float32)
    d = x.shape[1]
    t0 = time.perf_counter()
    index: Any
    if spec.kind == "flat":
        index = faiss.IndexFlatIP(d)
    elif spec.kind == "hnsw":
        index = faiss.IndexHNSWFlat(d, spec.m, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efConstruction = spec.ef_construction
    else:
        if d % spec.pq_m:
            raise ValueError(f"pq_m={spec.pq_m} must divide the dimension {d}")
        factory = f"IVF{spec.nlist},PQ{spec.pq_m}x8" + (",RFlat" if spec.refine else "")
        index = faiss.index_factory(d, factory, faiss.METRIC_INNER_PRODUCT)
        rng = np.random.default_rng(seed)
        sample = x[rng.choice(x.shape[0], size=min(spec.train_size, x.shape[0]), replace=False)]
        index.train(sample)
    index.add(x)
    set_search_params(index, spec)
    return BuiltIndex(spec, index, time.perf_counter() - t0)


def set_search_params(index: Any, spec: IndexSpec) -> None:
    """Apply the search-time parameters (`ef_search` or `nprobe`)."""
    if spec.kind == "hnsw":
        index.hnsw.efSearch = spec.ef_search
    elif spec.kind == "ivfpq":
        faiss.extract_index_ivf(index).nprobe = spec.nprobe
        if spec.refine:
            refine: Any = faiss.downcast_index(index)
            refine.k_factor = spec.refine_factor


def search(index: Any, queries: FloatArray, k: int) -> tuple[FloatArray, IntArray]:
    """Top-k item rows and scores per query (rows of -1 when fewer than k are found)."""
    q = np.ascontiguousarray(queries, dtype=np.float32)
    scores, rows = index.search(q, k)
    return scores.astype(np.float32), rows.astype(np.int64)


def recall_vs_exact(approx: IntArray, exact: IntArray) -> float:
    """Mean fraction of each query's exact top-k found by the approximate top-k."""
    k = exact.shape[1]
    hits = [np.intersect1d(a[a >= 0], e).size for a, e in zip(approx, exact, strict=True)]
    return float(np.mean(hits) / k)


def save(built: BuiltIndex, path: Path) -> None:
    """Write the index and a JSON sidecar with its spec."""
    import json  # noqa: PLC0415

    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(built.index, str(path))
    sidecar = {"spec": asdict(built.spec), "build_seconds": built.build_seconds, **built.meta}
    path.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")


def load(path: Path) -> BuiltIndex:
    """Read an index written by `save`, restoring its search-time parameters."""
    import json  # noqa: PLC0415

    sidecar = json.loads(path.with_suffix(".json").read_text())
    spec = IndexSpec(**sidecar["spec"])
    index = faiss.read_index(str(path))
    set_search_params(index, spec)
    return BuiltIndex(spec, index, float(sidecar["build_seconds"]))
