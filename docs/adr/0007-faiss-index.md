# 7. HNSW index for candidate retrieval

Date: 2026-10-04

## Status

Accepted

## Context

The API retrieves 200 candidates per request from 65,723 two-tower item vectors (dimension 128, unit length, inner product) inside a 50 ms p99 budget for the whole request. The index must lose essentially no recall relative to exact search, because the ranker can only reorder what retrieval finds.

## Options considered

1. Exact search (`IndexFlatIP`): no recall loss, about 1 ms per query at this size, cost grows linearly with the catalog.
2. HNSW (`IndexHNSWFlat`, inner product): graph search; graph degree M, construction depth efConstruction, search depth efSearch.
3. IVF-PQ (`IVF{nlist},PQ32x8`): coarse clusters with 8-bit product-quantized codes, optionally re-ranked with exact vectors (`RFlat`). It uses about 10 times less memory without re-ranking.

## Decision

HNSW with M = 16, efConstruction = 200, efSearch = 400. The search fetches 400 items so that 200 remain after removing items the user has already rated, and efSearch must be at least the fetch size.

Benchmark on the final model's vectors, all 6,463 warm validation users as queries, single-threaded, one query at a time (`docs/faiss-benchmark.md`, plot `docs/faiss-recall-latency.png`):

| Index | recall@200 vs exact | Recall@100 | p50 ms | p99 ms | size MB |
| --- | ---: | ---: | ---: | ---: | ---: |
| Exact | 1.0000 | 0.2569 | 1.04 | 2.51 | 33.7 |
| **HNSW M=16, ef=400** | **0.9992** | **0.2569** | **0.27** | **0.39** | 43.1 |
| HNSW M=32, ef=400 | 0.9994 | 0.2569 | 0.37 | 0.62 | 51.5 |
| IVF256-PQ32, nprobe=32 | 0.8817 | 0.2513 | 0.22 | 0.43 | 2.9 |
| IVF256-PQ32 + refine, nprobe=32 | 0.9941 | 0.2568 | 0.70 | 3.81 | 36.5 |
| IVF1024-PQ32 + refine, nprobe=64 (latency suspect, see below) | 0.9942 | 0.2565 | 0.24 | 0.35 | 36.9 |

- HNSW at ef = 400 finds 99.92% of the exact top 200 and leaves downstream Recall@100 unchanged (0.2569, same as exact), at a quarter of exact search's median latency. Larger M or ef buys nothing measurable.
- IVF-PQ without re-ranking loses 10 to 31% of the exact neighbors and 0.006 to 0.023 Recall@100. Re-ranking recovers recall but needs the full vectors again (36.5 MB), which removes the memory advantage.
- Exact search would also meet the budget at this catalog size (p99 2.5 ms). HNSW is chosen for headroom: exact cost grows linearly with the catalog (an estimated 16 ms per query at a million items, extrapolated from the measured median, not measured).
- The serving engine re-scores the retrieved candidates exactly with the item vectors, so the retrieval-score feature the ranker sees does not depend on the index type.
- On macOS, FAISS cannot share a process with PyTorch or LightGBM (each bundles an OpenMP runtime), so vectors are exported by a separate PyTorch process and the benchmark imports FAISS only.

## Consequences

- `make build-index` exports vectors, runs the benchmark, and saves the chosen index (`INDEX_CHOICE`) with its item IDs; the API restores efSearch from the sidecar file.
- Latencies were measured on a laptop with other jobs running, so they are noisy, including some medians: IVF1024 with re-ranking measured a 0.46 ms median at nprobe = 32 and 0.24 ms at nprobe = 64, which cannot both be right since a larger nprobe does more work. The HNSW choice does not depend on those IVF rows; it rests on recall and on HNSW's consistent sub-millisecond medians across all settings.

## Sources

- FAISS guidelines for choosing an index: https://github.com/facebookresearch/faiss/wiki/Guidelines-to-choose-an-index
- FAISS indexes overview: https://github.com/facebookresearch/faiss/wiki/Faiss-indexes
- FAISS threading: https://github.com/facebookresearch/faiss/wiki/Threads-and-asynchronous-calls
- FAISS changelog (1.14 HNSW inner-product ordering): https://github.com/facebookresearch/faiss/blob/main/CHANGELOG.md
- Malkov and Yashunin, "Efficient and robust approximate nearest neighbor search using Hierarchical Navigable Small World graphs": https://arxiv.org/abs/1603.09320
- Jégou, Douze, Schmid, "Product quantization for nearest neighbor search": https://doi.org/10.1109/TPAMI.2010.57
- FAISS and PyTorch OpenMP conflict on macOS: https://github.com/facebookresearch/faiss/issues/4273
