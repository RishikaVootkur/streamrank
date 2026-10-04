# 9. Serving architecture

Date: 2026-10-04

## Status

Accepted

## Context

A request must return 10 movies in under 50 ms at p99 on a laptop, using the user's latest events (including ones that arrived seconds ago through the stream), batch features from the feature store, the user tower, the index, and the ranker. Development happens on macOS, where PyTorch, FAISS, and LightGBM cannot share a process; the service itself runs in a Linux container.

## Options considered

1. Serve the PyTorch user tower and the LightGBM booster directly.
2. Export both models to ONNX and run them with ONNX Runtime, next to FAISS, in one FastAPI process.
3. A separate model server (Triton, TorchServe) behind the API. Extra moving parts for two small models.

## Decision

Option 2.

- One FastAPI process, one uvicorn worker per container. CPU work (ONNX Runtime, FAISS, NumPy) runs in a worker thread; each native library gets one thread (`OMP_NUM_THREADS=1`, ONNX Runtime intra-op 1, no spinning). Throughput scales by adding replicas.
- State per user lives in Redis: the last 200 events (for the user tower input) and the full seen set (for filtering), read with one `MGET`. The batch builder loads them from training history, and the streaming job appends new events, so a click reaches the next recommendation within about half a second (ADR 0010).
- Batch features: user statistics from the Feast online store per request; item statistics for all 65,723 items loaded from the online store at startup into NumPy arrays. The online store is materialized at the training cutoff, which is the serving clock, and the API refuses to start if any item statistic is newer than that clock.
- Retrieval: HNSW (ADR 0007), fetching past the user's seen items, with exact re-scoring of candidates from the item vectors. Users with more seen items than the fetch can cover are scored exactly over the whole catalog.
- Ranking: the same feature formulas as training, vectorized in NumPy (parity test against the offline Polars code), scored by the ONNX ranker (parity 9.2e-7 against LightGBM).
- Unknown users and users who have seen everything get a 30-day recent-popularity list.
- Operations: `/livez`, `/readyz` (after models, warm-up, and a Redis ping), Prometheus histograms per stage with buckets dense around 50 ms, JSON logs, OpenTelemetry traces when an OTLP endpoint is configured.
- Image: `python:3.12.13-slim-trixie` with the serve dependency group only (no PyTorch or LightGBM), non-root, Debian security updates applied (the CI Trivy scan fails on critical CVEs).

## Results

k6 against the container (2 CPUs): p99 6.7 ms at 250 requests per second with no errors or dropped requests; the tail degrades at 300 (p99 97 ms). Server-side median 4.8 ms, of which the user tower takes 3.0 ms. Details in `docs/load-test.md`. `make smoke` passes.

## Consequences

- Training-serving skew is guarded by four parity checks: ONNX user tower (1.8e-7), ONNX ranker (9.2e-7), NumPy against Polars features (1e-6), and left padding against the training query layout.
- The serving clock is the data clock (the training cutoff, moved forward by replayed events), so offline and online feature values agree.

## Sources

- FastAPI lifespan events: https://fastapi.tiangolo.com/advanced/events/
- ONNX Runtime threading: https://onnxruntime.ai/docs/performance/tune-performance/threading.html
- PyTorch ONNX export (dynamo): https://docs.pytorch.org/docs/stable/onnx_export.html
- Feast online server performance tuning: https://feast.dev/blog/feast-online-server-performance-tuning/
- Prometheus client multiprocess caveats: https://prometheus.github.io/client_python/multiprocess/
- k6 constant arrival rate executor: https://grafana.com/docs/k6/latest/using-k6/scenarios/executors/constant-arrival-rate/
