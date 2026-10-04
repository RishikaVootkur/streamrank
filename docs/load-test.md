# Serving load test

Setup: the API container (`make up`) limited to 2 CPUs with one uvicorn worker, on an Apple Silicon laptop under Docker Desktop (8 GB VM). k6 2.3.0 runs in the same Compose network (`make load`) with a constant-arrival-rate scenario: 20 s of warm-up at a quarter of the rate, then the measured phase. 90% of requests are validation users from the serving build's user list; 10% are unknown users who get the popularity fallback. Each request asks for `k = 10`.

Latency is measured by k6 per HTTP request (client side, so it includes the Docker network hop). The 250 request-per-second row was re-run after fixing the dropped-iteration threshold (it now keys on the scenario name), on an otherwise idle machine.

| Rate (req/s) | Duration | p50 ms | p95 ms | p99 ms | max ms | Errors | Dropped | p99 < 50 ms |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :---: |
| 100 | 2 min | 5.41 | 6.56 | 7.60 | 18.9 | 0 | 0 | yes |
| 200 | 1 min | 4.98 | 5.82 | 6.92 | 12.2 | 0 | 0 | yes |
| **250** | 1 min | **5.20** | **6.00** | **6.84** | 41.7 | 0 | 0 | **yes** |
| 300 | 1 min | 5.11 | 6.36 | 96.8 | 256 | 0 | 4 | no |
| 500 | 1 min | 933 | 1,065 | 1,136 | 1,585 | 0 | 11,030 | no |

"Dropped" counts requests k6 could not start on schedule because earlier ones were still running. The service holds a 50 ms p99 with nothing dropped up to at least 250 requests per second on 2 CPUs. At 300 the median is unchanged but queueing appears in the tail. At 500 the container is saturated: only about 316 requests per second were actually served (18,970 in the 60-second load phase), so that row describes overload, not a 500 request-per-second service. More throughput means more replicas behind a load balancer (the Kubernetes deployment in M10 adds a HorizontalPodAutoscaler), not a faster request.

## Where the time goes

Server-side stage timings over 200 sequential requests from known users (the API returns them per request):

| Stage | p50 ms | p95 ms |
| --- | ---: | ---: |
| Redis state read (sequence and seen set) | 0.12 | 0.16 |
| User tower (ONNX Runtime, 200 events) | 2.95 | 3.60 |
| Retrieval (HNSW, exact re-scoring) | 0.56 | 1.43 |
| User features (Feast online store) | 0.22 | 0.30 |
| Ranking (features and ONNX ranker, 200 candidates) | 0.86 | 1.00 |
| **Total** | **4.82** | **6.50** |

The user tower dominates: a 2-layer transformer over 200 positions on one CPU thread. Unknown users skip the models and return in about 2.6 ms.

Raw k6 summaries are written to `artifacts/serving/load_summary.json` by `make load` (not committed).
