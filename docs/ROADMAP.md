# Roadmap

Each milestone is tracked by a GitHub issue and lands through one or more pull requests.

## Milestones

- [x] M0 Scaffold, CI, guards
- [x] M1 Data ingest, validation, temporal split, synthetic generator
- [x] M2 Evaluation library and baselines
- [x] M3 PySpark offline features and Feast
- [x] M4 Two-tower retrieval with sequential user tower
- [x] M5 FAISS index
- [x] M6 LightGBM LambdaMART ranker
- [x] M7 Serving API
- [x] M8 Streaming session features
- [x] M9 Online test simulation
- [x] M10 Observability, demo UI, Kubernetes
- [x] M11 Final run, docs, release

## Scope notes

- Kubernetes (kind with Helm and HPA) is in scope: the development machine has 16 GB of RAM.

## Results

### Data split (M1)

Global temporal split of MovieLens 32M: T1 = 2020-11-05, T2 = 2022-02-01 (5% of interactions each for validation and test). Details in [data-split.md](data-split.md) and [ADR 0002](adr/0002-global-temporal-split.md).

| Partition | Interactions | Users | Users with train history |
| --- | ---: | ---: | ---: |
| train | 28,800,182 | 186,687 | |
| val | 1,600,010 | 14,113 | 6,742 |
| test | 1,600,012 | 13,718 | 5,199 |

### Baselines (M2)

Full split, validation partition: 6,463 warm users ranked against all 65,723 catalog items, training items excluded. Mean with 95% bootstrap interval over users. Tuned on the 10% sample by Recall@100. Details in [baselines.md](baselines.md) and [ADR 0003](adr/0003-evaluation-and-baselines.md).

| Model | Recall@10 | Recall@100 | Recall@200 | NDCG@10 |
| --- | ---: | ---: | ---: | ---: |
| Popularity | 0.0665 [0.0632, 0.0701] | 0.1411 [0.1368, 0.1455] | 0.1998 [0.1940, 0.2053] | 0.0695 [0.0660, 0.0731] |
| Recent popularity (30 days) | 0.1150 [0.1106, 0.1197] | 0.2341 [0.2281, 0.2403] | 0.3268 [0.3201, 0.3338] | 0.1155 [0.1114, 0.1199] |
| Item KNN | 0.1066 [0.1024, 0.1109] | 0.2311 [0.2252, 0.2370] | 0.3384 [0.3319, 0.3452] | 0.1087 [0.1044, 0.1130] |
| EASE | 0.1464 [0.1415, 0.1515] | **0.2854 [0.2795, 0.2912]** | 0.3902 [0.3833, 0.3965] | 0.1484 [0.1432, 0.1534] |
| ALS | 0.1300 [0.1253, 0.1347] | 0.2643 [0.2583, 0.2701] | 0.3670 [0.3607, 0.3737] | 0.1281 [0.1234, 0.1325] |

### Offline features (M3)

`make features` on MovieLens 32M (snapshots from 2019-11-01): Spark in Docker writes 1,279,732 user and 5,552,647 item snapshot rows in 66 s, then Feast loads 288,533 entity keys into Redis (118 s end to end). Spark output equals the reference definitions on every row in parity tests, and 50 random real rows recomputed from raw ratings had 0 mismatches. See [ADR 0004](adr/0004-offline-features-spark.md) and [ADR 0005](adr/0005-feature-store-feast.md).

### Streaming session features (M8)

10,000 replayed MovieLens events after the training cutoff (631 users): online session features in Redis equal the offline batch computation for every user (0 mismatches). Click-to-Redis latency p50 529 ms, p95 540 ms. See [ADR 0010](adr/0010-streaming-session-features.md).

### Two-tower retrieval (M4)

Validation users outside the 10% sample (5,816 users), full catalog, mean with 95% bootstrap interval; difference is paired against EASE on the same users. Trained on MPS, logged to MLflow. Details in [ADR 0006](adr/0006-two-tower-retrieval.md) and [segment analysis](two-tower-segments.md).

| Model | Recall@100 | Recall@200 | NDCG@10 |
| --- | ---: | ---: | ---: |
| Two-tower (sequential user tower) | 0.2562 [0.2494, 0.2628] | 0.3604 | 0.1161 |
| EASE (best baseline) | 0.2865 | 0.3907 | 0.1478 |
| Difference | -0.0303 [-0.0368, -0.0235] | | |

Ablations (10% sample, Recall@100): full 0.2306; no sequence encoder 0.1913; no content features 0.1834; no log-Q 0.1792; uniform training windows 0.1937.

### FAISS index (M5)

HNSW (M = 16, efSearch = 400) over 65,723 item vectors: recall@200 against exact search 0.9992, downstream Recall@100 unchanged from exact (0.2569), single-thread latency p50 0.27 ms and p99 0.39 ms. Recall versus latency for HNSW and IVF-PQ settings in [faiss-benchmark.md](faiss-benchmark.md); choice in [ADR 0007](adr/0007-faiss-index.md).

### LambdaMART ranker (M6)

On 2,908 held-out validation users (their validation labels were never used to train or tune the ranker or the retrieval model): NDCG@10 rises from 0.1155 [0.1088, 0.1222] in retrieval order to 0.1504 [0.1435, 0.1576], a lift of **+0.0349 [+0.0298, +0.0401]**. Against EASE on the same users the two-stage system is level: NDCG@10 difference +0.0033 [-0.0044, +0.0108]. Top features by mean |SHAP|: retrieval rank, 30-day item popularity, genre affinity ([plot](ranker-shap.png), [ADR 0008](adr/0008-lambdamart-ranker.md)).

### Serving API (M7)

FastAPI with the ONNX user tower, HNSW retrieval, Feast online features, the ONNX ranker, and a popularity fallback. k6 against the container (2 CPUs, one worker): **p99 6.8 ms at 250 requests per second** with no errors (p99 97 ms at 300). Server-side median 4.8 ms. `make smoke` passes. See [load-test.md](load-test.md) and [ADR 0009](adr/0009-serving-architecture.md).

### Online test simulation (M9)

Team-draft interleaving over replayed validation events, 2,000 held-out users, 23,197 impressions per run; preference for B is averaged over users with a 95% bootstrap interval ([ADR 0011](adr/0011-online-test-simulation.md)):

| A | B | Preference for B [95% CI] | Winner |
| --- | --- | ---: | --- |
| Retrieval order | Retrieval + ranker (features frozen at cutoff) | +0.018 [-0.030, +0.065] | tie |
| Retrieval order | Retrieval + ranker (features refreshed daily) | -0.027 [-0.072, +0.020] | tie |
| Recent popularity | Retrieval order | **+0.371 [+0.323, +0.417]** | retrieval |

The ranker's offline NDCG lift does not carry over to next-event replay; the sanity pair shows the test can detect a real difference.

### Kubernetes (M10)

Helm chart on a one-node kind cluster: API Deployment, Redis started from a snapshot of the serving state, and a CPU HPA (60% of a 500m request, 1 to 4 replicas). Under k6 at 150 requests per second from one replica, the HPA scaled 1 to 3 within 30 s of saturation and to 4 a minute later; no errors or restarts, median 6.5 ms, p99 1.9 s and 243 dropped requests from the minutes before scale-out. Steady state at 4 replicas with kept-alive connections: p99 16.5 ms. Load testing led to a cap on concurrent engine work in the API. See [ADR 0012](adr/0012-kubernetes-deployment.md).

### Final run (M11)

Retrieval refitted on train + validation for the 5 epochs early stopping chose on validation, and scored on the test period (6,601 warm users) with the validation-trained ranker and baselines refitted on the same history. Nothing was tuned or early-stopped on test labels. Two-tower Recall@100 0.2410 [0.2350, 0.2467] against EASE 0.2697 [0.2642, 0.2757]. Two-stage NDCG@10 0.1337 [0.1292, 0.1384] against EASE 0.1594 [0.1544, 0.1649]. The ranker's lift over retrieval order is +0.0237 [+0.0206, +0.0270]. All numbers are in [results.md](results.md), written by `make report`.

## Blocked or changed

- M4: the two-tower retrieval model does not beat EASE beyond the interval (Recall@100 -0.030 [-0.037, -0.023]). Tried: recent-biased training windows (+0.025 on the sample), temperature and dropout tuning, inclusion-probability log-Q. The gap is analyzed in ADR 0006: it ties EASE for very short and very long histories and loses most for moderate histories and recently active users. The two-tower model stays as the retrieval stage because EASE cannot be served from a nearest-neighbor index or embed new items, and the ranker stage recovers ranking quality.
- M4: the user tower has no static user features. Each event's embedding carries the positive flag and the time gap since the previous event, and user statistics (activity, rating mean, tenure) are ranker features.
- M6/M8: session features are computed online and parity-tested (ADR 0010), but the ranker was trained on batch features only. Its training examples are built at one cutoff, and almost no user has events in the 30 minutes before it, so session features would be constant in training. Recent activity reaches recommendations through the user tower's sequence, which the stream updates on every event.
- M11: on the test period the two-stage system trails EASE on NDCG@10 (-0.0257 [-0.0310, -0.0207]), where it tied on validation (+0.0033 [-0.0044, +0.0108]). The ranker still adds a clear gain over retrieval order, and the candidates it reorders already trail EASE (ADR 0006). No further tuning was done on test.
- M7: p99 under 50 ms held up to 250 requests per second on a quiet laptop, but only up to about 100 with other work running (load average about 5): p99 41 ms at 100, 114 ms at 150. The single API worker is bound by Python's global interpreter lock, so more worker processes or replicas (as on kind) are the way to add headroom. Serving numbers for both conditions are in [results.md](results.md).
- SHAP importance for the ranker uses LightGBM's built-in TreeSHAP (`pred_contrib=True`). The `shap` package currently resolves to an old `llvmlite` that fails to build with NumPy 2.5.
