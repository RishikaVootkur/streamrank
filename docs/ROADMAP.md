# Roadmap

Each milestone is tracked by a GitHub issue and lands through one or more pull requests.

## Milestones

- [x] M0 Scaffold, CI, guards
- [x] M1 Data ingest, validation, temporal split, synthetic generator
- [x] M2 Evaluation library and baselines
- [x] M3 PySpark offline features and Feast
- [x] M4 Two-tower retrieval with sequential user tower
- [ ] M5 FAISS index
- [ ] M6 LightGBM LambdaMART ranker
- [ ] M7 Serving API
- [x] M8 Streaming session features
- [ ] M9 Online test simulation
- [ ] M10 Observability, demo UI, Kubernetes
- [ ] M11 Final run, docs, release

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

Ablations (10% sample, Recall@100): full 0.2306; no sequence encoder 0.1913; no content features 0.1834; no log-Q 0.1792; uniform training windows 0.1896.

## Blocked or changed

- M4: the two-tower retrieval model does not beat EASE beyond the interval (Recall@100 -0.030 [-0.037, -0.023]). Tried: recent-biased training windows (+0.025 on the sample), temperature and dropout tuning, inclusion-probability log-Q. The gap is analyzed in ADR 0006: it ties EASE for very short and very long histories and loses most for moderate histories and recently active users. The two-tower model stays as the retrieval stage because EASE cannot be served from a nearest-neighbor index or embed new items, and the ranker stage recovers ranking quality.
- SHAP importance for the ranker uses LightGBM's built-in TreeSHAP (`pred_contrib=True`). The `shap` package currently resolves to an old `llvmlite` that fails to build with NumPy 2.5.
