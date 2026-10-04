# Results

Written by `make report` from saved artifacts.

## Final run

Final run on the test period: 6,601 warm test users, full-catalog ranking, 95% bootstrap intervals over users. Retrieval models were refitted on train + validation history; nothing was tuned or early-stopped on test labels.

| System | Recall@100 | NDCG@10 | Recall@10 |
| --- | ---: | ---: | ---: |
| EASE (best baseline) | 0.2697 [0.2642, 0.2757] | 0.1594 [0.1544, 0.1649] | 0.1527 [0.1478, 0.1580] |
| Two-tower retrieval | 0.2410 [0.2350, 0.2467] | 0.1100 [0.1060, 0.1145] | 0.1102 [0.1061, 0.1147] |
| Two-tower + LambdaMART ranker | 0.2410 [0.2350, 0.2467] | 0.1337 [0.1292, 0.1384] | 0.1333 [0.1287, 0.1380] |

- Two-tower against EASE, Recall@100: -0.0287 [-0.0350, -0.0228] (worse).
- Ranker against retrieval order, NDCG@10: +0.0237 [+0.0206, +0.0270] (better).
- Two-stage against EASE, NDCG@10: -0.0257 [-0.0310, -0.0207] (worse).
- The ranker reorders the same 200 candidates, so Recall@100 is shared by both two-tower rows. It was trained on validation labels and applied unchanged, with features as of the test cutoff and retrieval scores from the refitted model.

### Baselines on test (tuned on validation)

| Model | Recall@10 | Recall@100 | NDCG@10 | MRR | Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.0724 [0.0691, 0.0761] | 0.1444 [0.1400, 0.1494] | 0.0754 [0.0718, 0.0793] | 0.1719 [0.1643, 0.1794] | 0.0044 |
| Recent popularity | 0.1183 [0.1137, 0.1226] | 0.2255 [0.2198, 0.2312] | 0.1089 [0.1052, 0.1128] | 0.2139 [0.2076, 0.2203] | 0.0028 |
| Item KNN | 0.1134 [0.1091, 0.1180] | 0.2230 [0.2179, 0.2285] | 0.1174 [0.1127, 0.1221] | 0.2419 [0.2338, 0.2509] | 0.0129 |
| EASE | 0.1527 [0.1478, 0.1580] | 0.2697 [0.2642, 0.2757] | 0.1594 [0.1544, 0.1649] | 0.3167 [0.3077, 0.3264] | 0.0260 |
| ALS | 0.1333 [0.1287, 0.1382] | 0.2524 [0.2470, 0.2586] | 0.1343 [0.1297, 0.1392] | 0.2687 [0.2605, 0.2776] | 0.0289 |

## Development results

### Baselines on validation

| Model | Recall@10 | Recall@100 | NDCG@10 | MRR | Coverage@10 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Popularity | 0.0665 [0.0632, 0.0701] | 0.1411 [0.1368, 0.1455] | 0.0695 [0.0660, 0.0731] | 0.1607 [0.1534, 0.1678] | 0.0048 |
| Recent popularity | 0.1150 [0.1106, 0.1197] | 0.2341 [0.2281, 0.2403] | 0.1155 [0.1114, 0.1199] | 0.2425 [0.2345, 0.2507] | 0.0039 |
| Item KNN | 0.1066 [0.1024, 0.1109] | 0.2311 [0.2252, 0.2370] | 0.1087 [0.1044, 0.1130] | 0.2272 [0.2196, 0.2350] | 0.0133 |
| EASE | 0.1464 [0.1415, 0.1515] | 0.2854 [0.2795, 0.2912] | 0.1484 [0.1432, 0.1534] | 0.2925 [0.2837, 0.3011] | 0.0276 |
| ALS | 0.1300 [0.1253, 0.1347] | 0.2643 [0.2583, 0.2701] | 0.1281 [0.1234, 0.1325] | 0.2541 [0.2458, 0.2621] | 0.0315 |

#### Two-tower ablations (validation, 10% user sample)

Recall@100 on the sample's users that early stopping did not use.

| Variant | Recall@100 |
| --- | ---: |
| Full model | 0.2306 [0.2049, 0.2572] |
| No sequence encoder (mean of history embeddings) | 0.1913 [0.1688, 0.2153] |
| No content features (item ID only) | 0.1834 [0.1619, 0.2088] |
| No log-Q correction | 0.1792 [0.1560, 0.2029] |
| Uniform training windows (no recent-window sampling) | 0.1937 [0.1709, 0.2181] |

### Ranker (validation)

NDCG@10 on 2,908 held-out validation users: retrieval order 0.1155 [0.1088, 0.1222], ranker 0.1504 [0.1435, 0.1576], lift +0.0349 [+0.0298, +0.0401]. Top features by mean |SHAP|: retrieval_rank (0.242), item_log_n_ratings_30d (0.233), genre_affinity (0.168), item_log_n_ratings_7d (0.134), item_mean_rating (0.114).

### Retrieval index

hnsw(M=16,efC=200,ef=400): recall@200 against exact search 0.9992, single-query p50 0.27 ms, p99 0.39 ms.

### Interleaving replay (validation events)

| A | B | Batch features | Preference for B | Winner |
| --- | --- | --- | ---: | --- |
| popular | retrieval | frozen at cutoff | +0.371 [+0.324, +0.420] | retrieval |
| retrieval | ranker | daily | -0.027 [-0.072, +0.020] | tie |
| retrieval | ranker | frozen at cutoff | +0.018 [-0.030, +0.065] | tie |

### Streaming parity

10,000 replayed events for 631 users: 0 mismatches between online and batch session features; event-to-Redis latency p50 529 ms.

### Data drift

1 of 8 columns drifted between 2020-08-07 to 2020-11-05 and 2021-11-03 to 2022-02-01: item_log_n_ratings_30d.

### Serving latency (k6)

| Run | p50 ms | p95 ms | p99 ms | Dropped |
| --- | ---: | ---: | ---: | ---: |
| Compose, 100 req/s | 5.4 | 6.6 | 7.6 | 0 |
| Compose, 200 req/s | 5.0 | 5.8 | 6.9 | 0 |
| Compose, 250 req/s | 5.2 | 6.0 | 6.8 | 0 |
| Compose, 300 req/s | 5.1 | 6.4 | 96.8 | 4 |
| kind, 150 req/s, HPA scale-out from 1 pod (new connection per request) | 6.1 | 2297.5 | 2503.1 | 721 |
| kind, 150 req/s, 4 pods | 5.7 | 8.1 | 16.5 | 34 |
