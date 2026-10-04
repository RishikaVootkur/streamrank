# StreamRank

A two-stage movie recommender that returns a user's next 10 movies in milliseconds on a laptop. A PyTorch two-tower model, with a causal transformer over the user's last 200 events, retrieves 200 candidates from the full catalog through a FAISS HNSW index. A LightGBM LambdaMART ranker then orders them using batch features from Feast. Every new event, streamed through Redpanda, updates the user's sequence and session features in Redis, so the next request reflects it. Built on MovieLens 32M with a strict temporal split, and evaluated against strong baselines with bootstrap confidence intervals.

![Demo: a user's recent history next to live recommendations](docs/images/demo.png)

## Results

<!-- results:begin -->
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

#### Two-tower ablations (validation, 10% user sample)

Recall@100 on the sample's users that early stopping did not use.

| Variant | Recall@100 |
| --- | ---: |
| Full model | 0.2306 [0.2049, 0.2572] |
| No sequence encoder (mean of history embeddings) | 0.1913 [0.1688, 0.2153] |
| No content features (item ID only) | 0.1834 [0.1619, 0.2088] |
| No log-Q correction | 0.1792 [0.1560, 0.2029] |
| Uniform training windows (no recent-window sampling) | 0.1937 [0.1709, 0.2181] |
<!-- results:end -->

More results are in [docs/results.md](docs/results.md): baselines on both periods, the ranker's SHAP importance, index benchmarks, the interleaving replay, and serving latency. Decisions and their trade-offs are in [docs/adr](docs/adr).

What the numbers say:

- EASE, a linear item-to-item model, is the strongest single model on both periods. On test, the two-tower model trails it on Recall@100, and also trails ALS (0.2524). [ADR 0006](docs/adr/0006-two-tower-retrieval.md) analyzes the gap by user segment.
- The two-tower model stays as the retrieval stage anyway. It serves from a nearest-neighbor index in under a millisecond over the whole catalog, and it embeds new items from their content. EASE keeps a dense item-by-item matrix (20,000 items here, 1.6 GB) and cannot score items it has not seen in training.
- The ranker adds a clear NDCG@10 gain over retrieval order on both periods (+0.035 on validation, +0.024 on test). The full system tied EASE on validation but trails it on test.
- A replay of real next events with team-draft interleaving tells a different story ([ADR 0011](docs/adr/0011-online-test-simulation.md)). There the ranker only ties retrieval order, while retrieval clearly beats recent popularity. The offline metric rewards long-horizon relevance; the replay rewards the very next event, which is what the sequential model was trained to predict.

## Architecture

```mermaid
flowchart LR
  subgraph Offline
    A[MovieLens 32M] --> B[Ingest and validate<br/>Polars, Pandera]
    B --> C[Temporal split]
    C --> D[PySpark features<br/>in Docker]
    D --> E[Feast offline store]
    C --> G[Two-tower training<br/>PyTorch on MPS]
    G --> H[Item vectors]
    H --> I[FAISS HNSW index]
    G --> J[LambdaMART ranker<br/>LightGBM, Optuna]
    E --> J
    G & J --> S[MLflow]
  end
  subgraph Online
    E --> F[(Redis<br/>Feast online store,<br/>user sequences, sessions)]
    K[Event replay] --> L[Redpanda]
    L --> M[Quix Streams job]
    M --> F
    N[Client] --> O[FastAPI]
    F --> O
    O --> P[User tower<br/>ONNX Runtime]
    P --> I
    I --> Q[200 candidates]
    Q --> R[Ranker<br/>ONNX Runtime]
    R --> T[Top 10]
    O --> U[Prometheus, Grafana]
  end
```

- **Temporal split.** Train before 2020-11-05, validation until 2022-02-01, test after. Features are computed point in time, and a test fails on leakage.
- **One feature definition module** feeds the Spark batch job and the streaming job. Parity tests compare Spark with Polars and the stream with batch.
- **Retrieval.** A SASRec-style user tower over the last 200 events and an item tower over ID, genres, year, and tags. Trained with sampled softmax over in-batch and random negatives, with log-Q correction.
- **Ranking.** 19 features: retrieval score and rank, item popularity over several windows, item age and ratings, user activity, genre affinity, and release-year gap. Tuned with Optuna, exported to ONNX, and checked for parity.
- **Serving.** FastAPI with ONNX Runtime and FAISS, a popularity fallback for unknown users, and Prometheus metrics. The same image runs in Docker Compose and on Kubernetes (kind, Helm, CPU autoscaling).

![Grafana serving dashboard](docs/images/grafana.png)

## Quickstart

You need [uv](https://docs.astral.sh/uv/), Docker, and GNU Make. No API keys or paid services.

```bash
make setup                 # Python 3.12 environment, .env from .env.example, git hooks
make data split            # download MovieLens 32M (checksum verified), split by time
make up features           # Redis, Postgres, MLflow; Spark features; Feast online store
make baselines train-retrieval build-index train-ranker
make export-onnx serving-artifacts
make up smoke              # API, demo, Grafana; one end-to-end request
make load                  # k6 load test
make final-run report      # refit on train + validation, score on test, write results
make kind-up helm-install  # optional: Kubernetes on kind with an HPA
```

[docs/SETUP.md](docs/SETUP.md) covers every stage, with run times and options. Run `make help` for all targets.

## Repository

| Path | Contents |
| --- | --- |
| `src/streamrank/` | data, features, models, retrieval, ranking, serving, streaming, simulation, monitoring, demo |
| `pipelines/spark/` | offline feature job |
| `feature_repo/` | Feast definitions |
| `services/` | Dockerfiles for the API, Spark, streaming job, MLflow, and demo |
| `deploy/` | Prometheus, Grafana dashboards, Helm chart, kind config |
| `loadtest/` | k6 script |
| `docs/` | roadmap, results, ADRs, model card, setup |

## Documentation

- [docs/results.md](docs/results.md): every result, written by `make report`
- [docs/model-card.md](docs/model-card.md): intended use, data, evaluation, limitations
- [docs/ROADMAP.md](docs/ROADMAP.md): milestones, per-milestone results, and what changed
- [docs/adr](docs/adr): architecture decision records

## Data

MovieLens 32M from [GroupLens](https://grouplens.org/datasets/movielens/32m/) (F. Maxwell Harper and Joseph A. Konstan, 2015, ACM TiiS). It is downloaded locally and never redistributed here. Tests use a synthetic generator with the same schema.

## License

MIT
