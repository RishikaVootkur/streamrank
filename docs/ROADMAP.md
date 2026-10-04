# Roadmap

Each milestone is tracked by a GitHub issue and lands through one or more pull requests.

## Milestones

- [x] M0 Scaffold, CI, guards
- [x] M1 Data ingest, validation, temporal split, synthetic generator
- [ ] M2 Evaluation library and baselines
- [ ] M3 PySpark offline features and Feast
- [ ] M4 Two-tower retrieval with sequential user tower
- [ ] M5 FAISS index
- [ ] M6 LightGBM LambdaMART ranker
- [ ] M7 Serving API
- [ ] M8 Streaming session features
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

## Blocked or changed

- SHAP importance for the ranker uses LightGBM's built-in TreeSHAP (`pred_contrib=True`). The `shap` package currently resolves to an old `llvmlite` that fails to build with NumPy 2.5.
