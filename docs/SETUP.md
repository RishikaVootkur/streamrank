# Setup

How to build StreamRank from a fresh clone: environment, data, every pipeline stage, the local stack, and the Kubernetes demo. Times are from an Apple Silicon laptop with 16 GB of RAM and Docker Desktop set to 8 GB.

## Prerequisites

- macOS or Linux, 16 GB RAM, about 30 GB of free disk.
- [uv](https://docs.astral.sh/uv/) (it installs Python 3.12 for the project), Docker with Compose v2, GNU Make, git.
- For the Kubernetes demo only: [kind](https://kind.sigs.k8s.io/), [Helm](https://helm.sh/), and kubectl.
- Training uses Apple MPS when available and falls back to CPU.

No API keys or paid services are needed.

## 1. Environment

```bash
git clone https://github.com/RishikaVootkur/streamrank.git
cd streamrank
make setup        # uv sync --all-groups, creates .env from .env.example, installs git hooks
```

Edit `.env` and replace the `change-me` passwords. `.env` is gitignored.

Check the install:

```bash
make lint typecheck test
```

## 2. Data

```bash
make data         # MovieLens 32M (about 240 MB), MD5-verified, extracted to data/raw
make split        # Pandera-validated Parquet, temporal split, 10% user sample
```

MovieLens files stay in `data/` (gitignored) and are never committed. Tests use the synthetic generator: `uv run python -m streamrank.data.synthetic --out <dir>`.

## 3. Features

```bash
make down               # Spark needs about 5 GB of Docker's memory: stop the serving stack
make infra              # Redis, Postgres, MLflow
make features           # Spark job in Docker, Feast apply, Redis online store filled up to the validation cutoff (2020-11-05)
make test-spark         # optional: Spark-vs-Polars parity tests inside the Spark image
```

## 4. Models

```bash
make baselines          # tune on the 10% sample, evaluate on full validation (about 12 min)
make train-retrieval    # two-tower on MPS with the chosen settings, about 9 min per epoch,
                        # early stopping on the sample's users
make build-index        # export vectors, benchmark FAISS indexes, save HNSW (INDEX_CHOICE)
make train-ranker       # candidates, ranker features, Optuna-tuned LambdaMART, ONNX export
make two-stage-report   # two-stage system against EASE on the ranker's held-out users
```

Experiments are logged to MLflow at http://localhost:5001.

## 5. Serving

```bash
make export-onnx        # user tower to ONNX, with a parity check against PyTorch
make serving-artifacts  # copy serving files and load every user's state into Redis
make up                 # API, Prometheus, Grafana, Streamlit demo
make smoke              # end-to-end request against the running API
make load               # k6 at LOAD_RATE requests per second (default 100) for LOAD_DURATION
```

| Service | URL |
| --- | --- |
| API | http://localhost:8000/recommendations/{user_id} |
| Demo | http://localhost:8501 |
| Grafana | http://localhost:3000 (dashboard "StreamRank serving") |
| Prometheus | http://localhost:9090 |
| MLflow | http://localhost:5001 |

## 6. Streaming, simulation, monitoring

```bash
make stream-parity      # replay 10,000 events through Redpanda and Quix; compare with batch
make simulate           # team-draft interleaving replays (about 4 min each)
make drift              # Evidently drift report, docs/drift.md
```

## 7. Final run and report

```bash
make final-run          # refit on train + validation, score everything on test (about 70 min)
make report             # docs/results.md and the README results table from saved results
```

## 8. Kubernetes (kind)

Stop the Compose stack first: with both running, Docker's 8 GB VM starts swapping.

```bash
make down
make kind-up            # cluster, Redis snapshot, images, metrics-server
make helm-install       # API, Redis, HPA; API at http://localhost:18000
make kind-load LOAD_RATE=150 LOAD_DURATION=4m   # watch the HPA add replicas
make kind-down
```

## Using data or artifacts from another folder

`ARTIFACTS_HOST_DIR` and `DATA_HOST_DIR` point Compose and the kind mounts at existing `artifacts/` and `data/` folders, for example a second clone that reuses one set of trained models:

```bash
ARTIFACTS_HOST_DIR=/path/to/artifacts DATA_HOST_DIR=/path/to/data make up smoke
```

## Troubleshooting

- PyTorch, FAISS, and LightGBM each bundle OpenMP on macOS and cannot share a process. The pipeline runs them in separate processes, and `make test` runs the FAISS and ranker tests in their own pytest processes.
- `implicit` warns unless BLAS is single-threaded: the Makefile sets `OPENBLAS_NUM_THREADS=1`.
- Port 5000 is taken by AirPlay on macOS, so MLflow uses 5001.
- `kind load docker-image` fails with Docker Desktop's containerd image store; `make kind-up` loads image archives.
