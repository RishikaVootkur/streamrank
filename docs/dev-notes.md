# Dev notes

Short facts a contributor needs to work on this repo. Keep under 120 lines.

## Machine
- Apple Silicon (arm64), macOS 26.6, 16 GB RAM, Docker Desktop with 8 GB memory.
- Kubernetes (kind) milestone is in scope (RAM is 16 GB).
- Host Python is 3.14; the project uses a uv-managed Python 3.12 venv.

## Conventions
- Commits: one-line Conventional Commits, imperative, 50 chars max, no body.
- Branches: feat/<issue>-<slug>, fix/..., docs/..., chore/...
- PR body sections: What, Why, How tested, Results, then `Closes #<issue>`.
- Never push to main; squash-merge PRs after CI is green.
- Never commit MovieLens data or samples. Tests use the synthetic generator.
- Local secrets live in `.env` (gitignored); `.env.example` has placeholders.

## Ports
API 8000, MLflow 5001, Grafana 3000, Prometheus 9090, Redis 6379,
Redpanda 19092, Redpanda Console 8080, Streamlit 8501, Postgres 5432.

## Data pipeline
- `make data` downloads and MD5-verifies MovieLens 32M into `data/` (gitignored).
- `make split` runs ingest (Pandera-validated Parquet in `data/processed/`) then the split
  (`data/split/full/` and `data/split/sample10/`), and rewrites `docs/data-split.md`.
- Processed columns: `user_id`, `item_id`, `rating`, `ts` (Unix seconds), plus `label`
  (rating >= 4) after the split.
- Every stage writes `manifest.json` (git SHA, config, data hash, seed, output hashes);
  read inputs with `verify_outputs()`.
- Synthetic data: `python -m streamrank.data.synthetic --out <dir>` writes MovieLens-style CSVs.

## Gotchas
- `.gitignore` ignores `data/` everywhere except `src/streamrank/data/`, and all `*.csv`.
- Pin GitHub Actions to full version tags; `astral-sh/setup-uv` has no floating major tag.
- `shap` does not build against NumPy 2.5; use LightGBM `pred_contrib=True` for SHAP values.
- `implicit` warns unless BLAS runs single-threaded: set `OPENBLAS_NUM_THREADS=1`.

## Evaluation
- `make baselines` (about 12 min): tunes on `data/split/sample10` val, evaluates on full val, writes
  `artifacts/baselines/` and `docs/baselines.md`. Best baseline: EASE, Recall@100 0.2854.
- New models: implement `fit(TrainData)` and `score(user_rows) -> (n, n_items)` and call
  `eval.evaluate.fit_and_evaluate`; compare with `metrics.paired_difference` on per-user arrays.
- CI installs every dependency group; torch resolves to the CPU wheel index on Linux.
