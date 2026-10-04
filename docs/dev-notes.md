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
