#!/usr/bin/env bash
# Tune the two-tower model on the 10% user sample: one change at a time.
set -euo pipefail
cd "$(dirname "$0")/.."
export MLFLOW_DISABLE_AGENT_HINT=1
mkdir -p artifacts/tuning

run() {
  local name="$1"; shift
  if [ -f "artifacts/retrieval/s10_${name}/manifest.json" ]; then
    echo "skip ${name} (done)"; return
  fi
  uv run python -m streamrank.models.train_retrieval \
    --split-dir data/split/sample10 --experiment retrieval-tuning \
    --run-name "s10_${name}" --epochs 12 --patience 3 "$@" \
    > "artifacts/tuning/${name}.log" 2>&1
  echo "done ${name}"
}

run default
run recent_0.5 --recent-window-prob 0.5
run recent_1.0 --recent-window-prob 1.0
run recent_0.5_temp_0.1 --recent-window-prob 0.5 --temperature 0.1
run recent_0.5_dropout_0.3 --recent-window-prob 0.5 --dropout 0.3
run recent_0.5_dim_64 --recent-window-prob 0.5 --dim 64
run recent_0.5_temp_0.2 --recent-window-prob 0.5 --temperature 0.2
run recent_0.5_temp_0.1_dropout_0.3 --recent-window-prob 0.5 --temperature 0.1 --dropout 0.3
