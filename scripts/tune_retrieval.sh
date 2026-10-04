#!/usr/bin/env bash
# Tune the two-tower model on the 10% user sample: one change at a time from the default.
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
run temp_0.1 --temperature 0.1
run temp_0.03 --temperature 0.03
run dim_64 --dim 64
run len_100 --max-len 100
run neg_4096 --n-random-negatives 4096
run dropout_0.1 --dropout 0.1
