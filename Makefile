.DEFAULT_GOAL := help
-include .env
REDIS_CONNECTION_STRING ?= localhost:$(or $(REDIS_PORT),6379)
export REDIS_CONNECTION_STRING
UV ?= uv
COMPOSE ?= docker compose

.PHONY: help setup lint format typecheck test test-integration up down data ingest split baselines retrieval-segments evaluate spark-image test-spark features-offline features train-retrieval build-index train-ranker export-onnx serving-artifacts smoke load stream stream-parity two-stage-report simulate drift final-run report


help: ## List targets
	@grep -E '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "%-18s %s\n", $$1, $$2}'

setup: ## Install all dependency groups and git hooks
	$(UV) sync --all-groups
	@test -f .env || cp .env.example .env
	$(UV) run pre-commit install

lint: ## Check lint and formatting
	$(UV) run ruff check .
	$(UV) run ruff format --check .

format: ## Apply lint fixes and formatting
	$(UV) run ruff check --fix .
	$(UV) run ruff format .

typecheck: ## Run mypy in strict mode on src/
	$(UV) run mypy

test: ## Run unit tests with coverage (FAISS and LightGBM tests in their own processes)
	$(UV) run pytest tests/unit --cov --cov-report=
	$(UV) run pytest tests/faiss --cov --cov-append --cov-report=
	$(UV) run pytest tests/ranking --cov --cov-append --cov-report=term --cov-fail-under=70

test-integration: ## Run integration tests against Compose services
	$(COMPOSE) --profile streaming up -d --wait redis postgres mlflow redpanda
	$(UV) run pytest tests/integration -m integration
	$(UV) run pytest tests/serving -m integration
	$(UV) run pytest tests/streaming -m integration

up: ## Start the local stack, including the API (needs serving artifacts)
	$(COMPOSE) --profile serving up -d --wait --build

down: ## Stop the local stack
	$(COMPOSE) --profile serving --profile load --profile streaming down

data: ## Download MovieLens 32M and verify its checksum
	$(UV) run python -m streamrank.data.download

ingest: ## Validate raw CSV files and write Parquet tables
	$(UV) run python -m streamrank.data.ingest

split: ingest ## Create the global temporal split and the 10% user sample
	$(UV) run python -m streamrank.data.split --stats-doc docs/data-split.md

baselines: ## Tune baselines on the 10% sample and evaluate them on the full validation split
	OPENBLAS_NUM_THREADS=1 $(UV) run python -m streamrank.eval.run_baselines --doc docs/baselines.md

retrieval-segments: ## Compare the two-tower model with EASE by user segment
	OPENBLAS_NUM_THREADS=1 $(UV) run python -m streamrank.eval.segments \
		--model-dir $(RETRIEVAL_MODEL)

two-stage-report: ## Compare the two-stage system with EASE on the ranker's evaluation users
	OPENBLAS_NUM_THREADS=1 $(UV) run python -m streamrank.eval.two_stage

evaluate: baselines ## Run every offline evaluation

SPARK_IMAGE ?= streamrank-spark:local
FEATURES_FROM ?= 2019-11-01

spark-image: ## Build the Spark job image
	docker build -f services/spark/Dockerfile --target runtime -t $(SPARK_IMAGE) .

test-spark: ## Run the Spark parity tests inside the Spark image
	docker build -f services/spark/Dockerfile --target test -t $(SPARK_IMAGE)-test .
	docker run --rm $(SPARK_IMAGE)-test

features-offline: spark-image ## Compute offline feature snapshots with Spark in Docker
	docker run --rm -e HADOOP_USER_NAME=spark -e GIT_SHA=$$(git rev-parse HEAD) \
		-v $(CURDIR)/data:/app/data $(SPARK_IMAGE) \
		--processed-dir data/processed --out-dir data/features --from-date $(FEATURES_FROM)

features: features-offline ## Compute features, register them in Feast, and load Redis
	$(COMPOSE) up -d --wait redis
	cd feature_repo && $(UV) run feast apply
	$(UV) run python -m streamrank.features.store --start $(FEATURES_FROM) \
		--end-at-cutoff data/split/full

RETRIEVAL_ARGS ?=

train-retrieval: ## Train the two-tower model on the full split and compare it with EASE
	MLFLOW_DISABLE_AGENT_HINT=1 $(UV) run python -m streamrank.models.train_retrieval \
		--split-dir data/split/full --early-stop-split data/split/sample10 \
		--run-name two_tower_full $(RETRIEVAL_ARGS)

RETRIEVAL_MODEL ?= artifacts/retrieval/two_tower_full
INDEX_CHOICE ?=

build-index: ## Export vectors, benchmark FAISS indexes, and save the chosen index
	$(UV) run python -m streamrank.models.export_vectors --model-dir $(RETRIEVAL_MODEL)
	$(UV) run python -m streamrank.retrieval.benchmark --model-dir $(RETRIEVAL_MODEL) \
		$(if $(INDEX_CHOICE),--choose "$(INDEX_CHOICE)",)

RANKER_ARGS ?=

train-ranker: ## Export candidates, build ranker features, and train the LambdaMART ranker
	$(UV) run python -m streamrank.models.candidates --model-dir $(RETRIEVAL_MODEL)
	$(UV) run python -m streamrank.ranking.build_features
	MLFLOW_DISABLE_AGENT_HINT=1 $(UV) run python -m streamrank.ranking.train_ranker \
		$(RANKER_ARGS)
	$(UV) run python -m streamrank.ranking.export_onnx

export-onnx: ## Export the trained user tower to ONNX and check parity with PyTorch
	MLFLOW_DISABLE_AGENT_HINT=1 $(UV) run python -m streamrank.serving.export_onnx \
		--model-dir $(RETRIEVAL_MODEL)

serving-artifacts: ## Collect serving artifacts and load user state into Redis
	$(COMPOSE) up -d --wait redis
	$(UV) run python -m streamrank.serving.build --model-dir $(RETRIEVAL_MODEL)

smoke: ## End-to-end request against the running stack
	$(UV) run python -m streamrank.serving.smoke

LOAD_RATE ?= 100
LOAD_DURATION ?= 2m

load: ## Load test the API with k6 (constant arrival rate)
	LOAD_RATE=$(LOAD_RATE) LOAD_DURATION=$(LOAD_DURATION) \
		$(COMPOSE) --profile serving --profile load run --rm k6

stream: ## Run the streaming session-feature job (Redpanda -> Redis)
	$(COMPOSE) --profile streaming up -d --wait redis redpanda
	$(UV) run python -m streamrank.streaming.app

stream-parity: ## Replay 10,000 events and check online session features against batch
	$(COMPOSE) --profile streaming up -d --wait redis redpanda
	$(UV) run python -m streamrank.streaming.parity --events 10000

simulate: ## Replay future events and interleave rankers (retrieval vs ranker, plus a sanity pair)
	$(UV) run python -m streamrank.simulation.simulate --pair retrieval,ranker
	$(UV) run python -m streamrank.simulation.simulate --pair retrieval,ranker --daily-features
	$(UV) run python -m streamrank.simulation.simulate --pair popular,retrieval

drift: ## Data drift report: recent rating events against the pre-cutoff window
	OPENBLAS_NUM_THREADS=1 $(UV) run python -m streamrank.monitoring.drift

FINAL_DIR ?= artifacts/final
# The validation run's settings; early stopping picked epoch 5 of its 10-epoch schedule.
FINAL_RETRIEVAL_ARGS ?= --recent-window-prob 0.5 --temperature 0.1 --dropout 0.3 \
	--epochs 10 --stop-after 5

final-run: ## Final run: refit on train + validation, score retrieval, ranker, baselines on test
	MLFLOW_DISABLE_AGENT_HINT=1 $(UV) run python -m streamrank.models.train_retrieval \
		--split-dir data/split/full --partition test --run-name two_tower_final \
		--out-dir $(FINAL_DIR)/retrieval $(FINAL_RETRIEVAL_ARGS)
	$(UV) run python -m streamrank.models.candidates --model-dir $(FINAL_DIR)/retrieval \
		--partition test --out-dir $(FINAL_DIR)/candidates
	$(UV) run python -m streamrank.ranking.build_features --partition test \
		--candidates-dir $(FINAL_DIR)/candidates --out-dir $(FINAL_DIR)/ranker_features
	OPENBLAS_NUM_THREADS=1 $(UV) run python -m streamrank.eval.two_stage --partition test \
		--ranker-dir artifacts/ranker \
		--features $(FINAL_DIR)/ranker_features/ranker_features_test.parquet \
		--out $(FINAL_DIR)/two_stage_test.json
	OPENBLAS_NUM_THREADS=1 $(UV) run python -m streamrank.eval.run_baselines --partition test \
		--out-dir $(FINAL_DIR)/baselines

report: ## Write docs/results.md and the README results table from saved results
	$(UV) run python -m streamrank.report
