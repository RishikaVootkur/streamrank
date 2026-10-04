.DEFAULT_GOAL := help
-include .env
REDIS_CONNECTION_STRING ?= localhost:$(or $(REDIS_PORT),6379)
export REDIS_CONNECTION_STRING
UV ?= uv
COMPOSE ?= docker compose

.PHONY: help setup lint format typecheck test test-integration up down data ingest split baselines evaluate spark-image test-spark features-offline features

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

test: ## Run unit tests with coverage
	$(UV) run pytest tests/unit --cov --cov-report=term --cov-fail-under=70

test-integration: ## Run integration tests against Compose services
	$(COMPOSE) up -d --wait redis postgres
	$(UV) run pytest tests/integration -m integration

up: ## Start the local stack
	$(COMPOSE) up -d --wait

down: ## Stop the local stack
	$(COMPOSE) down

data: ## Download MovieLens 32M and verify its checksum
	$(UV) run python -m streamrank.data.download

ingest: ## Validate raw CSV files and write Parquet tables
	$(UV) run python -m streamrank.data.ingest

split: ingest ## Create the global temporal split and the 10% user sample
	$(UV) run python -m streamrank.data.split --stats-doc docs/data-split.md

baselines: ## Tune baselines on the 10% sample and evaluate them on the full validation split
	OPENBLAS_NUM_THREADS=1 $(UV) run python -m streamrank.eval.run_baselines --doc docs/baselines.md

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
	$(UV) run python -m streamrank.features.store --start $(FEATURES_FROM)
