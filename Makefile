.DEFAULT_GOAL := help
UV ?= uv
COMPOSE ?= docker compose

.PHONY: help setup lint format typecheck test test-integration up down data ingest split

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
