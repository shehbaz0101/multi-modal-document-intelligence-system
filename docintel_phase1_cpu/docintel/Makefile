.PHONY: help install infra-up infra-down infra-logs quickstart test lint format eval clean

help:
	@echo "DocIntel — Multi-Modal Document Intelligence"
	@echo ""
	@echo "Infra:"
	@echo "  make infra-up       Start Qdrant + OpenSearch + Postgres"
	@echo "  make infra-down     Stop and remove containers (keeps volumes)"
	@echo "  make infra-logs     Tail infra logs"
	@echo "  make infra-reset    Wipe volumes (DESTRUCTIVE)"
	@echo ""
	@echo "Dev:"
	@echo "  make install        Install Python deps with uv"
	@echo "  make quickstart     End-to-end demo: ingest sample + ask a question"
	@echo "  make test           Run unit tests"
	@echo "  make lint           Ruff lint"
	@echo "  make format         Ruff format"
	@echo "  make eval           Run the eval harness against data/eval/set.jsonl"
	@echo ""
	@echo "Clean:"
	@echo "  make clean          Remove local data/ and cache/"

install:
	uv venv
	uv pip install -e ".[dev]"

infra-up:
	docker compose up -d
	@echo "Waiting for services to be healthy..."
	@for i in 1 2 3 4 5 6 7 8 9 10; do \
		if docker compose ps --format json | grep -q '"Health":"healthy"'; then \
			echo "✓ services healthy"; break; \
		fi; sleep 3; \
	done

infra-down:
	docker compose down

infra-logs:
	docker compose logs -f --tail 100

infra-reset:
	docker compose down -v

quickstart:
	python scripts/quickstart.py

test:
	pytest -v

lint:
	ruff check .

format:
	ruff format .

eval:
	python -m evaluation.harness --eval data/eval/set.jsonl

clean:
	rm -rf data/ .pytest_cache .ruff_cache __pycache__ */__pycache__ */*/__pycache__
