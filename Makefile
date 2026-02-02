# =============================================================================
# NGSE Makefile - Common development commands
# =============================================================================

.PHONY: help install dev run test lint format clean docker-build docker-run ingest

# Default target
help:
	@echo "NGSE - Next-Gen Search Engine"
	@echo ""
	@echo "Available commands:"
	@echo "  make install     - Install production dependencies"
	@echo "  make dev         - Install development dependencies"
	@echo "  make run         - Run the development server"
	@echo "  make test        - Run tests"
	@echo "  make lint        - Run linters"
	@echo "  make format      - Format code"
	@echo "  make clean       - Clean build artifacts"
	@echo "  make ingest      - Run data ingestion"
	@echo "  make docker-build- Build Docker image"
	@echo "  make docker-run  - Run with Docker Compose"

# =============================================================================
# INSTALLATION
# =============================================================================

install:
	pip install -r requirements.txt

dev: install
	pip install pytest pytest-asyncio pytest-cov black isort flake8 mypy

# =============================================================================
# RUNNING
# =============================================================================

run:
	python main.py

run-prod:
	gunicorn main:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

ingest:
	python scripts/run_ingestion.py

# =============================================================================
# TESTING
# =============================================================================

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=src --cov-report=html

# =============================================================================
# CODE QUALITY
# =============================================================================

format:
	black src/ tests/ main.py
	isort src/ tests/ main.py

lint:
	flake8 src/ tests/ main.py --max-line-length=120
	mypy src/ --ignore-missing-imports

# =============================================================================
# CLEANUP
# =============================================================================

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	rm -rf htmlcov/ .coverage dist/ build/

clean-indices:
	rm -rf data/indices/*
	@echo "Indices cleared. Run 'make ingest' to rebuild."

# =============================================================================
# DOCKER
# =============================================================================

docker-build:
	docker build -t ngse:latest .

docker-run:
	docker-compose up -d

docker-stop:
	docker-compose down

docker-logs:
	docker-compose logs -f ngse-api

docker-prod:
	docker-compose --profile production up -d
