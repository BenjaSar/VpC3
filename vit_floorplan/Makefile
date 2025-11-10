.PHONY: help install dev-install test lint format clean eda train docker-build docker-run

help:
	@echo "Available commands:"
	@echo "  install      - Install production dependencies"
	@echo "  dev-install  - Install development dependencies"
	@echo "  test         - Run tests"
	@echo "  lint         - Run linting"
	@echo "  format       - Format code"
	@echo "  clean        - Clean generated files"
	@echo "  eda          - Run EDA analysis (FIRST STEP ⭐)"
	@echo "  train        - Start training"
	@echo "  docker-build - Build Docker image"
	@echo "  docker-run   - Run Docker container"

install:
	pip install -r requirements/base.txt
	pip install -e .

dev-install:
	pip install -r requirements/dev.txt
	pip install -e .

test:
	pytest tests/ -v --cov=src

lint:
	flake8 src/ --max-line-length=120
	isort --check-only src/

format:
	black src/
	isort src/

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf build/ dist/ *.egg-info

eda:
	python scripts/eda_analysis.py

train:
	python scripts/train.py

docker-build:
	docker-compose -f docker/docker-compose.yml build

docker-run:
	docker-compose -f docker/docker-compose.yml up
