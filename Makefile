.PHONY: install install-dev download-data eda train evaluate test clean mlflow-ui help

# Variables
PYTHON := python
PIP := pip
PYTEST := pytest
MLFLOW := mlflow

# Colors for output
BLUE := \033[0;34m
GREEN := \033[0;32m
YELLOW := \033[1;33m
NC := \033[0m # No Color

help:  ## Show this help message
	@echo "$(BLUE)CubiCasa5K Room Detection - Makefile Commands$(NC)"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | awk 'BEGIN {FS = ":.*?## "}; {printf "$(GREEN)%-20s$(NC) %s\n", $$1, $$2}'

install:  ## Install project dependencies
	@echo "$(BLUE)Installing dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	@echo "$(GREEN)Dependencies installed successfully!$(NC)"

install-dev:  ## Install development dependencies
	@echo "$(BLUE)Installing development dependencies...$(NC)"
	$(PIP) install -r requirements.txt
	$(PIP) install pytest pytest-cov black flake8 mypy
	@echo "$(GREEN)Dev dependencies installed!$(NC)"

download-data:  ## Download CubiCasa5K dataset
	@echo "$(BLUE)Downloading CubiCasa5K dataset...$(NC)"
	$(PYTHON) scripts/download_dataset.py
	@echo "$(GREEN)Dataset downloaded!$(NC)"

eda:  ## Run exploratory data analysis
	@echo "$(BLUE)Running EDA with MLflow...$(NC)"
	$(PYTHON) scripts/eda_analysis.py
	@echo "$(GREEN)EDA completed! Check mlruns/ for results$(NC)"

train:  ## Train the model
	@echo "$(BLUE)Training Swin Mask R-CNN...$(NC)"
	$(PYTHON) scripts/train.py --config configs/train_config.yaml
	@echo "$(GREEN)Training completed!$(NC)"

train-resume:  ## Resume training from checkpoint
	@echo "$(BLUE)Resuming training...$(NC)"
	$(PYTHON) scripts/train.py --resume checkpoints/last_checkpoint.pth
	@echo "$(GREEN)Training resumed!$(NC)"

evaluate:  ## Evaluate trained model
	@echo "$(BLUE)Evaluating model...$(NC)"
	$(PYTHON) scripts/evaluate.py --checkpoint checkpoints/best_model.pth
	@echo "$(GREEN)Evaluation completed!$(NC)"

test:  ## Run all tests
	@echo "$(BLUE)Running tests...$(NC)"
	$(PYTEST) tests/ -v
	@echo "$(GREEN)Tests completed!$(NC)"

test-cov:  ## Run tests with coverage
	@echo "$(BLUE)Running tests with coverage...$(NC)"
	$(PYTEST) --cov=src --cov-report=html --cov-report=term tests/
	@echo "$(GREEN)Coverage report generated in htmlcov/$(NC)"

inference:  ## Run inference on sample image
	@echo "$(BLUE)Running inference...$(NC)"
	$(PYTHON) scripts/inference.py \
		--checkpoint checkpoints/best_model.pth \
		--image results/synthetic_floorplan.png \
		--output results/inference_result.png
	@echo "$(GREEN)Inference completed!$(NC)"

mlflow-ui:  ## Start MLflow UI
	@echo "$(BLUE)Starting MLflow UI at http://localhost:5000...$(NC)"
	$(MLFLOW) ui

format:  ## Format code with black
	@echo "$(BLUE)Formatting code...$(NC)"
	black src/ scripts/ tests/
	@echo "$(GREEN)Code formatted!$(NC)"

lint:  ## Lint code with flake8
	@echo "$(BLUE)Linting code...$(NC)"
	flake8 src/ scripts/ tests/
	@echo "$(GREEN)Linting completed!$(NC)"

type-check:  ## Type check with mypy
	@echo "$(BLUE)Type checking...$(NC)"
	mypy src/
	@echo "$(GREEN)Type checking completed!$(NC)"

clean:  ## Clean temporary files
	@echo "$(BLUE)Cleaning temporary files...$(NC)"
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete
	find . -type d -name '*.egg-info' -delete
	rm -rf .pytest_cache
	rm -rf .mypy_cache
	rm -rf htmlcov
	rm -rf .coverage
	@echo "$(GREEN)Cleanup completed!$(NC)"

clean-all: clean  ## Clean everything (including checkpoints and results)
	@echo "$(YELLOW)WARNING: This will delete checkpoints and results!$(NC)"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		rm -rf checkpoints/*; \
		rm -rf results/*; \
		rm -rf mlruns/*; \
		echo "$(GREEN)All data cleaned!$(NC)"; \
	fi

docker-build:  ## Build Docker image
	@echo "$(BLUE)Building Docker image...$(NC)"
	docker build -t cubicasa5k-detection:latest -f docker/Dockerfile .
	@echo "$(GREEN)Docker image built!$(NC)"

docker-run:  ## Run Docker container
	@echo "$(BLUE)Running Docker container...$(NC)"
	docker run --gpus all -p 5000:5000 cubicasa5k-detection:latest
	@echo "$(GREEN)Container running!$(NC)"

setup:  ## Initial project setup
	@echo "$(BLUE)Setting up project...$(NC)"
	mkdir -p data checkpoints results mlruns logs
	cp .env.example .env
	@echo "$(GREEN)Project setup completed!$(NC)"

notebook:  ## Start Jupyter notebook server
	@echo "$(BLUE)Starting Jupyter notebook...$(NC)"
	jupyter notebook notebooks/

# Default target
.DEFAULT_GOAL := help
