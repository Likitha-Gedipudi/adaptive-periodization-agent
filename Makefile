.PHONY: install install-dev lint format test test-cov train tune evaluate clean help

# Python interpreter
PYTHON = python3
PIP = pip

# Directories
SRC_DIR = src
TEST_DIR = tests
DATA_DIR = data
EXPERIMENTS_DIR = experiments

# Default target
.DEFAULT_GOAL := help

help:
	@echo "Adaptive Periodization Agent - Available Commands"
	@echo "================================================="
	@echo ""
	@echo "Setup:"
	@echo "  make install       Install production dependencies"
	@echo "  make install-dev   Install development dependencies"
	@echo "  make setup         Full setup (install-dev + pre-commit)"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run linting checks (flake8, mypy)"
	@echo "  make format        Format code with black and isort"
	@echo "  make check         Run all quality checks"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run unit tests"
	@echo "  make test-cov      Run tests with coverage report"
	@echo "  make test-fast     Run tests excluding slow markers"
	@echo ""
	@echo "Training:"
	@echo "  make train         Run training with default config"
	@echo "  make train-quick   Quick training run (10 episodes)"
	@echo "  make tune          Run hyperparameter tuning"
	@echo "  make evaluate      Evaluate trained model"
	@echo ""
	@echo "Data:"
	@echo "  make data-synth    Generate synthetic dataset"
	@echo "  make data-process  Preprocess data"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Remove generated files"
	@echo "  make tensorboard   Launch TensorBoard"

# ============================================
# Setup
# ============================================

install:
	$(PIP) install -e .

install-dev:
	$(PIP) install -e ".[dev,notebook]"

setup: install-dev
	pre-commit install
	mkdir -p $(DATA_DIR)/raw $(DATA_DIR)/processed $(EXPERIMENTS_DIR)
	@echo "Setup complete!"

# ============================================
# Code Quality
# ============================================

lint:
	flake8 $(SRC_DIR) $(TEST_DIR) --max-line-length=100 --ignore=E203,W503
	mypy $(SRC_DIR) --ignore-missing-imports

format:
	black $(SRC_DIR) $(TEST_DIR)
	isort $(SRC_DIR) $(TEST_DIR)

check: format lint
	@echo "All checks passed!"

# ============================================
# Testing
# ============================================

test:
	pytest $(TEST_DIR) -v

test-cov:
	pytest $(TEST_DIR) --cov=$(SRC_DIR) --cov-report=html --cov-report=term-missing
	@echo "Coverage report generated in htmlcov/"

test-fast:
	pytest $(TEST_DIR) -v -m "not slow"

# ============================================
# Training & Evaluation
# ============================================

train:
	$(PYTHON) -m src.training.train

train-quick:
	$(PYTHON) -m src.training.train --episodes 10 --synthetic

tune:
	$(PYTHON) -m src.training.tune

evaluate:
	$(PYTHON) -m src.evaluation.evaluate

# ============================================
# Data
# ============================================

data-synth:
	$(PYTHON) -c "from src.data.synthetic_data import generate_and_save_dataset; generate_and_save_dataset()"

data-process:
	$(PYTHON) -m src.data.preprocess

# ============================================
# Utilities
# ============================================

tensorboard:
	tensorboard --logdir=runs

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ htmlcov/ .coverage
	@echo "Cleaned up generated files"
