.PHONY: help test run-regression run-classification run-hyperparam-scan clean

help:
	@echo "Gradient Boosting from Scratch - Makefile"
	@echo ""
	@echo "Available targets:"
	@echo "  make test                  - Run all unit tests"
	@echo "  make run-regression        - Run regression experiments (California Housing)"
	@echo "  make run-classification    - Run classification experiments (Breast Cancer)"
	@echo "  make run-hyperparam-scan   - Run comprehensive hyperparameter grid search"
	@echo "  make clean                 - Remove generated files and caches"
	@echo ""

test:
	@echo "Running unit tests..."
	pytest tests/test_core.py -v

run-regression:
	@echo "Running regression experiments..."
	python experiments/regression_run.py

run-classification:
	@echo "Running classification experiments..."
	python experiments/classification_run.py

run-hyperparam-scan:
	@echo "Running hyperparameter scan (this may take a while)..."
	python experiments/hyperparam_scan.py

clean:
	@echo "Cleaning generated files..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true
	find . -type f -name "*.pyo" -delete 2>/dev/null || true
	find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@echo "Clean complete!"
