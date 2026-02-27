# Makefile for GP Regression and Classification Project
# Compatible with Windows (use `make` via Git Bash or WSL, or run commands directly in PowerShell)

.PHONY: help install test run-regression run-classification run-toy-cov run-hyperparam run-all clean

help:
	@echo "Available targets:"
	@echo "  make install           - Install dependencies"
	@echo "  make test              - Run unit tests"
	@echo "  make run-regression    - Run regression experiment"
	@echo "  make run-classification - Run classification experiment"
	@echo "  make run-toy-cov       - Run 2D covariance visualisation"
	@echo "  make run-hyperparam    - Run hyperparameter scan"
	@echo "  make run-all           - Run all experiments"
	@echo "  make clean             - Clean output files"

install:
	pip install -r requirements.txt

test:
	pytest tests/test_gp.py -v

run-regression:
	python experiments/regression_demo.py

run-classification:
	python experiments/classification_demo.py

run-toy-cov:
	python experiments/toy_2d_covariance.py

run-hyperparam:
	python experiments/hyperparam_scan.py

run-all: run-regression run-classification run-toy-cov run-hyperparam
	@echo ""
	@echo "=========================================="
	@echo "All experiments completed successfully!"
	@echo "Results saved to experiments/outputs/"
	@echo "=========================================="

clean:
	rm -rf experiments/outputs/*.png
	rm -rf experiments/outputs/*.csv
	rm -rf src/gp/__pycache__
	rm -rf tests/__pycache__
	rm -rf .pytest_cache
	@echo "Cleaned output files and cache"
