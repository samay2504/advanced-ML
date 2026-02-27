# Gaussian Process Regression and Classification

**Implementation of Algorithms 2.1, 3.1, and 3.2 from "Gaussian Processes for Machine Learning" by Rasmussen and Williams**

This project implements exact Gaussian Process regression and Laplace-approximation binary classification from scratch, with comprehensive experiments demonstrating uncertainty quantification.

## Features

- ✅ **Exact GP Regression** (Algorithm 2.1) with Cholesky decomposition
- ✅ **GP Classification with Laplace Approximation** (Algorithms 3.1 & 3.2)
- ✅ **Hyperparameter Optimisation** via log marginal likelihood (L-BFGS-B)
- ✅ **Multiple Kernels**: RBF, Matérn (ν=3/2, 5/2), Rational Quadratic, White Noise
- ✅ **Numerical Stability**: Adaptive jitter, stable Cholesky, clamped probabilities
- ✅ **Comprehensive Experiments**: Regression, classification, 2D covariance visualisation, hyperparameter scans
- ✅ **Unit Tests**: Full test coverage with pytest
- ✅ **Production-Grade Code**: Type hints, docstrings, logging, British English

## Installation

### Requirements

- Python 3.8+
- Dependencies listed in `requirements.txt`

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Alternatively, use make
make install
```

## Quick Start

### Run All Experiments

```bash
make run-all
```

This executes all experiments and saves results to `experiments/outputs/`.

### Individual Experiments

```bash
# GP Regression on diabetes dataset
make run-regression

# GP Classification on breast cancer dataset
make run-classification

# 2D covariance structure visualisation
make run-toy-cov

# Hyperparameter grid search
make run-hyperparam
```

### Run Tests

```bash
make test
```

### Jupyter Notebook

```bash
jupyter notebook notebooks/brief_demo.ipynb
```

## Project Structure

```
.
├── README.md                    # This file
├── requirements.txt             # Dependencies
├── Makefile                     # Build automation
├── report.md                    # Technical report
├── src/gp/                      # Core GP library
│   ├── __init__.py
│   ├── base.py                  # Numerical utilities
│   ├── kernels.py               # Kernel functions
│   ├── regression.py            # Algorithm 2.1
│   └── classification.py        # Algorithms 3.1 & 3.2
├── experiments/                 # Reproducible experiments
│   ├── regression_demo.py       # Diabetes dataset
│   ├── classification_demo.py   # Breast cancer dataset
│   ├── toy_2d_covariance.py     # Covariance visualisation
│   ├── hyperparam_scan.py       # Grid search
│   └── outputs/                 # Generated figures and CSVs
├── notebooks/
│   └── brief_demo.ipynb         # Interactive demonstration
└── tests/
    └── test_gp.py               # Unit tests
```

## Algorithm Implementations

### Algorithm 2.1: GP Regression

Implements exact inference for Gaussian Process regression:

1. **Compute kernel matrices**: K = k(X,X) + σ²ₙI, K* = k(X, X*), k** = k(X*, X*)
2. **Cholesky decomposition**: L such that LL^T = K
3. **Precompute**: α = L^T \ (L \ y)
4. **Predictive mean**: f̄* = K*^T α
5. **Predictive variance**: v = L \ K*; Var[f*] = k** - v^T v

**Code**: [`src/gp/regression.py`](src/gp/regression.py)

### Algorithms 3.1 & 3.2: GP Classification (Laplace)

Implements binary classification via Laplace approximation:

**Algorithm 3.1 (Posterior Mode via Newton's Method)**:
- Initialise f = 0
- Iterate until convergence:
  - W = -∇∇ log p(y|f) (negative Hessian)
  - b = Wf + ∇ log p(y|f)
  - B = I + W^(1/2) K W^(1/2)
  - L = cholesky(B)
  - a = b - W^(1/2) L^T \ (L \ (W^(1/2) K b))
  - f ← Ka

**Algorithm 3.2 (Predictions)**:
- Compute predictive mean: f̄* = k(x*)^T ∇ log p(y|f̂)
- Compute predictive variance: v = L \ (W^(1/2) k(x*)); Var[f*] = k(x*,x*) - v^T v
- Averaged probability: π̄* = ∫ σ(f*) N(f* | f̄*, Var[f*]) df*

**Code**: [`src/gp/classification.py`](src/gp/classification.py)

## Usage Examples

### Regression

```python
from src.gp.regression import GaussianProcessRegressor
from src.gp.kernels import RBF

# Create and fit GP
gp = GaussianProcessRegressor(
    kernel=RBF(length_scale=1.0, variance=1.0),
    noise=0.1,
    optimise=True,  # Optimise hyperparameters
    n_restarts=2,
    random_state=42
)

gp.fit(X_train, y_train)

# Predictions with uncertainty
mean, std = gp.predict(X_test, return_std=True)

# Full covariance matrix
mean, cov = gp.predict_f_cov(X_test)

# Log marginal likelihood
lml = gp.log_marginal_likelihood_value()
```

### Classification

```python
from src.gp.classification import GaussianProcessClassifier
from src.gp.kernels import RBF

# Create and fit GP classifier
gp = GaussianProcessClassifier(
    kernel=RBF(length_scale=1.0),
    likelihood='logistic',  # or 'probit'
    max_iter=50,
    tol=1e-6,
    random_state=42
)

gp.fit(X_train, y_train)

# Predict probabilities
proba = gp.predict_proba(X_test)

# Predict class labels
y_pred = gp.predict(X_test)

# Latent function posterior
mean, cov = gp.predict_f_cov(X_test)
```

## Experiments

### 1. Regression Demo

**Dataset**: Diabetes (n=442, d=10)  
**Task**: Predict disease progression with uncertainty quantification

**Outputs**:
- `regression_predictions.png`: Predictive mean ±2σ on 1D PCA projection
- `regression_optimisation_trace.png`: Log marginal likelihood during optimisation
- `regression_predictions.csv`: Test predictions with confidence intervals

**Key Metrics**:
- RMSE
- Negative Log Predictive Density (NLPD)
- Log Marginal Likelihood

### 2. Classification Demo

**Dataset**: Breast Cancer (n=569, d=30)  
**Task**: Binary classification with uncertainty quantification

**Outputs**:
- `classification_roc_curve.png`: ROC curve
- `classification_calibration.png`: Calibration curve
- `classification_2d_heatmap.png`: Probability and uncertainty heatmaps (2D PCA)

**Key Metrics**:
- ROC AUC
- Log Loss
- Accuracy

### 3. Toy 2D Covariance

**Task**: Visualise covariance structure on small 2D dataset

**Outputs**:
- `toy_2d_mean_variance.png`: Mean, variance, and std dev heatmaps
- `toy_2d_covariance_structure.png`: Full covariance matrix and slices

**Demonstrates**:
- How covariance decays with distance
- Impact of lengthscale on correlation structure
- Relationship between data density and uncertainty

### 4. Hyperparameter Scan

**Task**: Grid search over lengthscale and noise variance

**Outputs**:
- `hyperparam_scan_regression.png`: LML and RMSE heatmaps
- `hyperparam_scan_classification.png`: LML and log-loss curves
- CSV files with detailed results

## Numerical Stability

This implementation includes several numerical safeguards:

1. **Adaptive Jitter**: Automatically increases diagonal regularisation if Cholesky fails
2. **Clamped Probabilities**: Ensures probabilities stay in [1e-9, 1-1e-9]
3. **Stable Cholesky Solve**: Two-step triangular solve instead of direct inversion
4. **Variance Clipping**: Ensures non-negative predictive variance
5. **Log-Space Optimisation**: Hyperparameters optimised in log space for unconstrained optimisation

## Computational Complexity

- **GP Regression Training**: O(n³) for Cholesky, O(n²) for α computation
- **GP Regression Prediction**: O(nm) for m test points (O(m²) for full covariance)
- **GP Classification Training**: O(kn³) for k Newton iterations
- **GP Classification Prediction**: O(nm) per prediction

For n > 2000, consider sparse approximations (not implemented here).

## Testing

Run the test suite:

```bash
pytest tests/test_gp.py -v
```

**Tests include**:
- ✅ Cholesky solve correctness
- ✅ Kernel symmetry and parameter roundtrip
- ✅ GP regression prediction shapes and interpolation
- ✅ GP classification convergence and probability bounds
- ✅ Determinism with fixed random seeds
- ✅ Log marginal likelihood values

## Code Quality

- **Type Hints**: All functions have complete type annotations
- **Docstrings**: Google-style docstrings for all public APIs
- **Logging**: Informative logging at INFO level (enable with `verbose=True`)
- **PEP8 Compliant**: Follow standard Python style guidelines
- **British English**: Consistent spelling in comments and strings

## References

1. Rasmussen, C. E., & Williams, C. K. I. (2006). *Gaussian Processes for Machine Learning*. MIT Press. [PDF](https://sisis.rz.htw-berlin.de/inh2012/12427576.pdf)

2. Chapter 2: Regression — Algorithm 2.1
3. Chapter 3: Classification — Algorithms 3.1 & 3.2

## Report

See [`report.md`](report.md) for a detailed technical report covering:
- Implementation details
- Numerical considerations
- Experimental results and interpretation
- Uncertainty quantification analysis
- Computational complexity discussion

## Limitations and Future Work

**Current Limitations**:
- O(n³) scaling limits to ~2000 training points
- No sparse GP approximations (FITC, VFE, etc.)
- Binary classification only (no multi-class)
- No automatic differentiation (gradients via finite differences for some kernels)

**Possible Extensions**:
- Sparse approximations (inducing points)
- Multi-class classification (one-vs-rest or softmax)
- Expectation Propagation (alternative to Laplace)
- Automatic Relevance Determination (ARD) kernels
- Deep Gaussian Processes

## Licence

MIT Licence — Free to use for academic and commercial purposes.

## Author

Implemented as part of Advanced Machine Learning coursework, following Rasmussen & Williams (2006).

---

**Questions?** See the [report](report.md) or inspect the well-documented source code in [`src/gp/`](src/gp/).
