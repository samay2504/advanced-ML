# Gradient Boosting from Scratch

A production-grade implementation of **Forward Stagewise Additive Modelling (Algorithm 10.3)** and **Gradient Tree Boosting (Algorithm 10.4)** from *The Elements of Statistical Learning* by Hastie, Tibshirani, and Friedman.

## Overview

This project implements gradient boosting for regression and binary classification **from scratch**, without using any off-the-shelf gradient boosting libraries (no `sklearn.ensemble.GradientBoosting*`, `xgboost`, `lightgbm`, or `catboost`). The boosting loop, loss functions, pseudo-residuals, leaf optimisation, shrinkage, and subsampling are all implemented manually.

### Key Features

- ✅ **Algorithm 10.4** for squared-error loss (regression)
- ✅ **Algorithm 10.4** for binomial deviance (classification)
- ✅ Pseudo-residual computation (negative gradients)
- ✅ Leaf optimisation using closed-form solutions (MSE) and Newton-Raphson (logistic loss)
- ✅ Shrinkage (learning rate) for regularisation
- ✅ Stochastic subsampling for variance reduction
- ✅ Comprehensive unit tests
- ✅ Reproducible experiments with visualisations
- ✅ Interactive Jupyter notebook demo

## Installation

```bash
# Clone or navigate to the repository
cd "d:\Projects2.0\Last Days Work\AML"

# Install dependencies
pip install -r requirements.txt
```

### Requirements

- Python 3.8+
- numpy >= 1.24.0
- scipy >= 1.10.0
- pandas >= 2.0.0
- scikit-learn >= 1.3.0 (for base learners and datasets only)
- matplotlib >= 3.7.0
- pytest >= 7.4.0
- jupyter >= 1.0.0

## Quick Start

### Run Unit Tests

```bash
# Run all tests
make test

# Or directly with pytest
pytest tests/test_core.py -v
```

### Run Experiments

```bash
# Regression experiment (California Housing)
make run-regression

# Classification experiment (Breast Cancer)
make run-classification

# Comprehensive hyperparameter scan
make run-hyperparam-scan
```

### Interactive Demo

```bash
# Launch Jupyter notebook
jupyter notebook notebooks/brief_demo.ipynb
```

## Project Structure

```
AML/
├── requirements.txt
├── Makefile
├── README.md
├── src/
│   └── gbt/
│       ├── __init__.py
│       ├── core.py          # Main GradientBoostingRegressor & Classifier
│       └── utils.py         # Loss functions, metrics, optimisation
├── experiments/
│   ├── regression_run.py           # California Housing experiments
│   ├── classification_run.py       # Breast Cancer experiments
│   └── hyperparam_scan.py          # Grid search over hyperparameters
├── notebooks/
│   └── brief_demo.ipynb            # Interactive demo and analysis
└── tests/
    └── test_core.py                # Unit tests
```

## Implementation Details

### Algorithm 10.4: Gradient Tree Boosting

**Initialisation:**
- Regression: $f_0(x) = \bar{y}$ (mean of targets)
- Classification: $f_0(x) = \log(\frac{\bar{y}}{1-\bar{y}})$ (log-odds)

**Boosting Loop** (for $m = 1, \ldots, M$):

1. **Compute pseudo-residuals** (negative gradients):
   - Regression: $r_{im} = y_i - f_{m-1}(x_i)$
   - Classification: $r_{im} = y_i - \sigma(f_{m-1}(x_i))$

2. **Fit weak learner**: Fit shallow decision tree to $(x_i, r_{im})$ yielding leaf regions $R_{jm}$

3. **Optimise leaf values**:
   - Regression: $\gamma_{jm} = \text{mean}(r_i : x_i \in R_{jm})$
   - Classification (Newton step): $\gamma_{jm} = \frac{\sum r_i}{\sum p_i(1-p_i)}$ where $p_i = \sigma(f_{m-1}(x_i))$

4. **Update model**: $f_m(x) = f_{m-1}(x) + \nu \sum_j \gamma_{jm} I(x \in R_{jm})$

where $\nu$ is the learning rate (shrinkage).

### Key Design Choices

- **Weak learners**: `sklearn.tree.DecisionTreeRegressor` (for fitting residuals in both regression and classification)
- **Leaf optimisation**: Analytical solutions for MSE; Newton-Raphson for logistic loss (LogitBoost approach)
- **Stochastic boosting**: Row subsampling per iteration with `subsample` parameter
- **Shrinkage**: Learning rate $\nu$ applied to all leaf updates for regularisation

### References

- Hastie, T., Tibshirani, R., & Friedman, J. (2009). *The Elements of Statistical Learning* (2nd ed.). Chapter 10.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *Annals of Statistics*, 29(5), 1189-1232.
- Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive logistic regression: a statistical view of boosting (LogitBoost). *Annals of Statistics*, 28(2), 337-407.
- Wikipedia: [Gradient Boosting](https://en.wikipedia.org/wiki/Gradient_boosting)
- Wikipedia: [LogitBoost](https://en.wikipedia.org/wiki/LogitBoost)

## API Reference

### `GradientBoostingRegressor`

```python
from gbt.core import GradientBoostingRegressor

gbr = GradientBoostingRegressor(
    n_estimators=100,      # Number of boosting iterations
    learning_rate=0.1,     # Shrinkage parameter ν
    max_depth=3,           # Maximum tree depth
    min_samples_leaf=1,    # Minimum samples per leaf
    subsample=1.0,         # Fraction of samples per iteration
    random_state=42,       # Random seed
    verbose=False          # Enable logging
)

gbr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
predictions = gbr.predict(X_test)
```

### `GradientBoostingClassifier`

```python
from gbt.core import GradientBoostingClassifier

gbc = GradientBoostingClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    min_samples_leaf=1,
    subsample=1.0,
    random_state=42,
    verbose=False
)

gbc.fit(X_train, y_train, X_val=X_val, y_val=y_val)
probabilities = gbc.predict_proba(X_test)  # Returns P(y=1|x)
predictions = gbc.predict(X_test)          # Returns {0, 1}
```

## Experiments & Results

### Regression (California Housing)

**Baseline:** Single DecisionTreeRegressor (max_depth=3)
- Test MSE: ~0.52

**Gradient Boosting:**
- n_estimators=150, learning_rate=0.1, max_depth=3, subsample=0.8
- Test MSE: ~0.30 (42% improvement)

### Classification (Breast Cancer)

**Baseline:** Single DecisionTreeClassifier (max_depth=3)
- Test Accuracy: ~0.93
- Test ROC AUC: ~0.95

**Gradient Boosting:**
- n_estimators=150, learning_rate=0.1, max_depth=3, subsample=0.8
- Test Accuracy: ~0.97
- Test ROC AUC: ~0.99

### Hyperparameter Effects

1. **Learning Rate** ($\nu$): Lower values (0.01-0.05) converge more slowly but generalise better. Optimal: 0.05-0.1.

2. **Number of Estimators** ($M$): More trees improve fit, but with diminishing returns beyond 100-200. Early stopping recommended.

3. **Tree Depth**: Shallow trees (2-4) provide strong regularisation. Deeper trees (5+) risk overfitting. Optimal: 3-4.

4. **Subsample**: Stochastic sampling (0.5-0.8) adds regularisation and speeds up training. Optimal: 0.5-0.8.

See `notebooks/brief_demo.ipynb` for detailed analysis and visualisations.

## Testing

Unit tests verify:
- Pseudo-residual computation matches theoretical formulas
- Leaf optimisation returns correct values (mean for MSE, Newton step for logistic)
- Single-tree boosting approximates residual fitting
- Determinism with fixed `random_state`
- Probability outputs are in [0, 1]
- Classification predictions are binary {0, 1}

Run tests:
```bash
pytest tests/test_core.py -v
```

## Code Style

- Type hints for all function signatures
- Docstrings following NumPy style
- Line length ≤ 100 characters
- Small, focused functions
- Vectorised NumPy operations
- Logging for verbosity control

Format code with `black`:
```bash
black src/ tests/ experiments/
```

## Reproducibility

All experiments use fixed `random_state=42`. Results are saved as CSV files and plots as PNG images in the `experiments/` directory.

## Limitations

- Binary classification only (not multiclass)
- No support for missing data
- No built-in feature importance calculation
- No early stopping (must specify n_estimators)

## License

This implementation is for educational purposes, demonstrating algorithms from *The Elements of Statistical Learning*.

## Author

Implemented as part of coursework for Advanced Machine Learning.

---

**Note:** This implementation prioritises correctness, clarity, and educational value over performance optimisation. For production use cases requiring high performance, consider libraries like XGBoost or LightGBM.
