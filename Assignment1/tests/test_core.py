"""
Unit tests for gradient boosting implementation.

Tests numerical correctness of:
- Pseudo-residual computation
- Leaf gamma optimisation
- Model fitting and prediction
- Determinism with random_state
"""

import numpy as np
import pytest
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.datasets import make_regression, make_classification
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path

# Add src directory to path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from gbt.core import GradientBoostingRegressor, GradientBoostingClassifier
from gbt.utils import (
    mse_negative_gradient, mse_optimal_gamma,
    logistic_negative_gradient, logistic_optimal_gamma,
    sigmoid
)


# =========================
# Test Loss Functions
# =========================

def test_mse_negative_gradient():
    """Test MSE pseudo-residuals match theoretical formula."""
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([1.5, 1.8, 3.2, 3.5])
    
    residuals = mse_negative_gradient(y_true, y_pred)
    expected = y_true - y_pred
    
    np.testing.assert_allclose(residuals, expected, rtol=1e-10)


def test_mse_optimal_gamma():
    """Test optimal leaf value for MSE is mean of residuals."""
    y_true = np.array([2.0, 3.0, 5.0, 7.0])
    y_pred = np.array([1.0, 2.5, 4.0, 6.0])
    
    gamma = mse_optimal_gamma(y_true, y_pred)
    expected_gamma = np.mean(y_true - y_pred)
    
    assert abs(gamma - expected_gamma) < 1e-10


def test_logistic_negative_gradient():
    """Test logistic loss pseudo-residuals."""
    y_true = np.array([0, 1, 1, 0])
    F = np.array([-0.5, 1.2, 0.3, -1.0])
    
    residuals = logistic_negative_gradient(y_true, F)
    
    # Should be y - sigmoid(F)
    p = sigmoid(F)
    expected = y_true - p
    
    np.testing.assert_allclose(residuals, expected, rtol=1e-10)


def test_sigmoid_stability():
    """Test sigmoid is numerically stable for large inputs."""
    x_large_pos = np.array([100.0, 500.0])
    x_large_neg = np.array([-100.0, -500.0])
    
    p_pos = sigmoid(x_large_pos)
    p_neg = sigmoid(x_large_neg)
    
    # Should be close to 1 and 0
    np.testing.assert_allclose(p_pos, 1.0, atol=1e-10)
    np.testing.assert_allclose(p_neg, 0.0, atol=1e-10)


# =========================
# Test GradientBoostingRegressor
# =========================

def test_regressor_single_tree_matches_dt():
    """
    Test that a single-tree boosting (n_estimators=1, learning_rate=1.0)
    approximately matches fitting DecisionTreeRegressor to residuals.
    """
    X, y = make_regression(n_samples=100, n_features=10, random_state=42)
    
    # Fit our boosting with 1 tree and no shrinkage
    gbr = GradientBoostingRegressor(
        n_estimators=1,
        learning_rate=1.0,
        max_depth=3,
        subsample=1.0,
        random_state=42
    )
    gbr.fit(X, y)
    y_pred_boost = gbr.predict(X)
    
    # Fit a decision tree to residuals from f_0 = mean(y)
    f0 = np.mean(y)
    residuals = y - f0
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X, residuals)
    y_pred_dt = f0 + dt.predict(X)
    
    # Should be very close
    np.testing.assert_allclose(y_pred_boost, y_pred_dt, rtol=1e-5)


def test_regressor_determinism():
    """Test that same random_state gives identical results."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=123)
    
    gbr1 = GradientBoostingRegressor(
        n_estimators=10,
        subsample=0.8,
        random_state=42
    )
    gbr1.fit(X, y)
    pred1 = gbr1.predict(X)
    
    gbr2 = GradientBoostingRegressor(
        n_estimators=10,
        subsample=0.8,
        random_state=42
    )
    gbr2.fit(X, y)
    pred2 = gbr2.predict(X)
    
    np.testing.assert_array_equal(pred1, pred2)


def test_regressor_learning_rate_effect():
    """Test that lower learning_rate reduces per-iteration impact."""
    X, y = make_regression(n_samples=100, n_features=5, random_state=42)
    
    # High learning rate
    gbr_high = GradientBoostingRegressor(
        n_estimators=5,
        learning_rate=1.0,
        max_depth=3,
        random_state=42
    )
    gbr_high.fit(X, y)
    
    # Low learning rate
    gbr_low = GradientBoostingRegressor(
        n_estimators=5,
        learning_rate=0.1,
        max_depth=3,
        random_state=42
    )
    gbr_low.fit(X, y)
    
    # High learning rate should have lower training error (may overfit)
    mse_high = np.mean((y - gbr_high.predict(X)) ** 2)
    mse_low = np.mean((y - gbr_low.predict(X)) ** 2)
    
    assert mse_high < mse_low


def test_regressor_validation_tracking():
    """Test that validation scores are tracked correctly."""
    X, y = make_regression(n_samples=200, n_features=5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    gbr = GradientBoostingRegressor(n_estimators=10, random_state=42)
    gbr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    assert len(gbr.train_scores_) == 10
    assert len(gbr.val_scores_) == 10


# =========================
# Test GradientBoostingClassifier
# =========================

def test_classifier_predict_proba_range():
    """Test that predicted probabilities are in [0, 1]."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )
    
    gbc = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gbc.fit(X, y)
    
    proba = gbc.predict_proba(X)
    
    assert np.all(proba >= 0.0)
    assert np.all(proba <= 1.0)


def test_classifier_predict_matches_proba():
    """Test that predict returns argmax of predict_proba."""
    X, y = make_classification(
        n_samples=100, n_features=10, n_classes=2, random_state=42
    )
    
    gbc = GradientBoostingClassifier(n_estimators=20, random_state=42)
    gbc.fit(X, y)
    
    proba = gbc.predict_proba(X)
    pred = gbc.predict(X)
    pred_from_proba = (proba >= 0.5).astype(int)
    
    np.testing.assert_array_equal(pred, pred_from_proba)


def test_classifier_determinism():
    """Test that same random_state gives identical results."""
    X, y = make_classification(n_samples=100, n_features=5, random_state=123)
    
    gbc1 = GradientBoostingClassifier(
        n_estimators=10,
        subsample=0.8,
        random_state=42
    )
    gbc1.fit(X, y)
    pred1 = gbc1.predict(X)
    
    gbc2 = GradientBoostingClassifier(
        n_estimators=10,
        subsample=0.8,
        random_state=42
    )
    gbc2.fit(X, y)
    pred2 = gbc2.predict(X)
    
    np.testing.assert_array_equal(pred1, pred2)


def test_classifier_improves_with_iterations():
    """Test that more iterations generally improve training accuracy."""
    X, y = make_classification(
        n_samples=200, n_features=10, n_informative=8, random_state=42
    )
    
    gbc_few = GradientBoostingClassifier(n_estimators=5, random_state=42)
    gbc_few.fit(X, y)
    acc_few = np.mean(gbc_few.predict(X) == y)
    
    gbc_many = GradientBoostingClassifier(n_estimators=50, random_state=42)
    gbc_many.fit(X, y)
    acc_many = np.mean(gbc_many.predict(X) == y)
    
    # More iterations should give better training accuracy
    assert acc_many >= acc_few


def test_classifier_validation_tracking():
    """Test that validation scores are tracked correctly."""
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    gbc = GradientBoostingClassifier(n_estimators=10, random_state=42)
    gbc.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    
    assert len(gbc.train_scores_) == 10
    assert len(gbc.val_scores_) == 10


# =========================
# Test Edge Cases
# =========================

def test_regressor_single_sample():
    """Test regressor on minimal data."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1.0, 2.0])
    
    gbr = GradientBoostingRegressor(n_estimators=5, random_state=42)
    gbr.fit(X, y)
    pred = gbr.predict(X)
    
    assert pred.shape == y.shape


def test_classifier_single_sample():
    """Test classifier on minimal data."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([0, 1])
    
    gbc = GradientBoostingClassifier(n_estimators=5, random_state=42)
    gbc.fit(X, y)
    pred = gbc.predict(X)
    proba = gbc.predict_proba(X)
    
    assert pred.shape == y.shape
    assert proba.shape == y.shape
    assert np.all(proba >= 0.0) and np.all(proba <= 1.0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
