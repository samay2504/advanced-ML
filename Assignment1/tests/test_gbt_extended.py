"""
Extended tests for gradient boosting implementation.

Author : Atharva Date
Roll No: B22AI045
Course : Advanced Machine Learning, IIT Jodhpur

Coverage:
- Loss function correctness (mse_loss, logistic_loss, logistic_optimal_gamma)
- Utility metric functions (compute_metrics_regression, compute_metrics_classification)
- Model state invariants after fit (f0_, estimators_, leaf_values_, train_scores_)
- Training loss monotonicity under deterministic subsampling
- Subsample stochasticity and reproducibility contracts
- Initialisation correctness for regressor (mean) and classifier (log-odds)
- Convergence on controlled toy problems
- Benchmark parity with sklearn GradientBoosting{Regressor,Classifier}
- Staged prediction consistency
- Input–output shape contracts
- Logistic optimal gamma Newton step correctness
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.datasets import make_classification, make_regression
from sklearn.ensemble import (
    GradientBoostingClassifier as SklearnGBC,
    GradientBoostingRegressor as SklearnGBR,
)
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

# ---------- path setup ----------
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from gbt.core import GradientBoostingClassifier, GradientBoostingRegressor
from gbt.utils import (
    compute_metrics_classification,
    compute_metrics_regression,
    logistic_loss,
    logistic_negative_gradient,
    logistic_optimal_gamma,
    mse_loss,
    mse_negative_gradient,
    sigmoid,
)


# =============================================================================
# Loss function correctness
# =============================================================================


class TestLossFunctions:
    """Verify loss values and gradients against independent implementations."""

    def test_mse_loss_known_value(self):
        """MSE loss = 0.5 * mean((y - ŷ)²)."""
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([1.0, 2.0, 3.0])
        assert mse_loss(y_true, y_pred) == pytest.approx(0.0, abs=1e-15)

        y_pred2 = np.array([2.0, 2.0, 2.0])
        expected = 0.5 * np.mean((y_true - y_pred2) ** 2)
        assert mse_loss(y_true, y_pred2) == pytest.approx(expected, rel=1e-10)

    def test_mse_loss_is_non_negative(self):
        rng = np.random.default_rng(0)
        y_true = rng.standard_normal(50)
        y_pred = rng.standard_normal(50)
        assert mse_loss(y_true, y_pred) >= 0.0

    def test_logistic_loss_matches_binary_cross_entropy(self):
        """Logistic loss must equal sklearn's log_loss up to tolerances."""
        from sklearn.metrics import log_loss as sklearn_log_loss

        rng = np.random.default_rng(7)
        y_true = rng.integers(0, 2, size=80).astype(float)
        raw_scores = rng.standard_normal(80)

        our_loss = logistic_loss(y_true, raw_scores)
        p = np.clip(sigmoid(raw_scores), 1e-15, 1 - 1e-15)
        sk_loss = sklearn_log_loss(y_true, p)

        assert our_loss == pytest.approx(sk_loss, rel=1e-6)

    def test_logistic_loss_decreases_with_better_predictions(self):
        """Better predictions (more confident and correct) should reduce loss."""
        y_true = np.array([1.0, 1.0, 0.0, 0.0])
        # Correct predictions with high confidence
        raw_good = np.array([5.0, 5.0, -5.0, -5.0])
        # Near-zero predictions (uncertain)
        raw_bad = np.zeros(4)
        assert logistic_loss(y_true, raw_good) < logistic_loss(y_true, raw_bad)

    def test_logistic_optimal_gamma_newton_step(self):
        """
        Verify Newton-step formula: γ* = Σr_i / Σp_i(1−p_i).

        For a leaf containing all samples, the optimal update is exactly the
        Newton-Raphson step derived from the logistic loss Hessian.
        """
        rng = np.random.default_rng(42)
        y_true = rng.integers(0, 2, size=20).astype(float)
        F = rng.standard_normal(20)

        neg_grad = logistic_negative_gradient(y_true, F)  # y - p
        gamma = logistic_optimal_gamma(y_true, F, neg_grad)

        p = sigmoid(F)
        expected = np.sum(neg_grad) / np.sum(p * (1.0 - p))
        assert gamma == pytest.approx(expected, rel=1e-8)

    def test_logistic_optimal_gamma_zero_denominator(self):
        """When Hessian denominator is zero, gamma should default to 0."""
        # All predictions saturated → p ≈ 1 → p(1-p) ≈ 0
        y_true = np.array([1.0, 1.0])
        raw = np.array([1000.0, 1000.0])
        neg_grad = logistic_negative_gradient(y_true, raw)
        gamma = logistic_optimal_gamma(y_true, raw, neg_grad)
        assert np.isfinite(gamma)
        assert gamma == pytest.approx(0.0, abs=1e-6)


# =============================================================================
# Metric utilities
# =============================================================================


class TestMetricUtilities:
    """compute_metrics_* return correct keys and numerically correct values."""

    def test_regression_metrics_keys(self):
        y = np.arange(10, dtype=float)
        pred = y + 0.1
        result = compute_metrics_regression(y, pred)
        assert {"mse", "rmse", "mae"}.issubset(result.keys())

    def test_regression_metrics_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        result = compute_metrics_regression(y, y)
        assert result["mse"] == pytest.approx(0.0, abs=1e-15)
        assert result["rmse"] == pytest.approx(0.0, abs=1e-15)
        assert result["mae"] == pytest.approx(0.0, abs=1e-15)

    def test_regression_metrics_known_values(self):
        y_true = np.array([1.0, 2.0, 3.0])
        y_pred = np.array([2.0, 2.0, 2.0])
        result = compute_metrics_regression(y_true, y_pred)
        expected_mse = np.mean((y_true - y_pred) ** 2)
        assert result["mse"] == pytest.approx(expected_mse, rel=1e-10)
        assert result["rmse"] == pytest.approx(np.sqrt(expected_mse), rel=1e-10)
        assert result["mae"] == pytest.approx(np.mean(np.abs(y_true - y_pred)), rel=1e-10)

    def test_classification_metrics_keys(self):
        y = np.array([0, 1, 0, 1, 1])
        proba = np.array([0.1, 0.9, 0.2, 0.8, 0.7])
        result = compute_metrics_classification(y, proba)
        assert {"log_loss", "accuracy", "roc_auc"}.issubset(result.keys())

    def test_classification_metrics_perfect_prediction(self):
        y = np.array([0, 1, 0, 1])
        proba = np.array([0.01, 0.99, 0.01, 0.99])  # near-perfect probabilities
        result = compute_metrics_classification(y, proba)
        assert result["accuracy"] == pytest.approx(1.0, abs=1e-9)
        assert result["roc_auc"] == pytest.approx(1.0, abs=1e-9)

    def test_classification_metrics_roc_auc_correct_binary(self):
        """ROC AUC must be 1.0 for a perfect probabilistic ranker on binary data."""
        y = np.array([0, 0, 1, 1, 1])
        # Perfect ranking: class-1 samples have strictly higher proba
        proba = np.array([0.1, 0.2, 0.7, 0.8, 0.9])
        result = compute_metrics_classification(y, proba)
        assert result["roc_auc"] == pytest.approx(1.0, abs=1e-9)

    def test_classification_metrics_roc_auc_two_classes_present(self):
        """roc_auc must be a finite float when both classes are present."""
        y = np.array([0, 1, 0, 1])
        proba = np.array([0.3, 0.6, 0.4, 0.8])
        result = compute_metrics_classification(y, proba)
        assert np.isfinite(result["roc_auc"])
        assert 0.0 <= result["roc_auc"] <= 1.0


# =============================================================================
# Model initialisation invariants
# =============================================================================


class TestModelStateAfterFit:
    """Verify internal state attributes after fitting."""

    def test_regressor_f0_equals_mean_of_y(self):
        """f0_ for regression must equal mean(y) (MSE global optimal)."""
        rng = np.random.default_rng(1)
        X = rng.standard_normal((50, 4))
        y = rng.standard_normal(50)

        gbr = GradientBoostingRegressor(n_estimators=5, random_state=0)
        gbr.fit(X, y)

        assert gbr.f0_ == pytest.approx(np.mean(y), rel=1e-10)

    def test_classifier_f0_equals_log_odds(self):
        """f0_ for classification = log(p/(1-p)), p = mean(y in {0,1})."""
        rng = np.random.default_rng(2)
        X = rng.standard_normal((60, 4))
        y = rng.integers(0, 2, size=60).astype(float)

        gbc = GradientBoostingClassifier(n_estimators=5, random_state=0)
        gbc.fit(X, y)

        p = np.mean(y)
        expected_f0 = np.log(p / (1.0 - p))
        assert gbc.f0_ == pytest.approx(expected_f0, rel=1e-10)

    def test_estimators_and_leaf_values_count(self):
        """estimators_ and leaf_values_ must have exactly n_estimators entries."""
        n_est = 8
        X, y = make_regression(n_samples=50, n_features=5, random_state=0)

        gbr = GradientBoostingRegressor(n_estimators=n_est, random_state=0)
        gbr.fit(X, y)

        assert len(gbr.estimators_) == n_est
        assert len(gbr.leaf_values_) == n_est

    def test_train_scores_length_equals_n_estimators(self):
        n_est = 12
        X, y = make_regression(n_samples=50, n_features=5, random_state=0)
        gbr = GradientBoostingRegressor(n_estimators=n_est, random_state=0)
        gbr.fit(X, y)
        assert len(gbr.train_scores_) == n_est

    def test_val_scores_empty_without_validation_data(self):
        X, y = make_regression(n_samples=50, n_features=5, random_state=0)
        gbr = GradientBoostingRegressor(n_estimators=5, random_state=0)
        gbr.fit(X, y)
        assert gbr.val_scores_ == []

    def test_classifier_state_after_fit(self):
        X, y = make_classification(n_samples=60, n_features=5, random_state=0)
        n_est = 7
        gbc = GradientBoostingClassifier(n_estimators=n_est, random_state=0)
        gbc.fit(X, y)
        assert len(gbc.estimators_) == n_est
        assert len(gbc.leaf_values_) == n_est
        assert len(gbc.train_scores_) == n_est


# =============================================================================
# Training loss monotonicity
# =============================================================================


class TestTrainingLossMonotonicity:
    """
    With full-sample deterministic boosting the training loss must be
    monotonically non-increasing.
    """

    def test_regressor_training_mse_non_increasing(self):
        X, y = make_regression(n_samples=100, n_features=10, random_state=7)
        gbr = GradientBoostingRegressor(
            n_estimators=30, learning_rate=0.3, max_depth=2,
            subsample=1.0, random_state=0
        )
        gbr.fit(X, y)
        scores = gbr.train_scores_
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 1e-10, (
                f"Training MSE increased from step {i-1} to {i}: "
                f"{scores[i-1]:.6f} → {scores[i]:.6f}"
            )

    def test_classifier_training_loss_non_increasing(self):
        X, y = make_classification(n_samples=100, n_features=10, random_state=7)
        gbc = GradientBoostingClassifier(
            n_estimators=30, learning_rate=0.3, max_depth=2,
            subsample=1.0, random_state=0
        )
        gbc.fit(X, y)
        scores = gbc.train_scores_
        for i in range(1, len(scores)):
            assert scores[i] <= scores[i - 1] + 1e-10, (
                f"Training logistic loss increased from step {i-1} to {i}: "
                f"{scores[i-1]:.6f} → {scores[i]:.6f}"
            )


# =============================================================================
# Subsample contracts
# =============================================================================


class TestSubsampleContracts:
    """Stochastic subsampling must differ across random states and be reproducible."""

    def test_subsample_introduces_variability(self):
        """Different seeds → different predictions when subsample < 1."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=0)

        pred_a = GradientBoostingRegressor(
            n_estimators=10, subsample=0.5, random_state=1
        ).fit(X, y).predict(X)

        pred_b = GradientBoostingRegressor(
            n_estimators=10, subsample=0.5, random_state=2
        ).fit(X, y).predict(X)

        # Highly unlikely to be identical with different seeds
        assert not np.allclose(pred_a, pred_b), (
            "Different random seeds with subsample<1 should produce different predictions"
        )

    def test_subsample_1_matches_full_batch(self):
        """subsample=1.0 must be identical to subsample=1.0 with any seed."""
        X, y = make_regression(n_samples=100, n_features=5, random_state=0)

        pred_a = GradientBoostingRegressor(
            n_estimators=10, subsample=1.0, random_state=42
        ).fit(X, y).predict(X)

        pred_b = GradientBoostingRegressor(
            n_estimators=10, subsample=1.0, random_state=99
        ).fit(X, y).predict(X)

        np.testing.assert_array_equal(pred_a, pred_b)


# =============================================================================
# Convergence on controlled problems
# =============================================================================


class TestConvergenceOnToyProblems:
    """The models must achieve low error on simple controlled datasets."""

    def test_regressor_low_mse_on_linear_problem(self):
        """Linear regression with enough estimators should reach near-zero MSE on training data."""
        rng = np.random.default_rng(10)
        X = rng.standard_normal((200, 3))
        y = 2 * X[:, 0] - 3 * X[:, 1] + 1.5 * X[:, 2]

        gbr = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=1.0, random_state=0
        )
        gbr.fit(X, y)

        train_r2 = r2_score(y, gbr.predict(X))
        assert train_r2 > 0.98, f"Expected R²>0.98 on training, got {train_r2:.4f}"

    def test_classifier_high_accuracy_on_separable_data(self):
        """Linearly separable data must be classified nearly perfectly on training set."""
        rng = np.random.default_rng(11)
        n = 100
        X = np.vstack([
            rng.standard_normal((n // 2, 2)) + np.array([3.0, 0.0]),
            rng.standard_normal((n // 2, 2)) + np.array([-3.0, 0.0]),
        ])
        y = np.array([1] * (n // 2) + [0] * (n // 2))

        gbc = GradientBoostingClassifier(
            n_estimators=50, learning_rate=0.1, max_depth=2,
            subsample=1.0, random_state=0
        )
        gbc.fit(X, y)
        acc = accuracy_score(y, gbc.predict(X))
        assert acc >= 0.98, f"Expected train accuracy ≥ 0.98, got {acc:.4f}"


# =============================================================================
# Parity with sklearn GradientBoosting{Regressor,Classifier}
# =============================================================================


class TestSklearnParity:
    """
    Our implementation should achieve R²/accuracy within reasonable range of
    sklearn's well-tested implementation on the same problem.
    """

    def test_regressor_r2_within_tolerance(self):
        X, y = make_regression(
            n_samples=300, n_features=10, n_informative=5, random_state=0
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

        ours = GradientBoostingRegressor(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=1.0, random_state=0
        ).fit(X_tr, y_tr)

        sk = SklearnGBR(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=1.0, random_state=0
        ).fit(X_tr, y_tr)

        our_r2 = r2_score(y_te, ours.predict(X_te))
        sk_r2 = r2_score(y_te, sk.predict(X_te))

        # Expect our R² to be within 0.1 of sklearn's
        assert abs(our_r2 - sk_r2) < 0.10, (
            f"R² gap too large: ours={our_r2:.3f}, sklearn={sk_r2:.3f}"
        )

    def test_classifier_accuracy_within_tolerance(self):
        X, y = make_classification(
            n_samples=300, n_features=10, n_informative=6, random_state=0
        )
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0)

        ours = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=1.0, random_state=0
        ).fit(X_tr, y_tr)

        sk = SklearnGBC(
            n_estimators=100, learning_rate=0.1, max_depth=3,
            subsample=1.0, random_state=0
        ).fit(X_tr, y_tr)

        our_acc = accuracy_score(y_te, ours.predict(X_te))
        sk_acc = accuracy_score(y_te, sk.predict(X_te))

        # Expect our accuracy to be within 5 percentage points of sklearn's
        assert abs(our_acc - sk_acc) < 0.05, (
            f"Accuracy gap too large: ours={our_acc:.3f}, sklearn={sk_acc:.3f}"
        )


# =============================================================================
# Output shape contracts
# =============================================================================


class TestOutputShapeContracts:
    """predict / predict_proba must produce outputs with correct shapes."""

    def test_regressor_predict_shape(self):
        X_tr, y_tr = make_regression(n_samples=50, n_features=5, random_state=0)
        X_te = np.random.default_rng(0).standard_normal((23, 5))

        gbr = GradientBoostingRegressor(n_estimators=5, random_state=0).fit(X_tr, y_tr)
        pred = gbr.predict(X_te)
        assert pred.shape == (23,)

    def test_classifier_predict_shape(self):
        X_tr, y_tr = make_classification(n_samples=50, n_features=5, random_state=0)
        X_te = np.random.default_rng(0).standard_normal((17, 5))

        gbc = GradientBoostingClassifier(n_estimators=5, random_state=0).fit(X_tr, y_tr)
        pred = gbc.predict(X_te)
        proba = gbc.predict_proba(X_te)

        assert pred.shape == (17,)
        assert proba.shape == (17,)

    def test_classifier_predict_labels_are_binary(self):
        X, y = make_classification(n_samples=60, n_features=4, random_state=0)
        gbc = GradientBoostingClassifier(n_estimators=10, random_state=0).fit(X, y)
        pred = gbc.predict(X)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_regressor_1d_feature(self):
        """Single-feature input should work without errors."""
        rng = np.random.default_rng(3)
        X = rng.standard_normal((40, 1))
        y = X[:, 0] ** 2 + rng.standard_normal(40) * 0.05
        gbr = GradientBoostingRegressor(n_estimators=10, random_state=0)
        gbr.fit(X, y)
        pred = gbr.predict(X)
        assert pred.shape == (40,)


# =============================================================================
# Staged prediction consistency
# =============================================================================


class TestStagedPredictionConsistency:
    """
    _predict_raw called with up_to_iteration=k must equal the prediction
    that would be obtained if the model was trained with only k estimators.
    """

    def test_staged_regressor_matches_partial_model(self):
        X, y = make_regression(n_samples=80, n_features=5, random_state=0)

        gbr_full = GradientBoostingRegressor(
            n_estimators=15, subsample=1.0, random_state=0
        ).fit(X, y)

        # Staged prediction at step 7 via full model
        staged = gbr_full._predict_raw(X, up_to_iteration=7)

        # Retrain with only 7 estimators (identical hyperparams)
        gbr_partial = GradientBoostingRegressor(
            n_estimators=7, subsample=1.0, random_state=0
        ).fit(X, y)

        np.testing.assert_allclose(staged, gbr_partial.predict(X), rtol=1e-8)

    def test_staged_regressor_final_matches_predict(self):
        X, y = make_regression(n_samples=80, n_features=5, random_state=0)
        n_est = 10
        gbr = GradientBoostingRegressor(
            n_estimators=n_est, subsample=1.0, random_state=0
        ).fit(X, y)

        staged_full = gbr._predict_raw(X, up_to_iteration=n_est)
        np.testing.assert_array_equal(staged_full, gbr.predict(X))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
