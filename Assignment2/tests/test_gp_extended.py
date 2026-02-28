"""
Extended tests for Gaussian Process regression and classification.

Author : Atharva Date
Roll No: B22AI045
Course : Advanced Machine Learning, IIT Jodhpur

Coverage:
- Kernel positive semi-definiteness (all three kernels)
- RBF kernel distance-monotonicity and self-covariance
- Matern-1.5 analytic formula verification
- Matern-2.5 analytic formula verification
- RationalQuadratic analytic formula verification
- Kernel diagonal API consistency
- Kernel parameter round-trip in get/set_params (log-space)
- GP regression: predict-before-fit raises RuntimeError
- GP regression: shape-mismatch raises ValueError
- GP regression: posterior variance strictly smaller at training points
- GP regression: uncertainty grows monotonically away from single training point
- GP regression: log marginal likelihood improves after hyperparameter optimisation
- GP regression with Matern and RationalQuadratic kernels
- GP regression: predict_f_cov shape and covariance symmetry
- GP classification: invalid likelihood raises ValueError
- GP classification: invalid labels raises ValueError
- GP classification: predict returns 0/1 labels
- GP classification: predict_f_cov shape and covariance symmetry
- GP classification: posterior mode consistent with ground truth on easy data
- GP classification: probit and logistic likelihoods produce valid, distinct predictions
- Log marginal likelihood: worse hyperparameters give strictly lower LML
"""

import sys
import os

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.gp.base import log_marginal_likelihood, stable_cholesky
from src.gp.kernels import Matern, RBF, RationalQuadratic
from src.gp.regression import GaussianProcessRegressor
from src.gp.classification import GaussianProcessClassifier


# =============================================================================
# Helpers
# =============================================================================

def _is_psd(K: np.ndarray, tol: float = 1e-6) -> bool:
    """Return True if K is symmetric positive semi-definite."""
    sym = np.allclose(K, K.T, atol=tol)
    eigs = np.linalg.eigvalsh(K)
    psd = bool(np.all(eigs >= -tol))
    return sym and psd


# =============================================================================
# Kernel: Positive Semi-Definiteness
# =============================================================================


class TestKernelPSD:
    """All kernels must produce PSD Gram matrices on arbitrary inputs."""

    @pytest.fixture(params=["rbf", "matern15", "matern25", "rq"])
    def kernel(self, request):
        return {
            "rbf": RBF(length_scale=1.5, variance=2.0),
            "matern15": Matern(nu=1.5, length_scale=1.0, variance=1.5),
            "matern25": Matern(nu=2.5, length_scale=0.8, variance=2.5),
            "rq": RationalQuadratic(length_scale=1.2, variance=1.0, alpha=0.5),
        }[request.param]

    def test_gram_matrix_is_psd(self, kernel):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((25, 3))
        K = kernel(X, X)
        assert _is_psd(K), f"Gram matrix not PSD for {kernel.__class__.__name__}"

    def test_gram_matrix_shape(self, kernel):
        rng = np.random.default_rng(1)
        n, m, d = 12, 17, 4
        X1 = rng.standard_normal((n, d))
        X2 = rng.standard_normal((m, d))
        K = kernel(X1, X2)
        assert K.shape == (n, m)

    def test_self_covariance_matches_variance(self, kernel):
        """k(x, x) must equal the signal variance for stationary kernels."""
        X = np.zeros((1, 3))
        K = kernel(X, X)
        assert K[0, 0] == pytest.approx(kernel.variance, rel=1e-8)


# =============================================================================
# Kernel: RBF analytic properties
# =============================================================================


class TestRBFKernel:
    """Verify RBF formula, monotonicity, and diagonal correctness."""

    def test_rbf_formula_known_value(self):
        """k(x, x') = σ² exp(−‖x−x'‖²/(2ℓ²))"""
        k = RBF(length_scale=2.0, variance=3.0)
        x1 = np.array([[1.0, 0.0]])
        x2 = np.array([[0.0, 0.0]])
        dist_sq = 1.0
        expected = 3.0 * np.exp(-0.5 * dist_sq / 4.0)
        K = k(x1, x2)
        assert K[0, 0] == pytest.approx(expected, rel=1e-10)

    def test_rbf_monotonically_decreasing_with_distance(self):
        """Covariance must decrease as points move further apart (1-D)."""
        k = RBF(length_scale=1.0, variance=1.0)
        anchor = np.array([[0.0]])
        distances = [0.0, 0.5, 1.0, 2.0, 5.0]
        cov_vals = [k(anchor, np.array([[d]]))[0, 0] for d in distances]
        for i in range(len(cov_vals) - 1):
            assert cov_vals[i] >= cov_vals[i + 1], (
                f"RBF covariance not monotone at distance {distances[i+1]}"
            )

    def test_rbf_diagonal_equals_variance_constant(self):
        """diag=True must return σ² for every point."""
        sigma2 = 4.7
        k = RBF(length_scale=1.0, variance=sigma2)
        X = np.random.default_rng(5).standard_normal((30, 4))
        diag = k(X, X, diag=True)
        np.testing.assert_allclose(diag, sigma2, rtol=1e-10)

    def test_rbf_get_set_params_round_trip(self):
        """Params encoded in log-space must survive get → set round-trip."""
        k = RBF(length_scale=3.14, variance=2.71)
        params = k.get_params()
        k2 = RBF()
        k2.set_params(params)
        assert k2.length_scale == pytest.approx(k.length_scale, rel=1e-10)
        assert k2.variance == pytest.approx(k.variance, rel=1e-10)


# =============================================================================
# Kernel: Matern analytic formula
# =============================================================================


class TestMaternKernel:
    """Verify Matern-3/2 and Matern-5/2 formula values."""

    def test_matern15_formula(self):
        """k_{3/2}(r) = σ²(1 + √3 r/ℓ) exp(−√3 r/ℓ)"""
        ell, sigma2 = 2.0, 3.0
        k = Matern(nu=1.5, length_scale=ell, variance=sigma2)
        x1 = np.array([[0.0]])
        x2 = np.array([[1.0]])
        r = 1.0
        scaled = np.sqrt(3.0) * r / ell
        expected = sigma2 * (1.0 + scaled) * np.exp(-scaled)
        K = k(x1, x2)
        assert K[0, 0] == pytest.approx(expected, rel=1e-8)

    def test_matern25_formula(self):
        """k_{5/2}(r) = σ²(1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(−√5 r/ℓ)"""
        ell, sigma2 = 1.5, 2.0
        k = Matern(nu=2.5, length_scale=ell, variance=sigma2)
        x1 = np.array([[0.0]])
        x2 = np.array([[1.5]])
        r = 1.5
        scaled = np.sqrt(5.0) * r / ell
        expected = sigma2 * (1.0 + scaled + scaled ** 2 / 3.0) * np.exp(-scaled)
        K = k(x1, x2)
        assert K[0, 0] == pytest.approx(expected, rel=1e-8)

    def test_matern_nu_1_5_and_2_5_differ(self):
        """ν=1.5 and ν=2.5 must produce different covariance matrices."""
        rng = np.random.default_rng(0)
        X = rng.standard_normal((10, 2))
        K15 = Matern(nu=1.5)(X, X)
        K25 = Matern(nu=2.5)(X, X)
        assert not np.allclose(K15, K25), "ν=1.5 and ν=2.5 should differ"

    def test_matern_invalid_nu_raises(self):
        """Unsupported ν must raise ValueError at construction."""
        with pytest.raises(ValueError):
            Matern(nu=3.5)

    def test_matern_self_covariance_equals_variance(self):
        for nu in [1.5, 2.5]:
            k = Matern(nu=nu, length_scale=1.0, variance=5.0)
            X = np.zeros((1, 2))
            assert k(X, X)[0, 0] == pytest.approx(5.0, rel=1e-8)


# =============================================================================
# Kernel: RationalQuadratic analytic formula
# =============================================================================


class TestRationalQuadraticKernel:
    """Verify RQ formula and behaviour."""

    def test_rq_formula_known_value(self):
        """k(r) = σ²(1 + r²/(2αℓ²))^{-α}"""
        ell, sigma2, alpha = 2.0, 1.5, 3.0
        k = RationalQuadratic(length_scale=ell, variance=sigma2, alpha=alpha)
        x1 = np.array([[0.0]])
        x2 = np.array([[2.0]])
        r2 = 4.0
        expected = sigma2 * (1.0 + r2 / (2.0 * alpha * ell ** 2)) ** (-alpha)
        K = k(x1, x2)
        assert K[0, 0] == pytest.approx(expected, rel=1e-8)

    def test_rq_approaches_rbf_for_large_alpha(self):
        """As α→∞, RQ kernel approaches RBF (same ℓ, σ²)."""
        ell, sigma2 = 1.0, 1.0
        X1 = np.array([[0.0]])
        X2 = np.array([[0.5]])

        K_rq = RationalQuadratic(length_scale=ell, variance=sigma2, alpha=1e6)(X1, X2)
        K_rbf = RBF(length_scale=ell, variance=sigma2)(X1, X2)
        assert abs(K_rq[0, 0] - K_rbf[0, 0]) < 1e-4

    def test_rq_get_set_params_round_trip(self):
        k = RationalQuadratic(length_scale=2.5, variance=1.3, alpha=4.0)
        params = k.get_params()
        k2 = RationalQuadratic()
        k2.set_params(params)
        assert k2.length_scale == pytest.approx(k.length_scale, rel=1e-10)
        assert k2.variance == pytest.approx(k.variance, rel=1e-10)
        assert k2.alpha == pytest.approx(k.alpha, rel=1e-10)


# =============================================================================
# GP Regression: API contracts
# =============================================================================


class TestGPRegressionAPIContracts:
    """Error handling and pre/post-fit state contracts."""

    def test_predict_before_fit_raises_runtime_error(self):
        gp = GaussianProcessRegressor(kernel=RBF(), optimise=False)
        X_test = np.random.default_rng(0).standard_normal((5, 2))
        with pytest.raises(RuntimeError, match="fitted"):
            gp.predict(X_test)

    def test_fit_shape_mismatch_raises_value_error(self):
        gp = GaussianProcessRegressor(kernel=RBF(), optimise=False)
        X = np.random.default_rng(0).standard_normal((10, 3))
        y = np.random.default_rng(0).standard_normal(9)  # wrong size
        with pytest.raises(ValueError):
            gp.fit(X, y)

    def test_predict_std_nonnegative(self):
        rng = np.random.default_rng(7)
        X_tr = rng.standard_normal((20, 2))
        y_tr = np.sin(X_tr[:, 0])
        X_te = rng.standard_normal((30, 2))

        gp = GaussianProcessRegressor(kernel=RBF(), noise=0.01, optimise=False)
        gp.fit(X_tr, y_tr)
        _, std = gp.predict(X_te, return_std=True)
        assert np.all(std >= 0)

    def test_fitted_state_populated_after_fit(self):
        rng = np.random.default_rng(8)
        X = rng.standard_normal((15, 2))
        y = rng.standard_normal(15)

        gp = GaussianProcessRegressor(kernel=RBF(), optimise=False)
        gp.fit(X, y)

        assert gp.X_train_ is not None
        assert gp.alpha_ is not None
        assert gp.L_ is not None
        assert np.isfinite(gp.log_marginal_likelihood_)


# =============================================================================
# GP Regression: posterior uncertainty properties
# =============================================================================


class TestGPRegressionPosteriorUncertainty:
    """The posterior must reflect reduced uncertainty near observed data."""

    def test_posterior_variance_lower_at_training_points(self):
        """Var[f(x)] at training points must be less than prior variance σ²."""
        rng = np.random.default_rng(20)
        sigma2 = 2.0
        X_tr = rng.standard_normal((15, 2))
        y_tr = rng.standard_normal(15)

        gp = GaussianProcessRegressor(
            kernel=RBF(variance=sigma2), noise=1e-4, optimise=False
        )
        gp.fit(X_tr, y_tr)

        _, std_at_train = gp.predict(X_tr, return_std=True)
        var_at_train = std_at_train ** 2

        # Posterior variance at training points must be well below prior variance
        assert np.all(var_at_train < sigma2), (
            "Posterior variance at training points must be less than prior variance"
        )

    def test_uncertainty_grows_away_from_training_data_1d(self):
        """In 1-D, uncertainty must increase as we move away from training data."""
        # Training data centred at zero
        X_tr = np.linspace(-1, 1, 10).reshape(-1, 1)
        y_tr = np.sin(X_tr[:, 0])

        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=0.5, variance=1.0),
            noise=1e-5,
            optimise=False
        )
        gp.fit(X_tr, y_tr)

        # Test points: nearby vs far
        X_near = np.array([[0.0]])   # within training range
        X_far = np.array([[10.0]])   # far outside

        _, std_near = gp.predict(X_near, return_std=True)
        _, std_far = gp.predict(X_far, return_std=True)

        assert std_far[0] > std_near[0], (
            f"Uncertainty at x=10 ({std_far[0]:.4f}) should exceed "
            f"uncertainty at x=0 ({std_near[0]:.4f})"
        )

    def test_posterior_covariance_psd(self):
        """Full posterior covariance from predict_f_cov must be PSD."""
        rng = np.random.default_rng(21)
        X_tr = rng.standard_normal((10, 2))
        y_tr = rng.standard_normal(10)
        X_te = rng.standard_normal((8, 2))

        gp = GaussianProcessRegressor(kernel=RBF(), noise=0.1, optimise=False)
        gp.fit(X_tr, y_tr)

        mean, cov = gp.predict_f_cov(X_te)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        assert _is_psd(cov, tol=1e-5), "Posterior covariance must be PSD"


# =============================================================================
# GP Regression: hyperparameter optimisation
# =============================================================================


class TestGPRegressionOptimisation:
    """Log marginal likelihood must not decrease after optimisation."""

    def test_optimised_lml_geq_initial_lml(self):
        """LML after optimisation must be ≥ LML with default hyperparameters."""
        rng = np.random.default_rng(30)
        n = 40
        X = rng.standard_normal((n, 2))
        y = np.sin(X[:, 0] * 2) + 0.1 * rng.standard_normal(n)

        # Unoptimised model
        gp_no_opt = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, variance=1.0),
            noise=0.01,
            optimise=False,
        )
        gp_no_opt.fit(X, y)
        lml_before = gp_no_opt.log_marginal_likelihood_

        # Optimised model (same starting kernel)
        gp_opt = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, variance=1.0),
            noise=0.01,
            optimise=True,
            n_restarts=0,
            random_state=0,
        )
        gp_opt.fit(X, y)
        lml_after = gp_opt.log_marginal_likelihood_

        assert lml_after >= lml_before - 1e-4, (
            f"Optimisation worsened LML: {lml_before:.4f} → {lml_after:.4f}"
        )

    def test_lml_worse_with_bad_hyperparameters(self):
        """Deliberately bad hyperparameters must yield lower LML."""
        rng = np.random.default_rng(31)
        n = 30
        X = rng.standard_normal((n, 2))
        y = np.cos(X[:, 0]) + 0.1 * rng.standard_normal(n)

        # Good hyperparameters (length_scale tailored to the data scale)
        gp_good = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, variance=1.0),
            noise=0.1, optimise=False
        )
        gp_good.fit(X, y)
        lml_good = gp_good.log_marginal_likelihood_

        # Bad hyperparameters (length_scale = 1e-4 → near-independent samples)
        gp_bad = GaussianProcessRegressor(
            kernel=RBF(length_scale=1e-4, variance=1.0),
            noise=0.1, optimise=False
        )
        gp_bad.fit(X, y)
        lml_bad = gp_bad.log_marginal_likelihood_

        assert lml_good > lml_bad, (
            f"Good hyperparameters ({lml_good:.2f}) should give higher LML "
            f"than bad ones ({lml_bad:.2f})"
        )


# =============================================================================
# GP Regression: multiple kernels
# =============================================================================


class TestGPRegressionMultipleKernels:
    """GP regression must work correctly with all implemented kernels."""

    @pytest.mark.parametrize("kernel", [
        RBF(length_scale=1.0, variance=1.0),
        Matern(nu=1.5, length_scale=1.0, variance=1.0),
        Matern(nu=2.5, length_scale=1.0, variance=1.0),
        RationalQuadratic(length_scale=1.0, variance=1.0, alpha=1.0),
    ])
    def test_prediction_shape_and_validity(self, kernel):
        rng = np.random.default_rng(40)
        X_tr = rng.standard_normal((20, 3))
        y_tr = rng.standard_normal(20)
        X_te = rng.standard_normal((12, 3))

        gp = GaussianProcessRegressor(kernel=kernel, noise=0.1, optimise=False)
        gp.fit(X_tr, y_tr)
        mean, std = gp.predict(X_te, return_std=True)

        assert mean.shape == (12,)
        assert std.shape == (12,)
        assert np.all(np.isfinite(mean))
        assert np.all(std >= 0)


# =============================================================================
# GP Regression: log marginal likelihood function
# =============================================================================


class TestLogMarginalLikelihoodFunction:
    """Verify the standalone log_marginal_likelihood utility."""

    def test_lml_is_finite_scalar(self):
        rng = np.random.default_rng(50)
        n = 20
        A = rng.standard_normal((n, n))
        K = A @ A.T + np.eye(n) * 0.5
        y = rng.standard_normal(n)

        lml = log_marginal_likelihood(y, K)
        assert np.isscalar(lml) or lml.ndim == 0
        assert np.isfinite(lml)

    def test_lml_uses_precomputed_cholesky(self):
        """Passing L should give the same result as not passing L."""
        rng = np.random.default_rng(51)
        n = 15
        A = rng.standard_normal((n, n))
        K = A @ A.T + np.eye(n) * 0.3
        y = rng.standard_normal(n)

        L = stable_cholesky(K)
        lml_with_L = log_marginal_likelihood(y, K, L=L)
        lml_without_L = log_marginal_likelihood(y, K)

        assert lml_with_L == pytest.approx(lml_without_L, rel=1e-8)


# =============================================================================
# GP Classification: API contracts
# =============================================================================


class TestGPClassificationAPIContracts:
    """Error handling for GPC."""

    def test_invalid_likelihood_raises_value_error(self):
        with pytest.raises(ValueError, match="likelihood"):
            GaussianProcessClassifier(likelihood="exponential")

    def test_invalid_labels_raises_value_error(self):
        gp = GaussianProcessClassifier()
        X = np.random.default_rng(0).standard_normal((10, 2))
        y = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0])  # multiclass → invalid
        with pytest.raises(ValueError):
            gp.fit(X, y)

    def test_predict_before_fit_raises_runtime_error(self):
        gp = GaussianProcessClassifier()
        X_test = np.random.default_rng(0).standard_normal((5, 2))
        with pytest.raises(RuntimeError, match="fitted"):
            gp.predict_proba(X_test)


# =============================================================================
# GP Classification: predictions
# =============================================================================


class TestGPClassificationPredictions:
    """Prediction output contracts for GPC."""

    @pytest.fixture(scope="class")
    def fitted_gpc(self):
        rng = np.random.default_rng(60)
        n = 60
        X_cls0 = rng.standard_normal((n // 2, 2)) + np.array([2.0, 2.0])
        X_cls1 = rng.standard_normal((n // 2, 2)) + np.array([-2.0, -2.0])
        X = np.vstack([X_cls0, X_cls1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))

        gp = GaussianProcessClassifier(
            kernel=RBF(length_scale=1.0),
            likelihood="logistic",
            max_iter=50,
            random_state=0,
        )
        gp.fit(X, y)
        return gp, X, y

    def test_predict_returns_binary_labels(self, fitted_gpc):
        gp, X, _ = fitted_gpc
        pred = gp.predict(X)
        assert pred.shape == (X.shape[0],)
        assert set(np.unique(pred)).issubset({0, 1})

    def test_predict_consistent_with_proba(self, fitted_gpc):
        gp, X, _ = fitted_gpc
        proba = gp.predict_proba(X)
        pred_from_proba = (proba[:, 1] >= 0.5).astype(int)
        pred = gp.predict(X)
        np.testing.assert_array_equal(pred, pred_from_proba)

    def test_predict_high_accuracy_on_separable_data(self, fitted_gpc):
        gp, X, y = fitted_gpc
        from sklearn.metrics import accuracy_score
        acc = accuracy_score(y, gp.predict(X))
        assert acc >= 0.90, f"Expected ≥90% accuracy on separable data, got {acc:.2f}"

    def test_proba_sums_to_one(self, fitted_gpc):
        gp, X, _ = fitted_gpc
        proba = gp.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(X.shape[0]), rtol=1e-6)

    def test_predict_f_cov_shape_and_symmetry(self, fitted_gpc):
        gp, X, _ = fitted_gpc
        X_test = X[:8]
        mean, cov = gp.predict_f_cov(X_test)
        assert mean.shape == (8,)
        assert cov.shape == (8, 8)
        np.testing.assert_allclose(cov, cov.T, atol=1e-8)


# =============================================================================
# GP Classification: logistic vs probit likelihoods
# =============================================================================


class TestGPClassificationLikelihoods:
    """Logistic and probit likelihoods must both produce valid, distinct outputs."""

    @pytest.fixture(scope="class")
    def easy_dataset(self):
        rng = np.random.default_rng(70)
        n = 40
        X = np.vstack([
            rng.standard_normal((n // 2, 2)) + np.array([1.5, 0.0]),
            rng.standard_normal((n // 2, 2)) - np.array([1.5, 0.0]),
        ])
        y = np.array([1] * (n // 2) + [0] * (n // 2))
        return X, y

    @pytest.mark.parametrize("likelihood", ["logistic", "probit"])
    def test_proba_bounds(self, easy_dataset, likelihood):
        X, y = easy_dataset
        gp = GaussianProcessClassifier(
            kernel=RBF(), likelihood=likelihood, max_iter=50, random_state=0
        )
        gp.fit(X, y)
        proba = gp.predict_proba(X)
        assert np.all(proba >= 0)
        assert np.all(proba <= 1)

    def test_logistic_and_probit_differ(self, easy_dataset):
        """The two likelihoods must produce distinguishable predictions."""
        X, y = easy_dataset
        X_test = np.array([[2.0, 0.0], [-2.0, 0.0]])

        gp_log = GaussianProcessClassifier(
            kernel=RBF(), likelihood="logistic", max_iter=50, random_state=0
        ).fit(X, y)

        gp_pro = GaussianProcessClassifier(
            kernel=RBF(), likelihood="probit", max_iter=50, random_state=0
        ).fit(X, y)

        proba_log = gp_log.predict_proba(X_test)
        proba_pro = gp_pro.predict_proba(X_test)

        # They should not be identical (different likelihoods)
        assert not np.allclose(proba_log, proba_pro, atol=1e-4), (
            "Logistic and probit likelihoods should produce different probabilities"
        )

    def test_newton_converges_both_likelihoods(self, easy_dataset):
        X, y = easy_dataset
        for likelihood in ["logistic", "probit"]:
            gp = GaussianProcessClassifier(
                kernel=RBF(), likelihood=likelihood, max_iter=50, tol=1e-6,
                random_state=0
            )
            gp.fit(X, y)
            assert gp.n_iter_ < 50, (
                f"Newton did not converge in 50 iters for {likelihood}"
            )
            assert np.all(gp.W_ >= 0), "Negative Hessian diagonal must be non-negative"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
