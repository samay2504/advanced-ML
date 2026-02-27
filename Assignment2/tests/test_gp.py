"""
Unit tests for GP regression and classification.

Tests core numerical operations and algorithm implementations.
"""

import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gp.base import stable_cholesky, cholesky_solve, log_marginal_likelihood
from src.gp.kernels import RBF, Matern, RationalQuadratic
from src.gp.regression import GaussianProcessRegressor
from src.gp.classification import GaussianProcessClassifier


class TestBaseOperations:
    """Test base numerical utilities."""
    
    def test_cholesky_solve(self):
        """Test that cholesky_solve matches direct solve."""
        np.random.seed(42)
        n = 50
        
        # Create random SPD matrix
        A = np.random.randn(n, n)
        K = A @ A.T + np.eye(n) * 0.1
        
        # Random vector
        b = np.random.randn(n)
        
        # Cholesky solve
        L = stable_cholesky(K, jitter=1e-8)
        x_chol = cholesky_solve(L, b)
        
        # Direct solve
        x_direct = np.linalg.solve(K, b)
        
        # Should match within numerical tolerance
        np.testing.assert_allclose(x_chol, x_direct, rtol=1e-6, atol=1e-8)
    
    def test_cholesky_solve_matrix(self):
        """Test cholesky_solve with matrix right-hand side."""
        np.random.seed(42)
        n = 30
        m = 5
        
        A = np.random.randn(n, n)
        K = A @ A.T + np.eye(n) * 0.1
        B = np.random.randn(n, m)
        
        L = stable_cholesky(K)
        X_chol = cholesky_solve(L, B)
        X_direct = np.linalg.solve(K, B)
        
        # Reasonable numerical tolerance for double precision
        np.testing.assert_allclose(X_chol, X_direct, rtol=5e-3, atol=1e-5)
    
    def test_stable_cholesky_with_jitter(self):
        """Test that stable_cholesky handles near-singular matrices."""
        # Create a nearly singular matrix
        n = 10
        K = np.ones((n, n)) * 0.1 + np.eye(n) * 1e-8
        
        # Should succeed with automatic jitter
        L = stable_cholesky(K, jitter=1e-6)
        
        # Verify it's lower triangular
        assert np.allclose(L, np.tril(L))
        
        # Verify approximate factorisation
        K_reconstructed = L @ L.T
        # Allow some slack due to added jitter
        assert np.allclose(K, K_reconstructed, atol=1e-4)


class TestKernels:
    """Test kernel implementations."""
    
    def test_rbf_kernel_symmetry(self):
        """Test RBF kernel is symmetric."""
        np.random.seed(42)
        kernel = RBF(length_scale=1.0, variance=2.0)
        
        X1 = np.random.randn(10, 3)
        X2 = np.random.randn(15, 3)
        
        K12 = kernel(X1, X2)
        K21 = kernel(X2, X1)
        
        np.testing.assert_allclose(K12, K21.T, rtol=1e-10)
    
    def test_rbf_kernel_diagonal(self):
        """Test RBF diagonal returns correct variance."""
        kernel = RBF(length_scale=1.0, variance=3.5)
        X = np.random.randn(20, 4)
        
        K_diag = kernel(X, X, diag=True)
        K_full_diag = np.diag(kernel(X, X))
        
        np.testing.assert_allclose(K_diag, K_full_diag, rtol=1e-10)
        np.testing.assert_allclose(K_diag, np.full(20, 3.5), rtol=1e-10)
    
    def test_kernel_params_roundtrip(self):
        """Test get/set params for kernels."""
        kernels_to_test = [
            RBF(length_scale=2.3, variance=4.5),
            Matern(nu=1.5, length_scale=1.7, variance=2.1),
            RationalQuadratic(length_scale=3.0, variance=1.5, alpha=2.0)
        ]
        
        for kernel in kernels_to_test:
            params = kernel.get_params()
            
            # Create new kernel and set params
            kernel_copy = kernel.__class__()
            kernel_copy.set_params(params)
            
            # Should produce same covariance
            X = np.random.randn(5, 2)
            K1 = kernel(X, X)
            K2 = kernel_copy(X, X)
            
            np.testing.assert_allclose(K1, K2, rtol=1e-10)


class TestGPRegression:
    """Test GP regression implementation."""
    
    def test_regression_prediction_shape(self):
        """Test prediction returns correct shapes."""
        np.random.seed(42)
        n_train, n_test, d = 20, 15, 3
        
        X_train = np.random.randn(n_train, d)
        y_train = np.random.randn(n_train)
        X_test = np.random.randn(n_test, d)
        
        gp = GaussianProcessRegressor(
            kernel=RBF(), noise=0.1, optimise=False, random_state=42
        )
        gp.fit(X_train, y_train)
        
        # Test mean only
        mean = gp.predict(X_test)
        assert mean.shape == (n_test,)
        
        # Test mean + std
        mean, std = gp.predict(X_test, return_std=True)
        assert mean.shape == (n_test,)
        assert std.shape == (n_test,)
        assert np.all(std >= 0)
        
        # Test mean + cov
        mean, cov = gp.predict(X_test, return_cov=True)
        assert mean.shape == (n_test,)
        assert cov.shape == (n_test, n_test)
        # Check symmetry
        np.testing.assert_allclose(cov, cov.T, rtol=1e-8)
    
    def test_regression_interpolation_no_noise(self):
        """Test GP interpolates exactly at training points with no noise."""
        np.random.seed(42)
        n, d = 10, 2
        
        X_train = np.random.randn(n, d)
        y_train = np.random.randn(n)
        
        gp = GaussianProcessRegressor(
            kernel=RBF(length_scale=1.0, variance=1.0),
            noise=1e-8,  # Tiny noise for numerical stability
            optimise=False,
            random_state=42
        )
        gp.fit(X_train, y_train)
        
        # Predict at training points
        y_pred = gp.predict(X_train)
        
        # Should match training values closely
        np.testing.assert_allclose(y_pred, y_train, rtol=1e-3, atol=1e-3)
    
    def test_regression_variance_nonnegative(self):
        """Test predictive variance is non-negative."""
        np.random.seed(42)
        X_train = np.random.randn(15, 3)
        y_train = np.random.randn(15)
        X_test = np.random.randn(20, 3)
        
        gp = GaussianProcessRegressor(
            kernel=RBF(), noise=0.1, optimise=False, random_state=42
        )
        gp.fit(X_train, y_train)
        
        _, std = gp.predict(X_test, return_std=True)
        
        assert np.all(std >= 0), "Standard deviation must be non-negative"
    
    def test_regression_determinism(self):
        """Test same random_state gives same results."""
        np.random.seed(42)
        X_train = np.random.randn(20, 3)
        y_train = np.random.randn(20)
        X_test = np.random.randn(10, 3)
        
        gp1 = GaussianProcessRegressor(
            kernel=RBF(), noise=0.1, optimise=True, 
            n_restarts=1, random_state=123
        )
        gp1.fit(X_train, y_train)
        pred1 = gp1.predict(X_test)
        
        gp2 = GaussianProcessRegressor(
            kernel=RBF(), noise=0.1, optimise=True,
            n_restarts=1, random_state=123
        )
        gp2.fit(X_train, y_train)
        pred2 = gp2.predict(X_test)
        
        np.testing.assert_allclose(pred1, pred2, rtol=1e-6)


class TestGPClassification:
    """Test GP classification with Laplace approximation."""
    
    def test_classification_convergence(self):
        """Test Newton's method converges on separable data."""
        np.random.seed(42)
        
        # Create linearly separable dataset
        n = 50
        X_class0 = np.random.randn(n // 2, 2) + np.array([2, 2])
        X_class1 = np.random.randn(n // 2, 2) + np.array([-2, -2])
        X = np.vstack([X_class0, X_class1])
        y = np.array([0] * (n // 2) + [1] * (n // 2))
        
        gp = GaussianProcessClassifier(
            kernel=RBF(length_scale=1.0),
            likelihood='logistic',
            max_iter=50,
            tol=1e-6,
            random_state=42
        )
        
        gp.fit(X, y)
        
        # Should converge
        assert gp.n_iter_ < 50, "Newton should converge on separable data"
        
        # W should be non-negative (second derivative property)
        assert np.all(gp.W_ >= 0), "Negative Hessian diagonal must be non-negative"
    
    def test_classification_predict_proba_bounds(self):
        """Test predict_proba returns valid probabilities."""
        np.random.seed(42)
        n_train, n_test = 40, 20
        
        X_train = np.random.randn(n_train, 3)
        y_train = (X_train[:, 0] > 0).astype(int)
        X_test = np.random.randn(n_test, 3)
        
        gp = GaussianProcessClassifier(
            kernel=RBF(), likelihood='logistic', random_state=42
        )
        gp.fit(X_train, y_train)
        
        proba = gp.predict_proba(X_test)
        
        # Check shape
        assert proba.shape == (n_test, 2)
        
        # Check bounds
        assert np.all(proba >= 0), "Probabilities must be >= 0"
        assert np.all(proba <= 1), "Probabilities must be <= 1"
        
        # Check probabilities sum to 1
        np.testing.assert_allclose(proba.sum(axis=1), np.ones(n_test), rtol=1e-6)
    
    def test_classification_both_likelihoods(self):
        """Test both logistic and probit likelihoods work."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)
        
        for likelihood in ['logistic', 'probit']:
            gp = GaussianProcessClassifier(
                kernel=RBF(),
                likelihood=likelihood,
                max_iter=50,
                random_state=42
            )
            
            gp.fit(X, y)
            
            # Should converge
            assert gp.n_iter_ < 50
            
            # Should make reasonable predictions
            X_test = np.array([[1, 1], [-1, -1]])
            proba = gp.predict_proba(X_test)
            
            # Point (1,1) should have high prob for class 1
            assert proba[0, 1] > 0.6, f"Failed for {likelihood} likelihood"
            
            # Point (-1,-1) should have high prob for class 0
            assert proba[1, 0] > 0.6, f"Failed for {likelihood} likelihood"


class TestLogMarginalLikelihood:
    """Test log marginal likelihood computation."""
    
    def test_lml_value_reasonable(self):
        """Test LML produces reasonable values."""
        np.random.seed(42)
        X = np.random.randn(30, 2)
        y = np.sin(X[:, 0]) + np.cos(X[:, 1]) + 0.1 * np.random.randn(30)
        
        gp = GaussianProcessRegressor(
            kernel=RBF(), noise=0.1, optimise=False, random_state=42
        )
        gp.fit(X, y)
        
        lml = gp.log_marginal_likelihood_value()
        
        # Should be finite
        assert np.isfinite(lml), "LML should be finite"
        
        # For this dataset, should be negative (but not too negative)
        assert lml < 0, "LML typically negative for noisy data"
        assert lml > -1000, "LML should not be extremely negative"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
