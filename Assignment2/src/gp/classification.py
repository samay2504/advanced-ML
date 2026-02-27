"""
Gaussian Process Classification with Laplace Approximation (Algorithms 3.1 & 3.2).

Implements binary GP classification using Newton's method to find the 
posterior mode and computing predictive probabilities via Gaussian approximation.
"""

import logging
import numpy as np
from typing import Optional, Tuple
from scipy.special import erf
from scipy.integrate import quad

from .base import stable_cholesky, clamp_probabilities
from .kernels import Kernel, RBF

logger = logging.getLogger(__name__)


class GaussianProcessClassifier:
    """
    Gaussian Process Classifier with Laplace Approximation.
    
    Implements Algorithms 3.1 (posterior mode via Newton) and 3.2 (predictions)
    for binary classification with non-Gaussian likelihoods.
    
    Parameters
    ----------
    kernel : Kernel, optional
        Covariance function. Defaults to RBF.
    likelihood : str, default='logistic'
        Likelihood function: 'logistic' (sigmoid) or 'probit'.
    max_iter : int, default=50
        Maximum Newton iterations for posterior mode.
    tol : float, default=1e-6
        Convergence tolerance for Newton method.
    noise : float, default=1e-6
        Small jitter for numerical stability.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable logging output.
    """
    
    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        likelihood: str = 'logistic',
        max_iter: int = 50,
        tol: float = 1e-6,
        noise: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.kernel = kernel if kernel is not None else RBF()
        if likelihood not in ['logistic', 'probit']:
            raise ValueError("likelihood must be 'logistic' or 'probit'")
        self.likelihood = likelihood
        self.max_iter = max_iter
        self.tol = tol
        self.noise = noise
        self.random_state = random_state
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.INFO)
        
        # Fitted state
        self.X_train_ = None
        self.y_train_ = None  # {-1, +1}
        self.f_hat_ = None    # Posterior mode
        self.W_ = None        # Negative Hessian at mode
        self.L_ = None        # Cholesky of B = I + W^{1/2} K W^{1/2}
        self.K_ = None        # Kernel matrix
        self.n_iter_ = None
        self.log_marginal_likelihood_ = None
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessClassifier':
        """
        Fit GP classifier via Laplace approximation (Algorithm 3.1).
        
        Uses Newton's method to find posterior mode f̂:
          1. Initialize f = 0
          2. Repeat until convergence:
             - W = -∇∇ log p(y|f)  (diagonal Hessian)
             - b = W f + ∇ log p(y|f)
             - B = I + W^{1/2} K W^{1/2}
             - L = cholesky(B)
             - a = b - W^{1/2} L^T \\ (L \\ (W^{1/2} K b))
             - f ← K a
        
        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Training inputs.
        y : np.ndarray, shape (n,)
            Training labels (0/1 or -1/+1).
            
        Returns
        -------
        self : GaussianProcessClassifier
            Fitted classifier.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        # Convert labels to {-1, +1}
        if set(np.unique(y)).issubset({0, 1}):
            y = 2 * y - 1
        elif not set(np.unique(y)).issubset({-1, 1}):
            raise ValueError("Labels must be binary (0/1 or -1/+1)")
        
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        self.K_ = self.kernel(X, X)
        
        # Algorithm 3.1: Newton iterations
        n = X.shape[0]
        f = np.zeros(n)  # Initialize at zero
        
        for iteration in range(self.max_iter):
            # Compute likelihood derivatives
            pi = self._sigmoid(f)
            grad_log_lik = self._gradient_log_likelihood(f, y)
            W = self._hessian_log_likelihood(f, y)  # Diagonal, return as vector
            
            # Newton step
            W_sqrt = np.sqrt(W)
            W_sqrt_K = W_sqrt[:, np.newaxis] * self.K_
            B = np.eye(n) + W_sqrt_K * W_sqrt[np.newaxis, :]
            
            try:
                L = stable_cholesky(B, jitter=self.noise)
            except np.linalg.LinAlgError:
                logger.warning(f"Cholesky failed at iteration {iteration}, stopping early")
                break
            
            b = W * f + grad_log_lik
            W_sqrt_K_b = W_sqrt_K.dot(b)
            
            # Solve L L^T c = W^{1/2} K b
            c = np.linalg.solve(L, W_sqrt_K_b)
            c = np.linalg.solve(L.T, c)
            
            a = b - W_sqrt * c
            f_new = self.K_.dot(a)
            
            # Check convergence
            delta = np.max(np.abs(f_new - f))
            f = f_new
            
            if delta < self.tol:
                if self.verbose:
                    logger.info(f"Newton converged in {iteration + 1} iterations (delta={delta:.2e})")
                break
        else:
            logger.warning(f"Newton did not converge in {self.max_iter} iterations")
        
        self.n_iter_ = iteration + 1
        self.f_hat_ = f
        self.W_ = W
        
        # Recompute final L for predictions
        W_sqrt = np.sqrt(W)
        W_sqrt_K = W_sqrt[:, np.newaxis] * self.K_
        B = np.eye(n) + W_sqrt_K * W_sqrt[np.newaxis, :]
        self.L_ = stable_cholesky(B, jitter=self.noise)
        
        # Approximate log marginal likelihood
        self.log_marginal_likelihood_ = self._approximate_log_marginal_likelihood()
        
        logger.info(f"GP classifier fitted with {n} samples")
        logger.info(f"Approximate log marginal likelihood: {self.log_marginal_likelihood_:.4f}")
        
        return self
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities (Algorithm 3.2).
        
        Computes averaged predictive probability:
          π̄_* = ∫ σ(f_*) N(f_* | f̄_*, V[f_*]) df_*
        
        Uses analytic solution for probit, numerical quadrature for logistic.
        
        Parameters
        ----------
        X : np.ndarray, shape (m, d)
            Test inputs.
            
        Returns
        -------
        proba : np.ndarray, shape (m, 2)
            Class probabilities [P(y=0), P(y=1)].
        """
        if self.X_train_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=np.float64)
        m = X.shape[0]
        
        # Step 3: Posterior mean of latent function
        k_star = self.kernel(self.X_train_, X)  # Shape (n, m)
        grad_log_lik = self._gradient_log_likelihood(self.f_hat_, self.y_train_)
        f_bar = k_star.T.dot(grad_log_lik)  # Shape (m,)
        
        # Step 4: Posterior variance
        W_sqrt = np.sqrt(self.W_)
        W_sqrt_k_star = W_sqrt[:, np.newaxis] * k_star  # Shape (n, m)
        v = np.linalg.solve(self.L_, W_sqrt_k_star)  # Shape (n, m)
        k_star_star = self.kernel(X, X, diag=True)  # Shape (m,)
        var_f = k_star_star - np.sum(v ** 2, axis=0)  # Shape (m,)
        var_f = np.maximum(var_f, 0.0)  # Numerical stability
        
        # Step 5: Averaged predictive probability
        if self.likelihood == 'probit':
            # Analytic solution for probit
            kappa = 1.0 / np.sqrt(1.0 + var_f)
            pi_star = self._probit_sigmoid(kappa * f_bar)
        else:
            # Numerical integration for logistic (use approximation)
            pi_star = self._logistic_averaged_probability(f_bar, var_f)
        
        pi_star = clamp_probabilities(pi_star)
        
        # Return as two-column array [P(y=0), P(y=1)]
        proba = np.column_stack([1 - pi_star, pi_star])
        return proba
    
    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """
        Predict class labels.
        
        Parameters
        ----------
        X : np.ndarray, shape (m, d)
            Test inputs.
        threshold : float, default=0.5
            Decision threshold for class 1.
            
        Returns
        -------
        labels : np.ndarray, shape (m,)
            Predicted class labels (0 or 1).
        """
        proba = self.predict_proba(X)
        return (proba[:, 1] >= threshold).astype(int)
    
    def predict_f_cov(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict latent function posterior mean and covariance.
        
        Parameters
        ----------
        X : np.ndarray, shape (m, d)
            Test inputs.
            
        Returns
        -------
        mean : np.ndarray, shape (m,)
            Posterior mean of latent function.
        cov : np.ndarray, shape (m, m)
            Posterior covariance matrix.
        """
        if self.X_train_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Posterior mean
        k_star = self.kernel(self.X_train_, X)
        grad_log_lik = self._gradient_log_likelihood(self.f_hat_, self.y_train_)
        mean = k_star.T.dot(grad_log_lik)
        
        # Posterior covariance
        W_sqrt = np.sqrt(self.W_)
        W_sqrt_k_star = W_sqrt[:, np.newaxis] * k_star
        v = np.linalg.solve(self.L_, W_sqrt_k_star)
        k_star_star = self.kernel(X, X)
        cov = k_star_star - v.T.dot(v)
        cov = 0.5 * (cov + cov.T)  # Ensure symmetry
        
        return mean, cov
    
    def _sigmoid(self, f: np.ndarray) -> np.ndarray:
        """Logistic sigmoid σ(f) = 1 / (1 + exp(-f))."""
        return 1.0 / (1.0 + np.exp(-np.clip(f, -500, 500)))
    
    def _probit_sigmoid(self, f: np.ndarray) -> np.ndarray:
        """Probit link function Φ(f) = (1 + erf(f/√2)) / 2."""
        return 0.5 * (1.0 + erf(f / np.sqrt(2.0)))
    
    def _gradient_log_likelihood(self, f: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute ∇_f log p(y|f).
        
        For logistic: ∇ log p(y|f) = (y+1)/2 - σ(f) = t - π
        For probit: ∇ log p(y|f) = y * N(yf|0,1) / Φ(yf)
        """
        if self.likelihood == 'logistic':
            pi = self._sigmoid(f)
            t = (y + 1.0) / 2.0  # Convert {-1,+1} to {0,1}
            return t - pi
        else:  # probit
            yf = y * f
            # N(yf|0,1) / Φ(yf)
            norm_pdf = np.exp(-0.5 * yf ** 2) / np.sqrt(2 * np.pi)
            norm_cdf = self._probit_sigmoid(yf)
            norm_cdf = np.clip(norm_cdf, 1e-10, 1 - 1e-10)
            return y * norm_pdf / norm_cdf
    
    def _hessian_log_likelihood(self, f: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Compute diagonal of -∇∇_f log p(y|f) = W.
        
        For logistic: W = π(1-π)
        For probit: More complex expression
        
        Returns
        -------
        W : np.ndarray, shape (n,)
            Diagonal of negative Hessian (must be non-negative).
        """
        if self.likelihood == 'logistic':
            pi = self._sigmoid(f)
            W = pi * (1.0 - pi)
        else:  # probit
            yf = y * f
            norm_pdf = np.exp(-0.5 * yf ** 2) / np.sqrt(2 * np.pi)
            norm_cdf = self._probit_sigmoid(yf)
            norm_cdf = np.clip(norm_cdf, 1e-10, 1 - 1e-10)
            
            z = norm_pdf / norm_cdf
            W = z * (z + yf)
        
        return np.maximum(W, 1e-10)  # Ensure positivity
    
    def _logistic_averaged_probability(self, f_bar: np.ndarray, var_f: np.ndarray) -> np.ndarray:
        """
        Approximate ∫ σ(f) N(f|f̄,V) df for logistic likelihood.
        
        Uses probit approximation: σ(f) ≈ Φ(κf) where κ² = π/8.
        Then integral becomes Φ(κf̄ / √(1 + κ²V)).
        """
        kappa_sq = np.pi / 8.0
        scaled_mean = f_bar / np.sqrt(1.0 + kappa_sq * var_f)
        return self._probit_sigmoid(scaled_mean)
    
    def _approximate_log_marginal_likelihood(self) -> float:
        """
        Compute approximate log marginal likelihood at mode.
        
        log q(y|X,θ) ≈ log p(y|f̂) - 0.5 * f̂^T K^{-1} f̂ - 0.5 * log|B|
        """
        # Log likelihood at mode
        if self.likelihood == 'logistic':
            log_lik = np.sum(np.log(self._sigmoid(self.y_train_ * self.f_hat_)))
        else:
            log_lik = np.sum(np.log(self._probit_sigmoid(self.y_train_ * self.f_hat_)))
        
        # Prior term: -0.5 * f̂^T K^{-1} f̂
        L_K = stable_cholesky(self.K_, jitter=self.noise)
        alpha = np.linalg.solve(L_K, self.f_hat_)
        alpha = np.linalg.solve(L_K.T, alpha)
        prior_term = -0.5 * np.dot(self.f_hat_, alpha)
        
        # Complexity term: -0.5 * log|B| = -sum(log(diag(L)))
        log_det_B = -np.sum(np.log(np.diag(self.L_)))
        
        return log_lik + prior_term + log_det_B
