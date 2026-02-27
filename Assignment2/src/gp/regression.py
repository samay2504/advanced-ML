"""
Gaussian Process Regression (Algorithm 2.1).

Implements exact GP regression with hyperparameter optimisation via 
log-marginal likelihood maximisation.
"""

import logging
import numpy as np
from typing import Optional, Tuple
from scipy.optimize import minimize

from .base import stable_cholesky, cholesky_solve, log_marginal_likelihood
from .kernels import Kernel, RBF

logger = logging.getLogger(__name__)


class GaussianProcessRegressor:
    """
    Gaussian Process Regressor implementing Algorithm 2.1.
    
    Performs exact GP inference for regression with Gaussian noise.
    Optionally optimises kernel hyperparameters via marginal likelihood.
    
    Parameters
    ----------
    kernel : Kernel, optional
        Covariance function. Defaults to RBF with unit parameters.
    noise : float, default=0.01
        Gaussian noise variance σ_n².
    optimise : bool, default=True
        Whether to optimise hyperparameters during fit.
    n_restarts : int, default=0
        Number of random restarts for optimisation.
    random_state : int, optional
        Random seed for reproducibility.
    verbose : bool, default=False
        Enable logging output.
    """
    
    def __init__(
        self,
        kernel: Optional[Kernel] = None,
        noise: float = 0.01,
        optimise: bool = True,
        n_restarts: int = 0,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        self.kernel = kernel if kernel is not None else RBF()
        self.noise = float(noise)
        self.optimise = optimise
        self.n_restarts = n_restarts
        self.random_state = random_state
        self.verbose = verbose
        
        if verbose:
            logger.setLevel(logging.INFO)
        
        # Fitted state
        self.X_train_ = None
        self.y_train_ = None
        self.L_ = None
        self.alpha_ = None
        self.log_marginal_likelihood_ = None
        self.optimisation_trace_ = []
        
        if random_state is not None:
            np.random.seed(random_state)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'GaussianProcessRegressor':
        """
        Fit Gaussian Process regression model (Algorithm 2.1).
        
        Computes:
          - K = k(X, X) + σ_n² I
          - L = cholesky(K)
          - α = L^T \\ (L \\ y)
        
        If optimise=True, maximises log marginal likelihood w.r.t. 
        kernel hyperparameters and noise variance.
        
        Parameters
        ----------
        X : np.ndarray, shape (n, d)
            Training inputs.
        y : np.ndarray, shape (n,)
            Training targets.
            
        Returns
        -------
        self : GaussianProcessRegressor
            Fitted estimator.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64).ravel()
        
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X and y have incompatible shapes: {X.shape[0]} vs {y.shape[0]}")
        
        self.X_train_ = X.copy()
        self.y_train_ = y.copy()
        
        if self.optimise:
            self._optimise_hyperparameters()
        else:
            self._compute_alpha()
        
        self.log_marginal_likelihood_ = self.log_marginal_likelihood_value()
        
        logger.info(f"GP fitted with {X.shape[0]} samples")
        logger.info(f"Log marginal likelihood: {self.log_marginal_likelihood_:.4f}")
        
        return self
    
    def _compute_alpha(self) -> None:
        """
        Compute precomputed quantities for prediction.
        
        Algorithm 2.1 steps 1-3:
          1. K = k(X, X) + σ_n² I
          2. L = cholesky(K)
          3. α = L^T \\ (L \\ y)
        """
        n = self.X_train_.shape[0]
        K = self.kernel(self.X_train_, self.X_train_)
        K += np.eye(n) * self.noise
        
        self.L_ = stable_cholesky(K, jitter=1e-6)
        self.alpha_ = cholesky_solve(self.L_, self.y_train_)
    
    def predict(
        self,
        X: np.ndarray,
        return_std: bool = False,
        return_cov: bool = False
    ) -> Tuple[np.ndarray, ...]:
        """
        Predict using GP regression (Algorithm 2.1 steps 4-5).
        
        Computes posterior mean and variance:
          - f̄_* = K_*^T α
          - v = L \\ K_*
          - V[f_*] = k(X_*, X_*) - v^T v
        
        Parameters
        ----------
        X : np.ndarray, shape (m, d)
            Test inputs.
        return_std : bool, default=False
            If True, return standard deviation along with mean.
        return_cov : bool, default=False
            If True, return full covariance matrix.
            
        Returns
        -------
        mean : np.ndarray, shape (m,)
            Posterior mean predictions.
        std : np.ndarray, shape (m,), optional
            Posterior standard deviations (if return_std=True).
        cov : np.ndarray, shape (m, m), optional
            Posterior covariance matrix (if return_cov=True).
        """
        if self.X_train_ is None:
            raise RuntimeError("Model must be fitted before prediction")
        
        X = np.asarray(X, dtype=np.float64)
        
        # Step 4: Posterior mean
        K_star = self.kernel(self.X_train_, X)  # Shape (n, m)
        mean = K_star.T.dot(self.alpha_)  # Shape (m,)
        
        if not return_std and not return_cov:
            return mean
        
        # Step 5: Posterior variance/covariance
        v = np.linalg.solve(self.L_, K_star)  # Shape (n, m)
        
        if return_cov:
            K_star_star = self.kernel(X, X)  # Shape (m, m)
            cov = K_star_star - v.T.dot(v)
            # Ensure symmetry and positive semi-definiteness
            cov = 0.5 * (cov + cov.T)
            return mean, cov
        
        if return_std:
            var = self.kernel(X, X, diag=True) - np.sum(v ** 2, axis=0)
            var = np.maximum(var, 0.0)  # Numerical stability
            std = np.sqrt(var)
            return mean, std
    
    def predict_f_cov(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict posterior mean and full covariance matrix.
        
        Parameters
        ----------
        X : np.ndarray, shape (m, d)
            Test inputs.
            
        Returns
        -------
        mean : np.ndarray, shape (m,)
            Posterior mean.
        cov : np.ndarray, shape (m, m)
            Posterior covariance matrix.
        """
        if X.shape[0] > 5000:
            logger.warning(
                f"Computing full covariance for {X.shape[0]} points. "
                "This may be memory-intensive."
            )
        
        return self.predict(X, return_cov=True)
    
    def log_marginal_likelihood_value(self) -> float:
        """
        Compute log marginal likelihood of current hyperparameters.
        
        log p(y|X,θ) = -0.5 * y^T K^{-1} y - 0.5 * log|K| - (n/2) * log(2π)
        
        Returns
        -------
        lml : float
            Log marginal likelihood.
        """
        if self.L_ is None:
            self._compute_alpha()
        
        n = self.X_train_.shape[0]
        K = self.kernel(self.X_train_, self.X_train_) + np.eye(n) * self.noise
        
        return log_marginal_likelihood(self.y_train_, K, L=self.L_)
    
    def _optimise_hyperparameters(self) -> None:
        """
        Optimise kernel hyperparameters and noise via log marginal likelihood.
        
        Uses L-BFGS-B with multiple random restarts. Hyperparameters are
        optimised in log space for unconstrained optimisation.
        """
        def objective(params: np.ndarray) -> float:
            """Negative log marginal likelihood."""
            # params = [kernel_params..., log(noise)]
            self.kernel.set_params(params[:-1])
            self.noise = np.exp(params[-1])
            
            self._compute_alpha()
            lml = self.log_marginal_likelihood_value()
            
            self.optimisation_trace_.append({
                'params': params.copy(),
                'lml': lml
            })
            
            return -lml  # Minimize negative LML
        
        # Initial parameters
        kernel_params = self.kernel.get_params()
        noise_param = np.log(self.noise)
        initial_params = np.concatenate([kernel_params, [noise_param]])
        
        # Bounds (in log space: allow values from e^-10 to e^10)
        bounds = [(-10, 10) for _ in initial_params]
        
        best_result = None
        best_lml = -np.inf
        
        # Multiple restarts
        for restart in range(self.n_restarts + 1):
            if restart == 0:
                x0 = initial_params
            else:
                # Random restart
                x0 = np.random.uniform(-2, 2, size=initial_params.shape)
            
            result = minimize(
                objective,
                x0=x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': 100}
            )
            
            lml = -result.fun
            
            if self.verbose:
                logger.info(f"Restart {restart}: LML = {lml:.4f}")
            
            if lml > best_lml:
                best_lml = lml
                best_result = result
        
        # Set best parameters
        best_params = best_result.x
        self.kernel.set_params(best_params[:-1])
        self.noise = np.exp(best_params[-1])
        self._compute_alpha()
        
        if self.verbose:
            logger.info(f"Optimisation converged to LML = {best_lml:.4f}")
            logger.info(f"Final noise: {self.noise:.6f}")
    
    def save(self, filepath: str) -> None:
        """Save fitted model to disk."""
        import pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)
        logger.info(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath: str) -> 'GaussianProcessRegressor':
        """Load fitted model from disk."""
        import pickle
        with open(filepath, 'rb') as f:
            model = pickle.load(f)
        logger.info(f"Model loaded from {filepath}")
        return model
