"""
Base utilities for Gaussian Process computations.

Provides numerically stable operations for GP inference including
Cholesky decomposition, solving linear systems, and data preprocessing.
"""

import logging
import numpy as np
from typing import Tuple, Optional

logger = logging.getLogger(__name__)


def stable_cholesky(K: np.ndarray, jitter: float = 1e-6) -> np.ndarray:
    """
    Compute stable Cholesky decomposition with automatic jitter adaptation.
    
    Returns lower-triangular L such that K ≈ LL^T. If the decomposition fails,
    progressively increases jitter (diagonal regularisation) until successful.
    
    Parameters
    ----------
    K : np.ndarray, shape (n, n)
        Symmetric positive (semi-)definite matrix.
    jitter : float, default=1e-6
        Initial diagonal regularisation term.
        
    Returns
    -------
    L : np.ndarray, shape (n, n)
        Lower-triangular Cholesky factor.
        
    Raises
    ------
    np.linalg.LinAlgError
        If decomposition fails even with maximum jitter.
    """
    n = K.shape[0]
    jitter_levels = [jitter, 1e-5, 1e-4, 1e-3, 1e-2]
    
    for j in jitter_levels:
        try:
            L = np.linalg.cholesky(K + np.eye(n) * j)
            if j > jitter:
                logger.warning(f"Cholesky required jitter={j:.2e} for stability")
            return L
        except np.linalg.LinAlgError:
            continue
    
    raise np.linalg.LinAlgError(
        f"Cholesky decomposition failed even with jitter={jitter_levels[-1]:.2e}. "
        "Matrix may be ill-conditioned."
    )


def cholesky_solve(L: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Solve K x = b given Cholesky factor L where K = LL^T.
    
    Efficient two-step triangular solve:
      1. Solve L y = b  (forward substitution)
      2. Solve L^T x = y  (backward substitution)
    
    Parameters
    ----------
    L : np.ndarray, shape (n, n)
        Lower-triangular Cholesky factor.
    b : np.ndarray, shape (n,) or (n, m)
        Right-hand side vector(s).
        
    Returns
    -------
    x : np.ndarray, shape (n,) or (n, m)
        Solution to K x = b.
    """
    y = np.linalg.solve(L, b)
    x = np.linalg.solve(L.T, y)
    return x


def log_marginal_likelihood(
    y: np.ndarray,
    K: np.ndarray,
    L: Optional[np.ndarray] = None,
    jitter: float = 1e-6
) -> float:
    """
    Compute log marginal likelihood for GP regression.
    
    log p(y|X,θ) = -0.5 * y^T K^{-1} y - 0.5 * log|K| - (n/2) * log(2π)
    
    Parameters
    ----------
    y : np.ndarray, shape (n,)
        Training targets.
    K : np.ndarray, shape (n, n)
        Covariance matrix (including noise).
    L : np.ndarray, optional
        Pre-computed Cholesky factor. If None, computed from K.
    jitter : float
        Regularisation for Cholesky if L not provided.
        
    Returns
    -------
    lml : float
        Log marginal likelihood.
    """
    n = len(y)
    
    if L is None:
        L = stable_cholesky(K, jitter=jitter)
    
    # Data fit term: -0.5 * y^T K^{-1} y
    alpha = cholesky_solve(L, y)
    data_fit = -0.5 * np.dot(y, alpha)
    
    # Complexity penalty: -0.5 * log|K| = -sum(log(diag(L)))
    log_det = -np.sum(np.log(np.diag(L)))
    
    # Normalisation constant
    const = -0.5 * n * np.log(2 * np.pi)
    
    return data_fit + log_det + const


def standardise(
    X: np.ndarray,
    mean: Optional[np.ndarray] = None,
    std: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Standardise features to zero mean and unit variance.
    
    Parameters
    ----------
    X : np.ndarray, shape (n, d)
        Input features.
    mean : np.ndarray, optional
        Pre-computed mean for transform.
    std : np.ndarray, optional
        Pre-computed standard deviation for transform.
        
    Returns
    -------
    X_scaled : np.ndarray, shape (n, d)
        Standardised features.
    mean : np.ndarray, shape (d,)
        Feature means.
    std : np.ndarray, shape (d,)
        Feature standard deviations.
    """
    if mean is None:
        mean = np.mean(X, axis=0)
    if std is None:
        std = np.std(X, axis=0)
        std[std < 1e-10] = 1.0  # Avoid division by zero
    
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def clamp_probabilities(probs: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    """
    Clamp probabilities to [eps, 1-eps] for numerical stability.
    
    Parameters
    ----------
    probs : np.ndarray
        Probability values.
    eps : float
        Small positive constant.
        
    Returns
    -------
    clamped : np.ndarray
        Clamped probabilities.
    """
    return np.clip(probs, eps, 1.0 - eps)
