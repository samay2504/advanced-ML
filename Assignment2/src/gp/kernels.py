"""
Kernel functions for Gaussian Processes.

Implements common covariance functions with hyperparameter gradients
for efficient marginal likelihood optimisation.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)


class Kernel(ABC):
    """Abstract base class for GP kernels."""
    
    @abstractmethod
    def __call__(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        diag: bool = False
    ) -> np.ndarray:
        """
        Compute covariance matrix K(X1, X2).
        
        Parameters
        ----------
        X1 : np.ndarray, shape (n, d)
            First set of inputs.
        X2 : np.ndarray, shape (m, d), optional
            Second set of inputs. If None, uses X1.
        diag : bool
            If True, return only diagonal of covariance matrix.
            
        Returns
        -------
        K : np.ndarray
            Covariance matrix, shape (n, m) or (n,) if diag=True.
        """
        pass
    
    @abstractmethod
    def get_params(self) -> np.ndarray:
        """Return hyperparameters as 1D array (in log space for optimization)."""
        pass
    
    @abstractmethod
    def set_params(self, params: np.ndarray) -> None:
        """Set hyperparameters from 1D array (in log space)."""
        pass
    
    @abstractmethod
    def gradient(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """
        Compute gradients of covariance matrix w.r.t. hyperparameters.
        
        Returns
        -------
        grads : dict
            Dictionary mapping parameter names to gradient matrices.
        """
        pass


class RBF(Kernel):
    """
    Radial Basis Function (Squared Exponential) kernel.
    
    k(x, x') = σ² exp(-||x - x'||² / (2 ℓ²))
    
    Parameters
    ----------
    length_scale : float, default=1.0
        Characteristic length scale ℓ.
    variance : float, default=1.0
        Signal variance σ².
    """
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = float(length_scale)
        self.variance = float(variance)
    
    def __call__(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        diag: bool = False
    ) -> np.ndarray:
        if X2 is None:
            X2 = X1
        
        if diag:
            return np.full(X1.shape[0], self.variance)
        
        # Compute squared Euclidean distances
        dists_sq = self._squared_distances(X1, X2)
        K = self.variance * np.exp(-0.5 * dists_sq / (self.length_scale ** 2))
        return K
    
    def _squared_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute matrix of squared Euclidean distances."""
        # ||x - x'||² = ||x||² - 2⟨x, x'⟩ + ||x'||²
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dists_sq = X1_sq + X2_sq.T - 2 * np.dot(X1, X2.T)
        return np.maximum(dists_sq, 0.0)  # Numerical stability
    
    def get_params(self) -> np.ndarray:
        """Return [log(length_scale), log(variance)]."""
        return np.array([np.log(self.length_scale), np.log(self.variance)])
    
    def set_params(self, params: np.ndarray) -> None:
        """Set from [log(length_scale), log(variance)]."""
        self.length_scale = np.exp(params[0])
        self.variance = np.exp(params[1])
    
    def gradient(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Compute gradients w.r.t. length_scale and variance."""
        if X2 is None:
            X2 = X1
        
        dists_sq = self._squared_distances(X1, X2)
        K = self(X1, X2)
        
        # ∂K/∂ℓ = K * dists_sq / ℓ³  (chain rule with log parameterisation)
        dK_dlength = K * dists_sq / (self.length_scale ** 3)
        
        # ∂K/∂σ² = K / σ²  (chain rule with log parameterisation)
        dK_dvariance = K / self.variance
        
        return {
            'length_scale': dK_dlength,
            'variance': dK_dvariance
        }


class Matern(Kernel):
    """
    Matérn kernel with fixed smoothness parameter ν.
    
    For ν = 3/2:
        k(r) = σ² (1 + √3 r/ℓ) exp(-√3 r/ℓ)
    
    For ν = 5/2:
        k(r) = σ² (1 + √5 r/ℓ + 5r²/(3ℓ²)) exp(-√5 r/ℓ)
    
    where r = ||x - x'||.
    
    Parameters
    ----------
    nu : float, default=1.5
        Smoothness parameter, either 1.5 (ν=3/2) or 2.5 (ν=5/2).
    length_scale : float, default=1.0
        Characteristic length scale ℓ.
    variance : float, default=1.0
        Signal variance σ².
    """
    
    def __init__(self, nu: float = 1.5, length_scale: float = 1.0, variance: float = 1.0):
        if nu not in [1.5, 2.5]:
            raise ValueError("Only ν ∈ {1.5, 2.5} supported")
        self.nu = nu
        self.length_scale = float(length_scale)
        self.variance = float(variance)
    
    def __call__(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        diag: bool = False
    ) -> np.ndarray:
        if X2 is None:
            X2 = X1
        
        if diag:
            return np.full(X1.shape[0], self.variance)
        
        dists = self._euclidean_distances(X1, X2)
        
        if self.nu == 1.5:
            scaled = np.sqrt(3.0) * dists / self.length_scale
            K = self.variance * (1.0 + scaled) * np.exp(-scaled)
        else:  # nu == 2.5
            scaled = np.sqrt(5.0) * dists / self.length_scale
            K = self.variance * (1.0 + scaled + scaled ** 2 / 3.0) * np.exp(-scaled)
        
        return K
    
    def _euclidean_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute matrix of Euclidean distances."""
        dists_sq = np.sum(X1 ** 2, axis=1, keepdims=True) + \
                   np.sum(X2 ** 2, axis=1, keepdims=True).T - \
                   2 * np.dot(X1, X2.T)
        return np.sqrt(np.maximum(dists_sq, 0.0))
    
    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.length_scale), np.log(self.variance)])
    
    def set_params(self, params: np.ndarray) -> None:
        self.length_scale = np.exp(params[0])
        self.variance = np.exp(params[1])
    
    def gradient(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Compute gradients (numerical for simplicity)."""
        if X2 is None:
            X2 = X1
        
        # Simple finite differences for gradient
        eps = 1e-7
        K = self(X1, X2)
        
        # Gradient w.r.t. length_scale
        l_orig = self.length_scale
        self.length_scale = l_orig * (1 + eps)
        K_plus = self(X1, X2)
        dK_dlength = (K_plus - K) / (l_orig * eps) * l_orig  # Chain rule for log param
        self.length_scale = l_orig
        
        # Gradient w.r.t. variance
        v_orig = self.variance
        self.variance = v_orig * (1 + eps)
        K_plus = self(X1, X2)
        dK_dvariance = (K_plus - K) / (v_orig * eps) * v_orig
        self.variance = v_orig
        
        return {
            'length_scale': dK_dlength,
            'variance': dK_dvariance
        }


class RationalQuadratic(Kernel):
    """
    Rational Quadratic kernel (infinite mixture of RBF kernels).
    
    k(x, x') = σ² (1 + ||x - x'||² / (2 α ℓ²))^(-α)
    
    Parameters
    ----------
    length_scale : float, default=1.0
        Characteristic length scale ℓ.
    variance : float, default=1.0
        Signal variance σ².
    alpha : float, default=1.0
        Scale mixture parameter α.
    """
    
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0, alpha: float = 1.0):
        self.length_scale = float(length_scale)
        self.variance = float(variance)
        self.alpha = float(alpha)
    
    def __call__(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        diag: bool = False
    ) -> np.ndarray:
        if X2 is None:
            X2 = X1
        
        if diag:
            return np.full(X1.shape[0], self.variance)
        
        dists_sq = self._squared_distances(X1, X2)
        base = 1.0 + dists_sq / (2.0 * self.alpha * self.length_scale ** 2)
        K = self.variance * (base ** (-self.alpha))
        return K
    
    def _squared_distances(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        X1_sq = np.sum(X1 ** 2, axis=1, keepdims=True)
        X2_sq = np.sum(X2 ** 2, axis=1, keepdims=True)
        dists_sq = X1_sq + X2_sq.T - 2 * np.dot(X1, X2.T)
        return np.maximum(dists_sq, 0.0)
    
    def get_params(self) -> np.ndarray:
        return np.array([
            np.log(self.length_scale),
            np.log(self.variance),
            np.log(self.alpha)
        ])
    
    def set_params(self, params: np.ndarray) -> None:
        self.length_scale = np.exp(params[0])
        self.variance = np.exp(params[1])
        self.alpha = np.exp(params[2])
    
    def gradient(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Compute gradients (numerical)."""
        if X2 is None:
            X2 = X1
        
        eps = 1e-7
        K = self(X1, X2)
        
        # Length scale gradient
        l_orig = self.length_scale
        self.length_scale = l_orig * (1 + eps)
        K_plus = self(X1, X2)
        dK_dlength = (K_plus - K) / (l_orig * eps) * l_orig
        self.length_scale = l_orig
        
        # Variance gradient
        v_orig = self.variance
        self.variance = v_orig * (1 + eps)
        K_plus = self(X1, X2)
        dK_dvariance = (K_plus - K) / (v_orig * eps) * v_orig
        self.variance = v_orig
        
        return {
            'length_scale': dK_dlength,
            'variance': dK_dvariance
        }


class WhiteKernel(Kernel):
    """
    White noise kernel (uncorrelated Gaussian noise).
    
    k(x, x') = σ² δ(x, x')
    
    Parameters
    ----------
    noise_level : float, default=1.0
        Noise variance σ².
    """
    
    def __init__(self, noise_level: float = 1.0):
        self.noise_level = float(noise_level)
    
    def __call__(
        self,
        X1: np.ndarray,
        X2: Optional[np.ndarray] = None,
        diag: bool = False
    ) -> np.ndarray:
        if X2 is None or X2 is X1:
            n = X1.shape[0]
            if diag:
                return np.full(n, self.noise_level)
            else:
                return np.eye(n) * self.noise_level
        else:
            # Different X1 and X2 → no correlation
            if diag:
                return np.zeros(X1.shape[0])
            else:
                return np.zeros((X1.shape[0], X2.shape[0]))
    
    def get_params(self) -> np.ndarray:
        return np.array([np.log(self.noise_level)])
    
    def set_params(self, params: np.ndarray) -> None:
        self.noise_level = np.exp(params[0])
    
    def gradient(self, X1: np.ndarray, X2: Optional[np.ndarray] = None) -> Dict[str, np.ndarray]:
        """Gradient of white noise kernel."""
        K = self(X1, X2)
        # ∂K/∂σ² = K / σ²
        dK_dnoise = K / self.noise_level
        return {'noise_level': dK_dnoise}
