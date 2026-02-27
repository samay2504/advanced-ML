"""
Utility functions for gradient boosting: loss functions, metrics, and optimisation.

References:
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
- Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive logistic regression: 
  a statistical view of boosting (LogitBoost).
"""

from typing import Tuple
import numpy as np
from scipy.optimize import minimize_scalar
from sklearn.metrics import mean_squared_error, log_loss, accuracy_score, roc_auc_score


# ===========================
# Loss Functions and Gradients
# ===========================

def mse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean squared error loss: L(y, f) = 0.5 * (y - f)^2."""
    return 0.5 * np.mean((y_true - y_pred) ** 2)


def mse_negative_gradient(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """
    Negative gradient (pseudo-residuals) for MSE loss.
    
    For L(y, f) = 0.5 * (y - f)^2, the negative gradient is:
    -∂L/∂f = y - f (the residuals).
    """
    return y_true - y_pred


def mse_optimal_gamma(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Optimal leaf value for MSE loss in a region.
    
    For squared error, γ* = argmin_γ Σ(y_i - (f_{m-1}(x_i) + γ))^2 
    over samples in the region, which is the mean of residuals.
    """
    residuals = y_true - y_pred
    return np.mean(residuals)


def logistic_loss(y_true: np.ndarray, y_pred_raw: np.ndarray) -> float:
    """
    Binomial deviance (logistic loss): L(y, F) = log(1 + exp(-2yF)),
    where y ∈ {-1, +1} and F is the raw score (before sigmoid).
    
    Alternatively, for y ∈ {0,1}: L = -y*log(p) - (1-y)*log(1-p), 
    where p = sigmoid(F).
    """
    # Convert to probabilities
    p = sigmoid(y_pred_raw)
    # Clip to avoid log(0)
    p = np.clip(p, 1e-15, 1 - 1e-15)
    return -np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p))


def logistic_negative_gradient(y_true: np.ndarray, y_pred_raw: np.ndarray) -> np.ndarray:
    """
    Negative gradient for logistic loss.
    
    For binomial deviance with y ∈ {0,1} and F the raw prediction:
    L(y, F) = -y*F + log(1 + exp(F))
    
    Gradient: ∂L/∂F = -y + exp(F)/(1 + exp(F)) = -y + p
    Negative gradient: y - p, where p = sigmoid(F).
    
    Reference: Friedman et al. (2000), "Additive logistic regression".
    """
    p = sigmoid(y_pred_raw)
    return y_true - p


def logistic_optimal_gamma(
    y_true: np.ndarray, 
    y_pred_raw: np.ndarray,
    negative_gradients: np.ndarray
) -> float:
    """
    Optimal leaf value for logistic loss using Newton-Raphson step.
    
    For logistic loss, the optimal γ in a region is found by minimising:
    Σ L(y_i, F_{m-1}(x_i) + γ)
    
    A one-step Newton-Raphson approximation (LogitBoost):
    γ* ≈ Σ r_i / Σ |r_i| * (1 - |r_i|)
    where r_i = y_i - p_i are the negative gradients.
    
    More precisely:
    γ* = Σ (y_i - p_i) / Σ p_i(1 - p_i)
    
    Reference: Friedman et al. (2000), LogitBoost Algorithm 6.
    """
    p = sigmoid(y_pred_raw)
    # Second derivative (Hessian diagonal): p(1-p)
    w = p * (1 - p)
    # Avoid division by zero
    w = np.clip(w, 1e-15, None)
    
    numerator = np.sum(negative_gradients)
    denominator = np.sum(w)
    
    if denominator == 0:
        return 0.0
    
    return numerator / denominator


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable sigmoid function."""
    return np.where(
        x >= 0,
        1 / (1 + np.exp(-x)),
        np.exp(x) / (1 + np.exp(x))
    )


# ===========================
# Metrics
# ===========================

def compute_metrics_regression(
    y_true: np.ndarray, 
    y_pred: np.ndarray
) -> dict:
    """Compute regression metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    
    return {
        "mse": mse,
        "rmse": rmse,
        "mae": mae
    }


def compute_metrics_classification(
    y_true: np.ndarray, 
    y_pred_proba: np.ndarray
) -> dict:
    """Compute classification metrics."""
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Clip probabilities for log_loss
    y_pred_proba_clipped = np.clip(y_pred_proba, 1e-15, 1 - 1e-15)
    
    logloss = log_loss(y_true, y_pred_proba_clipped)
    accuracy = accuracy_score(y_true, y_pred)
    
    # ROC AUC only if both classes present
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_pred_proba)
    else:
        auc = np.nan
    
    return {
        "log_loss": logloss,
        "accuracy": accuracy,
        "roc_auc": auc
    }
