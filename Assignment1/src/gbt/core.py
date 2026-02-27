"""
Core gradient boosting implementations.

Implements Algorithm 10.3 (Forward Stagewise Additive Modelling) and 
Algorithm 10.4 (Gradient Tree Boosting) from "The Elements of Statistical Learning"
(Hastie, Tibshirani, Friedman, 2009).

References:
- Hastie, T., Tibshirani, R., & Friedman, J. (2009). The Elements of Statistical
  Learning (2nd ed.). Springer. Chapter 10.
- Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine.
  Annals of Statistics, 29(5), 1189-1232.
- Friedman, J., Hastie, T., & Tibshirani, R. (2000). Additive logistic regression: 
  a statistical view of boosting (LogitBoost). Annals of Statistics, 28(2), 337-407.
"""

from typing import Optional, List, Tuple, Dict
import logging
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier

from .utils import (
    mse_loss, mse_negative_gradient, mse_optimal_gamma,
    logistic_loss, logistic_negative_gradient, logistic_optimal_gamma,
    sigmoid, compute_metrics_regression, compute_metrics_classification
)


class GradientBoostingBase:
    """
    Base class for gradient boosting models.
    
    Implements stochastic gradient boosting with shrinkage (learning rate) and
    row subsampling. The boosting loop follows Algorithm 10.4 from ESL.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        learning_rate: float = 0.1,
        max_depth: int = 3,
        min_samples_leaf: int = 1,
        subsample: float = 1.0,
        random_state: Optional[int] = None,
        verbose: bool = False
    ):
        """
        Args:
            n_estimators: Number of boosting stages (M).
            learning_rate: Shrinkage parameter ν ∈ (0, 1]. Multiplies tree contributions.
            max_depth: Maximum depth of individual trees (controls J, terminal nodes).
            min_samples_leaf: Minimum samples required in a leaf node.
            subsample: Fraction of samples to use per iteration (stochastic boosting).
            random_state: Random seed for reproducibility.
            verbose: Enable logging output.
        """
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.subsample = subsample
        self.random_state = random_state
        self.verbose = verbose
        
        # Model state
        self.f0_: float = 0.0  # Initial constant prediction
        self.estimators_: List = []  # Weak learners
        self.leaf_values_: List[Dict[int, float]] = []  # Optimal gamma per leaf per tree
        
        # Training history
        self.train_scores_: List[float] = []
        self.val_scores_: List[float] = []
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        if self.verbose:
            logging.basicConfig(level=logging.INFO)
    
    def _subsample_indices(self, n_samples: int, rng: np.random.Generator) -> np.ndarray:
        """Generate subsampled indices for stochastic boosting."""
        if self.subsample < 1.0:
            n_subsample = max(1, int(self.subsample * n_samples))
            indices = rng.choice(n_samples, size=n_subsample, replace=False)
            return np.sort(indices)
        else:
            return np.arange(n_samples)


class GradientBoostingRegressor(GradientBoostingBase):
    """
    Gradient Tree Boosting for regression (Algorithm 10.4 with squared-error loss).
    
    Implements:
    1. Initialisation: f_0(x) = argmin_γ Σ L(y_i, γ) = mean(y) for MSE.
    2. For m = 1 to M:
       a. Compute pseudo-residuals: r_im = -∂L/∂f = y_i - f_{m-1}(x_i).
       b. Fit tree to {(x_i, r_im)} yielding regions R_jm.
       c. For each region j: γ_jm = argmin_γ Σ_{x_i ∈ R_jm} L(y_i, f_{m-1}(x_i) + γ).
          For MSE: γ_jm = mean(residuals in R_jm).
       d. Update: f_m(x) = f_{m-1}(x) + ν * Σ_j γ_jm I(x ∈ R_jm).
    
    Reference: ESL Section 10.9, Algorithm 10.4.
    """
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "GradientBoostingRegressor":
        """
        Fit gradient boosting regressor.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets, shape (n_samples,).
            X_val: Optional validation features for tracking generalisation.
            y_val: Optional validation targets.
        
        Returns:
            self
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        
        # Step 1: Initialise f_0(x) = argmin_γ Σ L(y_i, γ) = mean(y) for MSE
        self.f0_ = np.mean(y)
        F_train = np.full(n_samples, self.f0_)  # Current predictions
        
        self.estimators_ = []
        self.leaf_values_ = []
        self.train_scores_ = []
        self.val_scores_ = []
        
        if self.verbose:
            self.logger.info(f"Initial f_0 = {self.f0_:.6f}")
        
        # Step 2: Boosting loop
        for m in range(self.n_estimators):
            # Subsample
            indices = self._subsample_indices(n_samples, rng)
            X_sub = X[indices]
            y_sub = y[indices]
            F_sub = F_train[indices]
            
            # (a) Compute pseudo-residuals (negative gradient)
            residuals = mse_negative_gradient(y_sub, F_sub)
            
            # (b) Fit tree to residuals
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_sub, residuals)
            
            # (c) Compute optimal gamma per leaf
            # Get leaf indices for all training samples
            leaf_indices_sub = tree.apply(X_sub)
            unique_leaves = np.unique(leaf_indices_sub)
            
            gamma_map = {}
            for leaf_id in unique_leaves:
                mask = (leaf_indices_sub == leaf_id)
                # γ_j = argmin_γ Σ L(y_i, F_{m-1}(x_i) + γ) = mean(residuals in leaf)
                gamma_map[leaf_id] = mse_optimal_gamma(y_sub[mask], F_sub[mask])
            
            # (d) Update predictions with shrinkage
            leaf_indices_train = tree.apply(X)
            update = np.array([gamma_map.get(leaf, 0.0) for leaf in leaf_indices_train])
            F_train += self.learning_rate * update
            
            # Store the tree and leaf values
            self.estimators_.append(tree)
            self.leaf_values_.append(gamma_map)
            
            # Track scores
            train_mse = mse_loss(y, F_train)
            self.train_scores_.append(train_mse)
            
            if X_val is not None and y_val is not None:
                F_val = self._predict_raw(X_val, up_to_iteration=m+1)
                val_mse = mse_loss(y_val, F_val)
                self.val_scores_.append(val_mse)
                
                if self.verbose and (m + 1) % 10 == 0:
                    self.logger.info(
                        f"Iteration {m+1}/{self.n_estimators}: "
                        f"train_mse={train_mse:.6f}, val_mse={val_mse:.6f}"
                    )
            elif self.verbose and (m + 1) % 10 == 0:
                self.logger.info(
                    f"Iteration {m+1}/{self.n_estimators}: train_mse={train_mse:.6f}"
                )
        
        return self
    
    def _predict_raw(self, X: np.ndarray, up_to_iteration: Optional[int] = None) -> np.ndarray:
        """
        Raw predictions (not probabilities).
        
        Args:
            X: Features, shape (n_samples, n_features).
            up_to_iteration: Use only first k estimators (for staged predictions).
        
        Returns:
            Predictions, shape (n_samples,).
        """
        n_estimators = up_to_iteration if up_to_iteration is not None else len(self.estimators_)
        
        F = np.full(X.shape[0], self.f0_)
        
        for m in range(n_estimators):
            tree = self.estimators_[m]
            gamma_map = self.leaf_values_[m]
            
            leaf_indices = tree.apply(X)
            update = np.array([gamma_map.get(leaf, 0.0) for leaf in leaf_indices])
            F += self.learning_rate * update
        
        return F
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict regression targets."""
        return self._predict_raw(X)


class GradientBoostingClassifier(GradientBoostingBase):
    """
    Gradient Tree Boosting for binary classification (Algorithm 10.4 with logistic loss).
    
    Uses binomial deviance (logistic loss) and Newton-Raphson leaf optimisation (LogitBoost).
    
    Implements:
    1. Initialisation: f_0(x) = log(p/(1-p)), where p = mean(y).
    2. For m = 1 to M:
       a. Pseudo-residuals: r_im = y_i - p_i, where p_i = sigmoid(F_{m-1}(x_i)).
       b. Fit tree to {(x_i, r_im)} yielding regions R_jm.
       c. For each region j: γ_jm via Newton step:
          γ_jm = Σ_{x_i ∈ R_jm} r_i / Σ_{x_i ∈ R_jm} p_i(1 - p_i).
       d. Update: F_m(x) = F_{m-1}(x) + ν * Σ_j γ_jm I(x ∈ R_jm).
    
    Predictions: p(x) = sigmoid(F_M(x)).
    
    References:
    - ESL Section 10.9, Algorithm 10.4.
    - Friedman et al. (2000), "Additive logistic regression" (LogitBoost).
    """
    
    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None
    ) -> "GradientBoostingClassifier":
        """
        Fit gradient boosting classifier.
        
        Args:
            X: Training features, shape (n_samples, n_features).
            y: Training targets {0, 1}, shape (n_samples,).
            X_val: Optional validation features.
            y_val: Optional validation targets.
        
        Returns:
            self
        """
        rng = np.random.default_rng(self.random_state)
        n_samples = X.shape[0]
        
        # Step 1: Initialise f_0(x) = log(p/(1-p)), where p = mean(y)
        p_init = np.mean(y)
        # Clip to avoid log(0) or division by zero
        p_init = np.clip(p_init, 1e-15, 1 - 1e-15)
        self.f0_ = np.log(p_init / (1 - p_init))
        
        F_train = np.full(n_samples, self.f0_)  # Current raw predictions
        
        self.estimators_ = []
        self.leaf_values_ = []
        self.train_scores_ = []
        self.val_scores_ = []
        
        if self.verbose:
            self.logger.info(f"Initial f_0 = {self.f0_:.6f}")
        
        # Step 2: Boosting loop
        for m in range(self.n_estimators):
            # Subsample
            indices = self._subsample_indices(n_samples, rng)
            X_sub = X[indices]
            y_sub = y[indices]
            F_sub = F_train[indices]
            
            # (a) Compute pseudo-residuals (negative gradient of logistic loss)
            residuals = logistic_negative_gradient(y_sub, F_sub)
            
            # (b) Fit tree to residuals
            tree = DecisionTreeRegressor(  # Note: we use regressor to fit continuous residuals
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state
            )
            tree.fit(X_sub, residuals)
            
            # (c) Compute optimal gamma per leaf using Newton-Raphson
            leaf_indices_sub = tree.apply(X_sub)
            unique_leaves = np.unique(leaf_indices_sub)
            
            gamma_map = {}
            for leaf_id in unique_leaves:
                mask = (leaf_indices_sub == leaf_id)
                # Newton step for logistic loss
                gamma_map[leaf_id] = logistic_optimal_gamma(
                    y_sub[mask], F_sub[mask], residuals[mask]
                )
            
            # (d) Update predictions with shrinkage
            leaf_indices_train = tree.apply(X)
            update = np.array([gamma_map.get(leaf, 0.0) for leaf in leaf_indices_train])
            F_train += self.learning_rate * update
            
            # Store the tree and leaf values
            self.estimators_.append(tree)
            self.leaf_values_.append(gamma_map)
            
            # Track scores
            p_train = sigmoid(F_train)
            train_loss = logistic_loss(y, F_train)
            self.train_scores_.append(train_loss)
            
            if X_val is not None and y_val is not None:
                F_val = self._predict_raw(X_val, up_to_iteration=m+1)
                val_loss = logistic_loss(y_val, F_val)
                self.val_scores_.append(val_loss)
                
                if self.verbose and (m + 1) % 10 == 0:
                    self.logger.info(
                        f"Iteration {m+1}/{self.n_estimators}: "
                        f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}"
                    )
            elif self.verbose and (m + 1) % 10 == 0:
                self.logger.info(
                    f"Iteration {m+1}/{self.n_estimators}: train_loss={train_loss:.6f}"
                )
        
        return self
    
    def _predict_raw(self, X: np.ndarray, up_to_iteration: Optional[int] = None) -> np.ndarray:
        """
        Raw predictions (log-odds, before sigmoid).
        
        Args:
            X: Features, shape (n_samples, n_features).
            up_to_iteration: Use only first k estimators.
        
        Returns:
            Raw predictions F(x), shape (n_samples,).
        """
        n_estimators = up_to_iteration if up_to_iteration is not None else len(self.estimators_)
        
        F = np.full(X.shape[0], self.f0_)
        
        for m in range(n_estimators):
            tree = self.estimators_[m]
            gamma_map = self.leaf_values_[m]
            
            leaf_indices = tree.apply(X)
            update = np.array([gamma_map.get(leaf, 0.0) for leaf in leaf_indices])
            F += self.learning_rate * update
        
        return F
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.
        
        Args:
            X: Features, shape (n_samples, n_features).
        
        Returns:
            Probabilities for class 1, shape (n_samples,).
        """
        F = self._predict_raw(X)
        return sigmoid(F)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Features, shape (n_samples, n_features).
        
        Returns:
            Predicted labels {0, 1}, shape (n_samples,).
        """
        proba = self.predict_proba(X)
        return (proba >= 0.5).astype(int)
