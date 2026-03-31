# Advanced Machine Learning — Viva Preparation Notes

> **Scope**: Core algorithm implementations only (excludes testing code).  
> **Format**: Each assignment is explained **in the order it executes** — from raw data all the way to final output.  
> Every concept is paired with the exact file and line where it lives in the code.

---

## Table of Contents

- [Assignment 1 — Gradient Boosting Trees](#assignment-1--gradient-boosting-trees)
- [Assignment 2 — Gaussian Processes](#assignment-2--gaussian-processes)
- [Assignment 3 — Variational Autoencoders](#assignment-3--variational-autoencoders)

---

# Assignment 1 — Gradient Boosting Trees

## What Is the Assignment Task?

**Goal**: Implement **Gradient Tree Boosting from scratch** without using any off-the-shelf boosting library (no XGBoost, LightGBM, or sklearn.ensemble.GradientBoosting). The task covers two sub-problems:

1. **Regression** on the *California Housing* dataset — predict median house value (MSE loss).
2. **Binary Classification** on the *Breast Cancer Wisconsin* dataset — classify malignant/benign (logistic/binomial deviance loss).

The implementation follows **Algorithm 10.4 (Gradient Tree Boosting)** from Hastie, Tibshirani & Friedman, *The Elements of Statistical Learning* (2009), Chapter 10.

**Source files**: `Assignment1/src/gbt/core.py`, `Assignment1/src/gbt/utils.py`  
**Experiment scripts**: `Assignment1/experiments/regression_run.py`, `classification_run.py`

---

## Pipeline Overview — Assignment 1

```
Raw Data
  |
  v
Step 1 -- Load & Preprocess Data  (StandardScaler, train/val/test split)
  |
  v
Step 2 -- Initialise Model  (compute F_0, the best constant baseline)
  |
  v
Step 3 -- Boosting Loop  [repeat M times]
  |          +-- 3a. Row subsampling  (stochastic boosting, optional)
  |          +-- 3b. Compute pseudo-residuals  (negative gradient of loss)
  |          +-- 3c. Fit a shallow regression tree to pseudo-residuals
  |          +-- 3d. Re-optimise leaf values  (optimal gamma per leaf)
  |          +-- 3e. Update running prediction:  F_m = F_{m-1} + nu * gamma
  |
  v
Step 4 -- Predict on new data  (replay the loop, accumulate all trees)
  |
  v
Step 5 -- Evaluate  (MSE for regression / log-loss, accuracy, AUC for classification)
```

---

## Step 1 — Load and Preprocess Data

The experiments load standard sklearn datasets, apply an 80/20 train-test split (with a further 80/20 train-val split), and standardise features using `StandardScaler`.

**Code** (`regression_run.py`, lines 26-50):
```python
data = fetch_california_housing()
X, y = data.data, data.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)   # fit only on train set
X_val   = scaler.transform(X_val)         # apply the same scale to val and test
X_test  = scaler.transform(X_test)
```

The validation set is passed to `gbr.fit(X_val=..., y_val=...)` to track the generalization loss curve during training. The test set is held back entirely for final evaluation.

---

## Step 2 — Model Initialisation (F_0)

Before any tree is added, the model starts with the **best constant prediction** — the value that minimises the loss over all training samples:

```
F_0(x) = argmin_{gamma}  sum_i  L(y_i, gamma)
```

| Mode | Loss | F_0 | Why |
|---|---|---|---|
| Regression | MSE = 0.5*(y-F)^2 | mean(y) | Mean minimises squared error |
| Classification | Logistic | log(p/(1-p)) | Log-odds of the base rate |

**Code — Regression** (`core.py`, lines 127-129):
```python
# Initialise f_0(x) = mean(y) for MSE
self.f0_ = np.mean(y)
F_train  = np.full(n_samples, self.f0_)   # every sample gets the same starting prediction
```

**Code — Classification** (`core.py`, lines 273-279):
```python
p_init   = np.mean(y)                             # fraction of positive labels
p_init   = np.clip(p_init, 1e-15, 1 - 1e-15)     # guard against log(0)
self.f0_ = np.log(p_init / (1 - p_init))          # log-odds initialisation
F_train  = np.full(n_samples, self.f0_)           # raw log-odds for all samples
```

`F_train` is a 1-D array of shape `(n_samples,)` holding the **running prediction**. It starts at `F_0` and is updated after every tree.

---

## Step 3a — Row Subsampling (Stochastic Boosting)

Before computing residuals, optionally draw a random fraction `subsample` of training rows **without replacement**. This acts as regularisation and speeds up training.

**Code** (`core.py`, lines 79-86, base class):
```python
def _subsample_indices(self, n_samples, rng):
    if self.subsample < 1.0:
        n_subsample = max(1, int(self.subsample * n_samples))
        indices = rng.choice(n_samples, size=n_subsample, replace=False)
        return np.sort(indices)
    return np.arange(n_samples)   # use all samples when subsample=1.0
```

**Used in loop** (`core.py`, lines 141-145 for regression):
```python
indices = self._subsample_indices(n_samples, rng)
X_sub   = X[indices]        # features for this iteration
y_sub   = y[indices]        # targets
F_sub   = F_train[indices]  # current predictions for these rows
```

---

## Step 3b — Compute Pseudo-Residuals (Negative Gradient)

The key insight of gradient boosting: treat function fitting as **gradient descent in function space**. At each step, compute the direction (negative gradient of the loss) that most reduces the current loss:

```
r_{im} = - dL(y_i, F_{m-1}(x_i)) / dF_{m-1}(x_i)
```

For MSE this equals ordinary residuals `y - F`. For other losses it generalises that concept.

**Code — MSE** (`utils.py`, lines 25-32):
```python
def mse_negative_gradient(y_true, y_pred):
    """ -d/dF [0.5*(y-F)^2] = y - F """
    return y_true - y_pred    # ordinary residuals
```

**Code — Logistic** (`utils.py`, lines 61-74):
```python
def logistic_negative_gradient(y_true, y_pred_raw):
    """
    Loss: L = -y*log(p) - (1-y)*log(1-p),  p = sigmoid(F)
    Gradient: dL/dF = p - y
    Negative gradient: y - p
    """
    p = sigmoid(y_pred_raw)   # convert raw log-odds to probability
    return y_true - p         # residual in probability space
```

**Called in the loop** (`core.py`, lines 147-148):
```python
residuals = mse_negative_gradient(y_sub, F_sub)
# or for classification:
residuals = logistic_negative_gradient(y_sub, F_sub)
```

---

## Step 3c — Fit a Shallow Decision Tree to Pseudo-Residuals

A **regression tree** is fitted to the pairs `{(x_i, r_{im})}`. The tree partitions the feature space into `J` leaf regions.

**Critical detail**: Even for the *classifier*, `DecisionTreeRegressor` is used — because pseudo-residuals are continuous floating-point values, not class labels.

**Code** (`core.py`, lines 151-156 for regression; lines 301-306 for classification — identical):
```python
tree = DecisionTreeRegressor(
    max_depth        = self.max_depth,        # controls J ~ 2^max_depth leaves
    min_samples_leaf = self.min_samples_leaf,
    random_state     = self.random_state
)
tree.fit(X_sub, residuals)   # fit to (features, pseudo-residuals)
```

`max_depth=3` means at most 8 leaves — each tree is a "stump-like" weak learner.

---

## Step 3d — Re-Optimise Leaf Values (Optimal Gamma per Leaf)

The tree was fitted to residuals, so its leaf values minimise squared residuals, not the actual loss. We **discard the tree's leaf predictions** and replace them with the optimal constant for the true loss in each leaf:

```
gamma_{jm} = argmin_{gamma}  sum_{x_i in R_{jm}} L(y_i, F_{m-1}(x_i) + gamma)
```

| Loss | Optimal gamma | Method |
|---|---|---|
| MSE | mean(y_i - F_{m-1}(x_i)) over the leaf | Closed-form |
| Logistic | sum(r_i) / sum(p_i*(1-p_i)) over the leaf | One-step Newton-Raphson |

**Code — MSE gamma** (`utils.py`, lines 35-43):
```python
def mse_optimal_gamma(y_true, y_pred):
    """Closed form: gamma* = mean of residuals in this leaf."""
    residuals = y_true - y_pred
    return np.mean(residuals)
```

**Code — Logistic gamma** (`utils.py`, lines 77-109):
```python
def logistic_optimal_gamma(y_true, y_pred_raw, negative_gradients):
    """
    Newton step: gamma* = sum(r_i) / sum(p_i*(1-p_i))
    Numerator  = first-order gradient sum
    Denominator = second-order Hessian sum = p*(1-p)
    """
    p           = sigmoid(y_pred_raw)
    w           = np.clip(p * (1 - p), 1e-15, None)   # Hessian diagonal
    numerator   = np.sum(negative_gradients)
    denominator = np.sum(w)
    return 0.0 if denominator == 0 else numerator / denominator
```

**Code — Build gamma map per leaf** (`core.py`, lines 160-167):
```python
leaf_indices_sub = tree.apply(X_sub)          # integer leaf-ID for each training sample
unique_leaves    = np.unique(leaf_indices_sub)

gamma_map = {}                                 # {leaf_id: optimal_gamma}
for leaf_id in unique_leaves:
    mask = (leaf_indices_sub == leaf_id)
    gamma_map[leaf_id] = mse_optimal_gamma(y_sub[mask], F_sub[mask])
    # or logistic_optimal_gamma for the classifier
```

`tree.apply(X)` routes every sample through the already-fitted tree and returns its leaf ID — no re-fitting involved.

---

## Step 3e — Update the Running Prediction (Shrinkage)

The ensemble prediction is updated by adding this tree's contribution, scaled by the **learning rate** nu:

```
F_m(x) = F_{m-1}(x)  +  nu * sum_j gamma_{jm} * I[x in R_{jm}]
```

A smaller nu means each tree contributes less — the model learns more slowly but generalises better.

**Code** (`core.py`, lines 169-176):
```python
leaf_indices_train = tree.apply(X)              # assign ALL training samples to leaves
update = np.array([
    gamma_map.get(leaf, 0.0) for leaf in leaf_indices_train
])                                              # gamma for each sample's leaf
F_train += self.learning_rate * update          # nu * gamma applied to every sample

self.estimators_.append(tree)                   # store tree for prediction
self.leaf_values_.append(gamma_map)             # store gamma map for prediction
```

`gamma_map.get(leaf, 0.0)`: if a leaf was not seen during subsampled training, default to 0 update.

---

## Step 4 — Prediction

At prediction time, the model replays the boosting loop — starting from `F_0` and applying each stored tree's leaf-gamma map in order.

**Code — `_predict_raw`** (`core.py`, lines 199-222 for regression; lines 351-374 for classification):
```python
def _predict_raw(self, X, up_to_iteration=None):
    n_est = up_to_iteration or len(self.estimators_)
    F = np.full(X.shape[0], self.f0_)           # start at constant baseline

    for m in range(n_est):
        tree      = self.estimators_[m]
        gamma_map = self.leaf_values_[m]
        leaf_ids  = tree.apply(X)               # route X through tree m
        update    = np.array([gamma_map.get(l, 0.0) for l in leaf_ids])
        F        += self.learning_rate * update  # accumulate each tree's contribution

    return F   # regression: return F directly
               # classification: apply sigmoid to get P(y=1)
```

**For classification** (`core.py`, lines 376-387):
```python
def predict_proba(self, X):
    F = self._predict_raw(X)    # raw log-odds
    return sigmoid(F)           # P(y=1 | x)

def predict(self, X):
    return (self.predict_proba(X) >= 0.5).astype(int)
```

---

## Step 5 — Evaluate

- **Regression**: MSE on train, val, test. `train_scores_` and `val_scores_` lists track MSE per iteration for learning-curve plots.
- **Classification**: Log-loss, accuracy, ROC-AUC, confusion matrix.

---

## Key Hyperparameters and Their Role

| Hyperparameter | Role | Where set |
|---|---|---|
| `n_estimators` M | Number of trees / boosting stages | `GradientBoostingBase.__init__` |
| `learning_rate` nu | Shrinkage — scales each tree's update | `F_train += lr * update` |
| `max_depth` | Controls tree complexity (J leaves) | `DecisionTreeRegressor(max_depth=...)` |
| `subsample` | Fraction of rows per iteration (stochastic) | `_subsample_indices` |
| `min_samples_leaf` | Minimum samples per leaf (regularises trees) | `DecisionTreeRegressor(...)` |

---

## Viva Q&A — Assignment 1

| Question | Concise Answer | Code location |
|---|---|---|
| What is the task? | Implement GBT from scratch for regression (California Housing) and classification (Breast Cancer) | `core.py`, `utils.py` |
| First thing done before any tree? | Initialise F_0: mean(y) for MSE, log-odds for logistic | `core.py` L127-129, L273-279 |
| Why pseudo-residuals, not raw residuals? | To extend boosting to any differentiable loss — they are the negative gradient | `utils.py` L25-32, L61-74 |
| Why `DecisionTreeRegressor` for classification? | Pseudo-residuals are continuous — we need regression, not classification | `core.py` L301 |
| Why re-compute gamma after tree fitting? | The tree minimised squared residuals, not the true loss — gamma is the exact optimal update | `utils.py` L35-109, `core.py` L160-167 |
| What does learning rate do? | Scales the leaf update `F += nu*gamma` — smaller nu reduces overfitting | `core.py` L172 |
| What is stochastic boosting? | Row subsampling per iteration — adds regularisation, reduces variance | `core.py` L79-86 |
| What is stored in `estimators_` and `leaf_values_`? | List of fitted trees; list of `{leaf_id: gamma}` dicts | `core.py` L174-176 |
| How does predict work at test time? | Start from F_0, route X through each stored tree, look up gamma, accumulate nu*gamma | `core.py` L199-222 |

---
---

# Assignment 2 — Gaussian Processes

## What Is the Assignment Task?

**Goal**: Implement **Gaussian Process regression and binary classification from scratch**, following algorithms directly from Rasmussen & Williams, *Gaussian Processes for Machine Learning* (2006):

- **Algorithm 2.1** — Exact GP Regression with hyperparameter optimisation via log marginal likelihood.
- **Algorithm 3.1** — Posterior mode via Newton's method (Laplace approximation for classification).
- **Algorithm 3.2** — Predictive probabilities for GP classification.

Applied to two tasks:
1. **Regression** on the *Diabetes* dataset — predict disease progression with uncertainty bands (RMSE, NLPD).
2. **Binary Classification** on the *Breast Cancer Wisconsin* dataset — classify malignant/benign with calibrated probabilities (ROC-AUC, calibration curve).

**Source files**: `Assignment2/src/gp/base.py`, `kernels.py`, `regression.py`, `classification.py`  
**Experiment scripts**: `Assignment2/experiments/regression_demo.py`, `classification_demo.py`

A Gaussian Process does not predict a single value — it predicts a **full Gaussian distribution** over possible function values at every test point, giving **uncertainty quantification for free**: high uncertainty far from training data or near class boundaries, low uncertainty near dense training data.

---

## Part A — GP Regression Pipeline

```
Raw Data (Diabetes dataset, 442 samples, 10 features)
  |
  v
Step A1 -- Load & Split Data  (train/test)
  |
  v
Step A2 -- Choose a Kernel  (encodes smoothness assumptions about f)
  |
  v
Step A3 -- Fit
  |         Compute K = k(X,X) + sigma_n^2 * I
  |         Cholesky: L = chol(K)
  |         Alpha:    alpha = L^T \ (L \ y)   [= K^{-1}y]
  v
Step A4 -- Optimise Hyperparameters
  |         Maximise log p(y|X,theta) via L-BFGS-B
  |         (multiple random restarts, log-space search)
  v
Step A5 -- Predict
  |         mean  = K_*^T * alpha
  |         var   = k(x_*,x_*) - v^T*v,   v = L^{-1}*K_*
  v
Step A6 -- Evaluate: RMSE, NLPD, predictive-band plots
```

---

### A1. Load and Split Data

**Code** (`regression_demo.py`, lines 47-53):
```python
diabetes = load_diabetes()
X, y     = diabetes.data, diabetes.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
```
The kernel hyperparameters (length-scale) adapt to the data scale during optimisation, so explicit standardisation is not strictly required.

---

### A2. Choose a Kernel

A kernel `k(x, x')` measures similarity between inputs and determines the smoothness of functions the GP can represent. All kernels inherit from an abstract `Kernel` base class (`kernels.py`, line 16).

#### RBF (Squared Exponential)
```
k(x, x') = sigma^2 * exp(-||x-x'||^2 / (2*l^2))
```
Infinitely differentiable. `l` = length-scale, `sigma^2` = signal variance.

**Code** (`Assignment2/src/gp/kernels.py`, lines 86-101):
```python
class RBF(Kernel):
    def __call__(self, X1, X2=None, diag=False):
        dists_sq = self._squared_distances(X1, X2)
        return self.variance * np.exp(-0.5 * dists_sq / (self.length_scale ** 2))

    def _squared_distances(self, X1, X2):
        # ||x-x'||^2 = ||x||^2 - 2<x,x'> + ||x'||^2  (vectorised, no loops)
        X1_sq = np.sum(X1**2, axis=1, keepdims=True)
        X2_sq = np.sum(X2**2, axis=1, keepdims=True)
        return np.maximum(X1_sq + X2_sq.T - 2 * X1 @ X2.T, 0.0)
```

#### Matern Kernel (nu=3/2 or nu=5/2)
```
nu=3/2: k(r) = sigma^2*(1 + sqrt(3)*r/l)*exp(-sqrt(3)*r/l)          -- once-differentiable
nu=5/2: k(r) = sigma^2*(1 + sqrt(5)*r/l + 5r^2/(3l^2))*exp(-sqrt(5)*r/l)  -- twice-differentiable
```

**Code** (`Assignment2/src/gp/kernels.py`, lines 183-190):
```python
if self.nu == 1.5:
    scaled = np.sqrt(3.0) * dists / self.length_scale
    K = self.variance * (1.0 + scaled) * np.exp(-scaled)
else:  # nu=2.5
    scaled = np.sqrt(5.0) * dists / self.length_scale
    K = self.variance * (1.0 + scaled + scaled**2 / 3.0) * np.exp(-scaled)
```

#### Hyperparameters Stored in Log Space

Kernels store parameters in log space so the optimiser can search over all reals without violating positivity (`Assignment2/src/gp/kernels.py`, lines 111-118):
```python
def get_params(self):
    return np.array([np.log(self.length_scale), np.log(self.variance)])

def set_params(self, params):
    self.length_scale = np.exp(params[0])   # always > 0
    self.variance     = np.exp(params[1])   # always > 0
```

---

### A3. Fit — Cholesky Decomposition and Alpha (Algorithm 2.1, Steps 1-3)

GP inference requires solving `K^{-1}*y`. Direct inversion is numerically unstable. The Cholesky decomposition `K = L*L^T` is used instead.

```
K      = k(X,X) + sigma_n^2 * I    (n x n)
L      = cholesky(K)                (lower triangular)
alpha  = L^T \ (L \ y)             = K^{-1}*y  without explicit inversion
```

**Code — Adaptive Jitter Cholesky** (`Assignment2/src/gp/base.py`, lines 39-54):
```python
def stable_cholesky(K, jitter=1e-6):
    """Try progressively larger jitter until Cholesky succeeds."""
    for j in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        try:
            return np.linalg.cholesky(K + np.eye(K.shape[0]) * j)
        except np.linalg.LinAlgError:
            continue
    raise np.linalg.LinAlgError("Cholesky failed even with max jitter")
```
Jitter is a small diagonal term that restores positive-definiteness when K is near-singular.

**Code — Two-Step Cholesky Solve** (`Assignment2/src/gp/base.py`, lines 57-79):
```python
def cholesky_solve(L, b):
    """Solve K*x = b given L such that K = L*L^T."""
    y = np.linalg.solve(L,   b)   # forward:  L*y = b
    x = np.linalg.solve(L.T, y)   # backward: L^T*x = y
    return x                       # x = K^{-1}*b
```

**Code — `_compute_alpha`** (`Assignment2/src/gp/regression.py`, lines 117-131):
```python
def _compute_alpha(self):
    n  = self.X_train_.shape[0]
    K  = self.kernel(self.X_train_, self.X_train_)   # n x n kernel matrix
    K += np.eye(n) * self.noise                       # K + sigma_n^2 * I

    self.L_     = stable_cholesky(K, jitter=1e-6)    # L = chol(K)
    self.alpha_ = cholesky_solve(self.L_, self.y_train_)  # alpha = K^{-1}*y
```
`alpha_` is precomputed and cached — all predictions use it without re-solving.

---

### A4. Optimise Hyperparameters — Log Marginal Likelihood

The hyperparameters `theta = (l, sigma^2, sigma_n^2)` are chosen by maximising the **log marginal likelihood**:

```
log p(y|X,theta) = -0.5 * y^T K^{-1} y  -  0.5 * log|K|  -  n/2 * log(2*pi)
                   -----------------         -----------       --------------
                    data-fit term             complexity          constant
```

The data-fit term rewards fitting training data; the log-determinant penalises overly complex models. Together they provide automatic Occam's razor.

**Code — Log Marginal Likelihood** (`Assignment2/src/gp/base.py`, lines 82-124):
```python
def log_marginal_likelihood(y, K, L=None, jitter=1e-6):
    if L is None:
        L = stable_cholesky(K, jitter)

    alpha    = cholesky_solve(L, y)
    data_fit = -0.5 * np.dot(y, alpha)            # -0.5 * y^T K^{-1} y

    log_det  = -np.sum(np.log(np.diag(L)))         # -0.5 * log|K| = -sum(log L_ii)
    const    = -0.5 * len(y) * np.log(2 * np.pi)

    return data_fit + log_det + const
```
`log|K| = 2*sum(log L_ii)` because `K = L*L^T` so `|K| = (prod L_ii)^2`.

**Code — L-BFGS-B Optimisation** (`Assignment2/src/gp/regression.py`, lines 236-302):
```python
def _optimise_hyperparameters(self):
    def objective(params):                              # minimise negative LML
        self.kernel.set_params(params[:-1])             # update l, sigma^2 in log space
        self.noise = np.exp(params[-1])                 # update noise sigma_n^2
        self._compute_alpha()                           # recompute L and alpha
        return -self.log_marginal_likelihood_value()    # negate to minimise

    initial = np.concatenate([self.kernel.get_params(), [np.log(self.noise)]])
    bounds  = [(-10, 10) for _ in initial]

    for restart in range(self.n_restarts + 1):
        x0 = initial if restart == 0 else np.random.uniform(-2, 2, initial.shape)
        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        # keep best result across restarts
```

---

### A5. Predict — Posterior Mean and Variance (Algorithm 2.1, Steps 4-5)

```
f_bar_*  = K_*^T * alpha                (posterior mean)
V[f_*]   = k(x_*,x_*) - v^T*v,   v = L^{-1} * K_*   (posterior variance)
```
The variance is the prior variance `k(x_*,x_*)` minus the reduction from observing training data.

**Code — `predict`** (`Assignment2/src/gp/regression.py`, lines 133-191):
```python
def predict(self, X, return_std=False, return_cov=False):
    K_star = self.kernel(self.X_train_, X)          # K_* = k(X_train, X_test)
    mean   = K_star.T @ self.alpha_                  # f_bar_* = K_*^T * alpha

    if return_std:
        v   = np.linalg.solve(self.L_, K_star)      # v = L^{-1} * K_*
        var = self.kernel(X, X, diag=True) - np.sum(v**2, axis=0)
        var = np.maximum(var, 0.0)                   # clip numerical negatives to 0
        return mean, np.sqrt(var)

    if return_cov:
        v   = np.linalg.solve(self.L_, K_star)
        cov = self.kernel(X, X) - v.T @ v
        cov = 0.5 * (cov + cov.T)                   # enforce exact symmetry
        return mean, cov
```

---

### A6. Evaluate

**Code** (`regression_demo.py`, lines 86-94):
```python
y_pred_mean, y_pred_std = gp.predict(X_test, return_std=True)

rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
# NLPD penalises both wrong mean and wrong uncertainty:
nlpd = negative_log_predictive_density(y_test, y_pred_mean, y_pred_std)
```

---

## Part B — GP Classification Pipeline

```
Raw Data (Breast Cancer dataset, 569 samples, 30 features)
  |
  v
Step B1 -- Load & Split Data
  |
  v
Step B2 -- Choose Kernel + Create GaussianProcessClassifier
  |
  v
Step B3 -- Fit: find posterior mode f_hat  [Algorithm 3.1 -- Newton iterations]
  |         Initialise f = 0
  |         Repeat until convergence:
  |           W  = diag(-d^2 log p(y|f)/df^2)     -- negative Hessian (curvature)
  |           b  = W*f + d log p(y|f)/df           -- Newton direction
  |           B  = I + W^{1/2} K W^{1/2}
  |           L  = chol(B)
  |           a  = b - W^{1/2} L^T\(L\(W^{1/2}*K*b))
  |           f  <- K*a
  v
Step B4 -- Predict probabilities  [Algorithm 3.2]
  |         f_bar_*  = k(X,x_*)^T * d log p(y|f_hat)/df
  |         V[f_*]   = k(x_*,x_*) - v^T*v
  |         pi_star ~= Phi(kappa*f_bar_* / sqrt(1 + kappa^2*V[f_*]))  [probit approx]
  v
Step B5 -- Evaluate: ROC-AUC, calibration curve, 2D probability heatmap
```

---

### B3. Fit — Newton's Method for Posterior Mode (Algorithm 3.1)

**Theory**: For classification the likelihood `p(y|f)` is non-Gaussian (sigmoid/probit). The exact posterior is intractable. The **Laplace approximation** finds the posterior **mode** f_hat using Newton's method, then fits a Gaussian at that point.

`W = diag(-d^2 log p(y|f)/df^2)` is the negative Hessian of the log-likelihood (diagonal). Points near the decision boundary (p ~ 0.5) have the highest W and contribute most to the Newton step.

**Code — Newton Loop** (`Assignment2/src/gp/classification.py`, lines 121-162):
```python
f = np.zeros(n)                     # initialise at zero

for iteration in range(self.max_iter):
    grad_log_lik = self._gradient_log_likelihood(f, y)   # d log p(y|f)/df
    W            = self._hessian_log_likelihood(f, y)    # diag of -d^2 log p(y|f)/df^2

    # Build B = I + W^{1/2} K W^{1/2}
    W_sqrt   = np.sqrt(W)
    W_sqrt_K = W_sqrt[:, np.newaxis] * self.K_           # W^{1/2} K
    B        = np.eye(n) + W_sqrt_K * W_sqrt[np.newaxis, :]

    L = stable_cholesky(B, jitter=self.noise)            # L = chol(B)

    b          = W * f + grad_log_lik                    # b = W*f + d log p
    W_sqrt_K_b = W_sqrt_K @ b
    c          = np.linalg.solve(L, W_sqrt_K_b)
    c          = np.linalg.solve(L.T, c)
    a          = b - W_sqrt * c
    f_new      = self.K_ @ a                             # f <- K*a

    if np.max(np.abs(f_new - f)) < self.tol:             # convergence check
        break
    f = f_new
```

**Gradient and Hessian for logistic likelihood** (`Assignment2/src/gp/classification.py`, lines 296-339):
```python
def _gradient_log_likelihood(self, f, y):
    # y in {-1,+1};  t = (y+1)/2 in {0,1}
    t  = (y + 1.0) / 2.0
    pi = self._sigmoid(f)
    return t - pi            # = t - sigma(f)

def _hessian_log_likelihood(self, f, y):
    pi = self._sigmoid(f)
    W  = pi * (1.0 - pi)     # W = sigma(f)*(1-sigma(f))
    return np.maximum(W, 1e-10)
```

---

### B4. Predict — Algorithm 3.2

The integral of `sigma(f)*N(f|mu,sigma^2)` has no closed form for logistic -> use the **probit approximation**: `sigma(f) ~= Phi(kappa*f)`, `kappa^2 = pi/8`.

**Code — `predict_proba`** (`Assignment2/src/gp/classification.py`, lines 180-231):
```python
def predict_proba(self, X):
    k_star       = self.kernel(self.X_train_, X)
    grad_log_lik = self._gradient_log_likelihood(self.f_hat_, self.y_train_)
    f_bar        = k_star.T @ grad_log_lik          # posterior mean f_bar_*

    W_sqrt = np.sqrt(self.W_)
    v      = np.linalg.solve(self.L_, W_sqrt[:, np.newaxis] * k_star)
    var_f  = np.maximum(self.kernel(X, X, diag=True) - np.sum(v**2, axis=0), 0.0)

    kappa_sq = np.pi / 8.0
    pi_star  = self._probit_sigmoid(f_bar / np.sqrt(1.0 + kappa_sq * var_f))

    return np.column_stack([1 - pi_star, pi_star])   # [P(y=0), P(y=1)]
```

---

## Viva Q&A — Assignment 2

| Question | Concise Answer | Code location |
|---|---|---|
| What is the task? | Implement GP regression (Alg 2.1) and GP classification (Alg 3.1/3.2) with uncertainty quantification | `regression.py`, `classification.py` |
| What does a GP predict that a standard model does not? | A full Gaussian distribution — mean + variance (uncertainty) at every test point | `regression.py` L133-191 |
| Why Cholesky instead of matrix inversion? | Numerically stable; avoids catastrophic cancellation; same O(n^3) complexity | `base.py` L39-54 |
| What is alpha? | Pre-computed `K^{-1}*y` — used in every prediction as the "weights" | `regression.py` L131 |
| What is the LML used for? | Selecting kernel hyperparameters — balances data fit vs. model complexity | `base.py` L82-124 |
| What is the Laplace approximation? | Approximate the intractable posterior with a Gaussian centred at the MAP estimate f_hat | `classification.py` L81-178 |
| What does W represent in Alg. 3.1? | Diagonal of the negative Hessian — local curvature of the log-likelihood | `classification.py` L315-339 |
| How is predictive uncertainty computed? | Prior variance minus variance explained by data: `k(x_*,x_*) - v^T*v` | `regression.py` L187-190 |
| Why use probit approximation? | The integral of sigma(f)*N(f|mu,sigma^2) has no closed form — probit makes it tractable | `classification.py` L341-350 |

---
---

# Assignment 3 — Variational Autoencoders

## What Is the Assignment Task?

**Goal**: Implement a **Variational Autoencoder (VAE)** and a **Conditional VAE (CVAE)** from scratch on MNIST handwritten digits, demonstrating:

1. **Part 1 (VAE)**: Unsupervised generative modelling — learn a structured latent space and generate new digit images from Gaussian noise. Visualise the latent space via PCA and latent interpolation.
2. **Part 2 (CVAE)**: Conditional generative modelling — condition on a class label so specific digits can be generated on demand.

Both models include: log-variance reparameterisation, KL annealing (beta-VAE schedule), mixed-precision (AMP) training, and early stopping.

**Source files**: `Assignment3/part1_vae.py`, `Assignment3/part2_cvae.py`

---

## Part 1 — VAE Pipeline

```
MNIST Dataset (60k train / 10k test, 28x28 greyscale pixels)
  |
  v
Step 1 -- Load Data & Set Hyperparameters
  |         INPUT_DIM=784, H_DIM=400, Z_DIM=20, BATCH_SIZE=128
  v
Step 2 -- Define VAE Architecture
  |         Encoder: 784 -> [400 ReLU] -> mu (20) and log_var (20)
  |         Decoder: 20  -> [400 ReLU] -> 784 raw logits
  v
Step 3 -- Forward Pass (per mini-batch)
  |         3a. Encode:          x -> (mu, log_var)
  |         3b. Reparameterize:  z = mu + sigma * eps,  eps ~ N(0,I)
  |         3c. Decode:          z -> x_hat_logits
  v
Step 4 -- Compute ELBO Loss
  |         Recon = BCEWithLogitsLoss(x_hat_logits, x) / batch_size
  |         KL    = -0.5 * sum(1 + log_var - mu^2 - exp(log_var)) / batch_size
  |         Loss  = Recon + beta * KL   (beta annealed 0->1)
  v
Step 5 -- Backward + AMP Update
  |         scaler.scale(loss).backward()
  |         scaler.step(optimizer); scaler.update()
  v
Step 6 -- Checkpointing + Early Stopping
  |         Save best model; stop if no improvement for PATIENCE epochs after annealing
  v
Step 7 -- Visualise Outputs
            Reconstructions, prior samples, latent space PCA, latent interpolation
```

---

### Step 1 — Load Data and Set Hyperparameters

**Code** (`part1_vae.py`, lines 54-103):
```python
INPUT_DIM           = 784    # 28x28 pixels flattened
H_DIM               = 400    # hidden layer width
Z_DIM               = 20     # latent dimension
BATCH_SIZE          = 128
NUM_EPOCHS          = 120
LR                  = 1e-3
KL_ANNEAL_EPOCHS    = 10     # ramp beta from 0->1 over first 10 epochs
EARLY_STOP_PATIENCE = 10

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=2, pin_memory=True)
```
`pin_memory=True` pre-pins host memory for faster CPU->GPU transfer. `shuffle=True` randomises batch order each epoch.

---

### Step 2 — VAE Architecture

The encoder has **two output heads** (mu and log_var) sharing a single hidden layer.

**Code — `VAE.__init__`** (`part1_vae.py`, lines 127-134):
```python
# Encoder: two separate linear heads after shared hidden
self.enc_hidden  = nn.Linear(input_dim, h_dim)   # 784 -> 400
self.enc_mu      = nn.Linear(h_dim, z_dim)        # 400 -> 20  (mu)
self.enc_log_var = nn.Linear(h_dim, z_dim)        # 400 -> 20  (log sigma^2)

# Decoder: single path back to pixel space
self.dec_hidden  = nn.Linear(z_dim, h_dim)        # 20  -> 400
self.dec_out     = nn.Linear(h_dim, input_dim)    # 400 -> 784 (raw logits)
```

`log_var` is used instead of `var` because it can take any real value — no positivity constraint needed. Variance is recovered by `sigma^2 = exp(log_var)`, which is always positive.

---

### Step 3a — Encode: x -> (mu, log_var)

Maps a flattened image to the parameters of the approximate posterior `q_phi(z|x) = N(z | mu, diag(sigma^2))`.

**Code — `VAE.encode`** (`part1_vae.py`, lines 136-149):
```python
def encode(self, x):
    h       = self.relu(self.enc_hidden(x))   # shared hidden representation
    mu      = self.enc_mu(h)                  # mean of q(z|x)
    log_var = self.enc_log_var(h)             # log variance of q(z|x)
    return mu, log_var
```

---

### Step 3b — Reparameterization Trick: (mu, log_var) -> z

**Problem**: Sampling `z ~ N(mu, sigma^2)` is stochastic — gradients cannot flow through it.  
**Solution**: Rewrite as a deterministic function of parameters plus fixed noise:

```
z = mu + sigma * eps,     eps ~ N(0, I)
```

Gradients now flow through `mu` and `sigma` normally.

**Code** (`part1_vae.py`, lines 151-164):
```python
@staticmethod
def reparameterize(mu, log_var):
    std     = torch.exp(0.5 * log_var)    # sigma = exp(log_var / 2)
    epsilon = torch.randn_like(std)       # eps ~ N(0,I), detached from graph
    return mu + std * epsilon             # z = mu + sigma*eps  (differentiable)
```
`torch.randn_like(std)` creates eps with the same shape/device but no gradient connection.

---

### Step 3c — Decode: z -> x_hat_logits

Maps the latent vector back to pixel space, outputting **raw logits** (before sigmoid).

**Code — `VAE.decode`** (`part1_vae.py`, lines 166-176):
```python
def decode(self, z):
    h = self.relu(self.dec_hidden(z))
    return self.dec_out(h)       # raw logits (pre-sigmoid), shape (B, 784)
```
Sigmoid is applied explicitly only during inference/generation (not during training — the fused loss handles it).

---

### Step 3d — Full Forward Pass

**Code — `VAE.forward`** (`part1_vae.py`, lines 178-192):
```python
def forward(self, x):
    mu, log_var     = self.encode(x)                    # Step 3a
    z               = self.reparameterize(mu, log_var)  # Step 3b
    x_hat_logits    = self.decode(z)                    # Step 3c
    return x_hat_logits, mu, log_var
```

---

### Step 4 — Compute ELBO Loss

The VAE maximises the **Evidence Lower BOund (ELBO)**:

```
ELBO = E_{q(z|x)}[log p(x|z)]  -  KL(q(z|x) || p(z))
     = Reconstruction term       -  Regularisation term
```

**Reconstruction loss**: Binary cross-entropy via fused `BCEWithLogitsLoss` (numerically stable and AMP-safe).  
**KL divergence** (closed-form for two Gaussians):
```
KL = -0.5 * sum_j (1 + log_var_j - mu_j^2 - exp(log_var_j))
```

**Code** (`part1_vae.py`, lines 199-224):
```python
bce_logits_loss_fn = nn.BCEWithLogitsLoss(reduction="sum")

def vae_loss(x, x_hat_logits, mu, log_var, beta=1.0):
    batch_size = x.size(0)

    recon = bce_logits_loss_fn(x_hat_logits, x) / batch_size   # Reconstruction

    kl    = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp()) / batch_size  # KL

    total = recon + beta * kl   # beta controls KL weight
    return total, recon, kl
```

---

### Step 5 — Training Loop: KL Annealing + AMP + Early Stopping

#### KL Annealing (beta-VAE Schedule)

**Posterior collapse**: early in training, if KL is large, the encoder collapses to the prior (mu->0, sigma->1) and the decoder ignores z entirely. Solution: start with `beta = 0` (pure reconstruction) and linearly increase to `beta = 1`.

**Code** (`part1_vae.py`, lines 249-251):
```python
beta = min(1.0, epoch / KL_ANNEAL_EPOCHS)
# epoch=1 -> beta=0.1,  epoch=10+ -> beta=1.0
```

#### Mixed-Precision (AMP) Training

`torch.amp.autocast` runs the forward pass in float16. The `GradScaler` prevents float16 underflow in gradients.

**Code** (`part1_vae.py`, lines 260-268):
```python
with torch.amp.autocast("cuda", dtype=torch.float16):
    x_hat, mu, log_var = model(x)
    loss, recon, kl    = vae_loss(x, x_hat, mu, log_var, beta)

optimizer.zero_grad(set_to_none=True)
scaler.scale(loss).backward()   # scale loss -> prevent float16 underflow
scaler.step(optimizer)          # unscale gradients -> check inf/NaN -> step
scaler.update()                 # update scaler factor
```

#### Early Stopping

Begins **after** the KL annealing phase (during annealing beta changes, making cross-epoch loss comparison invalid).

**Code** (`part1_vae.py`, lines 306-322):
```python
if epoch > KL_ANNEAL_EPOCHS:
    if avg_total < best_loss:
        best_loss        = avg_total
        patience_counter = 0
        torch.save(model.state_dict(), best_model_path)
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PATIENCE:
            break
else:
    best_loss = avg_total              # during annealing: always save
    torch.save(model.state_dict(), best_model_path)
```

---

### Step 6 — Generation: Sample from Prior

**Code** (`part1_vae.py`, lines 402-405):
```python
z       = torch.randn(64, Z_DIM, device=device)          # 64 samples from N(0,I)
samples = torch.sigmoid(model.decode(z)).view(-1, 1, 28, 28).cpu()
# sigmoid converts raw logits to pixel probabilities in (0,1)
```

### Step 7 — Latent Space Visualisation

**PCA of latent means** (`part1_vae.py`, lines 419-458): encode all 10 000 test images, project their mu vectors to 2D via PCA, colour by digit label.

**Latent interpolation** (`part1_vae.py`, lines 462-515):
```python
mu_a, _ = model.encode(img_a)
mu_b, _ = model.encode(img_b)

for alpha in np.linspace(0, 1, num_steps):
    z_interp = (1 - alpha) * mu_a + alpha * mu_b     # straight line in latent space Z
    decoded  = torch.sigmoid(model.decode(z_interp))
```
Smooth morphing demonstrates the continuity of the learned latent space.

---

## Part 2 — CVAE Pipeline

The CVAE extends the VAE by conditioning every step on a class label. The label is encoded as a **one-hot vector** of length 10 and concatenated to the encoder input and decoder input.

---

### One-Hot Encoding of Labels

**Code** (`part2_cvae.py`, lines 111-125):
```python
def one_hot(labels, num_classes=NUM_CLASSES, device="cuda"):
    return torch.zeros(labels.size(0), num_classes, device=device).scatter_(
        1, labels.unsqueeze(1), 1   # place 1 at position labels[i] in row i
    )
```

---

### Conditional Architecture — Key Structural Change

| Component | VAE input | CVAE input | Size difference |
|---|---|---|---|
| `enc_hidden` | `x` (784) | `x concat one_hot(y)` (794) | +10 |
| `dec_hidden` | `z` (20) | `z concat one_hot(y)` (30) | +10 |

**Code — `CVAE.__init__`** (`part2_cvae.py`, lines 154-161):
```python
self.enc_hidden = nn.Linear(input_dim + num_classes, h_dim)  # 784+10=794 -> 400
self.dec_hidden = nn.Linear(z_dim + num_classes, h_dim)      # 20+10=30 -> 400
```

**Code — `CVAE.encode`** (`part2_cvae.py`, lines 163-179):
```python
def encode(self, x, y_onehot):
    inp = torch.cat([x, y_onehot], dim=1)   # (B, 794)
    h   = self.relu(self.enc_hidden(inp))
    return self.enc_mu(h), self.enc_log_var(h)
```

**Code — `CVAE.decode`** (`part2_cvae.py`, lines 196-209):
```python
def decode(self, z, y_onehot):
    inp = torch.cat([z, y_onehot], dim=1)   # (B, 30)
    h   = self.relu(self.dec_hidden(inp))
    return self.dec_out(h)
```

**Code — `CVAE.forward`** (`part2_cvae.py`, lines 211-227):
```python
def forward(self, x, y_onehot):
    mu, log_var  = self.encode(x, y_onehot)
    z            = self.reparameterize(mu, log_var)
    x_hat_logits = self.decode(z, y_onehot)
    return x_hat_logits, mu, log_var
```

---

### CVAE Training Loop — Label Passed at Every Step

**Code** (`part2_cvae.py`, lines 291-299):
```python
for i, (x, y) in enumerate(loop):
    x        = x.view(-1, INPUT_DIM).to(device, non_blocking=True)
    y_onehot = one_hot(y.to(device), device=device)

    with torch.amp.autocast("cuda", dtype=torch.float16):
        x_hat, mu, log_var = model(x, y_onehot)   # label passed at every step
        loss, recon, kl    = cvae_loss(x, x_hat, mu, log_var, beta)
```
The ELBO loss formula is identical to the VAE — conditioning does not change the loss form.

---

### Conditional Generation — Specific Digit on Demand

**Code** (`part2_cvae.py`, lines 442-449):
```python
for digit in range(NUM_CLASSES):         # digit = 0,1,...,9
    z        = torch.randn(8, Z_DIM, device=device)
    labels   = torch.full((8,), digit, dtype=torch.long, device=device)
    y_onehot = one_hot(labels, device=device)
    samples  = torch.sigmoid(model.decode(z, y_onehot)).view(-1, 1, 28, 28)
```
Keeping `y` fixed and varying `z` gives diversity within a class. Varying `y` gives different digit shapes.

---

### CVAE Latent Interpolation — Interpolating the Label Too

**Code** (`part2_cvae.py`, lines 509-513):
```python
for step, alpha in enumerate(np.linspace(0, 1, num_steps)):
    z_interp = (1 - alpha) * mu_1 + alpha * mu_7    # interpolate latent code
    y_interp = (1 - alpha) * y_oh_1 + alpha * y_oh_7  # interpolate class label
    decoded  = torch.sigmoid(model.decode(z_interp, y_interp))
```
A soft one-hot produces an intermediate class conditioning — the decoder morphs between digit shapes.

---

## VAE vs CVAE — Side-by-Side Comparison

| Aspect | VAE | CVAE |
|---|---|---|
| What does it learn? | Any digit from noise | Specific digit from noise + label |
| Encoder input | `x` (784) | `x concat one_hot(y)` (794) |
| Decoder input | `z` (20) | `z concat one_hot(y)` (30) |
| `enc_hidden` | `Linear(784, 400)` | `Linear(794, 400)` |
| `dec_hidden` | `Linear(20, 400)` | `Linear(30, 400)` |
| Generation | Uncontrolled | Controlled by label |
| Interpolation | Interpolate `z` only | Interpolate `z` AND `y` |

---

## Viva Q&A — Assignment 3

| Question | Concise Answer | Code location |
|---|---|---|
| What is the task? | Implement VAE (unsupervised generation) and CVAE (class-conditional generation) on MNIST | `part1_vae.py`, `part2_cvae.py` |
| What is the ELBO? | Evidence Lower BOund = Reconstruction loss + KL divergence | `part1_vae.py` L199-224 |
| Why reparameterize? | Sampling breaks gradient flow — mu + sigma*eps makes it differentiable | `part1_vae.py` L151-164 |
| Why `log_var` not `var`? | log can be any real -> no positivity constraint; sigma^2 = exp(log_var) always positive | `part1_vae.py` L148 |
| What is posterior collapse? | Encoder collapses to prior (KL->0), decoder ignores z — prevented by KL annealing | `part1_vae.py` L249-251 |
| What does KL annealing do? | Ramps beta 0->1 — starts as autoencoder, gradually becomes full VAE | `part1_vae.py` L249-251 |
| Why `BCEWithLogitsLoss`? | Fused sigmoid+log is numerically stable and safe under float16 AMP | `part1_vae.py` L199-221 |
| What does GradScaler do? | Scales loss before backprop to prevent float16 underflow; unscales before optimizer step | `part1_vae.py` L266-268 |
| Why early stopping only after annealing? | beta changes during annealing, making cross-epoch loss comparison invalid | `part1_vae.py` L306-322 |
| How does CVAE differ structurally? | Label one-hot concatenated to encoder AND decoder input; those layers are 10 units wider | `part2_cvae.py` L154-161 |
| How do you generate a specific digit with CVAE? | Sample z~N(0,I), set y_onehot to desired digit, call `decode(z, y_onehot)` | `part2_cvae.py` L442-449 |
| What does interpolating the label do? | Soft one-hot creates intermediate class conditioning — decoder morphs between digits | `part2_cvae.py` L509-513 |

---

# Quick Concept-to-File Reference

| Concept | File | Lines |
|---|---|---|
| **Assignment 1** | | |
| GBT Regression init (F_0 = mean(y)) | `Assignment1/src/gbt/core.py` | 127-129 |
| GBT Classification init (log-odds) | `Assignment1/src/gbt/core.py` | 273-279 |
| MSE pseudo-residuals | `Assignment1/src/gbt/utils.py` | 25-32 |
| Logistic pseudo-residuals | `Assignment1/src/gbt/utils.py` | 61-74 |
| MSE leaf gamma (mean) | `Assignment1/src/gbt/utils.py` | 35-43 |
| Logistic leaf gamma (Newton step) | `Assignment1/src/gbt/utils.py` | 77-109 |
| Gamma map construction | `Assignment1/src/gbt/core.py` | 160-167 |
| Model update with shrinkage | `Assignment1/src/gbt/core.py` | 169-176 |
| Row subsampling | `Assignment1/src/gbt/core.py` | 79-86 |
| Predict (accumulate trees) | `Assignment1/src/gbt/core.py` | 199-222 |
| Sigmoid utility | `Assignment1/src/gbt/utils.py` | 112-118 |
| **Assignment 2** | | |
| Adaptive jitter Cholesky | `Assignment2/src/gp/base.py` | 39-54 |
| Cholesky solve (K^{-1}*b) | `Assignment2/src/gp/base.py` | 57-79 |
| Log marginal likelihood | `Assignment2/src/gp/base.py` | 82-124 |
| RBF kernel | `Assignment2/src/gp/kernels.py` | 68-137 |
| Matern kernel | `Assignment2/src/gp/kernels.py` | 140-232 |
| Rational Quadratic kernel | `Assignment2/src/gp/kernels.py` | 235-316 |
| Hyperparams in log space | `Assignment2/src/gp/kernels.py` | 111-118 |
| Compute alpha | `Assignment2/src/gp/regression.py` | 117-131 |
| GP predict (posterior mean+var) | `Assignment2/src/gp/regression.py` | 133-191 |
| Hyperparameter optimisation | `Assignment2/src/gp/regression.py` | 236-302 |
| Newton loop (Alg 3.1) | `Assignment2/src/gp/classification.py` | 121-162 |
| Gradient + Hessian of log-lik | `Assignment2/src/gp/classification.py` | 296-339 |
| Predictive probability (Alg 3.2) | `Assignment2/src/gp/classification.py` | 180-231 |
| **Assignment 3** | | |
| VAE architecture | `Assignment3/part1_vae.py` | 109-192 |
| CVAE architecture | `Assignment3/part2_cvae.py` | 131-228 |
| One-hot encoding (CVAE) | `Assignment3/part2_cvae.py` | 111-125 |
| Encode step (VAE) | `Assignment3/part1_vae.py` | 136-149 |
| Encode step (CVAE) | `Assignment3/part2_cvae.py` | 163-179 |
| Reparameterization trick | `Assignment3/part1_vae.py` | 151-164 |
| Decode step (VAE) | `Assignment3/part1_vae.py` | 166-176 |
| Decode step (CVAE) | `Assignment3/part2_cvae.py` | 196-209 |
| ELBO loss (VAE) | `Assignment3/part1_vae.py` | 199-224 |
| ELBO loss (CVAE) | `Assignment3/part2_cvae.py` | 237-259 |
| KL annealing | `Assignment3/part1_vae.py` | 249-251 |
| AMP training step | `Assignment3/part1_vae.py` | 260-268 |
| Early stopping | `Assignment3/part1_vae.py` | 306-322 |
| Generation from prior (VAE) | `Assignment3/part1_vae.py` | 402-405 |
| Conditional generation (CVAE) | `Assignment3/part2_cvae.py` | 442-449 |
| Latent interpolation (VAE) | `Assignment3/part1_vae.py` | 462-515 |
| Latent interpolation + label (CVAE) | `Assignment3/part2_cvae.py` | 473-527 |
