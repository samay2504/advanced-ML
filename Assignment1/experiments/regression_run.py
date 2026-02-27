"""
Regression experiment on California Housing dataset.

Demonstrates Algorithm 10.4 for squared-error loss with hyperparameter analysis.
"""

import sys
sys.path.insert(0, 'd:\\Projects2.0\\Last Days Work\\AML\\src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error

from gbt.core import GradientBoostingRegressor

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)


def load_and_prepare_data():
    """Load California Housing dataset and split."""
    print("Loading California Housing dataset...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Further split train into train/val for tracking
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    # Standardise (helps with convergence)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def baseline_comparison(X_train, X_test, y_train, y_test):
    """Baseline: single DecisionTreeRegressor."""
    print("\n" + "="*60)
    print("Baseline: Single Decision Tree Regressor")
    print("="*60)
    
    dt = DecisionTreeRegressor(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    
    train_mse = mean_squared_error(y_train, dt.predict(X_train))
    test_mse = mean_squared_error(y_test, dt.predict(X_test))
    
    print(f"Train MSE: {train_mse:.6f}")
    print(f"Test MSE:  {test_mse:.6f}")
    
    return train_mse, test_mse


def experiment_iterations(X_train, X_val, X_test, y_train, y_val, y_test):
    """Experiment: effect of number of iterations."""
    print("\n" + "="*60)
    print("Experiment 1: Effect of n_estimators")
    print("="*60)
    
    n_estimators_list = [50, 100, 300]
    results = []
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for idx, n_est in enumerate(n_estimators_list):
        print(f"\nFitting with n_estimators={n_est}...")
        
        gbr = GradientBoostingRegressor(
            n_estimators=n_est,
            learning_rate=0.1,
            max_depth=3,
            subsample=1.0,
            random_state=42,
            verbose=False
        )
        gbr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        test_mse = mean_squared_error(y_test, gbr.predict(X_test))
        print(f"Test MSE: {test_mse:.6f}")
        
        results.append({
            'n_estimators': n_est,
            'test_mse': test_mse,
            'final_train_mse': gbr.train_scores_[-1],
            'final_val_mse': gbr.val_scores_[-1]
        })
        
        # Plot learning curves
        ax = axes[idx]
        ax.plot(gbr.train_scores_, label='Train', linewidth=2)
        ax.plot(gbr.val_scores_, label='Validation', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('MSE')
        ax.set_title(f'n_estimators={n_est}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_iterations.png', dpi=150)
    print("\nSaved plot: regression_iterations.png")
    
    return pd.DataFrame(results)


def experiment_learning_rate(X_train, X_val, X_test, y_train, y_val, y_test):
    """Experiment: effect of learning rate (shrinkage)."""
    print("\n" + "="*60)
    print("Experiment 2: Effect of learning_rate")
    print("="*60)
    
    learning_rates = [0.01, 0.1, 0.2]
    results = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for lr in learning_rates:
        print(f"\nFitting with learning_rate={lr}...")
        
        gbr = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=lr,
            max_depth=3,
            subsample=1.0,
            random_state=42,
            verbose=False
        )
        gbr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        test_mse = mean_squared_error(y_test, gbr.predict(X_test))
        print(f"Test MSE: {test_mse:.6f}")
        
        results.append({
            'learning_rate': lr,
            'test_mse': test_mse
        })
        
        # Plot validation curves
        ax.plot(gbr.val_scores_, label=f'lr={lr}', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Effect of Learning Rate on Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_learning_rate.png', dpi=150)
    print("\nSaved plot: regression_learning_rate.png")
    
    return pd.DataFrame(results)


def experiment_max_depth(X_train, X_val, X_test, y_train, y_val, y_test):
    """Experiment: effect of tree complexity (max_depth)."""
    print("\n" + "="*60)
    print("Experiment 3: Effect of max_depth")
    print("="*60)
    
    max_depths = [2, 3, 5]
    results = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for depth in max_depths:
        print(f"\nFitting with max_depth={depth}...")
        
        gbr = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=depth,
            subsample=1.0,
            random_state=42,
            verbose=False
        )
        gbr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        test_mse = mean_squared_error(y_test, gbr.predict(X_test))
        print(f"Test MSE: {test_mse:.6f}")
        
        results.append({
            'max_depth': depth,
            'test_mse': test_mse
        })
        
        # Plot validation curves
        ax.plot(gbr.val_scores_, label=f'depth={depth}', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Effect of Tree Depth on Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_max_depth.png', dpi=150)
    print("\nSaved plot: regression_max_depth.png")
    
    return pd.DataFrame(results)


def experiment_subsample(X_train, X_val, X_test, y_train, y_val, y_test):
    """Experiment: effect of stochastic subsampling."""
    print("\n" + "="*60)
    print("Experiment 4: Effect of subsample (Stochastic Boosting)")
    print("="*60)
    
    subsamples = [0.5, 0.8, 1.0]
    results = []
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for subsample in subsamples:
        print(f"\nFitting with subsample={subsample}...")
        
        gbr = GradientBoostingRegressor(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            subsample=subsample,
            random_state=42,
            verbose=False
        )
        gbr.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        test_mse = mean_squared_error(y_test, gbr.predict(X_test))
        print(f"Test MSE: {test_mse:.6f}")
        
        results.append({
            'subsample': subsample,
            'test_mse': test_mse
        })
        
        # Plot validation curves
        ax.plot(gbr.val_scores_, label=f'subsample={subsample}', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation MSE')
    ax.set_title('Effect of Subsampling on Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_subsample.png', dpi=150)
    print("\nSaved plot: regression_subsample.png")
    
    return pd.DataFrame(results)


def final_model_and_summary(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train final model with best hyperparameters."""
    print("\n" + "="*60)
    print("Final Model with Optimised Hyperparameters")
    print("="*60)
    
    # Best hyperparameters (from experiments above)
    best_params = {
        'n_estimators': 200,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8
    }
    
    print(f"\nBest hyperparameters: {best_params}")
    
    # Combine train + val for final training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    gbr_final = GradientBoostingRegressor(
        **best_params,
        random_state=42,
        verbose=False
    )
    gbr_final.fit(X_train_full, y_train_full)
    
    train_mse = mean_squared_error(y_train_full, gbr_final.predict(X_train_full))
    test_mse = mean_squared_error(y_test, gbr_final.predict(X_test))
    
    print(f"\nFinal Train MSE: {train_mse:.6f}")
    print(f"Final Test MSE:  {test_mse:.6f}")
    
    return test_mse


def main():
    """Run all regression experiments."""
    print("="*60)
    print("Gradient Boosting Regression Experiments")
    print("California Housing Dataset")
    print("="*60)
    
    # Load data
    X_train, X_val, X_test, y_train, y_val, y_test = load_and_prepare_data()
    
    # Baseline
    baseline_comparison(X_train, X_test, y_train, y_test)
    
    # Experiments
    results_iterations = experiment_iterations(X_train, X_val, X_test, y_train, y_val, y_test)
    results_lr = experiment_learning_rate(X_train, X_val, X_test, y_train, y_val, y_test)
    results_depth = experiment_max_depth(X_train, X_val, X_test, y_train, y_val, y_test)
    results_subsample = experiment_subsample(X_train, X_val, X_test, y_train, y_val, y_test)
    
    # Save results
    results_iterations.to_csv('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_iterations_results.csv', index=False)
    results_lr.to_csv('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_learning_rate_results.csv', index=False)
    results_depth.to_csv('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_max_depth_results.csv', index=False)
    results_subsample.to_csv('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_subsample_results.csv', index=False)
    
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    print("\nEffect of n_estimators:")
    print(results_iterations.to_string(index=False))
    print("\nEffect of learning_rate:")
    print(results_lr.to_string(index=False))
    print("\nEffect of max_depth:")
    print(results_depth.to_string(index=False))
    print("\nEffect of subsample:")
    print(results_subsample.to_string(index=False))
    
    # Final model
    final_test_mse = final_model_and_summary(X_train, X_val, X_test, y_train, y_val, y_test)
    
    print("\n" + "="*60)
    print("Regression Experiments Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
