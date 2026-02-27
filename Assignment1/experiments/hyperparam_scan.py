"""
Comprehensive hyperparameter scan for gradient boosting.

Performs grid search over key hyperparameters and saves results.
"""

import sys
sys.path.insert(0, 'd:\\Projects2.0\\Last Days Work\\AML\\src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from itertools import product
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, roc_auc_score

from gbt.core import GradientBoostingRegressor, GradientBoostingClassifier

np.random.seed(42)


def prepare_regression_data():
    """Load and prepare California housing data."""
    print("Loading regression data...")
    data = fetch_california_housing()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def prepare_classification_data():
    """Load and prepare breast cancer data."""
    print("Loading classification data...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    return X_train, X_test, y_train, y_test


def regression_grid_search():
    """Grid search for regression."""
    print("\n" + "="*60)
    print("Hyperparameter Grid Search - Regression")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_regression_data()
    
    # Define grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [2, 3, 5],
        'subsample': [0.5, 0.8, 1.0]
    }
    
    results = []
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    
    print(f"\nTotal combinations: {total_combinations}")
    print("Running grid search...")
    
    combo_idx = 0
    for n_est, lr, depth, subsample in product(
        param_grid['n_estimators'],
        param_grid['learning_rate'],
        param_grid['max_depth'],
        param_grid['subsample']
    ):
        combo_idx += 1
        print(f"\n[{combo_idx}/{total_combinations}] Testing: "
              f"n_est={n_est}, lr={lr}, depth={depth}, subsample={subsample}")
        
        try:
            gbr = GradientBoostingRegressor(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth,
                subsample=subsample,
                random_state=42,
                verbose=False
            )
            gbr.fit(X_train, y_train)
            
            train_mse = mean_squared_error(y_train, gbr.predict(X_train))
            test_mse = mean_squared_error(y_test, gbr.predict(X_test))
            
            results.append({
                'n_estimators': n_est,
                'learning_rate': lr,
                'max_depth': depth,
                'subsample': subsample,
                'train_mse': train_mse,
                'test_mse': test_mse
            })
            
            print(f"  Train MSE: {train_mse:.6f}, Test MSE: {test_mse:.6f}")
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_mse')
    
    # Save results
    output_path = 'd:\\Projects2.0\\Last Days Work\\AML\\experiments\\regression_grid_search.csv'
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")
    
    print("\n" + "="*60)
    print("Top 10 Configurations (by Test MSE)")
    print("="*60)
    print(df_results.head(10).to_string(index=False))
    
    return df_results


def classification_grid_search():
    """Grid search for classification."""
    print("\n" + "="*60)
    print("Hyperparameter Grid Search - Classification")
    print("="*60)
    
    X_train, X_test, y_train, y_test = prepare_classification_data()
    
    # Define grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'learning_rate': [0.01, 0.1, 0.2],
        'max_depth': [2, 3, 5],
        'subsample': [0.5, 0.8, 1.0]
    }
    
    results = []
    total_combinations = np.prod([len(v) for v in param_grid.values()])
    
    print(f"\nTotal combinations: {total_combinations}")
    print("Running grid search...")
    
    combo_idx = 0
    for n_est, lr, depth, subsample in product(
        param_grid['n_estimators'],
        param_grid['learning_rate'],
        param_grid['max_depth'],
        param_grid['subsample']
    ):
        combo_idx += 1
        print(f"\n[{combo_idx}/{total_combinations}] Testing: "
              f"n_est={n_est}, lr={lr}, depth={depth}, subsample={subsample}")
        
        try:
            gbc = GradientBoostingClassifier(
                n_estimators=n_est,
                learning_rate=lr,
                max_depth=depth,
                subsample=subsample,
                random_state=42,
                verbose=False
            )
            gbc.fit(X_train, y_train)
            
            train_acc = accuracy_score(y_train, gbc.predict(X_train))
            test_proba = gbc.predict_proba(X_test)
            test_acc = accuracy_score(y_test, gbc.predict(X_test))
            test_auc = roc_auc_score(y_test, test_proba)
            
            results.append({
                'n_estimators': n_est,
                'learning_rate': lr,
                'max_depth': depth,
                'subsample': subsample,
                'train_acc': train_acc,
                'test_acc': test_acc,
                'test_auc': test_auc
            })
            
            print(f"  Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}, AUC: {test_auc:.4f}")
        
        except Exception as e:
            print(f"  Error: {e}")
            continue
    
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('test_auc', ascending=False)
    
    # Save results
    output_path = 'd:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_grid_search.csv'
    df_results.to_csv(output_path, index=False)
    print(f"\nSaved results to: {output_path}")
    
    print("\n" + "="*60)
    print("Top 10 Configurations (by Test AUC)")
    print("="*60)
    print(df_results.head(10).to_string(index=False))
    
    return df_results


def plot_hyperparameter_effects(df_reg, df_clf):
    """Create visualisations of hyperparameter effects."""
    print("\n" + "="*60)
    print("Creating Hyperparameter Effect Plots")
    print("="*60)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    
    # Regression plots
    hyperparams = ['n_estimators', 'learning_rate', 'max_depth', 'subsample']
    
    for idx, param in enumerate(hyperparams):
        ax = axes[0, idx]
        grouped = df_reg.groupby(param)['test_mse'].agg(['mean', 'std'])
        
        ax.errorbar(
            grouped.index, grouped['mean'], yerr=grouped['std'],
            marker='o', capsize=5, linewidth=2, markersize=8
        )
        ax.set_xlabel(param)
        ax.set_ylabel('Test MSE')
        ax.set_title(f'Regression: {param} Effect')
        ax.grid(True, alpha=0.3)
    
    # Classification plots
    for idx, param in enumerate(hyperparams):
        ax = axes[1, idx]
        grouped = df_clf.groupby(param)['test_auc'].agg(['mean', 'std'])
        
        ax.errorbar(
            grouped.index, grouped['mean'], yerr=grouped['std'],
            marker='o', capsize=5, linewidth=2, markersize=8
        )
        ax.set_xlabel(param)
        ax.set_ylabel('Test AUC')
        ax.set_title(f'Classification: {param} Effect')
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\hyperparameter_effects.png', dpi=150)
    print("\nSaved plot: hyperparameter_effects.png")


def main():
    """Run comprehensive hyperparameter scan."""
    print("="*60)
    print("Comprehensive Hyperparameter Scan")
    print("="*60)
    
    # Regression
    df_reg = regression_grid_search()
    
    # Classification
    df_clf = classification_grid_search()
    
    # Visualise effects
    plot_hyperparameter_effects(df_reg, df_clf)
    
    print("\n" + "="*60)
    print("Hyperparameter Scan Complete!")
    print("="*60)
    print("\nKey Findings:")
    print("\nRegression - Best Configuration:")
    print(df_reg.iloc[0].to_string())
    print("\nClassification - Best Configuration:")
    print(df_clf.iloc[0].to_string())


if __name__ == "__main__":
    main()
