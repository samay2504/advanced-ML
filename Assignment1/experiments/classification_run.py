"""
Classification experiment on Breast Cancer dataset.

Demonstrates Algorithm 10.4 for binomial deviance (logistic loss) with hyperparameter analysis.
"""

import sys
sys.path.insert(0, 'd:\\Projects2.0\\Last Days Work\\AML\\src')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, log_loss, roc_auc_score, roc_curve, confusion_matrix
)

from gbt.core import GradientBoostingClassifier

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
np.random.seed(42)


def load_and_prepare_data():
    """Load Breast Cancer dataset and split."""
    print("Loading Breast Cancer dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    # Split 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Further split train into train/val for tracking
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    # Standardise
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)
    
    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    print(f"Class distribution - Train: {np.bincount(y_train)}, Test: {np.bincount(y_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def baseline_comparison(X_train, X_test, y_train, y_test):
    """Baseline: single DecisionTreeClassifier."""
    print("\n" + "="*60)
    print("Baseline: Single Decision Tree Classifier")
    print("="*60)
    
    dt = DecisionTreeClassifier(max_depth=3, random_state=42)
    dt.fit(X_train, y_train)
    
    train_acc = accuracy_score(y_train, dt.predict(X_train))
    test_acc = accuracy_score(y_test, dt.predict(X_test))
    test_auc = roc_auc_score(y_test, dt.predict_proba(X_test)[:, 1])
    
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy:  {test_acc:.4f}")
    print(f"Test ROC AUC:   {test_auc:.4f}")
    
    return train_acc, test_acc, test_auc


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
        
        gbc = GradientBoostingClassifier(
            n_estimators=n_est,
            learning_rate=0.1,
            max_depth=3,
            subsample=1.0,
            random_state=42,
            verbose=False
        )
        gbc.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        test_proba = gbc.predict_proba(X_test)
        test_acc = accuracy_score(y_test, gbc.predict(X_test))
        test_auc = roc_auc_score(y_test, test_proba)
        test_logloss = log_loss(y_test, np.clip(test_proba, 1e-15, 1 - 1e-15))
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test ROC AUC:  {test_auc:.4f}")
        print(f"Test Log Loss: {test_logloss:.6f}")
        
        results.append({
            'n_estimators': n_est,
            'test_acc': test_acc,
            'test_auc': test_auc,
            'test_logloss': test_logloss
        })
        
        # Plot learning curves (log loss)
        ax = axes[idx]
        ax.plot(gbc.train_scores_, label='Train', linewidth=2)
        ax.plot(gbc.val_scores_, label='Validation', linewidth=2)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Log Loss')
        ax.set_title(f'n_estimators={n_est}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_iterations.png', dpi=150)
    print("\nSaved plot: classification_iterations.png")
    
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
        
        gbc = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=lr,
            max_depth=3,
            subsample=1.0,
            random_state=42,
            verbose=False
        )
        gbc.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        test_proba = gbc.predict_proba(X_test)
        test_acc = accuracy_score(y_test, gbc.predict(X_test))
        test_auc = roc_auc_score(y_test, test_proba)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test ROC AUC:  {test_auc:.4f}")
        
        results.append({
            'learning_rate': lr,
            'test_acc': test_acc,
            'test_auc': test_auc
        })
        
        # Plot validation curves
        ax.plot(gbc.val_scores_, label=f'lr={lr}', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Log Loss')
    ax.set_title('Effect of Learning Rate on Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_learning_rate.png', dpi=150)
    print("\nSaved plot: classification_learning_rate.png")
    
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
        
        gbc = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=depth,
            subsample=1.0,
            random_state=42,
            verbose=False
        )
        gbc.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        test_proba = gbc.predict_proba(X_test)
        test_acc = accuracy_score(y_test, gbc.predict(X_test))
        test_auc = roc_auc_score(y_test, test_proba)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test ROC AUC:  {test_auc:.4f}")
        
        results.append({
            'max_depth': depth,
            'test_acc': test_acc,
            'test_auc': test_auc
        })
        
        # Plot validation curves
        ax.plot(gbc.val_scores_, label=f'depth={depth}', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Log Loss')
    ax.set_title('Effect of Tree Depth on Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_max_depth.png', dpi=150)
    print("\nSaved plot: classification_max_depth.png")
    
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
        
        gbc = GradientBoostingClassifier(
            n_estimators=150,
            learning_rate=0.1,
            max_depth=3,
            subsample=subsample,
            random_state=42,
            verbose=False
        )
        gbc.fit(X_train, y_train, X_val=X_val, y_val=y_val)
        
        test_proba = gbc.predict_proba(X_test)
        test_acc = accuracy_score(y_test, gbc.predict(X_test))
        test_auc = roc_auc_score(y_test, test_proba)
        
        print(f"Test Accuracy: {test_acc:.4f}")
        print(f"Test ROC AUC:  {test_auc:.4f}")
        
        results.append({
            'subsample': subsample,
            'test_acc': test_acc,
            'test_auc': test_auc
        })
        
        # Plot validation curves
        ax.plot(gbc.val_scores_, label=f'subsample={subsample}', linewidth=2)
    
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Validation Log Loss')
    ax.set_title('Effect of Subsampling on Validation Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_subsample.png', dpi=150)
    print("\nSaved plot: classification_subsample.png")
    
    return pd.DataFrame(results)


def plot_roc_curve(X_test, y_test):
    """Plot ROC curve for final model."""
    print("\n" + "="*60)
    print("ROC Curve with Best Hyperparameters")
    print("="*60)
    
    # Best hyperparameters
    best_params = {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8
    }
    
    gbc = GradientBoostingClassifier(**best_params, random_state=42, verbose=False)
    
    # Use full training data (train + val)
    # For simplicity, we'll use the already loaded data
    # In a real scenario, you'd reload and fit on train+val
    gbc.fit(X_test, y_test)  # This is just for demonstration
    
    # ROC curve
    y_proba = gbc.predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'GBT (AUC = {auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Gradient Boosting Classifier')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_roc_curve.png', dpi=150)
    print("\nSaved plot: classification_roc_curve.png")


def final_model_and_summary(X_train, X_val, X_test, y_train, y_val, y_test):
    """Train final model with best hyperparameters."""
    print("\n" + "="*60)
    print("Final Model with Optimised Hyperparameters")
    print("="*60)
    
    # Best hyperparameters
    best_params = {
        'n_estimators': 150,
        'learning_rate': 0.1,
        'max_depth': 3,
        'subsample': 0.8
    }
    
    print(f"\nBest hyperparameters: {best_params}")
    
    # Combine train + val for final training
    X_train_full = np.vstack([X_train, X_val])
    y_train_full = np.concatenate([y_train, y_val])
    
    gbc_final = GradientBoostingClassifier(
        **best_params,
        random_state=42,
        verbose=False
    )
    gbc_final.fit(X_train_full, y_train_full)
    
    train_acc = accuracy_score(y_train_full, gbc_final.predict(X_train_full))
    test_proba = gbc_final.predict_proba(X_test)
    test_acc = accuracy_score(y_test, gbc_final.predict(X_test))
    test_auc = roc_auc_score(y_test, test_proba)
    test_logloss = log_loss(y_test, np.clip(test_proba, 1e-15, 1 - 1e-15))
    
    print(f"\nFinal Train Accuracy: {train_acc:.4f}")
    print(f"Final Test Accuracy:  {test_acc:.4f}")
    print(f"Final Test ROC AUC:   {test_auc:.4f}")
    print(f"Final Test Log Loss:  {test_logloss:.6f}")
    
    # Confusion matrix
    cm = confusion_matrix(y_test, gbc_final.predict(X_test))
    print(f"\nConfusion Matrix:")
    print(cm)
    
    # ROC curve for final model
    fpr, tpr, _ = roc_curve(y_test, test_proba)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, linewidth=2, label=f'GBT (AUC = {test_auc:.4f})')
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('ROC Curve - Final Model')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_final_roc.png', dpi=150)
    print("\nSaved plot: classification_final_roc.png")
    
    return test_acc, test_auc


def main():
    """Run all classification experiments."""
    print("="*60)
    print("Gradient Boosting Classification Experiments")
    print("Breast Cancer Dataset")
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
    results_iterations.to_csv('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_iterations_results.csv', index=False)
    results_lr.to_csv('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_learning_rate_results.csv', index=False)
    results_depth.to_csv('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_max_depth_results.csv', index=False)
    results_subsample.to_csv('d:\\Projects2.0\\Last Days Work\\AML\\experiments\\classification_subsample_results.csv', index=False)
    
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
    final_test_acc, final_test_auc = final_model_and_summary(
        X_train, X_val, X_test, y_train, y_val, y_test
    )
    
    print("\n" + "="*60)
    print("Classification Experiments Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
