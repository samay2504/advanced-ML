"""
Hyperparameter grid search for GP regression and classification.

Performs small grid search over kernel lengthscale and noise variance,
evaluating log marginal likelihood and predictive performance.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, log_loss

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gp.regression import GaussianProcessRegressor
from src.gp.classification import GaussianProcessClassifier
from src.gp.kernels import RBF

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def hyperparameter_scan_regression():
    """Grid search for GP regression hyperparameters."""
    
    logger.info("=" * 70)
    logger.info("REGRESSION HYPERPARAMETER SCAN")
    logger.info("=" * 70)
    
    # Load data
    logger.info("Loading diabetes dataset...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Use smaller subset for faster grid search
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    logger.info(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}")
    
    # Hyperparameter grid
    length_scales = [0.5, 1.0, 2.0, 5.0]
    noise_levels = [0.1, 1.0, 10.0, 50.0]
    
    results = []
    
    logger.info(f"\nGrid search: {len(length_scales)}×{len(noise_levels)} = {len(length_scales) * len(noise_levels)} combinations")
    
    for i, length_scale in enumerate(length_scales):
        for j, noise in enumerate(noise_levels):
            logger.info(f"Testing ℓ={length_scale:.2f}, σ_n²={noise:.2f}...")
            
            try:
                kernel = RBF(length_scale=length_scale, variance=100.0)
                gp = GaussianProcessRegressor(
                    kernel=kernel,
                    noise=noise,
                    optimise=False,  # Fixed hyperparameters
                    random_state=42,
                    verbose=False
                )
                
                gp.fit(X_train, y_train)
                
                # Compute metrics
                lml = gp.log_marginal_likelihood_value()
                
                y_pred_mean = gp.predict(X_val)
                rmse = np.sqrt(mean_squared_error(y_val, y_pred_mean))
                
                results.append({
                    'length_scale': length_scale,
                    'noise': noise,
                    'log_marginal_likelihood': lml,
                    'val_rmse': rmse
                })
                
                logger.info(f"  → LML: {lml:.2f}, Val RMSE: {rmse:.2f}")
                
            except Exception as e:
                logger.warning(f"  → Failed: {str(e)}")
                continue
    
    # Save results
    results_df = pd.DataFrame(results)
    
    return results_df


def hyperparameter_scan_classification():
    """Grid search for GP classification hyperparameters."""
    
    logger.info("\n" + "=" * 70)
    logger.info("CLASSIFICATION HYPERPARAMETER SCAN")
    logger.info("=" * 70)
    
    # Load data
    logger.info("Loading breast cancer dataset...")
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # Use smaller subset for faster grid search
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    logger.info(f"Training: {X_train.shape[0]}, Validation: {X_val.shape[0]}")
    
    # Hyperparameter grid
    length_scales = [1.0, 3.0, 5.0, 10.0]
    likelihoods = ['logistic', 'probit']
    
    results = []
    
    logger.info(f"\nGrid search: {len(length_scales)}×{len(likelihoods)} = {len(length_scales) * len(likelihoods)} combinations")
    
    for length_scale in length_scales:
        for likelihood in likelihoods:
            logger.info(f"Testing ℓ={length_scale:.2f}, likelihood={likelihood}...")
            
            try:
                kernel = RBF(length_scale=length_scale, variance=1.0)
                gp = GaussianProcessClassifier(
                    kernel=kernel,
                    likelihood=likelihood,
                    max_iter=50,
                    random_state=42,
                    verbose=False
                )
                
                gp.fit(X_train, y_train)
                
                # Compute metrics
                approx_lml = gp.log_marginal_likelihood_
                
                y_pred_proba = gp.predict_proba(X_val)
                logloss = log_loss(y_val, y_pred_proba)
                
                results.append({
                    'length_scale': length_scale,
                    'likelihood': likelihood,
                    'approx_log_marginal_likelihood': approx_lml,
                    'val_log_loss': logloss,
                    'n_iterations': gp.n_iter_
                })
                
                logger.info(f"  → Approx LML: {approx_lml:.2f}, Val Log Loss: {logloss:.3f}, Iters: {gp.n_iter_}")
                
            except Exception as e:
                logger.warning(f"  → Failed: {str(e)}")
                continue
    
    # Save results
    results_df = pd.DataFrame(results)
    
    return results_df


def plot_regression_results(results_df, output_dir):
    """Create visualisations for regression grid search."""
    
    # Pivot for heatmaps
    lml_pivot = results_df.pivot(index='noise', columns='length_scale', 
                                  values='log_marginal_likelihood')
    rmse_pivot = results_df.pivot(index='noise', columns='length_scale', 
                                   values='val_rmse')
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Log marginal likelihood heatmap
    ax = axes[0]
    im1 = ax.imshow(lml_pivot.values, cmap='viridis', aspect='auto')
    ax.set_xticks(range(len(lml_pivot.columns)))
    ax.set_yticks(range(len(lml_pivot.index)))
    ax.set_xticklabels([f'{v:.1f}' for v in lml_pivot.columns])
    ax.set_yticklabels([f'{v:.1f}' for v in lml_pivot.index])
    ax.set_xlabel('Length Scale ℓ', fontsize=12)
    ax.set_ylabel('Noise Variance σ_n²', fontsize=12)
    ax.set_title('Log Marginal Likelihood', fontsize=13, fontweight='bold')
    
    # Annotate values
    for i in range(len(lml_pivot.index)):
        for j in range(len(lml_pivot.columns)):
            val = lml_pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color='white' if val < lml_pivot.values.mean() else 'black', 
                       fontsize=9, weight='bold')
    
    plt.colorbar(im1, ax=ax, label='LML')
    
    # Validation RMSE heatmap
    ax = axes[1]
    im2 = ax.imshow(rmse_pivot.values, cmap='Reds', aspect='auto')
    ax.set_xticks(range(len(rmse_pivot.columns)))
    ax.set_yticks(range(len(rmse_pivot.index)))
    ax.set_xticklabels([f'{v:.1f}' for v in rmse_pivot.columns])
    ax.set_yticklabels([f'{v:.1f}' for v in rmse_pivot.index])
    ax.set_xlabel('Length Scale ℓ', fontsize=12)
    ax.set_ylabel('Noise Variance σ_n²', fontsize=12)
    ax.set_title('Validation RMSE', fontsize=13, fontweight='bold')
    
    # Annotate values
    for i in range(len(rmse_pivot.index)):
        for j in range(len(rmse_pivot.columns)):
            val = rmse_pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f'{val:.1f}', ha='center', va='center', 
                       color='white' if val > rmse_pivot.values.mean() else 'black', 
                       fontsize=9, weight='bold')
    
    plt.colorbar(im2, ax=ax, label='RMSE')
    
    plt.suptitle('GP Regression Hyperparameter Grid Search', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'hyperparam_scan_regression.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Regression plot saved to {plot_path}")
    plt.close()


def plot_classification_results(results_df, output_dir):
    """Create visualisations for classification grid search."""
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Group by likelihood
    for likelihood in ['logistic', 'probit']:
        subset = results_df[results_df['likelihood'] == likelihood]
        
        axes[0].plot(subset['length_scale'], subset['approx_log_marginal_likelihood'], 
                    'o-', linewidth=2, markersize=8, label=likelihood.capitalize())
        axes[1].plot(subset['length_scale'], subset['val_log_loss'], 
                    's-', linewidth=2, markersize=8, label=likelihood.capitalize())
    
    axes[0].set_xlabel('Length Scale ℓ', fontsize=12)
    axes[0].set_ylabel('Approx. Log Marginal Likelihood', fontsize=12)
    axes[0].set_title('Model Fit (Laplace Approx.)', fontsize=13, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Length Scale ℓ', fontsize=12)
    axes[1].set_ylabel('Validation Log Loss', fontsize=12)
    axes[1].set_title('Predictive Performance', fontsize=13, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle('GP Classification Hyperparameter Grid Search', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'hyperparam_scan_classification.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Classification plot saved to {plot_path}")
    plt.close()


def main():
    """Run hyperparameter grid search experiments."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    # Regression scan
    reg_results = hyperparameter_scan_regression()
    
    # Save results
    reg_csv_path = os.path.join(output_dir, 'hyperparam_scan_regression.csv')
    reg_results.to_csv(reg_csv_path, index=False)
    logger.info(f"\nRegression results saved to {reg_csv_path}")
    
    # Plot
    plot_regression_results(reg_results, output_dir)
    
    # Print best configuration
    best_lml_idx = reg_results['log_marginal_likelihood'].idxmax()
    best_rmse_idx = reg_results['val_rmse'].idxmin()
    
    logger.info("\n" + "=" * 70)
    logger.info("REGRESSION: BEST HYPERPARAMETERS")
    logger.info("=" * 70)
    logger.info("By Log Marginal Likelihood:")
    logger.info(f"  ℓ={reg_results.loc[best_lml_idx, 'length_scale']:.2f}, "
               f"σ_n²={reg_results.loc[best_lml_idx, 'noise']:.2f}, "
               f"LML={reg_results.loc[best_lml_idx, 'log_marginal_likelihood']:.2f}")
    logger.info("By Validation RMSE:")
    logger.info(f"  ℓ={reg_results.loc[best_rmse_idx, 'length_scale']:.2f}, "
               f"σ_n²={reg_results.loc[best_rmse_idx, 'noise']:.2f}, "
               f"RMSE={reg_results.loc[best_rmse_idx, 'val_rmse']:.2f}")
    logger.info("=" * 70)
    
    # Classification scan
    clf_results = hyperparameter_scan_classification()
    
    # Save results
    clf_csv_path = os.path.join(output_dir, 'hyperparam_scan_classification.csv')
    clf_results.to_csv(clf_csv_path, index=False)
    logger.info(f"\nClassification results saved to {clf_csv_path}")
    
    # Plot
    plot_classification_results(clf_results, output_dir)
    
    # Print best configuration
    best_lml_idx = clf_results['approx_log_marginal_likelihood'].idxmax()
    best_loss_idx = clf_results['val_log_loss'].idxmin()
    
    logger.info("\n" + "=" * 70)
    logger.info("CLASSIFICATION: BEST HYPERPARAMETERS")
    logger.info("=" * 70)
    logger.info("By Approximate Log Marginal Likelihood:")
    logger.info(f"  ℓ={clf_results.loc[best_lml_idx, 'length_scale']:.2f}, "
               f"likelihood={clf_results.loc[best_lml_idx, 'likelihood']}, "
               f"Approx LML={clf_results.loc[best_lml_idx, 'approx_log_marginal_likelihood']:.2f}")
    logger.info("By Validation Log Loss:")
    logger.info(f"  ℓ={clf_results.loc[best_loss_idx, 'length_scale']:.2f}, "
               f"likelihood={clf_results.loc[best_loss_idx, 'likelihood']}, "
               f"Log Loss={clf_results.loc[best_loss_idx, 'val_log_loss']:.3f}")
    logger.info("=" * 70)
    
    logger.info("\nHyperparameter scan completed successfully!")


if __name__ == '__main__':
    main()
