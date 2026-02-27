"""
Regression demonstration using diabetes dataset.

Fits GP regressor with hyperparameter optimisation and produces
predictive plots with uncertainty quantification.
"""

import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gp.regression import GaussianProcessRegressor
from src.gp.kernels import RBF

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def negative_log_predictive_density(y_true, y_pred, y_std):
    """
    Compute negative log predictive density (NLPD).
    
    NLPD = -log N(y|μ,σ²) = 0.5 * log(2π) + 0.5 * log(σ²) + 0.5 * (y-μ)²/σ²
    """
    var = y_std ** 2 + 1e-10  # Avoid division by zero
    nlpd = 0.5 * np.log(2 * np.pi * var) + 0.5 * ((y_true - y_pred) ** 2) / var
    return np.mean(nlpd)


def main():
    """Run regression experiment on diabetes dataset."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Loading diabetes dataset...")
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    logger.info(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    
    # Fit GP regressor
    logger.info("Fitting GP regressor with RBF kernel...")
    kernel = RBF(length_scale=1.0, variance=100.0)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        noise=1.0,
        optimise=True,
        n_restarts=2,
        random_state=42,
        verbose=True
    )
    
    gp.fit(X_train, y_train)
    
    # Report hyperparameters
    logger.info("\n" + "=" * 60)
    logger.info("OPTIMISED HYPERPARAMETERS")
    logger.info("=" * 60)
    logger.info(f"Kernel length scale: {gp.kernel.length_scale:.6f}")
    logger.info(f"Kernel variance: {gp.kernel.variance:.6f}")
    logger.info(f"Noise variance: {gp.noise:.6f}")
    logger.info(f"Log marginal likelihood: {gp.log_marginal_likelihood_:.4f}")
    logger.info("=" * 60 + "\n")
    
    # Predictions
    logger.info("Generating predictions on test set...")
    y_pred_mean, y_pred_std = gp.predict(X_test, return_std=True)
    
    # Metrics
    rmse = np.sqrt(mean_squared_error(y_test, y_pred_mean))
    nlpd = negative_log_predictive_density(y_test, y_pred_mean, y_pred_std)
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET METRICS")
    logger.info("=" * 60)
    logger.info(f"RMSE: {rmse:.4f}")
    logger.info(f"NLPD: {nlpd:.4f}")
    logger.info("=" * 60 + "\n")
    
    # Save predictions
    results_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred_mean': y_pred_mean,
        'y_pred_std': y_pred_std,
        'lower_95': y_pred_mean - 1.96 * y_pred_std,
        'upper_95': y_pred_mean + 1.96 * y_pred_std
    })
    results_path = os.path.join(output_dir, 'regression_predictions.csv')
    results_df.to_csv(results_path, index=False)
    logger.info(f"Predictions saved to {results_path}")
    
    # Visualisation: 1D projection via PCA
    logger.info("Creating visualisations...")
    pca = PCA(n_components=1, random_state=42)
    X_train_1d = pca.fit_transform(X_train).ravel()
    X_test_1d = pca.transform(X_test).ravel()
    
    # Sort for plotting
    sort_idx = np.argsort(X_test_1d)
    X_test_1d_sorted = X_test_1d[sort_idx]
    y_test_sorted = y_test[sort_idx]
    y_pred_sorted = y_pred_mean[sort_idx]
    y_std_sorted = y_pred_std[sort_idx]
    
    # Plot
    plt.figure(figsize=(12, 6))
    
    # Training data
    plt.scatter(X_train_1d, y_train, c='blue', alpha=0.3, s=20, label='Training data')
    
    # Test data
    plt.scatter(X_test_1d_sorted, y_test_sorted, c='red', alpha=0.6, s=30, 
                label='Test data', zorder=5)
    
    # Predictions with uncertainty
    plt.plot(X_test_1d_sorted, y_pred_sorted, 'g-', linewidth=2, 
             label='Predictive mean', zorder=4)
    plt.fill_between(
        X_test_1d_sorted,
        y_pred_sorted - 2 * y_std_sorted,
        y_pred_sorted + 2 * y_std_sorted,
        alpha=0.2,
        color='green',
        label='±2σ confidence'
    )
    
    plt.xlabel('First Principal Component', fontsize=12)
    plt.ylabel('Disease Progression', fontsize=12)
    plt.title('GP Regression on Diabetes Dataset\n(1D PCA Projection)', fontsize=14, fontweight='bold')
    plt.legend(loc='best', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    plot_path = os.path.join(output_dir, 'regression_predictions.png')
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    logger.info(f"Plot saved to {plot_path}")
    plt.close()
    
    # Plot optimisation trace
    if gp.optimisation_trace_:
        lml_values = [entry['lml'] for entry in gp.optimisation_trace_]
        
        plt.figure(figsize=(10, 5))
        plt.plot(lml_values, 'b-', linewidth=1.5)
        plt.xlabel('Optimisation Iteration', fontsize=12)
        plt.ylabel('Log Marginal Likelihood', fontsize=12)
        plt.title('Hyperparameter Optimisation Trace', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        trace_path = os.path.join(output_dir, 'regression_optimisation_trace.png')
        plt.savefig(trace_path, dpi=150, bbox_inches='tight')
        logger.info(f"Optimisation trace saved to {trace_path}")
        plt.close()
    
    logger.info("\nRegression experiment completed successfully!")


if __name__ == '__main__':
    main()
