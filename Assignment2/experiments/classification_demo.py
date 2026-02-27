"""
Binary classification demonstration using breast cancer dataset.

Fits GP classifier with Laplace approximation and produces
ROC curves, calibration plots, and 2D probability heatmaps.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, log_loss, roc_curve
from sklearn.calibration import calibration_curve

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gp.classification import GaussianProcessClassifier
from src.gp.kernels import RBF

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Run classification experiment on breast cancer dataset."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Loading breast cancer dataset...")
    cancer = load_breast_cancer()
    X, y = cancer.data, cancer.target
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
    logger.info(f"Class distribution (train): {np.bincount(y_train)}")
    
    # Fit GP classifier
    logger.info("Fitting GP classifier with Laplace approximation...")
    kernel = RBF(length_scale=5.0, variance=1.0)
    gp = GaussianProcessClassifier(
        kernel=kernel,
        likelihood='logistic',
        max_iter=50,
        tol=1e-6,
        random_state=42,
        verbose=True
    )
    
    gp.fit(X_train, y_train)
    
    # Report convergence
    logger.info("\n" + "=" * 60)
    logger.info("CONVERGENCE DIAGNOSTICS")
    logger.info("=" * 60)
    logger.info(f"Newton iterations: {gp.n_iter_}")
    logger.info(f"Approximate log marginal likelihood: {gp.log_marginal_likelihood_:.4f}")
    logger.info(f"Kernel length scale: {gp.kernel.length_scale:.6f}")
    logger.info(f"Kernel variance: {gp.kernel.variance:.6f}")
    logger.info("=" * 60 + "\n")
    
    # Predictions
    logger.info("Generating predictions on test set...")
    y_pred_proba = gp.predict_proba(X_test)
    y_pred = gp.predict(X_test)
    
    # Metrics
    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    logloss = log_loss(y_test, y_pred_proba)
    accuracy = np.mean(y_pred == y_test)
    
    logger.info("\n" + "=" * 60)
    logger.info("TEST SET METRICS")
    logger.info("=" * 60)
    logger.info(f"ROC AUC: {roc_auc:.4f}")
    logger.info(f"Log Loss: {logloss:.4f}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info("=" * 60 + "\n")
    
    # ROC curve
    logger.info("Creating ROC curve...")
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba[:, 1])
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'GP Classifier (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Breast Cancer Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = os.path.join(output_dir, 'classification_roc_curve.png')
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    logger.info(f"ROC curve saved to {roc_path}")
    plt.close()
    
    # Calibration curve
    logger.info("Creating calibration curve...")
    prob_true, prob_pred = calibration_curve(
        y_test, y_pred_proba[:, 1], n_bins=10, strategy='uniform'
    )
    
    plt.figure(figsize=(8, 6))
    plt.plot(prob_pred, prob_true, 'o-', linewidth=2, markersize=8, label='GP Classifier')
    plt.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfect Calibration')
    plt.xlabel('Mean Predicted Probability', fontsize=12)
    plt.ylabel('Fraction of Positives', fontsize=12)
    plt.title('Calibration Curve - GP Classification', fontsize=14, fontweight='bold')
    plt.legend(loc='upper left', fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    calib_path = os.path.join(output_dir, 'classification_calibration.png')
    plt.savefig(calib_path, dpi=150, bbox_inches='tight')
    logger.info(f"Calibration curve saved to {calib_path}")
    plt.close()
    
    # 2D visualisation via PCA
    logger.info("Creating 2D probability heatmap (PCA projection)...")
    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)
    
    # Create grid for contour plot
    x_min, x_max = X_train_pca[:, 0].min() - 1, X_train_pca[:, 0].max() + 1
    y_min, y_max = X_train_pca[:, 1].min() - 1, X_train_pca[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    # Predict on grid (transform back to original space)
    grid_pca = np.c_[xx.ravel(), yy.ravel()]
    # Note: This is an approximation - we're predicting on incomplete PCA space
    # For proper prediction, we'd need to inverse transform, but that introduces artifacts
    # Instead, we'll fit a separate GP on the 2D PCA space for visualisation
    
    logger.info("Fitting separate GP on 2D PCA space for visualisation...")
    gp_pca = GaussianProcessClassifier(
        kernel=RBF(length_scale=2.0, variance=1.0),
        likelihood='logistic',
        max_iter=50,
        random_state=42,
        verbose=False
    )
    gp_pca.fit(X_train_pca, y_train)
    
    Z = gp_pca.predict_proba(grid_pca)[:, 1]
    Z = Z.reshape(xx.shape)
    
    # Plot
    plt.figure(figsize=(12, 5))
    
    # Contour plot
    plt.subplot(1, 2, 1)
    contour = plt.contourf(xx, yy, Z, levels=20, cmap='RdYlBu_r', alpha=0.8)
    plt.colorbar(contour, label='P(y=1)')
    
    # Plot training points
    plt.scatter(
        X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1],
        c='blue', marker='o', s=30, alpha=0.6, edgecolors='k', linewidth=0.5,
        label='Class 0 (train)'
    )
    plt.scatter(
        X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
        c='red', marker='s', s=30, alpha=0.6, edgecolors='k', linewidth=0.5,
        label='Class 1 (train)'
    )
    
    plt.xlabel('First Principal Component', fontsize=11)
    plt.ylabel('Second Principal Component', fontsize=11)
    plt.title('GP Classification Probability (2D PCA)', fontsize=12, fontweight='bold')
    plt.legend(loc='best', fontsize=9)
    plt.grid(True, alpha=0.3)
    
    # Uncertainty visualisation (predictive variance of latent function)
    plt.subplot(1, 2, 2)
    _, grid_cov = gp_pca.predict_f_cov(grid_pca)
    grid_var = np.diag(grid_cov).reshape(xx.shape)
    
    contour_var = plt.contourf(xx, yy, grid_var, levels=20, cmap='viridis', alpha=0.8)
    plt.colorbar(contour_var, label='Predictive Variance')
    
    plt.scatter(
        X_train_pca[y_train == 0, 0], X_train_pca[y_train == 0, 1],
        c='white', marker='o', s=20, alpha=0.8, edgecolors='k', linewidth=0.5
    )
    plt.scatter(
        X_train_pca[y_train == 1, 0], X_train_pca[y_train == 1, 1],
        c='white', marker='s', s=20, alpha=0.8, edgecolors='k', linewidth=0.5
    )
    
    plt.xlabel('First Principal Component', fontsize=11)
    plt.ylabel('Second Principal Component', fontsize=11)
    plt.title('Predictive Uncertainty (Latent Variance)', fontsize=12, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    plt.suptitle('GP Classification on Breast Cancer Dataset', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    heatmap_path = os.path.join(output_dir, 'classification_2d_heatmap.png')
    plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
    logger.info(f"2D heatmap saved to {heatmap_path}")
    plt.close()
    
    logger.info("\nClassification experiment completed successfully!")
    logger.info("\nInterpretation:")
    logger.info("- Uncertainty is highest near the decision boundary (where classes overlap)")
    logger.info("- Uncertainty also increases in low-density regions far from training data")
    logger.info("- The kernel lengthscale controls smoothness; smaller values → more variable predictions")


if __name__ == '__main__':
    main()
