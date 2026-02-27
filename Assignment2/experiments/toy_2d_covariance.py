"""
Toy 2D covariance visualisation.

Demonstrates GP predictive covariance structure on a simple 2D dataset,
showing how uncertainty relates to data density and kernel parameters.
"""

import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.gp.regression import GaussianProcessRegressor
from src.gp.kernels import RBF

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


def main():
    """Create toy 2D dataset and visualise covariance structure."""
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(__file__), 'outputs')
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("Generating toy 2D dataset...")
    
    # Create toy training data: 8 points in 2D space
    np.random.seed(42)
    X_train = np.array([
        [1.0, 1.0],
        [1.5, 2.0],
        [2.0, 1.5],
        [3.5, 3.5],
        [4.0, 4.5],
        [4.5, 4.0],
        [1.0, 4.0],
        [4.0, 1.0]
    ])
    
    # Target values (simple function with some structure)
    y_train = np.sin(X_train[:, 0]) + np.cos(X_train[:, 1]) + 0.1 * np.random.randn(len(X_train))
    
    logger.info(f"Training data: {X_train.shape[0]} points")
    
    # Fit GP regressor
    logger.info("Fitting GP regressor...")
    kernel = RBF(length_scale=0.8, variance=1.0)
    gp = GaussianProcessRegressor(
        kernel=kernel,
        noise=0.05,
        optimise=False,  # Use fixed hyperparameters for interpretability
        random_state=42,
        verbose=False
    )
    
    gp.fit(X_train, y_train)
    
    # Create test grid (40x40)
    grid_size = 40
    x1_range = np.linspace(0, 5, grid_size)
    x2_range = np.linspace(0, 5, grid_size)
    xx1, xx2 = np.meshgrid(x1_range, x2_range)
    X_test = np.c_[xx1.ravel(), xx2.ravel()]
    
    logger.info(f"Test grid: {X_test.shape[0]} points ({grid_size}x{grid_size})")
    
    # Predict mean and full covariance
    logger.info("Computing predictive mean and covariance...")
    mean, cov = gp.predict_f_cov(X_test)
    var = np.diag(cov)  # Marginal variances
    
    # Reshape for plotting
    mean_grid = mean.reshape(grid_size, grid_size)
    var_grid = var.reshape(grid_size, grid_size)
    std_grid = np.sqrt(var_grid)
    
    # Create comprehensive visualisation
    logger.info("Creating visualisations...")
    
    # Figure 1: Mean, Variance, and Training Data
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    
    # Predictive mean
    ax = axes[0]
    im1 = ax.contourf(xx1, xx2, mean_grid, levels=20, cmap='viridis', alpha=0.9)
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, s=150, 
               edgecolors='white', linewidths=2, cmap='viridis', 
               vmin=mean_grid.min(), vmax=mean_grid.max(), zorder=10)
    for i, (x, y) in enumerate(X_train):
        ax.annotate(f'{i+1}', (x, y), fontsize=9, ha='center', va='center', 
                   color='white', weight='bold')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Predictive Mean f̄(x)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.colorbar(im1, ax=ax, label='Mean')
    
    # Marginal variance
    ax = axes[1]
    im2 = ax.contourf(xx1, xx2, var_grid, levels=20, cmap='Reds', alpha=0.9)
    ax.scatter(X_train[:, 0], X_train[:, 1], c='white', s=150, 
               edgecolors='black', linewidths=2, marker='o', zorder=10)
    for i, (x, y) in enumerate(X_train):
        ax.annotate(f'{i+1}', (x, y), fontsize=9, ha='center', va='center', 
                   color='black', weight='bold')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Predictive Variance Var[f(x)]', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.colorbar(im2, ax=ax, label='Variance')
    
    # Standard deviation
    ax = axes[2]
    im3 = ax.contourf(xx1, xx2, std_grid, levels=20, cmap='YlOrRd', alpha=0.9)
    contour = ax.contour(xx1, xx2, std_grid, levels=6, colors='black', 
                        linewidths=0.5, alpha=0.4)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.2f')
    ax.scatter(X_train[:, 0], X_train[:, 1], c='white', s=150, 
               edgecolors='black', linewidths=2, marker='o', zorder=10)
    for i, (x, y) in enumerate(X_train):
        ax.annotate(f'{i+1}', (x, y), fontsize=9, ha='center', va='center', 
                   color='black', weight='bold')
    ax.set_xlabel('x₁', fontsize=12)
    ax.set_ylabel('x₂', fontsize=12)
    ax.set_title('Predictive Std Dev σ(x)', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
    plt.colorbar(im3, ax=ax, label='Std Dev')
    
    plt.suptitle(f'GP Regression on 2D Toy Dataset (ℓ={kernel.length_scale:.2f}, σ²={kernel.variance:.2f})', 
                 fontsize=14, fontweight='bold', y=1.00)
    plt.tight_layout()
    
    fig1_path = os.path.join(output_dir, 'toy_2d_mean_variance.png')
    plt.savefig(fig1_path, dpi=150, bbox_inches='tight')
    logger.info(f"Mean/variance plot saved to {fig1_path}")
    plt.close()
    
    # Figure 2: Full Covariance Matrix and Slices
    fig = plt.figure(figsize=(16, 5))
    
    # Full covariance matrix heatmap
    ax1 = plt.subplot(1, 3, 1)
    im = ax1.imshow(cov, cmap='coolwarm', aspect='auto', interpolation='nearest')
    ax1.set_xlabel('Test Point Index', fontsize=11)
    ax1.set_ylabel('Test Point Index', fontsize=11)
    ax1.set_title(f'Full Covariance Matrix\n({X_test.shape[0]}×{X_test.shape[0]})', 
                  fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Covariance')
    
    # Select a few test points for covariance slices
    test_indices = [
        grid_size * (grid_size // 4) + (grid_size // 4),      # Lower-left quadrant
        grid_size * (grid_size // 2) + (grid_size // 2),      # Center
        grid_size * (3 * grid_size // 4) + (3 * grid_size // 4)  # Upper-right quadrant
    ]
    
    # Covariance slice 1
    ax2 = plt.subplot(1, 3, 2)
    for idx in test_indices:
        cov_slice = cov[idx, :].reshape(grid_size, grid_size)
        test_pt = X_test[idx]
        
        # Plot as contour
        contour = ax2.contour(xx1, xx2, cov_slice, levels=8, linewidths=1.5, alpha=0.7)
        ax2.clabel(contour, inline=True, fontsize=7)
    
    # Mark selected points
    for idx in test_indices:
        pt = X_test[idx]
        ax2.scatter(pt[0], pt[1], c='red', s=100, marker='x', linewidths=3, zorder=10)
    
    ax2.scatter(X_train[:, 0], X_train[:, 1], c='blue', s=80, 
               edgecolors='white', linewidths=1.5, marker='o', zorder=9, alpha=0.7)
    ax2.set_xlabel('x₁', fontsize=11)
    ax2.set_ylabel('x₂', fontsize=11)
    ax2.set_title('Covariance Slices\n(from selected test points)', 
                  fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Cross-sections of covariance
    ax3 = plt.subplot(1, 3, 3)
    center_idx = grid_size * (grid_size // 2) + (grid_size // 2)
    
    # Horizontal slice through center
    h_slice_idx = grid_size // 2
    cov_h = cov[center_idx, h_slice_idx * grid_size:(h_slice_idx + 1) * grid_size]
    
    # Vertical slice through center
    v_slice_idx = grid_size // 2
    v_indices = [v_slice_idx + i * grid_size for i in range(grid_size)]
    cov_v = cov[center_idx, v_indices]
    
    ax3.plot(x1_range, cov_h, 'b-', linewidth=2, label='Horizontal slice (through center)')
    ax3.plot(x2_range, cov_v, 'r--', linewidth=2, label='Vertical slice (through center)')
    ax3.axhline(y=0, color='k', linestyle=':', linewidth=1, alpha=0.5)
    ax3.set_xlabel('Distance along slice', fontsize=11)
    ax3.set_ylabel('Covariance with center point', fontsize=11)
    ax3.set_title('Covariance Cross-Sections', fontsize=12, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    plt.suptitle('Predictive Covariance Structure', fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    fig2_path = os.path.join(output_dir, 'toy_2d_covariance_structure.png')
    plt.savefig(fig2_path, dpi=150, bbox_inches='tight')
    logger.info(f"Covariance structure plot saved to {fig2_path}")
    plt.close()
    
    # Print interpretation
    logger.info("\n" + "=" * 70)
    logger.info("INTERPRETATION OF UNCERTAINTY STRUCTURE")
    logger.info("=" * 70)
    logger.info("1. Marginal Variance (Uncertainty):")
    logger.info("   - Lowest near training points (GP interpolates the data)")
    logger.info("   - Increases with distance from training data")
    logger.info("   - Reflects both data density and kernel lengthscale")
    logger.info("")
    logger.info("2. Covariance Structure:")
    logger.info("   - Points close in input space have high covariance")
    logger.info("   - Covariance decays with distance (controlled by lengthscale)")
    logger.info("   - Off-diagonal covariance → predictions at nearby points are correlated")
    logger.info("")
    logger.info("3. Kernel Lengthscale Impact:")
    logger.info(f"   - Current ℓ={kernel.length_scale:.2f} defines 'local' scale")
    logger.info("   - Smaller ℓ → more wiggly functions, faster variance increase")
    logger.info("   - Larger ℓ → smoother functions, gradual variance increase")
    logger.info("=" * 70)
    
    logger.info("\nToy 2D covariance visualisation completed successfully!")


if __name__ == '__main__':
    main()
