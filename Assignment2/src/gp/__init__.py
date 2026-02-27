"""
Gaussian Process Library for Regression and Classification.

Implements algorithms from "Gaussian Processes for Machine Learning" 
by Rasmussen and Williams.
"""

from .regression import GaussianProcessRegressor
from .classification import GaussianProcessClassifier
from . import kernels

__all__ = [
    'GaussianProcessRegressor',
    'GaussianProcessClassifier',
    'kernels',
]
