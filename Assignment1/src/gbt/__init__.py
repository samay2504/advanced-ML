"""
Gradient Boosting Tree implementation from scratch.

Implements Algorithms 10.3 (Forward Stagewise Additive Modelling) and 10.4 
(Gradient Tree Boosting) from "The Elements of Statistical Learning" 
by Hastie, Tibshirani, and Friedman.
"""

from .core import GradientBoostingRegressor, GradientBoostingClassifier

__version__ = "0.1.0"
__all__ = ["GradientBoostingRegressor", "GradientBoostingClassifier"]
