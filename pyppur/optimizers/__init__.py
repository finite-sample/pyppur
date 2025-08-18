"""
Optimization methods for projection pursuit.
"""

from .grid_optimizer import GridOptimizer
from .scipy_optimizer import ScipyOptimizer

__all__ = ["GridOptimizer", "ScipyOptimizer"]
