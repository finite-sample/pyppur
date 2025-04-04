"""
Base class for optimization methods.
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any, Callable

import numpy as np


class BaseOptimizer(ABC):
    """
    Abstract base class for projection pursuit optimizers.
    
    All optimizer implementations should inherit from this class
    and implement the required methods.
    """
    
    def __init__(
        self,
        objective_func: Callable,
        n_components: int,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs
    ):
        """
        Initialize the optimizer.
        
        Args:
            objective_func: Objective function to minimize
            n_components: Number of projection components
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            **kwargs: Additional keyword arguments
        """
        self.objective_func = objective_func
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.verbose = verbose
        self.kwargs = kwargs
        
        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)
    
    @abstractmethod
    def optimize(
        self, 
        X: np.ndarray, 
        initial_guess: Optional[np.ndarray] = None, 
        **kwargs
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Optimize the projection directions.
        
        Args:
            X: Input data, shape (n_samples, n_features)
            initial_guess: Optional initial guess for projection directions
            **kwargs: Additional arguments for the objective function
            
        Returns:
            Tuple[np.ndarray, float, Dict[str, Any]]: 
                - Optimized projection directions, shape (n_components, n_features)
                - Final objective value
                - Additional optimizer information
        """
        pass