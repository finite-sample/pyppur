"""
SciPy-based optimizer for projection pursuit.
"""

from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np
from scipy.optimize import minimize

from pyppur.optimizers.base import BaseOptimizer


class ScipyOptimizer(BaseOptimizer):
    """
    Optimizer using SciPy's optimization methods.

    This optimizer leverages SciPy's optimization functionality,
    particularly the L-BFGS-B method which is well-suited for
    projection pursuit problems.
    """

    def __init__(
        self,
        objective_func: Callable,
        n_components: int,
        method: str = "L-BFGS-B",
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the SciPy optimizer.

        Args:
            objective_func: Objective function to minimize
            n_components: Number of projection components
            method: SciPy optimization method (default: "L-BFGS-B")
            max_iter: Maximum number of iterations
            tol: Tolerance for convergence
            random_state: Random seed for reproducibility
            verbose: Whether to print progress information
            **kwargs: Additional keyword arguments for the optimizer
        """
        super().__init__(
            objective_func=objective_func,
            n_components=n_components,
            max_iter=max_iter,
            tol=tol,
            random_state=random_state,
            verbose=verbose,
            **kwargs,
        )
        self.method = method

    def optimize(
        self, X: np.ndarray, initial_guess: Optional[np.ndarray] = None, **kwargs: Any
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Optimize the projection directions using SciPy's optimization methods.

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
        n_features = X.shape[1]

        # If no initial guess provided, use PCA or random initialization
        if initial_guess is None:
            if self.verbose:
                print("No initial guess provided, using random initialization")
            initial_guess_matrix = np.random.randn(self.n_components, n_features)
            initial_guess_matrix = initial_guess_matrix / np.linalg.norm(
                initial_guess_matrix, axis=1, keepdims=True
            )
            initial_guess_flat = initial_guess_matrix.flatten()
        else:
            # Ensure correct shape and normalization
            if initial_guess.shape != (self.n_components, n_features):
                if initial_guess.size == self.n_components * n_features:
                    initial_guess = initial_guess.reshape(self.n_components, n_features)
                else:
                    raise ValueError(
                        f"Initial guess shape {initial_guess.shape} does not match "
                        f"expected shape ({self.n_components}, {n_features})"
                    )

            # Normalize
            initial_guess = initial_guess / np.linalg.norm(
                initial_guess, axis=1, keepdims=True
            )
            initial_guess_flat = initial_guess.flatten()

        # Set up optimization options
        options = {"maxiter": self.max_iter, "gtol": self.tol, "disp": self.verbose}

        # Additional options from kwargs
        options.update(self.kwargs.get("options", {}))

        # Run optimization
        k = self.n_components

        # Objective function with proper keyword argument handling
        def objective_wrapper(a_flat: np.ndarray) -> float:
            return self.objective_func(a_flat, X, k, **kwargs)

        result = minimize(
            objective_wrapper,
            initial_guess_flat,
            method=self.method,
            options=options,
        )

        # Reshape and normalize the result
        a_matrix = result.x.reshape(self.n_components, n_features)
        a_matrix = a_matrix / np.linalg.norm(a_matrix, axis=1, keepdims=True)

        # Prepare additional information
        info = {
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "nfev": result.nfev,
            "nit": result.nit if hasattr(result, "nit") else None,
        }

        return a_matrix, result.fun, info
