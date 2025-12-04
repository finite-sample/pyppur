"""
SciPy-based optimizer for projection pursuit.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.optimize import minimize

from pyppur.optimizers.base import BaseOptimizer


def normalize_projection_directions(
    a_flat: np.ndarray, n_components: int, n_features: int
) -> np.ndarray:
    """
    Normalize the encoder projection directions to unit norm.

    Args:
        a_flat: Flattened parameter vector
        n_components: Number of projection components
        n_features: Number of features

    Returns:
        np.ndarray: Normalized parameter vector
    """
    # For tied weights, normalize only the encoder part
    # For untied weights, normalize only the encoder part (first half)
    encoder_params = n_components * n_features

    # Extract and normalize encoder
    a_encoder = a_flat[:encoder_params].reshape(n_components, n_features)
    norms = np.linalg.norm(a_encoder, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
    a_encoder_normalized = a_encoder / norms

    # Replace encoder part with normalized version
    a_flat_normalized = a_flat.copy()
    a_flat_normalized[:encoder_params] = a_encoder_normalized.flatten()

    return a_flat_normalized


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
        random_state: int | None = None,
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
        self, X: np.ndarray, initial_guess: np.ndarray | None = None, **kwargs: Any
    ) -> tuple[np.ndarray, float, dict[str, Any]]:
        """
        Optimize the projection directions using SciPy's optimization methods.

        Args:
            X: Input data, shape (n_samples, n_features)
            initial_guess: Optional initial guess for projection directions
            **kwargs: Additional arguments for the objective function

        Returns:
            tuple[np.ndarray, float, dict[str, Any]]:
                - Optimized projection directions, shape (n_components, n_features)
                - Final objective value
                - Additional optimizer information
        """
        n_features = X.shape[1]

        # Check if this is for untied weights (reconstruction with separate decoder)
        is_untied = (
            hasattr(self.objective_func, "tied_weights")
            and not self.objective_func.tied_weights
        )
        expected_params = self.n_components * n_features * (2 if is_untied else 1)

        # If no initial guess provided, use PCA or random initialization
        if initial_guess is None:
            if self.verbose:
                print("No initial guess provided, using random initialization")
            initial_guess_matrix = np.random.randn(self.n_components, n_features)
            initial_guess_matrix = initial_guess_matrix / np.linalg.norm(
                initial_guess_matrix, axis=1, keepdims=True
            )
            initial_guess_flat = initial_guess_matrix.flatten()

            # Add decoder for untied weights
            if is_untied:
                decoder_guess = np.random.randn(self.n_components, n_features) * 0.1
                initial_guess_flat = np.concatenate(
                    [initial_guess_flat, decoder_guess.flatten()]
                )
        else:
            # Handle both tied and untied weights initial guess
            if initial_guess.size == expected_params:
                # Already correct size (flat or can be flattened)
                initial_guess_flat = initial_guess.flatten()
            elif initial_guess.shape == (self.n_components, n_features):
                # Only encoder provided, add decoder for untied weights
                initial_guess_flat = initial_guess.flatten()
                if is_untied:
                    decoder_guess = np.random.randn(self.n_components, n_features) * 0.1
                    initial_guess_flat = np.concatenate(
                        [initial_guess_flat, decoder_guess.flatten()]
                    )
            else:
                raise ValueError(
                    f"Initial guess size {initial_guess.size} does not match "
                    f"expected size {expected_params} for "
                    f"{'untied' if is_untied else 'tied'} weights"
                )

        # Set up optimization options based on method
        if self.method == "L-BFGS-B":
            options = {
                "maxfun": self.max_iter * 100,  # Max function evaluations
                "gtol": self.tol,  # Gradient tolerance
                "ftol": 2.2e-9,  # Function tolerance
            }
        elif self.method == "SLSQP":
            options = {
                "maxiter": self.max_iter,  # Max iterations
                "ftol": self.tol,  # Function tolerance
            }
        else:
            # Default options for other methods
            options = {"maxiter": self.max_iter}

        # Additional options from kwargs
        options.update(self.kwargs.get("options", {}))

        # Run optimization
        k = self.n_components

        # Objective function with proper keyword argument handling and normalization
        def objective_wrapper(a_flat: np.ndarray) -> float:
            # Normalize projection directions before evaluation
            a_flat_normalized = normalize_projection_directions(a_flat, k, n_features)
            return self.objective_func(a_flat_normalized, X, k, **kwargs)

        result = minimize(
            objective_wrapper,
            initial_guess_flat,
            method=self.method,
            options=options,
        )

        # Prepare additional information
        info = {
            "success": result.success,
            "status": result.status,
            "message": result.message,
            "nfev": result.nfev,
            "nit": result.nit if hasattr(result, "nit") else None,
        }

        # Normalize and reshape the result
        result_normalized = normalize_projection_directions(
            result.x, self.n_components, n_features
        )

        # For tied weights, return just the encoder matrix
        # For untied weights, we need to handle this differently in the main class
        if (
            hasattr(self.objective_func, "tied_weights")
            and not self.objective_func.tied_weights
        ):
            # Return the full parameter vector (will be handled by main class)
            return result_normalized, result.fun, info
        else:
            # Return just the encoder matrix for tied weights
            encoder_params = self.n_components * n_features
            a_matrix = result_normalized[:encoder_params].reshape(
                self.n_components, n_features
            )
            return a_matrix, result.fun, info
