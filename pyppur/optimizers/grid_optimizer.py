"""
Grid-based optimizer for projection pursuit.
"""

from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np

from pyppur.optimizers.base import BaseOptimizer


class GridOptimizer(BaseOptimizer):
    """
    Optimizer using a grid-based search approach.

    This optimizer is particularly useful for projection indices that are not
    differentiable or have many local minima. It systematically explores the
    space of projection directions using a grid-based approach.
    """

    def __init__(
        self,
        objective_func: Callable,
        n_components: int,
        n_directions: int = 250,
        n_iterations: int = 10,
        max_iter: int = 1000,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        verbose: bool = False,
        **kwargs,
    ):
        """
        Initialize the grid optimizer.

        Args:
            objective_func: Objective function to minimize
            n_components: Number of projection components
            n_directions: Number of random directions to generate per iteration
            n_iterations: Number of refinement iterations
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
        self.n_directions = n_directions
        self.n_iterations = n_iterations

    def _generate_random_directions(
        self, n_features: int, n_directions: int
    ) -> np.ndarray:
        """
        Generate random unit directions.

        Args:
            n_features: Number of features in the data
            n_directions: Number of directions to generate

        Returns:
            np.ndarray: Random directions, shape (n_directions, n_features)
        """
        directions = np.random.randn(n_directions, n_features)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        return directions

    def _optimize_sequential(
        self, X: np.ndarray, initial_directions: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, float, List[float]]:
        """
        Optimize projection directions sequentially.

        This method finds one direction at a time, optimizing each direction
        while keeping the previous ones fixed.

        Args:
            X: Input data, shape (n_samples, n_features)
            initial_directions: Optional initial directions
            **kwargs: Additional arguments for the objective function

        Returns:
            Tuple[np.ndarray, float, List[float]]:
                - Optimized projection directions
                - Final objective value
                - Loss values for each component
        """
        n_samples, n_features = X.shape
        best_directions = np.zeros((self.n_components, n_features))
        loss_values = []

        # Use initial directions if provided
        if (
            initial_directions is not None
            and initial_directions.shape[0] >= self.n_components
        ):
            for i in range(self.n_components):
                best_directions[i] = initial_directions[i]

            # Normalize
            best_directions = best_directions / np.linalg.norm(
                best_directions, axis=1, keepdims=True
            )

        # Optimize each direction sequentially
        for component in range(self.n_components):
            if self.verbose:
                print(f"Optimizing component {component+1}/{self.n_components}")

            best_loss = np.inf
            best_direction = None

            # If we already have an initial direction for this component, use it
            if (
                initial_directions is not None
                and component < initial_directions.shape[0]
            ):
                current_direction = initial_directions[component]
                current_direction = current_direction / np.linalg.norm(
                    current_direction
                )

                # Set this direction in the best_directions array
                best_directions[component] = current_direction

                # Evaluate with this direction
                projections = X @ best_directions[: component + 1].T
                objective_args = (
                    (X, component + 1)
                    if len(kwargs) == 0
                    else (X, component + 1, *kwargs.values())
                )

                current_loss = self.objective_func(
                    best_directions[: component + 1].flatten(), *objective_args
                )

                if current_loss < best_loss:
                    best_loss = current_loss
                    best_direction = current_direction

            # Perform iterations to refine the direction
            for iteration in range(self.n_iterations):
                if self.verbose:
                    print(f"  Iteration {iteration+1}/{self.n_iterations}")

                # Generate random directions
                if iteration == 0 and best_direction is None:
                    # First iteration: completely random directions
                    directions = self._generate_random_directions(
                        n_features, self.n_directions
                    )
                else:
                    # Later iterations: perturb the best direction so far
                    if best_direction is None:
                        best_direction = self._generate_random_directions(
                            n_features, 1
                        )[0]

                    # Generate perturbation scale based on iteration
                    scale = 1.0 / (iteration + 1)

                    # Generate perturbations around the best direction
                    directions = np.random.randn(self.n_directions, n_features) * scale
                    directions = directions + best_direction
                    directions = directions / np.linalg.norm(
                        directions, axis=1, keepdims=True
                    )

                # Evaluate each direction
                for i, direction in enumerate(directions):
                    # Set this direction in the best_directions array
                    best_directions[component] = direction

                    # Evaluate with this direction
                    projections = X @ best_directions[: component + 1].T
                    objective_args = (
                        (X, component + 1)
                        if len(kwargs) == 0
                        else (X, component + 1, *kwargs.values())
                    )

                    loss = self.objective_func(
                        best_directions[: component + 1].flatten(), *objective_args
                    )

                    if loss < best_loss:
                        best_loss = loss
                        best_direction = direction

                if self.verbose:
                    print(f"    Best loss: {best_loss}")

            # Update the best directions with the best one found
            best_directions[component] = best_direction
            loss_values.append(best_loss)

            if self.verbose:
                print(f"  Component {component+1} optimized, loss: {best_loss}")

        # Final evaluation with all components
        objective_args = (
            (X, self.n_components)
            if len(kwargs) == 0
            else (X, self.n_components, *kwargs.values())
        )
        final_loss = self.objective_func(best_directions.flatten(), *objective_args)

        return best_directions, final_loss, loss_values

    def optimize(
        self, X: np.ndarray, initial_guess: Optional[np.ndarray] = None, **kwargs
    ) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Optimize the projection directions using a grid-based approach.

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
        # Run sequential optimization
        best_directions, final_loss, loss_values = self._optimize_sequential(
            X, initial_directions=initial_guess, **kwargs
        )

        # Prepare additional information
        info = {
            "success": True,
            "message": "Grid optimization completed",
            "n_iterations": self.n_iterations,
            "n_directions": self.n_directions,
            "loss_per_component": loss_values,
        }

        return best_directions, final_loss, info
