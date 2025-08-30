"""
Main implementation of Projection Pursuit for dimensionality reduction.
"""

import time
import warnings
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from pyppur.objectives import Objective
from pyppur.objectives.base import BaseObjective
from pyppur.objectives.distance import DistanceObjective
from pyppur.objectives.reconstruction import ReconstructionObjective
from pyppur.optimizers import ScipyOptimizer
from pyppur.utils.metrics import (
    compute_distance_distortion,
    compute_silhouette,
    compute_trustworthiness,
    evaluate_embedding,
)
from pyppur.utils.preprocessing import standardize_data


class ProjectionPursuit:
    """
    Implementation of Projection Pursuit for dimensionality reduction.

    This class provides methods to find optimal projections by minimizing
    either reconstruction loss or distance distortion. It supports both
    initialization strategies and different optimizers.

    Attributes:
        n_components (int): Number of projection dimensions
        objective (Objective): Optimization objective (distance distortion or reconstruction)
        alpha (float): Steepness parameter for the ridge function
        max_iter (int): Maximum number of iterations for optimization
        tol (float): Tolerance for optimization convergence
        random_state (Optional[int]): Random seed for reproducibility
        optimizer (str): Optimization method ('L-BFGS-B' recommended)
        n_init (int): Number of random initializations
        verbose (bool): Whether to print progress information
        center (bool): Whether to center the data
        scale (bool): Whether to scale the data
        weight_by_distance (bool): Whether to weight distance distortion by inverse of original distances
    """

    def __init__(
        self,
        n_components: int = 2,
        objective: Union[Objective, str] = Objective.DISTANCE_DISTORTION,
        alpha: float = 1.0,
        max_iter: int = 500,
        tol: float = 1e-6,
        random_state: Optional[int] = None,
        optimizer: str = "L-BFGS-B",
        n_init: int = 3,
        verbose: bool = False,
        center: bool = True,
        scale: bool = True,
        weight_by_distance: bool = False,
    ):
        """
        Initialize a ProjectionPursuit model.

        Args:
            n_components: Number of projection dimensions to use
            objective: Optimization objective, either "distance_distortion" or "reconstruction"
            alpha: Steepness parameter for the ridge function g(z) = tanh(alpha * z)
            max_iter: Maximum number of iterations for optimization
            tol: Tolerance for optimization convergence
            random_state: Random seed for reproducibility
            optimizer: Optimization method ('L-BFGS-B' recommended)
            n_init: Number of random initializations to try
            verbose: Whether to print progress information
            center: Whether to center the data
            scale: Whether to scale the data
            weight_by_distance: Whether to weight distance distortion by inverse of original distances
        """
        self.n_components = n_components

        if isinstance(objective, str):
            try:
                self.objective = Objective(objective)
            except ValueError:
                # List the valid objective types directly
                raise ValueError(
                    f"Objective must be one of {[Objective.DISTANCE_DISTORTION, Objective.RECONSTRUCTION]}"
                )
        else:
            self.objective = objective

        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.optimizer = optimizer
        self.n_init = n_init
        self.verbose = verbose
        self.center = center
        self.scale = scale
        self.weight_by_distance = weight_by_distance

        # Private attributes
        self._fitted = False
        self._x_loadings: Optional[np.ndarray] = None
        self._scaler: Optional[StandardScaler] = None
        self._loss_curve: List[float] = []
        self._best_loss = np.inf
        self._fit_time = 0.0
        self._objective_func: Optional[BaseObjective] = None
        self._optimizer_info: Dict[str, Any] = {}

        # Set random seed if provided
        if random_state is not None:
            np.random.seed(random_state)

    def fit(self, X: np.ndarray) -> "ProjectionPursuit":
        """
        Fit the ProjectionPursuit model to the data.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            self: The fitted model
        """
        start_time = time.time()

        # Check input
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        n_samples, n_features = X.shape

        if self.n_components > n_features:
            warnings.warn(
                f"n_components={self.n_components} must be <= n_features={n_features}. "
                f"Setting n_components={n_features}"
            )
            self.n_components = n_features

        # Scale data if requested
        if self.center or self.scale:
            X_scaled, self._scaler = standardize_data(X, self.center, self.scale)
        else:
            X_scaled = X
            self._scaler = None

        # Initialize objective function
        if self.objective == Objective.RECONSTRUCTION:
            self._objective_func = ReconstructionObjective(alpha=self.alpha)
        else:  # DISTANCE_DISTORTION
            # Compute pairwise distances for distance distortion
            dist_X = squareform(pdist(X_scaled, metric="euclidean"))

            # Create weight matrix if requested
            if self.weight_by_distance:
                weight_matrix = 1.0 / (
                    dist_X + 0.1
                )  # Add small constant to avoid division by zero
                np.fill_diagonal(weight_matrix, 0)  # Ignore self-distances
                weight_matrix = weight_matrix / weight_matrix.sum()  # Normalize
            else:
                weight_matrix = None

            self._objective_func = DistanceObjective(
                alpha=self.alpha, weight_by_distance=self.weight_by_distance
            )
            objective_kwargs = {"dist_X": dist_X, "weight_matrix": weight_matrix}

        # Try multiple initializations and keep the best result
        best_loss = np.inf
        best_a = None

        # Try PCA initialization
        if self.verbose:
            print("Trying PCA initialization...")

        pca = PCA(n_components=self.n_components)
        _ = pca.fit_transform(X_scaled)
        a0_pca = pca.components_  # Use PCA directions as starting point

        # Create optimizer
        optimizer = ScipyOptimizer(
            objective_func=self._objective_func,
            n_components=self.n_components,
            method=self.optimizer,
            max_iter=self.max_iter,
            tol=self.tol,
            random_state=self.random_state,
            verbose=self.verbose,
        )

        # Run optimization with PCA initialization
        if self.objective == Objective.RECONSTRUCTION:
            a_matrix_pca, loss_pca, info_pca = optimizer.optimize(X_scaled, a0_pca)
        else:  # DISTANCE_DISTORTION
            a_matrix_pca, loss_pca, info_pca = optimizer.optimize(
                X_scaled, a0_pca, dist_X=dist_X, weight_matrix=weight_matrix
            )

        if loss_pca < best_loss:
            best_loss = loss_pca
            best_a = a_matrix_pca
            self._optimizer_info = info_pca
            self._loss_curve.append(loss_pca)

        # Try random initializations
        for i in range(self.n_init):
            if self.verbose:
                print(f"Random initialization {i+1}/{self.n_init}...")

            np.random.seed(
                self.random_state + i if self.random_state is not None else None
            )
            a0_random = np.random.randn(self.n_components, n_features)

            # Normalize each direction
            norms = np.linalg.norm(a0_random, axis=1, keepdims=True)
            a0_random = a0_random / norms

            # Run optimization with random initialization
            if self.objective == Objective.RECONSTRUCTION:
                a_matrix_random, loss_random, info_random = optimizer.optimize(
                    X_scaled, a0_random
                )
            else:  # DISTANCE_DISTORTION
                a_matrix_random, loss_random, info_random = optimizer.optimize(
                    X_scaled, a0_random, dist_X=dist_X, weight_matrix=weight_matrix
                )

            if loss_random < best_loss:
                best_loss = loss_random
                best_a = a_matrix_random
                self._optimizer_info = info_random
                self._loss_curve.append(loss_random)

        if self.verbose:
            print(f"Best optimization loss: {best_loss}")

        # Store the best result
        self._best_loss = best_loss
        self._x_loadings = best_a
        self._fitted = True
        self._fit_time = time.time() - start_time

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply dimensionality reduction to X.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            np.ndarray: Transformed data, shape (n_samples, n_components)
        """
        if not self._fitted:
            raise ValueError(
                "This ProjectionPursuit instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

        # Check input
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        # Scale data if model was fitted with scaling
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        # Project the data using the optimal projection directions
        assert self._x_loadings is not None
        Z = X_scaled @ self._x_loadings.T

        # Apply ridge function
        assert self._objective_func is not None
        Z_transformed = self._objective_func.g(Z, self.alpha)

        return Z_transformed

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit the model with X and apply dimensionality reduction on X.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            np.ndarray: Transformed data, shape (n_samples, n_components)
        """
        self.fit(X)
        return self.transform(X)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """
        Reconstruct X from the projected data.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            np.ndarray: Reconstructed data, shape (n_samples, n_features)
        """
        if not self._fitted:
            raise ValueError(
                "This ProjectionPursuit instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

        # Check input
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        # Scale data if model was fitted with scaling
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        # Reconstruct the data
        assert self._x_loadings is not None
        assert self._objective_func is not None
        if isinstance(self._objective_func, ReconstructionObjective):
            X_hat = self._objective_func.reconstruct(X_scaled, self._x_loadings)
        else:
            # For distance distortion, manually reconstruct
            Z = X_scaled @ self._x_loadings.T
            G = self._objective_func.g(Z, self.alpha)
            X_hat = G @ self._x_loadings

        # Inverse transform if scaling was applied
        if self._scaler is not None:
            X_hat = self._scaler.inverse_transform(X_hat)

        return X_hat

    def reconstruction_error(self, X: np.ndarray) -> float:
        """
        Compute the reconstruction error for X.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            float: Mean squared reconstruction error
        """
        X_hat = self.reconstruct(X)
        return np.mean((X - X_hat) ** 2)

    def distance_distortion(self, X: np.ndarray) -> float:
        """
        Compute the distance distortion for X.

        Args:
            X: Input data, shape (n_samples, n_features)

        Returns:
            float: Mean squared distance distortion
        """
        if not self._fitted:
            raise ValueError(
                "This ProjectionPursuit instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

        # Check input
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        # Scale data if model was fitted with scaling
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        # Compute pairwise distances in original space
        dist_X = squareform(pdist(X_scaled, metric="euclidean"))

        # Project and compute distances in projected space
        Z = self.transform(X)
        dist_Z = squareform(pdist(Z, metric="euclidean"))

        # Compute distance distortion
        distortion = np.mean((dist_X - dist_Z) ** 2)

        return distortion

    def compute_trustworthiness(self, X: np.ndarray, n_neighbors: int = 5) -> float:
        """
        Compute the trustworthiness score for the dimensionality reduction.

        Trustworthiness measures how well the local structure is preserved.
        A score of 1.0 indicates perfect trustworthiness, while a score of 0.0
        indicates that the local structure is not preserved at all.

        Args:
            X: Input data, shape (n_samples, n_features)
            n_neighbors: Number of neighbors to consider for trustworthiness

        Returns:
            float: Trustworthiness score between 0.0 and 1.0
        """
        if not self._fitted:
            raise ValueError(
                "This ProjectionPursuit instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

        # Check input
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        # Scale data if model was fitted with scaling
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        # Project the data
        Z = self.transform(X)

        # Compute trustworthiness (adjust n_neighbors if needed)
        n_samples = X_scaled.shape[0]
        effective_neighbors = min(n_neighbors, max(1, int(n_samples / 2) - 1))
        trust = compute_trustworthiness(X_scaled, Z, n_neighbors=effective_neighbors)

        return trust

    def compute_silhouette(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Compute the silhouette score for the dimensionality reduction.

        Silhouette score measures how well clusters are separated.
        A score close to 1.0 indicates that clusters are well separated,
        while a score close to -1.0 indicates poor separation.

        Args:
            X: Input data, shape (n_samples, n_features)
            labels: Cluster labels for each sample

        Returns:
            float: Silhouette score between -1.0 and 1.0
        """
        if not self._fitted:
            raise ValueError(
                "This ProjectionPursuit instance is not fitted yet. "
                "Call 'fit' before using this method."
            )

        # Check input
        X = np.asarray(X)
        if X.ndim != 2:
            raise ValueError(f"Expected 2D array, got {X.ndim}D array instead")

        # Project the data
        Z = self.transform(X)

        # Check if we have enough samples for each label
        unique_labels, counts = np.unique(labels, return_counts=True)
        if any(counts < 2):
            warnings.warn(
                "Some labels have fewer than 2 samples, silhouette score may be undefined"
            )
            return np.nan

        # Compute silhouette score
        silhouette = compute_silhouette(Z, labels)

        return silhouette

    def evaluate(
        self, X: np.ndarray, labels: Optional[np.ndarray] = None, n_neighbors: int = 5
    ) -> Dict[str, float]:
        """
        Evaluate the dimensionality reduction with multiple metrics.

        Args:
            X: Input data, shape (n_samples, n_features)
            labels: Optional cluster labels for silhouette score
            n_neighbors: Number of neighbors for trustworthiness

        Returns:
            Dict[str, float]: Dictionary with evaluation metrics
        """
        metrics = {}

        # Scale data if model was fitted with scaling
        if self._scaler is not None:
            X_scaled = self._scaler.transform(X)
        else:
            X_scaled = X

        # Transform data
        Z = self.transform(X)

        # Distance distortion
        metrics["distance_distortion"] = compute_distance_distortion(X_scaled, Z)

        # Reconstruction error
        metrics["reconstruction_error"] = self.reconstruction_error(X)

        # Trustworthiness (adjust n_neighbors if needed)
        n_samples = X_scaled.shape[0]
        effective_neighbors = min(n_neighbors, max(1, int(n_samples / 2) - 1))
        metrics["trustworthiness"] = compute_trustworthiness(
            X_scaled, Z, effective_neighbors
        )

        # Silhouette score (if labels provided)
        if labels is not None:
            metrics["silhouette"] = compute_silhouette(Z, labels)

        return metrics

    @property
    def x_loadings_(self) -> np.ndarray:
        """
        Get the projection directions.

        Returns:
            np.ndarray: Projection directions, shape (n_components, n_features)
        """
        if not self._fitted:
            raise ValueError(
                "This ProjectionPursuit instance is not fitted yet. "
                "Call 'fit' before using this method."
            )
        assert self._x_loadings is not None
        return self._x_loadings

    @property
    def loss_curve_(self) -> List[float]:
        """
        Get the loss curve during optimization.

        Returns:
            List[float]: Loss values during optimization
        """
        return self._loss_curve

    @property
    def best_loss_(self) -> float:
        """
        Get the best loss value achieved.

        Returns:
            float: Best loss value
        """
        return self._best_loss

    @property
    def fit_time_(self) -> float:
        """
        Get the time taken to fit the model.

        Returns:
            float: Time in seconds
        """
        return self._fit_time

    @property
    def optimizer_info_(self) -> Dict[str, Any]:
        """
        Get additional information from the optimizer.

        Returns:
            Dict[str, Any]: Optimizer information
        """
        return self._optimizer_info
