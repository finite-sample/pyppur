"""
Distance distortion objective for projection pursuit.
"""

from typing import Any, Literal

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr

from pyppur.objectives.base import BaseObjective

DistanceMetric = Literal["mse", "correlation", "spearman"]


class DistanceObjective(BaseObjective):
    """Distance distortion objective function for projection pursuit.

    This objective minimizes the difference between pairwise distances
    in the original space and the projected space. Can optionally apply
    ridge function nonlinearity before distance computation.

    The distance_metric parameter controls how distance preservation is measured:
    - 'mse': Mean squared error between distance matrices (default, scale-sensitive)
    - 'correlation': Negative Pearson correlation (scale-invariant, recommended)
    - 'spearman': Negative Spearman rank correlation (scale and monotonic-invariant)
    """

    def __init__(
        self,
        alpha: float = 0.1,
        weight_by_distance: bool = False,
        use_nonlinearity: bool = True,
        distance_metric: DistanceMetric = "correlation",
        **kwargs: Any,
    ) -> None:
        """Initialize the distance distortion objective.

        Args:
            alpha: Steepness parameter for the ridge function.
            weight_by_distance: Whether to weight distortion by inverse of
                original distances.
            use_nonlinearity: Whether to apply ridge function before computing
                distances.
            distance_metric: How to measure distance preservation. Options:
                - 'mse': Mean squared error (scale-sensitive, original behavior)
                - 'correlation': Negative Pearson correlation (scale-invariant)
                - 'spearman': Negative Spearman rank correlation
            **kwargs: Additional keyword arguments.
        """
        super().__init__(alpha=alpha, **kwargs)
        self.weight_by_distance = weight_by_distance
        self.use_nonlinearity = use_nonlinearity
        if distance_metric not in ("mse", "correlation", "spearman"):
            raise ValueError(
                f"distance_metric must be 'mse', 'correlation', or 'spearman', "
                f"got '{distance_metric}'"
            )
        self.distance_metric = distance_metric

    def __call__(
        self,
        a_flat: np.ndarray,
        X: np.ndarray,
        k: int,
        dist_X: np.ndarray | None = None,
        weight_matrix: np.ndarray | None = None,
        **kwargs: Any,
    ) -> float:
        """Compute the distance distortion objective.

        Args:
            a_flat: Flattened projection directions.
            X: Input data.
            k: Number of projections.
            dist_X: Pairwise distances in original space (optional).
            weight_matrix: Optional weight matrix for distances.
            **kwargs: Additional arguments.

        Returns:
            Distance distortion value (to be minimized).
        """
        # Reshape the flat parameter vector into a matrix
        a_matrix = a_flat.reshape(k, X.shape[1])

        # Note: Normalization is now handled in the optimizer, not here

        # Compute distances in original space if not provided
        if dist_X is None:
            dist_X = squareform(pdist(X, metric="euclidean"))

        # Create weight matrix if requested and not provided
        if self.weight_by_distance and weight_matrix is None:
            # Weight by inverse of distances (emphasize preserving small distances)
            weight_matrix = 1.0 / (dist_X + 0.1)  # Add small constant
            np.fill_diagonal(weight_matrix, 0)  # Ignore self-distances
            weight_matrix = weight_matrix / weight_matrix.sum()  # Normalize

        # Project the data
        Y = X @ a_matrix.T

        if self.use_nonlinearity:
            # Apply ridge function before computing distances
            Z = self.g(Y, self.alpha)
        else:
            # Use linear projections for distance computation
            Z = Y

        # Compute distances in projection space
        dist_Z = squareform(pdist(Z, metric="euclidean"))

        # Calculate the distortion based on the chosen metric
        if self.distance_metric == "correlation":
            # Use Pearson correlation (scale-invariant)
            # Extract upper triangular elements (excluding diagonal)
            triu_idx = np.triu_indices_from(dist_X, k=1)
            d_orig_flat = dist_X[triu_idx]
            d_embed_flat = dist_Z[triu_idx]

            # Compute Pearson correlation
            if np.std(d_embed_flat) < 1e-10:
                # If all embedded distances are the same, correlation is undefined
                return 1.0

            corr = np.corrcoef(d_orig_flat, d_embed_flat)[0, 1]
            # Minimize negative correlation (maximize correlation)
            loss = -corr

        elif self.distance_metric == "spearman":
            # Use Spearman rank correlation (scale and monotonic-invariant)
            triu_idx = np.triu_indices_from(dist_X, k=1)
            d_orig_flat = dist_X[triu_idx]
            d_embed_flat = dist_Z[triu_idx]

            if np.std(d_embed_flat) < 1e-10:
                return 1.0

            result = spearmanr(d_orig_flat, d_embed_flat)
            corr = float(result.correlation)  # type: ignore[union-attr]
            loss = -corr

        else:  # mse
            # Mean squared error (original behavior, scale-sensitive)
            if weight_matrix is not None:
                loss = np.mean(weight_matrix * (dist_X - dist_Z) ** 2)
            else:
                loss = np.mean((dist_X - dist_Z) ** 2)

        return float(loss)
