"""Distance distortion objective for projection pursuit."""

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
        # Part of the BaseObjective calling convention; this objective takes
        # no extra options.
        **kwargs: Any,  # noqa: ARG002
    ) -> float:
        """Compute the distance distortion objective.

        Args:
            a_flat: Flattened projection directions.
            X: Input data.
            k: Number of projections.
            dist_X: Optional original distances as a square matrix. The
                correlation metrics also accept the condensed order from pdist.
            weight_matrix: Optional weight matrix for distances.
            **kwargs: Additional arguments.

        Returns:
            Distance distortion value (to be minimized).
        """
        # Reshape the flat parameter vector into a matrix
        a_matrix = a_flat.reshape(k, X.shape[1])

        # Note: Normalization is now handled in the optimizer, not here

        # Project the data
        Y = X @ a_matrix.T

        # Ridge function before computing distances, or linear projections.
        Z = self.g(Y, self.alpha) if self.use_nonlinearity else Y

        projected = pdist(Z, metric="euclidean")
        if self.distance_metric != "mse":
            # pdist already orders pairs like the strict upper triangle.
            if dist_X is None:
                original = pdist(X, metric="euclidean")
            elif dist_X.ndim == 2:
                original = dist_X[np.triu_indices_from(dist_X, k=1)]
            else:
                original = dist_X
            if np.std(projected) < 1e-10:
                return 1.0
            if self.distance_metric == "correlation":
                return -float(np.corrcoef(original, projected)[0, 1])
            result = spearmanr(original, projected)
            return -float(result.correlation)  # type: ignore[union-attr]

        # Compute distances in original space if not provided
        if dist_X is None:
            dist_X = squareform(pdist(X, metric="euclidean"))

        # Create weight matrix if requested and not provided
        if self.weight_by_distance and weight_matrix is None:
            # Weight by inverse of distances (emphasize preserving small distances)
            weight_matrix = 1.0 / (dist_X + 0.1)  # Add small constant
            np.fill_diagonal(weight_matrix, 0)  # Ignore self-distances
            weight_matrix = weight_matrix / weight_matrix.sum()  # Normalize

        dist_Z = squareform(projected)
        # Mean squared error (original behavior, scale-sensitive)
        if weight_matrix is not None:
            loss = np.mean(weight_matrix * (dist_X - dist_Z) ** 2)
        else:
            loss = np.mean((dist_X - dist_Z) ** 2)

        return float(loss)
