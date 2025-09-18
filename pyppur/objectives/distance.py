"""
Distance distortion objective for projection pursuit.
"""

from typing import Any, Dict, Optional

import numpy as np
from scipy.spatial.distance import pdist, squareform

from pyppur.objectives.base import BaseObjective


# Rename the class to match the import
class DistanceObjective(BaseObjective):
    """
    Distance distortion objective function for projection pursuit.

    This objective minimizes the difference between pairwise distances
    in the original space and the projected space. Can optionally apply
    ridge function nonlinearity before distance computation.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        weight_by_distance: bool = False,
        use_nonlinearity: bool = True,
        **kwargs,
    ):
        """
        Initialize the distance distortion objective.

        Args:
            alpha: Steepness parameter for the ridge function
            weight_by_distance: Whether to weight distortion by inverse of
                original distances
            use_nonlinearity: Whether to apply ridge function before computing
                distances
            **kwargs: Additional keyword arguments
        """
        super().__init__(alpha=alpha, **kwargs)
        self.weight_by_distance = weight_by_distance
        self.use_nonlinearity = use_nonlinearity

    def __call__(
        self,
        a_flat: np.ndarray,
        X: np.ndarray,
        k: int,
        dist_X: Optional[np.ndarray] = None,
        weight_matrix: Optional[np.ndarray] = None,
        **kwargs,
    ) -> float:
        """
        Compute the distance distortion objective.

        Args:
            a_flat: Flattened projection directions
            X: Input data
            k: Number of projections
            dist_X: Pairwise distances in original space (optional)
            weight_matrix: Optional weight matrix for distances
            **kwargs: Additional arguments

        Returns:
            float: Distance distortion value (to be minimized)
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

        # Calculate the distortion with optional weighting
        if weight_matrix is not None:
            loss = np.mean(weight_matrix * (dist_X - dist_Z) ** 2)
        else:
            loss = np.mean((dist_X - dist_Z) ** 2)

        return loss


# Alias for backward compatibility
DistanceDistortionObjective = DistanceObjective
