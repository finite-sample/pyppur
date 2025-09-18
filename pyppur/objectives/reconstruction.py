"""
Reconstruction loss objective for projection pursuit.
"""

from typing import Optional

import numpy as np

from pyppur.objectives.base import BaseObjective


class ReconstructionObjective(BaseObjective):
    """
    Reconstruction loss objective function for projection pursuit.

    This objective minimizes the reconstruction error when
    projecting and reconstructing data. Supports both tied-weights
    (encoder=decoder) and free decoder configurations.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        tied_weights: bool = True,
        l2_reg: float = 0.0,
        **kwargs,
    ):
        """
        Initialize the reconstruction objective.

        Args:
            alpha: Steepness parameter for the ridge function
            tied_weights: If True, use tied weights (B=A). If False, learn separate
                decoder B
            l2_reg: L2 regularization strength for decoder weights (when
                tied_weights=False)
            **kwargs: Additional keyword arguments
        """
        super().__init__(alpha=alpha, **kwargs)
        self.tied_weights = tied_weights
        self.l2_reg = l2_reg

    def __call__(self, a_flat: np.ndarray, X: np.ndarray, k: int, **kwargs) -> float:
        """
        Compute the reconstruction objective.

        Args:
            a_flat: Flattened parameters (encoder A, and decoder B if untied)
            X: Input data
            k: Number of projections
            **kwargs: Additional arguments

        Returns:
            float: Reconstruction loss value (to be minimized)
        """
        n_features = X.shape[1]

        if self.tied_weights:
            # Tied weights: only encoder parameters
            a_matrix = a_flat.reshape(k, n_features)
            # Note: Normalization is now handled in the optimizer, not here
            b_matrix = a_matrix  # Decoder = encoder
        else:
            # Untied weights: encoder A and decoder B parameters
            total_encoder_params = k * n_features
            total_decoder_params = k * n_features

            if len(a_flat) != total_encoder_params + total_decoder_params:
                raise ValueError(
                    f"Expected {total_encoder_params + total_decoder_params} "
                    f"parameters for untied weights, got {len(a_flat)}"
                )

            # Split parameters
            a_matrix = a_flat[:total_encoder_params].reshape(k, n_features)
            b_matrix = a_flat[total_encoder_params:].reshape(k, n_features)

            # Note: Normalization is now handled in the optimizer, not here

        # Project the data
        Z = self.g(X @ a_matrix.T, self.alpha)

        # Reconstruct the data
        X_hat = Z @ b_matrix

        # Mean squared reconstruction error
        loss = np.mean((X - X_hat) ** 2)

        # Add L2 regularization for decoder (when untied)
        if not self.tied_weights and self.l2_reg > 0:
            loss += self.l2_reg * np.mean(b_matrix**2)

        return loss

    def reconstruct(
        self, X: np.ndarray, a_matrix: np.ndarray, b_matrix: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Reconstruct data from projections.

        Args:
            X: Input data
            a_matrix: Encoder projection matrix
            b_matrix: Decoder matrix (if None, uses tied weights with a_matrix)

        Returns:
            np.ndarray: Reconstructed data
        """
        # Project the data
        Z = self.g(X @ a_matrix.T, self.alpha)

        # Reconstruct the data
        if b_matrix is None:
            # Use tied weights
            X_hat = Z @ a_matrix
        else:
            # Use separate decoder
            X_hat = Z @ b_matrix

        return X_hat
