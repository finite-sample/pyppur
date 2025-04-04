"""
Reconstruction loss objective for projection pursuit.
"""
from typing import Optional, Dict, Any
import numpy as np
from pyppur.objectives.base import BaseObjective

class ReconstructionObjective(BaseObjective):
    """
    Reconstruction loss objective function for projection pursuit.
    
    This objective minimizes the reconstruction error when 
    projecting and reconstructing data.
    """
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        """
        Initialize the reconstruction objective.
        
        Args:
            alpha: Steepness parameter for the ridge function
            **kwargs: Additional keyword arguments
        """
        super().__init__(alpha=alpha, **kwargs)
    
    def __call__(
        self, 
        a_flat: np.ndarray, 
        X: np.ndarray, 
        k: int, 
        **kwargs
    ) -> float:
        """
        Compute the reconstruction objective.
        
        Args:
            a_flat: Flattened projection directions
            X: Input data
            k: Number of projections
            **kwargs: Additional arguments
            
        Returns:
            float: Reconstruction loss value (to be minimized)
        """
        # Reshape the flat parameter vector into a matrix
        a_matrix = a_flat.reshape(k, X.shape[1])
        
        # Normalize projection directions
        a_matrix = a_matrix / np.linalg.norm(a_matrix, axis=1, keepdims=True)
        
        # Project the data
        Z = self.g(X @ a_matrix.T, self.alpha)
        
        # Reconstruct the data
        X_hat = Z @ a_matrix
        
        # Mean squared reconstruction error
        loss = np.mean((X - X_hat) ** 2)
        
        return loss
    
    def reconstruct(self, X: np.ndarray, a_matrix: np.ndarray) -> np.ndarray:
        """
        Reconstruct data from projections.
        
        Args:
            X: Input data
            a_matrix: Projection matrix
        
        Returns:
            np.ndarray: Reconstructed data
        """
        # Project the data
        Z = self.g(X @ a_matrix.T, self.alpha)
        
        # Reconstruct the data
        X_hat = Z @ a_matrix
        
        return X_hat