"""
Base class for objective functions.
"""
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Dict, Any
import numpy as np

class Objective:
    """
    Objective types for projection pursuit.
    """
    DISTANCE_DISTORTION = "distance_distortion"
    RECONSTRUCTION = "reconstruction"

    def __new__(cls, value=None):
        """
        Create an objective type from a string value.
        
        Args:
            value: Objective type to validate
        
        Returns:
            str: Validated objective type
        
        Raises:
            ValueError: If the objective type is invalid
        """
        if value is None:
            return cls.DISTANCE_DISTORTION
        
        if value in (cls.DISTANCE_DISTORTION, cls.RECONSTRUCTION):
            return value
        
        raise ValueError(f"Invalid objective type: {value}")

# Base abstract objective class
class BaseObjective(ABC):
    """
    Abstract base class for projection pursuit objective functions.
    """
    
    def __init__(self, alpha: float = 1.0, **kwargs):
        """
        Initialize the objective function.
        
        Args:
            alpha: Steepness parameter for ridge functions
            **kwargs: Additional keyword arguments
        """
        
        self.alpha = alpha
        self.kwargs = kwargs
    
    @abstractmethod
    def __call__(self, a_flat: np.ndarray, X: np.ndarray, k: int, **kwargs) -> float:
        """
        Compute the objective function value.
        
        Args:
            a_flat: Flattened projection directions
            X: Input data
            k: Number of projections
            **kwargs: Additional arguments
            
        Returns:
            float: Objective function value
        """
        pass
    
    @staticmethod
    def g(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Apply the ridge function (non-linearity) to projected data.
        
        Args:
            z: Input data, shape (n_samples, n_components)
            alpha: Steepness parameter for the ridge function
            
        Returns:
            np.ndarray: Transformed data with the same shape as z
        """
        return np.tanh(alpha * z)
    
    @staticmethod
    def grad_g(z: np.ndarray, alpha: float = 1.0) -> np.ndarray:
        """
        Compute the gradient of the ridge function.
        
        Args:
            z: Input data, shape (n_samples, n_components)
            alpha: Steepness parameter for the ridge function
            
        Returns:
            np.ndarray: Gradient values with the same shape as z
        """
        t = np.tanh(alpha * z)
        return alpha * (1 - t**2)