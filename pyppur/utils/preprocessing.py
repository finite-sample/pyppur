"""
Preprocessing utilities for projection pursuit.
"""

from typing import Optional, Tuple

import numpy as np
from sklearn.preprocessing import StandardScaler


def standardize_data(
    X: np.ndarray,
    center: bool = True,
    scale: bool = True,
    scaler: Optional[StandardScaler] = None,
) -> Tuple[np.ndarray, StandardScaler]:
    """
    Standardize data for projection pursuit.

    Args:
        X: Input data, shape (n_samples, n_features)
        center: Whether to center the data
        scale: Whether to scale the data to unit variance
        scaler: Optional pre-fitted scaler for transform-only operation

    Returns:
        Tuple[np.ndarray, StandardScaler]: Standardized data and the scaler
    """
    if scaler is None:
        scaler = StandardScaler(with_mean=center, with_std=scale)
        X_std = scaler.fit_transform(X)
    else:
        X_std = scaler.transform(X)

    return X_std, scaler
