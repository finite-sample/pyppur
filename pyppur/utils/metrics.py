"""
Evaluation metrics for dimensionality reduction.
"""

from __future__ import annotations

import warnings

import numpy as np
from scipy.spatial.distance import pdist, squareform
from sklearn.manifold import trustworthiness as sklearn_trustworthiness
from sklearn.metrics import silhouette_score as sklearn_silhouette_score


def compute_trustworthiness(
    X_original: np.ndarray, X_embedded: np.ndarray, n_neighbors: int = 5
) -> float:
    """
    Compute the trustworthiness score for dimensionality reduction.

    Trustworthiness measures how well local neighborhoods are preserved.

    Args:
        X_original: Original high-dimensional data
        X_embedded: Low-dimensional embedding
        n_neighbors: Number of neighbors to consider

    Returns:
        float: Trustworthiness score in range [0, 1]
    """
    return sklearn_trustworthiness(X_original, X_embedded, n_neighbors=n_neighbors)


def compute_silhouette(X_embedded: np.ndarray, labels: np.ndarray) -> float:
    """
    Compute the silhouette score for the embedding.

    The silhouette score measures how well clusters are separated.

    Args:
        X_embedded: Low-dimensional embedding
        labels: Cluster or class labels

    Returns:
        float: Silhouette score in range [-1, 1]
    """
    unique_labels, counts = np.unique(labels, return_counts=True)
    if len(unique_labels) < 2:
        warnings.warn("Silhouette score requires at least two clusters")
        return np.nan

    if any(counts < 2):
        warnings.warn(
            "Some labels have fewer than 2 samples, silhouette may be undefined"
        )
        return np.nan

    return sklearn_silhouette_score(X_embedded, labels)


def compute_distance_distortion(
    X_original: np.ndarray, X_embedded: np.ndarray
) -> float:
    """
    Compute the distance distortion between original and embedded spaces.

    Distance distortion measures how well pairwise distances are preserved.

    Args:
        X_original: Original high-dimensional data
        X_embedded: Low-dimensional embedding

    Returns:
        float: Mean squared distance distortion
    """
    # Compute pairwise distances in original space
    dist_original = squareform(pdist(X_original, metric="euclidean"))

    # Compute pairwise distances in embedded space
    dist_embedded = squareform(pdist(X_embedded, metric="euclidean"))

    # Calculate distortion
    distortion = np.mean((dist_original - dist_embedded) ** 2)

    return distortion


def evaluate_embedding(
    X_original: np.ndarray,
    X_embedded: np.ndarray,
    labels: np.ndarray | None = None,
    n_neighbors: int = 5,
) -> dict[str, float]:
    """
    Evaluate the quality of an embedding using multiple metrics.

    Args:
        X_original: Original high-dimensional data
        X_embedded: Low-dimensional embedding
        labels: Optional cluster or class labels
        n_neighbors: Number of neighbors for trustworthiness

    Returns:
        dict[str, float]: Dictionary with evaluation metrics
    """
    metrics = {}

    # Trustworthiness
    metrics["trustworthiness"] = compute_trustworthiness(
        X_original, X_embedded, n_neighbors
    )

    # Distance distortion
    metrics["distance_distortion"] = compute_distance_distortion(X_original, X_embedded)

    # Silhouette score (if labels provided)
    if labels is not None:
        metrics["silhouette"] = compute_silhouette(X_embedded, labels)

    return metrics
