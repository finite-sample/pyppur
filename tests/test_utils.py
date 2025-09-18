"""
Tests for the utils module.
"""

import numpy as np
import pytest

from pyppur.utils.metrics import (
    compute_distance_distortion,
    compute_silhouette,
    compute_trustworthiness,
    evaluate_embedding,
)
from pyppur.utils.preprocessing import standardize_data
from pyppur.utils.visualization import (
    plot_comparison,
    plot_embedding,
)


@pytest.fixture
def sample_data():
    """Fixture for sample data and embedding."""
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(50, 10)  # 50 samples, 10 features

    # Generate labels (3 classes)
    labels = np.random.randint(0, 3, size=50)

    # Generate a 2D embedding
    embedding = np.random.randn(50, 2)

    return X, labels, embedding


def test_metrics(sample_data):
    """Test the metrics functions."""
    X, labels, embedding = sample_data

    # Test trustworthiness
    trust = compute_trustworthiness(X, embedding, n_neighbors=5)
    assert isinstance(trust, float)
    assert 0.0 <= trust <= 1.0

    # Test silhouette
    silhouette = compute_silhouette(embedding, labels)
    assert isinstance(silhouette, float)
    assert -1.0 <= silhouette <= 1.0

    # Test distance distortion
    distortion = compute_distance_distortion(X, embedding)
    assert isinstance(distortion, float)
    assert distortion >= 0.0

    # Test evaluate_embedding
    metrics = evaluate_embedding(X, embedding, labels, n_neighbors=5)
    assert "trustworthiness" in metrics
    assert "distance_distortion" in metrics
    assert "silhouette" in metrics

    # Test evaluate_embedding without labels
    metrics = evaluate_embedding(X, embedding)
    assert "trustworthiness" in metrics
    assert "distance_distortion" in metrics
    assert "silhouette" not in metrics


def test_preprocessing():
    """Test the preprocessing functions."""
    # Generate sample data
    np.random.seed(42)
    X = np.random.randn(50, 10)  # 50 samples, 10 features

    # Test standardize_data with centering and scaling
    X_std1, scaler1 = standardize_data(X, center=True, scale=True)
    assert X_std1.shape == X.shape
    assert np.allclose(X_std1.mean(axis=0), 0.0, atol=1e-10)
    assert np.allclose(X_std1.std(axis=0), 1.0, atol=1e-10)

    # Test standardize_data with centering only
    X_std2, scaler2 = standardize_data(X, center=True, scale=False)
    assert X_std2.shape == X.shape
    assert np.allclose(X_std2.mean(axis=0), 0.0, atol=1e-10)
    assert not np.allclose(X_std2.std(axis=0), 1.0)

    # Test standardize_data with scaling only
    X_std3, scaler3 = standardize_data(X, center=False, scale=True)
    assert X_std3.shape == X.shape
    assert not np.allclose(X_std3.mean(axis=0), 0.0)

    # Test standardize_data with pre-fitted scaler
    X_new = np.random.randn(10, 10)
    X_std4, _ = standardize_data(X_new, scaler=scaler1)
    assert X_std4.shape == X_new.shape


def test_visualization_errors():
    """Test error handling in visualization functions."""
    # Generate sample data
    np.random.seed(42)
    embedding_2d = np.random.randn(50, 2)
    embedding_3d = np.random.randn(50, 3)
    embedding_4d = np.random.randn(50, 4)
    labels = np.random.randint(0, 3, size=50)

    # Test plot_embedding with invalid embedding dimensionality
    with pytest.raises(ValueError):
        plot_embedding(embedding_4d, labels)

    # Test plot_comparison with invalid embedding dimensionality
    embeddings = {"Method 1": embedding_2d, "Method 2": embedding_3d}
    with pytest.raises(ValueError):
        plot_comparison(embeddings, labels)
