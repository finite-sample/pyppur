"""
Tests for the ProjectionPursuit class.
"""

import numpy as np
import pytest
from sklearn.datasets import load_digits, make_swiss_roll
from sklearn.preprocessing import StandardScaler

from pyppur import Objective, ProjectionPursuit


@pytest.fixture
def digits_data():
    """Fixture for digits dataset."""
    digits = load_digits()
    X = digits.data[:100]  # Subset for faster testing
    y = digits.target[:100]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y


@pytest.fixture
def swiss_roll_data():
    """Fixture for swiss roll dataset."""
    X, t = make_swiss_roll(n_samples=100, noise=0.1, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    # Create labels based on the position in the roll
    labels = np.zeros(len(t))
    labels[t < np.percentile(t, 33)] = 0
    labels[(t >= np.percentile(t, 33)) & (t < np.percentile(t, 66))] = 1
    labels[t >= np.percentile(t, 66)] = 2
    return X_scaled, labels.astype(int)


def test_init():
    """Test initialization of ProjectionPursuit."""
    # Default initialization
    pp = ProjectionPursuit()
    assert pp.n_components == 2
    assert pp.objective == Objective.DISTANCE_DISTORTION
    assert pp.alpha == 1.0

    # Custom initialization
    pp = ProjectionPursuit(
        n_components=3,
        objective=Objective.RECONSTRUCTION,
        alpha=1.5,
        max_iter=1000,
        random_state=42,
    )
    assert pp.n_components == 3
    assert pp.objective == Objective.RECONSTRUCTION
    assert pp.alpha == 1.5
    assert pp.max_iter == 1000
    assert pp.random_state == 42

    # Test string objective
    pp = ProjectionPursuit(objective="reconstruction")
    assert pp.objective == Objective.RECONSTRUCTION

    # Test invalid objective
    with pytest.raises(ValueError):
        ProjectionPursuit(objective="invalid")


def test_distance_distortion_pipeline(digits_data):
    """Test the full pipeline with distance distortion objective."""
    X, y = digits_data

    pp = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        n_init=1,  # Speed up test
        max_iter=10,  # Speed up test
        random_state=42,
    )

    # Test fit
    pp.fit(X)
    assert pp._fitted
    assert pp.x_loadings_.shape == (2, X.shape[1])

    # Test transform
    Z = pp.transform(X)
    assert Z.shape == (X.shape[0], 2)

    # Test fit_transform
    pp = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        n_init=1,
        max_iter=10,
        random_state=42,
    )
    Z = pp.fit_transform(X)
    assert Z.shape == (X.shape[0], 2)

    # Test transform on new data
    X_subset = X[:10]
    Z_subset = pp.transform(X_subset)
    assert Z_subset.shape == (10, 2)

    # Test reconstruction
    X_hat = pp.reconstruct(X)
    assert X_hat.shape == X.shape

    # Test evaluation metrics
    metrics = pp.evaluate(X, y)
    assert "distance_distortion" in metrics
    assert "reconstruction_error" in metrics
    assert "trustworthiness" in metrics
    assert "silhouette" in metrics


def test_reconstruction_pipeline(digits_data):
    """Test the full pipeline with reconstruction objective."""
    X, y = digits_data

    pp = ProjectionPursuit(
        n_components=2,
        objective=Objective.RECONSTRUCTION,
        n_init=1,  # Speed up test
        max_iter=10,  # Speed up test
        random_state=42,
    )

    # Test fit
    pp.fit(X)
    assert pp._fitted
    assert pp.x_loadings_.shape == (2, X.shape[1])

    # Test transform
    Z = pp.transform(X)
    assert Z.shape == (X.shape[0], 2)

    # Test reconstruction
    X_hat = pp.reconstruct(X)
    assert X_hat.shape == X.shape

    # Test reconstruction error
    recon_error = pp.reconstruction_error(X)
    assert isinstance(recon_error, float)
    assert recon_error >= 0.0


def test_non_linear_data(swiss_roll_data):
    """Test on non-linear data to ensure it works well."""
    X, y = swiss_roll_data

    # Test with distance distortion
    pp_dist = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        n_init=1,
        max_iter=10,
        random_state=42,
    )
    Z_dist = pp_dist.fit_transform(X)
    assert Z_dist.shape == (X.shape[0], 2)

    # Test with reconstruction
    pp_recon = ProjectionPursuit(
        n_components=2,
        objective=Objective.RECONSTRUCTION,
        n_init=1,
        max_iter=10,
        random_state=42,
    )
    Z_recon = pp_recon.fit_transform(X)
    assert Z_recon.shape == (X.shape[0], 2)

    # Compare metrics
    metrics_dist = pp_dist.evaluate(X, y)
    metrics_recon = pp_recon.evaluate(X, y)

    # Both methods should produce results
    assert "trustworthiness" in metrics_dist
    assert "trustworthiness" in metrics_recon


def test_error_handling():
    """Test error handling in the ProjectionPursuit class."""
    pp = ProjectionPursuit()

    # Calling methods before fitting
    with pytest.raises(ValueError):
        pp.transform(np.random.randn(10, 5))

    with pytest.raises(ValueError):
        pp.reconstruct(np.random.randn(10, 5))

    with pytest.raises(ValueError):
        pp.x_loadings_

    # Incorrect input shape
    pp = ProjectionPursuit()
    with pytest.raises(ValueError):
        pp.fit(np.random.randn(10))  # 1D array

    # n_components > n_features
    X = np.random.randn(10, 3)
    pp = ProjectionPursuit(n_components=5)
    with pytest.warns(UserWarning):
        pp.fit(X)
    assert pp.n_components == 3  # Should be adjusted
