"""
Tests for the optimizers module.
"""

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from pyppur.objectives.distance import DistanceDistortionObjective
from pyppur.objectives.reconstruction import ReconstructionObjective
from pyppur.optimizers import GridOptimizer, ScipyOptimizer


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    # Use a subset of the digits dataset
    digits = load_digits()
    X = digits.data[:100]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled


def test_scipy_optimizer_reconstruction(sample_data):
    """Test SciPy optimizer with reconstruction objective."""
    X = sample_data
    n_components = 2

    # Create objective function
    objective = ReconstructionObjective(alpha=1.0)

    # Create optimizer
    optimizer = ScipyOptimizer(
        objective_func=objective,
        n_components=n_components,
        method="L-BFGS-B",
        max_iter=10,  # Low for testing
        random_state=42,
        verbose=False,
    )

    # Generate initial directions
    np.random.seed(42)
    initial_guess = np.random.randn(n_components, X.shape[1])
    initial_guess = initial_guess / np.linalg.norm(initial_guess, axis=1, keepdims=True)

    # Optimize
    directions, loss, info = optimizer.optimize(X, initial_guess)

    # Check results
    assert directions.shape == (n_components, X.shape[1])
    assert isinstance(loss, float)
    assert loss > 0.0
    assert "success" in info
    assert "nfev" in info


def test_scipy_optimizer_distance(sample_data):
    """Test SciPy optimizer with distance distortion objective."""
    X = sample_data
    n_components = 2

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    dist_X = squareform(pdist(X, metric="euclidean"))

    # Create objective function
    objective = DistanceDistortionObjective(alpha=1.0)

    # Create optimizer
    optimizer = ScipyOptimizer(
        objective_func=objective,
        n_components=n_components,
        method="L-BFGS-B",
        max_iter=10,  # Low for testing
        random_state=42,
        verbose=False,
    )

    # Generate initial directions
    np.random.seed(42)
    initial_guess = np.random.randn(n_components, X.shape[1])
    initial_guess = initial_guess / np.linalg.norm(initial_guess, axis=1, keepdims=True)

    # Optimize
    directions, loss, info = optimizer.optimize(X, initial_guess, dist_X=dist_X)

    # Check results
    assert directions.shape == (n_components, X.shape[1])
    assert isinstance(loss, float)
    assert loss > 0.0
    assert "success" in info
    assert "nfev" in info


def test_grid_optimizer_reconstruction(sample_data):
    """Test Grid optimizer with reconstruction objective."""
    X = sample_data
    n_components = 2

    # Create objective function
    objective = ReconstructionObjective(alpha=1.0)

    # Create optimizer
    optimizer = GridOptimizer(
        objective_func=objective,
        n_components=n_components,
        n_directions=10,  # Low for testing
        n_iterations=2,  # Low for testing
        random_state=42,
        verbose=False,
    )

    # Generate initial directions
    np.random.seed(42)
    initial_guess = np.random.randn(n_components, X.shape[1])
    initial_guess = initial_guess / np.linalg.norm(initial_guess, axis=1, keepdims=True)

    # Optimize
    directions, loss, info = optimizer.optimize(X, initial_guess)

    # Check results
    assert directions.shape == (n_components, X.shape[1])
    assert isinstance(loss, float)
    assert loss > 0.0
    assert "success" in info
    assert "loss_per_component" in info
    assert len(info["loss_per_component"]) == n_components


def test_grid_optimizer_distance(sample_data):
    """Test Grid optimizer with distance distortion objective."""
    X = sample_data
    n_components = 2

    # Compute pairwise distances
    from scipy.spatial.distance import pdist, squareform

    dist_X = squareform(pdist(X, metric="euclidean"))

    # Create objective function
    objective = DistanceDistortionObjective(alpha=1.0)

    # Create optimizer
    optimizer = GridOptimizer(
        objective_func=objective,
        n_components=n_components,
        n_directions=10,  # Low for testing
        n_iterations=2,  # Low for testing
        random_state=42,
        verbose=False,
    )

    # Generate initial directions
    np.random.seed(42)
    initial_guess = np.random.randn(n_components, X.shape[1])
    initial_guess = initial_guess / np.linalg.norm(initial_guess, axis=1, keepdims=True)

    # Optimize
    directions, loss, info = optimizer.optimize(X, initial_guess, dist_X=dist_X)

    # Check results
    assert directions.shape == (n_components, X.shape[1])
    assert isinstance(loss, float)
    assert loss > 0.0
    assert "success" in info
    assert "loss_per_component" in info
    assert len(info["loss_per_component"]) == n_components
