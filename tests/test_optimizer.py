"""
Tests for the optimizers module.
"""

import numpy as np
import pytest
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from pyppur.objectives.distance import DistanceObjective
from pyppur.objectives.reconstruction import ReconstructionObjective
from pyppur.optimizers import GridOptimizer, ScipyOptimizer


@pytest.fixture
def sample_data():
    """Fixture for sample data."""
    # Use a subset of the digits dataset
    digits = load_digits()
    X = digits.data[:100]  # type: ignore[attr-defined]
    scaler = StandardScaler()
    return scaler.fit_transform(X)


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

    # Create objective function with MSE metric (loss is always >= 0)
    objective = DistanceObjective(alpha=1.0, distance_metric="mse")

    # Create optimizer
    optimizer = ScipyOptimizer(
        objective_func=objective,
        n_components=n_components,
        method="L-BFGS-B",
        max_iter=10,
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

    # Create objective function with MSE metric (loss is always >= 0)
    objective = DistanceObjective(alpha=1.0, distance_metric="mse")

    # Create optimizer
    optimizer = GridOptimizer(
        objective_func=objective,
        n_components=n_components,
        n_directions=10,
        n_iterations=2,
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


def _sin_loss(a_flat: np.ndarray) -> float:
    return float(np.sin(a_flat).sum())


class _RecordingObjective:
    """Deterministic objective that records every call the optimizer makes."""

    def __init__(self) -> None:
        self.calls: list[tuple[np.ndarray, np.ndarray, int, tuple]] = []

    def __call__(self, a_flat: np.ndarray, X: np.ndarray, k: int, *extra) -> float:
        self.calls.append((a_flat.copy(), X, k, extra))
        return _sin_loss(a_flat)


def test_grid_optimizer_evaluation_protocol():
    """Pin how GridOptimizer calls the objective and picks each component."""
    rng = np.random.RandomState(3)
    X = rng.randn(8, 4)
    n_features = X.shape[1]
    initial_guess = rng.randn(1, n_features)
    dist_X = object()
    objective = _RecordingObjective()
    optimizer = GridOptimizer(
        objective_func=objective,
        n_components=2,
        n_directions=3,
        n_iterations=2,
        random_state=0,
    )

    directions, loss, info = optimizer.optimize(X, initial_guess, dist_X=dist_X)

    # component 0: initial direction + 2 iterations x 3 candidates; component 1:
    # 2 x 3 candidates; then one final evaluation of all components.
    ks = [k for _, _, k, _ in objective.calls]
    assert ks == [1] * 7 + [2] * 6 + [2]
    for a_flat, x_arg, k, extra in objective.calls:
        assert x_arg is X
        assert extra == (dist_X,)
        assert a_flat.shape == (k * n_features,)

    first_flat = objective.calls[0][0]
    np.testing.assert_array_equal(
        first_flat, initial_guess[0] / np.linalg.norm(initial_guess[0])
    )

    for component, calls in enumerate([objective.calls[:7], objective.calls[7:13]]):
        losses = [_sin_loss(a) for a, _, _, _ in calls]
        best = calls[int(np.argmin(losses))][0]
        assert info["loss_per_component"][component] == min(losses)
        np.testing.assert_array_equal(
            directions[component], best[component * n_features :]
        )
        if component == 1:
            for a_flat, _, _, _ in calls:
                np.testing.assert_array_equal(a_flat[:n_features], directions[0])

    np.testing.assert_array_equal(objective.calls[-1][0], directions.flatten())
    assert loss == _sin_loss(directions.flatten())
