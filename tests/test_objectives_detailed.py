"""
Detailed tests for objective functions.
"""

import numpy as np
import pytest
from scipy.spatial.distance import pdist, squareform

from pyppur.objectives.base import BaseObjective, Objective
from pyppur.objectives.distance import DistanceDistortionObjective
from pyppur.objectives.reconstruction import ReconstructionObjective


def test_base_objective_ridge_functions():
    """Test the base objective ridge functions."""
    # Test g function (tanh)
    z = np.array([[0.0, 1.0, -1.0], [2.0, -2.0, 0.5]])
    alpha = 1.5

    g_result = BaseObjective.g(z, alpha)
    expected = np.tanh(alpha * z)

    assert np.allclose(g_result, expected)

    # Test gradient
    grad_result = BaseObjective.grad_g(z, alpha)
    expected_grad = alpha * (1 - np.tanh(alpha * z) ** 2)

    assert np.allclose(grad_result, expected_grad)


def test_distance_distortion_objective():
    """Test distance distortion objective."""
    np.random.seed(42)
    X = np.random.randn(20, 5)
    n_components = 2

    # Compute distance matrix
    dist_X = squareform(pdist(X, metric="euclidean"))

    # Create objective
    objective = DistanceDistortionObjective(alpha=1.0)

    # Create projection directions
    A = np.random.randn(n_components, X.shape[1])
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    a_flat = A.flatten()

    # Test objective function
    loss = objective(a_flat, X, n_components, dist_X=dist_X)
    assert isinstance(loss, float)
    assert loss >= 0.0

    # Test with weight matrix
    weight_matrix = np.random.rand(*dist_X.shape)
    weight_matrix = weight_matrix / weight_matrix.sum()
    np.fill_diagonal(weight_matrix, 0)

    objective_weighted = DistanceDistortionObjective(alpha=1.0, weight_by_distance=True)
    loss_weighted = objective_weighted(
        a_flat, X, n_components, dist_X=dist_X, weight_matrix=weight_matrix
    )
    assert isinstance(loss_weighted, float)
    assert loss_weighted >= 0.0


def test_reconstruction_objective():
    """Test reconstruction objective."""
    np.random.seed(42)
    X = np.random.randn(20, 5)
    n_components = 2

    # Create objective
    objective = ReconstructionObjective(alpha=1.0)

    # Create projection directions
    A = np.random.randn(n_components, X.shape[1])
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    a_flat = A.flatten()

    # Test objective function
    loss = objective(a_flat, X, n_components)
    assert isinstance(loss, float)
    assert loss >= 0.0

    # Test reconstruction
    X_reconstructed = objective.reconstruct(X, A)
    assert X_reconstructed.shape == X.shape
    assert isinstance(X_reconstructed, np.ndarray)


def test_objective_validation_edge_cases():
    """Test objective validation edge cases."""
    # Test with None
    assert Objective() == Objective.DISTANCE_DISTORTION

    # Test with valid strings
    assert Objective("distance_distortion") == Objective.DISTANCE_DISTORTION
    assert Objective("reconstruction") == Objective.RECONSTRUCTION

    # Test with invalid string
    with pytest.raises(ValueError, match="Invalid objective type"):
        Objective("invalid_objective")


def test_ridge_function_properties():
    """Test properties of ridge functions."""
    # Test symmetry
    z = np.array([[-2.0, -1.0, 0.0, 1.0, 2.0]])
    alpha = 1.0

    g_result = BaseObjective.g(z, alpha)

    # tanh is an odd function: f(-x) = -f(x)
    assert np.allclose(g_result[0, :2], -g_result[0, 3:][::-1], atol=1e-10)

    # tanh(0) = 0
    assert np.isclose(g_result[0, 2], 0.0, atol=1e-10)

    # Test gradient at zero
    grad_result = BaseObjective.grad_g(np.array([[0.0]]), alpha)
    assert np.isclose(grad_result[0, 0], alpha)


def test_objective_with_different_alphas():
    """Test objectives with different alpha values."""
    np.random.seed(42)
    X = np.random.randn(10, 3)
    n_components = 2
    A = np.random.randn(n_components, X.shape[1])
    A = A / np.linalg.norm(A, axis=1, keepdims=True)
    a_flat = A.flatten()

    # Test reconstruction objective with different alphas
    alphas = [0.5, 1.0, 2.0, 5.0]
    losses = []

    for alpha in alphas:
        obj = ReconstructionObjective(alpha=alpha)
        loss = obj(a_flat, X, n_components)
        losses.append(loss)
        assert loss >= 0.0

    # Different alphas should generally give different losses
    assert not all(np.isclose(losses[0], loss) for loss in losses[1:])
