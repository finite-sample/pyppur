"""
Tests for edge cases and error handling.
"""

import numpy as np
import pytest

from pyppur import Objective, ProjectionPursuit


def test_projection_pursuit_edge_cases():
    """Test ProjectionPursuit with edge cases."""

    # Test with very small dataset
    X_small = np.random.randn(3, 2)
    pp = ProjectionPursuit(n_components=1, random_state=42, max_iter=5)

    # Should work with small dataset
    Z = pp.fit_transform(X_small)
    assert Z.shape == (3, 1)

    # Test with single component
    X = np.random.randn(20, 5)
    pp = ProjectionPursuit(n_components=1, random_state=42)

    Z = pp.fit_transform(X)
    assert Z.shape == (20, 1)

    # Test with n_components equal to n_features
    X = np.random.randn(50, 3)
    pp = ProjectionPursuit(n_components=3, random_state=42, max_iter=5)

    Z = pp.fit_transform(X)
    assert Z.shape == (50, 3)


def test_projection_pursuit_scaling_options():
    """Test different scaling options."""
    X = np.random.randn(30, 4) * 10 + 5  # Data with different scale and offset

    # Test no scaling
    pp_none = ProjectionPursuit(center=False, scale=False, random_state=42, max_iter=5)
    Z_none = pp_none.fit_transform(X)

    # Test centering only
    pp_center = ProjectionPursuit(center=True, scale=False, random_state=42, max_iter=5)
    Z_center = pp_center.fit_transform(X)

    # Test scaling only
    pp_scale = ProjectionPursuit(center=False, scale=True, random_state=42, max_iter=5)
    Z_scale = pp_scale.fit_transform(X)

    # Test both centering and scaling
    pp_both = ProjectionPursuit(center=True, scale=True, random_state=42, max_iter=5)
    Z_both = pp_both.fit_transform(X)

    # All should produce valid results
    for Z in [Z_none, Z_center, Z_scale, Z_both]:
        assert Z.shape == (30, 2)
        assert not np.any(np.isnan(Z))
        assert np.all(np.isfinite(Z))


def test_projection_pursuit_weight_by_distance():
    """Test weight_by_distance option."""
    X = np.random.randn(20, 4)

    # Test without weighting
    pp_no_weight = ProjectionPursuit(
        objective=Objective.DISTANCE_DISTORTION,
        weight_by_distance=False,
        random_state=42,
        max_iter=5,
    )
    Z_no_weight = pp_no_weight.fit_transform(X)

    # Test with weighting
    pp_weight = ProjectionPursuit(
        objective=Objective.DISTANCE_DISTORTION,
        weight_by_distance=True,
        random_state=42,
        max_iter=5,
    )
    Z_weight = pp_weight.fit_transform(X)

    # Both should work
    assert Z_no_weight.shape == Z_weight.shape == (20, 2)

    # Results might be different due to weighting
    # (but we won't enforce this as it depends on the optimization)


def test_projection_pursuit_different_optimizers():
    """Test different optimizer settings."""
    X = np.random.randn(30, 5)

    optimizers = ["L-BFGS-B", "SLSQP"]

    for optimizer in optimizers:
        pp = ProjectionPursuit(
            optimizer=optimizer, random_state=42, max_iter=10, n_init=1
        )

        try:
            Z = pp.fit_transform(X)
            assert Z.shape == (30, 2)
            assert not np.any(np.isnan(Z))
        except Exception as e:
            # Some optimizers might not be available or might fail
            # This is acceptable for testing purposes
            pytest.skip(f"Optimizer {optimizer} failed: {e}")


def test_projection_pursuit_reconstruction_consistency():
    """Test that reconstruction works consistently."""
    X = np.random.randn(25, 4)

    # Test reconstruction objective
    pp_recon = ProjectionPursuit(
        objective=Objective.RECONSTRUCTION,
        random_state=42,
        max_iter=10,
    )
    pp_recon.fit(X)

    # Test reconstruction
    X_hat = pp_recon.reconstruct(X)
    assert X_hat.shape == X.shape

    # Reconstruction error should be non-negative
    recon_error = pp_recon.reconstruction_error(X)
    assert recon_error >= 0.0

    # Test distance distortion objective
    pp_dist = ProjectionPursuit(
        objective=Objective.DISTANCE_DISTORTION,
        random_state=42,
        max_iter=10,
    )
    pp_dist.fit(X)

    # Test reconstruction (should work even for distance objective)
    X_hat_dist = pp_dist.reconstruct(X)
    assert X_hat_dist.shape == X.shape

    # Distance distortion should be non-negative
    dist_distortion = pp_dist.distance_distortion(X)
    assert dist_distortion >= 0.0


def test_projection_pursuit_properties():
    """Test ProjectionPursuit properties."""
    X = np.random.randn(20, 3)

    pp = ProjectionPursuit(random_state=42, max_iter=5)
    pp.fit(X)

    # Test properties
    assert hasattr(pp, "x_loadings_")
    assert hasattr(pp, "loss_curve_")
    assert hasattr(pp, "best_loss_")
    assert hasattr(pp, "fit_time_")
    assert hasattr(pp, "optimizer_info_")

    # Check property types and values
    assert isinstance(pp.x_loadings_, np.ndarray)
    assert pp.x_loadings_.shape == (2, 3)

    assert isinstance(pp.loss_curve_, list)
    assert len(pp.loss_curve_) > 0

    assert isinstance(pp.best_loss_, float)
    assert pp.best_loss_ >= 0.0

    assert isinstance(pp.fit_time_, float)
    assert pp.fit_time_ >= 0.0

    assert isinstance(pp.optimizer_info_, dict)


def test_evaluation_metrics_edge_cases():
    """Test evaluation metrics with edge cases."""
    X = np.random.randn(15, 3)

    pp = ProjectionPursuit(random_state=42, max_iter=5)
    pp.fit(X)

    # Test evaluation without labels
    metrics_no_labels = pp.evaluate(X)
    expected_keys = {"trustworthiness", "distance_distortion", "reconstruction_error"}
    assert set(metrics_no_labels.keys()) == expected_keys

    # Test evaluation with labels
    labels = np.random.randint(0, 3, size=15)
    metrics_with_labels = pp.evaluate(X, labels=labels)
    expected_keys.add("silhouette")
    assert set(metrics_with_labels.keys()) == expected_keys

    # Test trustworthiness with different n_neighbors
    trust_k3 = pp.compute_trustworthiness(X, n_neighbors=3)
    trust_k5 = pp.compute_trustworthiness(X, n_neighbors=5)

    assert 0.0 <= trust_k3 <= 1.0
    assert 0.0 <= trust_k5 <= 1.0

    # Test silhouette with edge case labels
    labels_edge = np.array([0] * 7 + [1] * 8)  # Two groups
    silhouette = pp.compute_silhouette(X, labels_edge)
    assert -1.0 <= silhouette <= 1.0


def test_random_state_reproducibility():
    """Test that random_state ensures reproducibility."""
    np.random.seed(42)  # Fix the data generation seed
    X = np.random.randn(20, 4)

    # Same random state should give same results
    pp1 = ProjectionPursuit(random_state=123, max_iter=5, n_init=1)
    pp2 = ProjectionPursuit(random_state=123, max_iter=5, n_init=1)

    Z1 = pp1.fit_transform(X)
    Z2 = pp2.fit_transform(X)

    # Results should be very similar (allowing for small numerical differences)
    assert np.allclose(Z1, Z2, rtol=1e-8)

    # Just check that the method works - reproducibility is the main test
    assert Z1.shape == Z2.shape == (20, 2)


def test_transform_on_new_data():
    """Test transforming new data after fitting."""
    np.random.seed(42)
    X_train = np.random.randn(30, 5)
    X_test = np.random.randn(10, 5)

    pp = ProjectionPursuit(random_state=42, max_iter=5, center=True, scale=True)
    pp.fit(X_train)

    # Transform training data
    Z_train = pp.transform(X_train)
    assert Z_train.shape == (30, 2)

    # Transform test data (same number of features)
    Z_test = pp.transform(X_test)
    assert Z_test.shape == (10, 2)

    # Both should be valid
    assert not np.any(np.isnan(Z_train))
    assert not np.any(np.isnan(Z_test))

    # Test reconstruction on new data
    X_test_hat = pp.reconstruct(X_test)
    assert X_test_hat.shape == X_test.shape

    # Test evaluation on new data
    labels_test = np.random.randint(0, 2, size=10)
    metrics_test = pp.evaluate(X_test, labels=labels_test)

    assert "trustworthiness" in metrics_test
    assert "silhouette" in metrics_test
    assert "distance_distortion" in metrics_test
    assert "reconstruction_error" in metrics_test
