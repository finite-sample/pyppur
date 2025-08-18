"""
Basic usage examples for pyppur.
"""

import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from pyppur import Objective, ProjectionPursuit
from pyppur.utils.visualization import plot_comparison, plot_embedding


def digits_example():
    """
    Example with the digits dataset.
    """
    # Load data
    digits = load_digits()
    X = digits.data
    y = digits.target

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Projection pursuit with distance distortion
    print("Running Projection Pursuit (Distance Distortion)...")
    pp_dist = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=1.5,  # Steepness of the ridge function
        n_init=1,  # Number of random initializations
        verbose=True,
    )

    # Fit and transform
    X_pp_dist = pp_dist.fit_transform(X_scaled)

    # Evaluate
    metrics_dist = pp_dist.evaluate(X_scaled, y)
    print("\nDistance Distortion Metrics:")
    for metric, value in metrics_dist.items():
        print(f"  {metric}: {value:.4f}")

    # Projection pursuit with reconstruction loss
    print("\nRunning Projection Pursuit (Reconstruction)...")
    pp_recon = ProjectionPursuit(
        n_components=2,
        objective=Objective.RECONSTRUCTION,
        alpha=1.5,
        n_init=1,
        verbose=True,
    )

    # Fit and transform
    X_pp_recon = pp_recon.fit_transform(X_scaled)

    # Evaluate
    metrics_recon = pp_recon.evaluate(X_scaled, y)
    print("\nReconstruction Metrics:")
    for metric, value in metrics_recon.items():
        print(f"  {metric}: {value:.4f}")

    # Compare embeddings
    embeddings = {"Distance Distortion": X_pp_dist, "Reconstruction": X_pp_recon}

    metrics = {"Distance Distortion": metrics_dist, "Reconstruction": metrics_recon}

    # Plot comparison
    fig = plot_comparison(embeddings, y, metrics)
    plt.tight_layout()
    plt.savefig("digits_comparison.png", dpi=300)
    plt.close()

    print("\nComparison plot saved as 'digits_comparison.png'")


if __name__ == "__main__":
    digits_example()
