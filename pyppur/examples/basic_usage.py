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

    # Projection pursuit with distance distortion (with nonlinearity)
    print("Running Projection Pursuit (Distance Distortion with Nonlinearity)...")
    pp_dist_nl = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=1.5,  # Steepness of the ridge function
        use_nonlinearity_in_distance=True,  # Apply tanh before distance computation
        n_init=1,  # Number of random initializations
        verbose=True,
    )

    # Fit and transform
    X_pp_dist_nl = pp_dist_nl.fit_transform(X_scaled)

    # Evaluate
    metrics_dist_nl = pp_dist_nl.evaluate(X_scaled, y)
    print("\nDistance Distortion (Nonlinear) Metrics:")
    for metric, value in metrics_dist_nl.items():
        print(f"  {metric}: {value:.4f}")

    # Projection pursuit with distance distortion (linear)
    print("\nRunning Projection Pursuit (Distance Distortion Linear)...")
    pp_dist_linear = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=1.5,
        use_nonlinearity_in_distance=False,  # Linear distance preservation
        n_init=1,
        verbose=True,
    )

    # Fit and transform
    X_pp_dist_linear = pp_dist_linear.fit_transform(X_scaled)

    # Evaluate
    metrics_dist_linear = pp_dist_linear.evaluate(X_scaled, y)
    print("\nDistance Distortion (Linear) Metrics:")
    for metric, value in metrics_dist_linear.items():
        print(f"  {metric}: {value:.4f}")

    # Projection pursuit with reconstruction loss (tied weights)
    print("\nRunning Projection Pursuit (Reconstruction Tied Weights)...")
    pp_recon_tied = ProjectionPursuit(
        n_components=2,
        objective=Objective.RECONSTRUCTION,
        alpha=1.5,
        tied_weights=True,  # Traditional tied-weights autoencoder
        n_init=1,
        verbose=True,
    )

    # Fit and transform
    X_pp_recon_tied = pp_recon_tied.fit_transform(X_scaled)

    # Evaluate
    metrics_recon_tied = pp_recon_tied.evaluate(X_scaled, y)
    print("\nReconstruction (Tied) Metrics:")
    for metric, value in metrics_recon_tied.items():
        print(f"  {metric}: {value:.4f}")

    # Projection pursuit with reconstruction loss (free decoder)
    print("\nRunning Projection Pursuit (Reconstruction Free Decoder)...")
    pp_recon_free = ProjectionPursuit(
        n_components=2,
        objective=Objective.RECONSTRUCTION,
        alpha=1.5,
        tied_weights=False,  # Separate encoder and decoder
        l2_reg=0.01,  # Regularize decoder weights
        n_init=1,
        verbose=True,
    )

    # Fit and transform
    X_pp_recon_free = pp_recon_free.fit_transform(X_scaled)

    # Evaluate
    metrics_recon_free = pp_recon_free.evaluate(X_scaled, y)
    print("\nReconstruction (Free Decoder) Metrics:")
    for metric, value in metrics_recon_free.items():
        print(f"  {metric}: {value:.4f}")

    # Compare embeddings
    embeddings = {
        "Dist (Nonlinear)": X_pp_dist_nl, 
        "Dist (Linear)": X_pp_dist_linear,
        "Recon (Tied)": X_pp_recon_tied, 
        "Recon (Free)": X_pp_recon_free
    }

    metrics = {
        "Dist (Nonlinear)": metrics_dist_nl,
        "Dist (Linear)": metrics_dist_linear, 
        "Recon (Tied)": metrics_recon_tied, 
        "Recon (Free)": metrics_recon_free
    }

    # Plot comparison
    fig = plot_comparison(embeddings, y, metrics)
    plt.tight_layout()
    plt.savefig("digits_comparison.png", dpi=300)
    plt.close()

    print("\nComparison plot saved as 'digits_comparison.png'")


if __name__ == "__main__":
    digits_example()
