"""Basic usage examples for pyppur."""

import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

from pyppur import Objective, ProjectionPursuit
from pyppur.utils.visualization import plot_comparison


def digits_example() -> None:
    """Example with the digits dataset."""
    # Load data
    digits = load_digits()
    X = digits.data  # type: ignore[attr-defined]
    y = digits.target  # type: ignore[attr-defined]

    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Projection pursuit with distance distortion (correlation metric - default)
    print("Running Projection Pursuit (Distance Distortion, correlation metric)...")
    pp_dist_nl = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=0.1,
        distance_metric="correlation",
        use_nonlinearity_in_distance=True,
        n_init=1,
        verbose=True,
    )

    # Fit and transform
    X_pp_dist_nl = pp_dist_nl.fit_transform(X_scaled)

    # Evaluate
    metrics_dist_nl = pp_dist_nl.evaluate(X_scaled, y)
    print("\nDistance Distortion (Correlation) Metrics:")
    for metric, value in metrics_dist_nl.items():
        print(f"  {metric}: {value:.4f}")

    # Projection pursuit with distance distortion (spearman metric)
    print("\nRunning Projection Pursuit (Distance Distortion, spearman metric)...")
    pp_dist_linear = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=0.1,
        distance_metric="spearman",
        use_nonlinearity_in_distance=True,
        n_init=1,
        verbose=True,
    )

    # Fit and transform
    X_pp_dist_linear = pp_dist_linear.fit_transform(X_scaled)

    # Evaluate
    metrics_dist_linear = pp_dist_linear.evaluate(X_scaled, y)
    print("\nDistance Distortion (Spearman) Metrics:")
    for metric, value in metrics_dist_linear.items():
        print(f"  {metric}: {value:.4f}")

    # Projection pursuit with reconstruction loss (tied weights)
    print("\nRunning Projection Pursuit (Reconstruction Tied Weights)...")
    pp_recon_tied = ProjectionPursuit(
        n_components=2,
        objective=Objective.RECONSTRUCTION,
        alpha=1.0,
        tied_weights=True,
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
        alpha=1.0,
        tied_weights=False,
        l2_reg=0.01,
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
        "Dist (Corr)": X_pp_dist_nl,
        "Dist (Spearman)": X_pp_dist_linear,
        "Recon (Tied)": X_pp_recon_tied,
        "Recon (Free)": X_pp_recon_free,
    }

    metrics = {
        "Dist (Corr)": metrics_dist_nl,
        "Dist (Spearman)": metrics_dist_linear,
        "Recon (Tied)": metrics_recon_tied,
        "Recon (Free)": metrics_recon_free,
    }

    # Plot comparison
    fig = plot_comparison(embeddings, y, metrics)
    plt.tight_layout()
    plt.savefig("digits_comparison.png", dpi=300)
    plt.close()

    print("\nComparison plot saved as 'digits_comparison.png'")


if __name__ == "__main__":
    digits_example()
