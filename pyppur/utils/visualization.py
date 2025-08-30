"""
Visualization utilities for projection pursuit results.
"""

from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure


def plot_embedding(
    X_embedded: np.ndarray,
    labels: Optional[np.ndarray] = None,
    title: str = "Projection Pursuit Embedding",
    metrics: Optional[Dict[str, float]] = None,
    figsize: Tuple[float, float] = (10, 8),
    cmap: str = "tab10",
    alpha: float = 0.7,
    s: float = 30.0,
    ax: Optional[Axes] = None,
) -> Tuple[Figure, Axes]:
    """
    Plot the results of a projection pursuit embedding.

    Args:
        X_embedded: Embedded data, shape (n_samples, 2) or (n_samples, 3)
        labels: Optional labels for coloring points
        title: Plot title
        metrics: Optional dictionary of metrics to include in title
        figsize: Figure size (width, height) in inches
        cmap: Colormap name
        alpha: Transparency of points
        s: Point size
        ax: Optional axes to plot on

    Returns:
        Tuple[Figure, Axes]: Figure and Axes objects
    """
    if X_embedded.shape[1] not in (2, 3):
        raise ValueError(
            "Can only plot 2D or 3D embeddings, " f"got {X_embedded.shape[1]}D"
        )

    is_3d = X_embedded.shape[1] == 3

    if ax is None:
        fig = plt.figure(figsize=figsize)
        if is_3d:
            ax = fig.add_subplot(111, projection="3d")
        else:
            ax = fig.add_subplot(111)
    else:
        fig = ax.figure

    # Add metrics to title if provided
    full_title = title
    if metrics is not None:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        full_title = f"{title}\n{metrics_str}"

    ax.set_title(full_title)

    # Plot the embedding
    if is_3d:
        if labels is not None:
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                X_embedded[:, 2],
                c=labels,
                cmap=cmap,
                alpha=alpha,
                s=s,
            )
            plt.colorbar(scatter, ax=ax, label="Class")
        else:
            ax.scatter(
                X_embedded[:, 0], X_embedded[:, 1], X_embedded[:, 2], alpha=alpha, s=s
            )
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_zlabel("Component 3")
    else:
        if labels is not None:
            scatter = ax.scatter(
                X_embedded[:, 0],
                X_embedded[:, 1],
                c=labels,
                cmap=cmap,
                alpha=alpha,
                s=s,
            )
            plt.colorbar(scatter, ax=ax, label="Class")
        else:
            ax.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=alpha, s=s)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    ax.grid(True, linestyle="--", alpha=0.7)

    return fig, ax


def plot_reconstruction(X, X_recon, n_samples=3):
    """
    Plot reconstructed samples alongside original samples.

    Args:
        X: Original data
        X_recon: Reconstructed data
        n_samples: Number of samples to plot

    Returns:
        matplotlib Figure
    """
    import matplotlib.pyplot as plt

    # Ensure X and X_recon are numpy arrays
    X = np.asarray(X)
    X_recon = np.asarray(X_recon)

    # Limit to specified number of samples
    n_samples = min(n_samples, X.shape[0])

    # Reshape data if it's 1D or has more than 2 dimensions
    if X.ndim > 2:
        X = X.reshape(X.shape[0], -1)
    if X_recon.ndim > 2:
        X_recon = X_recon.reshape(X_recon.shape[0], -1)

    # Reshape for visualization
    X_viz = X[:n_samples]
    X_recon_viz = X_recon[:n_samples]

    # Create figure
    fig, axes = plt.subplots(2, n_samples, figsize=(n_samples * 3, 6))

    # Normalize for consistent color scaling
    from matplotlib.colors import Normalize

    norm = Normalize(
        vmin=min(X_viz.min(), X_recon_viz.min()),
        vmax=max(X_viz.max(), X_recon_viz.max()),
    )

    # Plot original and reconstructed samples
    for i in range(n_samples):
        # Check if data can be reshaped to a square image
        n_features = X_viz.shape[1]
        sqrt_features = int(np.sqrt(n_features))

        if sqrt_features**2 == n_features:
            # Square image data - reshape and plot as image
            original = X_viz[i].reshape(sqrt_features, sqrt_features)
            im1 = axes[0, i].imshow(original, cmap="viridis", norm=norm)
            axes[0, i].set_title("Original")
            axes[0, i].axis("off")

            reconstructed = X_recon_viz[i].reshape(sqrt_features, sqrt_features)
            im2 = axes[1, i].imshow(reconstructed, cmap="viridis", norm=norm)
            axes[1, i].set_title("Reconstructed")
            axes[1, i].axis("off")
        else:
            # Non-image data - plot as line plots
            axes[0, i].plot(X_viz[i], "b-", alpha=0.7)
            axes[0, i].set_title("Original")
            axes[0, i].grid(True, alpha=0.3)

            axes[1, i].plot(X_recon_viz[i], "r-", alpha=0.7)
            axes[1, i].set_title("Reconstructed")
            axes[1, i].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_comparison(
    embeddings: Dict[str, np.ndarray],
    labels: Optional[np.ndarray] = None,
    metrics: Optional[Dict[str, Dict[str, float]]] = None,
    title: Optional[str] = None,  # <-- add this line
    figsize: Tuple[float, float] = (15, 5),
    cmap: str = "tab10",
    alpha: float = 0.7,
    s: float = 30.0,
) -> Figure:
    """
    Plot a comparison of multiple embeddings.

    Args:
        embeddings: Dictionary of embeddings {name: embedded_data}
        labels: Optional labels for coloring points
        metrics: Optional dictionary of metrics for each embedding
        title: Optional overall figure title
        figsize: Figure size (width, height) in inches
        cmap: Colormap name
        alpha: Transparency of points
        s: Point size

    Returns:
        Figure: matplotlib Figure object
    """
    n_plots = len(embeddings)

    # Check that all embeddings have the same dimensionality
    dimensions = [embedding.shape[1] for embedding in embeddings.values()]
    if len(set(dimensions)) > 1:
        raise ValueError("All embeddings must have the same number of dimensions")

    # Check that dimensions are supported (2D or 3D)
    embedding_dim = dimensions[0]
    if embedding_dim not in (2, 3):
        raise ValueError(f"Can only plot 2D or 3D embeddings, got {embedding_dim}D")

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)

    if n_plots == 1:
        axes = [axes]

    for i, (name, embedding) in enumerate(embeddings.items()):

        # Get metrics for this embedding if available
        plot_metrics = None
        if metrics is not None and name in metrics:
            plot_metrics = metrics[name]

        # Plot embedding
        _, _ = plot_embedding(
            embedding,
            labels,
            title=name,
            metrics=plot_metrics,
            cmap=cmap,
            alpha=alpha,
            s=s,
            ax=axes[i],
        )

    # Add overall title if provided
    if title:
        fig.suptitle(title, fontsize=16, y=1.02)

    plt.tight_layout()

    return fig
