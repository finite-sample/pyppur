"""
Visualization utilities for projection pursuit results.
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from mpl_toolkits.mplot3d import Axes3D


def _plot_2d_embedding(
    X_embedded: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Projection Pursuit Embedding",
    metrics: dict[str, float] | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "tab10",
    alpha: float = 0.7,
    s: float = 30.0,
    ax: Axes | None = None,
) -> tuple[Figure, Axes]:
    """Plot 2D embedding."""
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Add metrics to title if provided
    full_title = title
    if metrics is not None:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        full_title = f"{title}\n{metrics_str}"

    ax.set_title(full_title)

    # Plot the 2D embedding
    if labels is not None:
        scatter = ax.scatter(
            X_embedded[:, 0],
            X_embedded[:, 1],
            c=labels,
            cmap=cmap,
            alpha=alpha,
            s=int(s),
        )
        plt.colorbar(scatter, ax=ax, label="Class")
    else:
        ax.scatter(X_embedded[:, 0], X_embedded[:, 1], alpha=alpha, s=int(s))

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True, linestyle="--", alpha=0.7)

    assert isinstance(fig, Figure)
    return fig, ax


def _plot_3d_embedding(
    X_embedded: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Projection Pursuit Embedding",
    metrics: dict[str, float] | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "tab10",
    alpha: float = 0.7,
    s: float = 30.0,
    ax: Axes3D | None = None,
) -> tuple[Figure, Axes3D]:
    """Plot 3D embedding."""
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111, projection="3d")
    else:
        fig = ax.get_figure()

    # Add metrics to title if provided
    full_title = title
    if metrics is not None:
        metrics_str = ", ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        full_title = f"{title}\n{metrics_str}"

    ax.set_title(full_title)

    # Plot the 3D embedding
    if labels is not None:
        scatter = ax.scatter(
            X_embedded[:, 0],
            X_embedded[:, 1],
            X_embedded[:, 2],  # type: ignore[arg-type]  # matplotlib 3D scatter typing issue
            c=labels,
            cmap=cmap,
            alpha=alpha,
            s=int(s),
        )
        plt.colorbar(scatter, ax=ax, label="Class")
    else:
        ax.scatter(
            X_embedded[:, 0],
            X_embedded[:, 1],
            X_embedded[:, 2],  # type: ignore[arg-type]  # matplotlib 3D scatter typing issue
            alpha=alpha,
            s=int(s),
        )

    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.set_zlabel("Component 3")
    ax.grid(True, linestyle="--", alpha=0.7)

    assert isinstance(fig, Figure)
    return fig, ax


def plot_embedding(
    X_embedded: np.ndarray,
    labels: np.ndarray | None = None,
    title: str = "Projection Pursuit Embedding",
    metrics: dict[str, float] | None = None,
    figsize: tuple[float, float] = (10, 8),
    cmap: str = "tab10",
    alpha: float = 0.7,
    s: float = 30.0,
    ax: Axes | Axes3D | None = None,
) -> tuple[Figure, Axes | Axes3D]:
    """Plot the results of a projection pursuit embedding.

    Args:
        X_embedded: Embedded data, shape (n_samples, 2) or (n_samples, 3).
        labels: Optional labels for coloring points.
        title: Plot title.
        metrics: Optional dictionary of metrics to include in title.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap name.
        alpha: Transparency of points.
        s: Point size.
        ax: Optional axes to plot on.

    Returns:
        Figure and Axes objects.
    """
    if X_embedded.shape[1] not in (2, 3):
        raise ValueError(
            f"Can only plot 2D or 3D embeddings, got {X_embedded.shape[1]}D"
        )

    # Delegate to the appropriate function based on dimensionality
    if X_embedded.shape[1] == 3:
        # For 3D, ax can only be None or Axes3D
        ax_3d = None if ax is None else ax
        if ax_3d is not None and not isinstance(ax_3d, Axes3D):
            raise ValueError("For 3D plots, ax must be an Axes3D instance or None")
        return _plot_3d_embedding(
            X_embedded, labels, title, metrics, figsize, cmap, alpha, s, ax_3d
        )
    else:
        # For 2D, ax can only be None or regular Axes
        ax_2d = None if ax is None else ax
        if ax_2d is not None and isinstance(ax_2d, Axes3D):
            raise ValueError("For 2D plots, ax must be a regular Axes instance or None")
        return _plot_2d_embedding(
            X_embedded, labels, title, metrics, figsize, cmap, alpha, s, ax_2d
        )


def plot_reconstruction(
    X: np.ndarray, X_recon: np.ndarray, n_samples: int = 3
) -> Figure:
    """Plot reconstructed samples alongside original samples.

    Args:
        X: Original data.
        X_recon: Reconstructed data.
        n_samples: Number of samples to plot.

    Returns:
        matplotlib Figure.
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
    embeddings: dict[str, np.ndarray],
    labels: np.ndarray | None = None,
    metrics: dict[str, dict[str, float]] | None = None,
    title: str | None = None,
    figsize: tuple[float, float] = (15, 5),
    cmap: str = "tab10",
    alpha: float = 0.7,
    s: float = 30.0,
) -> Figure:
    """Plot a comparison of multiple embeddings.

    Args:
        embeddings: Dictionary of embeddings {name: embedded_data}.
        labels: Optional labels for coloring points.
        metrics: Optional dictionary of metrics for each embedding.
        title: Optional overall figure title.
        figsize: Figure size (width, height) in inches.
        cmap: Colormap name.
        alpha: Transparency of points.
        s: Point size.

    Returns:
        matplotlib Figure object.
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
