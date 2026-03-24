"""
Deep investigation into pyppur's failure modes and potential rescues.

Hypotheses to test:
1. Scale mismatch: tanh bounds output to [-1,1], distances can't match
2. Optimization stuck: gradient descent failing to find good solutions
3. Wrong metric: MSE on distances isn't the right objective
4. Alpha misconfiguration: steepness parameter needs tuning

Potential rescues:
1. Scale-invariant metrics (correlation, rank-based)
2. Relative distance preservation instead of absolute
3. Different nonlinearities or learnable alpha
4. Stress normalization (like Sammon mapping)
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_digits, make_swiss_roll, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

from pyppur import ProjectionPursuit, Objective


def diagnose_scale_mismatch(X, name):
    """Diagnose if scale mismatch is the problem."""
    print(f"\n{'='*60}")
    print(f"SCALE MISMATCH DIAGNOSIS: {name}")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Original distances
    d_orig = pdist(X_scaled, metric="euclidean")
    print(f"Original distances: min={d_orig.min():.2f}, max={d_orig.max():.2f}, mean={d_orig.mean():.2f}")

    # PCA embedding distances
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    d_pca = pdist(X_pca, metric="euclidean")
    print(f"PCA distances: min={d_pca.min():.2f}, max={d_pca.max():.2f}, mean={d_pca.mean():.2f}")

    # pyppur with tanh - what are the embedded distances?
    pp = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        use_nonlinearity_in_distance=True,
        random_state=42,
        n_init=3,
    )
    X_pp = pp.fit_transform(X_scaled)
    d_pp = pdist(X_pp, metric="euclidean")
    print(f"pyppur (tanh) distances: min={d_pp.min():.2f}, max={d_pp.max():.2f}, mean={d_pp.mean():.2f}")

    # The theoretical max distance with tanh is 2*sqrt(k) since each dim bounded in [-1,1]
    k = 2
    theoretical_max = 2 * np.sqrt(k)
    print(f"Theoretical max with tanh (k={k}): {theoretical_max:.2f}")

    # Compute scale ratio
    scale_ratio = d_orig.max() / d_pp.max() if d_pp.max() > 0 else float('inf')
    print(f"Scale ratio (orig/pp): {scale_ratio:.2f}x")

    return {
        "orig_range": (d_orig.min(), d_orig.max()),
        "pp_range": (d_pp.min(), d_pp.max()),
        "scale_ratio": scale_ratio,
    }


def test_scaled_objective(X, name):
    """Test if scaling the embedded distances helps."""
    print(f"\n{'='*60}")
    print(f"SCALED DISTANCE TEST: {name}")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Original distances
    d_orig = pdist(X_scaled, metric="euclidean")

    # pyppur embedding
    pp = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        use_nonlinearity_in_distance=True,
        random_state=42,
        n_init=3,
    )
    X_pp = pp.fit_transform(X_scaled)
    d_pp = pdist(X_pp, metric="euclidean")

    # Unscaled correlation
    corr_unscaled, _ = pearsonr(d_orig, d_pp)

    # If we SCALE the embedded distances to match original range
    # This is what the method SHOULD be optimizing if using correlation
    d_pp_scaled = d_pp * (d_orig.std() / d_pp.std()) if d_pp.std() > 0 else d_pp
    d_pp_scaled = d_pp_scaled - d_pp_scaled.mean() + d_orig.mean()
    mse_scaled = np.mean((d_orig - d_pp_scaled) ** 2)
    mse_unscaled = np.mean((d_orig - d_pp) ** 2)

    print(f"MSE (unscaled distances): {mse_unscaled:.4f}")
    print(f"MSE (scaled distances): {mse_scaled:.4f}")
    print(f"Correlation: {corr_unscaled:.4f}")

    # Rank correlation (Spearman) - invariant to monotonic transforms
    rank_corr, _ = spearmanr(d_orig, d_pp)
    print(f"Rank correlation (Spearman): {rank_corr:.4f}")

    return {
        "mse_unscaled": mse_unscaled,
        "mse_scaled": mse_scaled,
        "pearson": corr_unscaled,
        "spearman": rank_corr,
    }


def test_alpha_sensitivity(X, name):
    """Test how alpha affects performance."""
    print(f"\n{'='*60}")
    print(f"ALPHA SENSITIVITY: {name}")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    d_orig = pdist(X_scaled, metric="euclidean")

    alphas = [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    results = []

    for alpha in alphas:
        pp = ProjectionPursuit(
            n_components=2,
            objective=Objective.DISTANCE_DISTORTION,
            alpha=alpha,
            use_nonlinearity_in_distance=True,
            random_state=42,
            n_init=3,
        )
        X_pp = pp.fit_transform(X_scaled)
        d_pp = pdist(X_pp, metric="euclidean")

        corr, _ = pearsonr(d_orig, d_pp)
        rank_corr, _ = spearmanr(d_orig, d_pp)
        results.append({
            "alpha": alpha,
            "pearson": corr,
            "spearman": rank_corr,
            "loss": pp.best_loss_,
        })
        print(f"  alpha={alpha:5.2f}: Pearson={corr:.4f}, Spearman={rank_corr:.4f}, loss={pp.best_loss_:.4f}")

    # Also test alpha very close to 0 (almost linear)
    pp_linear = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        use_nonlinearity_in_distance=False,  # Linear mode
        random_state=42,
        n_init=3,
    )
    X_pp_linear = pp_linear.fit_transform(X_scaled)
    d_pp_linear = pdist(X_pp_linear, metric="euclidean")
    corr_linear, _ = pearsonr(d_orig, d_pp_linear)
    rank_corr_linear, _ = spearmanr(d_orig, d_pp_linear)
    print(f"  Linear (no tanh): Pearson={corr_linear:.4f}, Spearman={rank_corr_linear:.4f}")

    return results


def test_optimization_quality(X, name):
    """Check if optimization is converging properly."""
    print(f"\n{'='*60}")
    print(f"OPTIMIZATION QUALITY: {name}")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Compare different numbers of initializations
    for n_init in [1, 5, 10, 20]:
        pp = ProjectionPursuit(
            n_components=2,
            objective=Objective.DISTANCE_DISTORTION,
            use_nonlinearity_in_distance=True,
            random_state=42,
            n_init=n_init,
            max_iter=1000,
        )
        pp.fit(X_scaled)
        print(f"  n_init={n_init:2d}: best_loss={pp.best_loss_:.6f}, fit_time={pp.fit_time_:.2f}s")


def test_outlier_robustness(n_samples=200):
    """Test if tanh helps with outlier robustness."""
    print(f"\n{'='*60}")
    print(f"OUTLIER ROBUSTNESS TEST")
    print("="*60)

    # Create data with outliers
    np.random.seed(42)
    X_clean = np.random.randn(n_samples, 10)

    # Add 5% outliers
    n_outliers = n_samples // 20
    outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
    X_outliers = X_clean.copy()
    X_outliers[outlier_indices] *= 10  # Make outliers 10x larger

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_outliers)

    # Test methods
    results = {}

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # MDS
    mds = MDS(n_components=2, normalized_stress="auto", random_state=42, max_iter=300)
    X_mds = mds.fit_transform(X_scaled)

    # pyppur with tanh
    pp_tanh = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        use_nonlinearity_in_distance=True,
        random_state=42,
    )
    X_pp_tanh = pp_tanh.fit_transform(X_scaled)

    # pyppur linear
    pp_linear = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        use_nonlinearity_in_distance=False,
        random_state=42,
    )
    X_pp_linear = pp_linear.fit_transform(X_scaled)

    # Measure: how much do outliers dominate the embedding?
    # Compute variance ratio: outlier variance / non-outlier variance
    non_outlier_mask = np.ones(n_samples, dtype=bool)
    non_outlier_mask[outlier_indices] = False

    methods = {
        "PCA": X_pca,
        "MDS": X_mds,
        "pyppur (tanh)": X_pp_tanh,
        "pyppur (linear)": X_pp_linear,
    }

    print("\nVariance ratio (outlier/non-outlier) - lower is more robust:")
    for name, X_embed in methods.items():
        outlier_var = np.var(X_embed[outlier_indices])
        non_outlier_var = np.var(X_embed[non_outlier_mask])
        ratio = outlier_var / non_outlier_var if non_outlier_var > 0 else float('inf')
        print(f"  {name:20s}: {ratio:.4f}")


def test_cluster_separation():
    """Test if pyppur preserves cluster structure."""
    print(f"\n{'='*60}")
    print(f"CLUSTER SEPARATION TEST")
    print("="*60)

    # Create well-separated clusters in high-D
    X, y = make_blobs(n_samples=300, n_features=50, centers=5,
                      cluster_std=1.0, random_state=42)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    d_orig = pdist(X_scaled, metric="euclidean")

    methods = {}

    # PCA
    pca = PCA(n_components=2)
    methods["PCA"] = pca.fit_transform(X_scaled)

    # MDS
    mds = MDS(n_components=2, normalized_stress="auto", random_state=42, max_iter=300)
    methods["MDS"] = mds.fit_transform(X_scaled)

    # pyppur variants
    for use_nl in [True, False]:
        name = f"pyppur ({'tanh' if use_nl else 'linear'})"
        pp = ProjectionPursuit(
            n_components=2,
            objective=Objective.DISTANCE_DISTORTION,
            use_nonlinearity_in_distance=use_nl,
            random_state=42,
            n_init=5,
        )
        methods[name] = pp.fit_transform(X_scaled)

    # Compute silhouette scores
    from sklearn.metrics import silhouette_score
    print("\nSilhouette scores (higher = better cluster separation):")
    for name, X_embed in methods.items():
        score = silhouette_score(X_embed, y)
        d_embed = pdist(X_embed, metric="euclidean")
        corr, _ = pearsonr(d_orig, d_embed)
        print(f"  {name:20s}: silhouette={score:.4f}, dist_corr={corr:.4f}")


def test_sammon_style_objective(X, name):
    """
    Test what happens if we use Sammon-style normalized stress.
    Sammon mapping uses: sum((d_X - d_Z)^2 / d_X) instead of sum((d_X - d_Z)^2)
    This naturally handles scale.
    """
    print(f"\n{'='*60}")
    print(f"SAMMON-STYLE ANALYSIS: {name}")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    d_orig = pdist(X_scaled, metric="euclidean")

    # MDS
    mds = MDS(n_components=2, normalized_stress="auto", random_state=42, max_iter=300)
    X_mds = mds.fit_transform(X_scaled)
    d_mds = pdist(X_mds, metric="euclidean")

    # pyppur
    pp = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        use_nonlinearity_in_distance=True,
        random_state=42,
        n_init=5,
    )
    X_pp = pp.fit_transform(X_scaled)
    d_pp = pdist(X_pp, metric="euclidean")

    # Compute Sammon stress for each
    def sammon_stress(d_orig, d_embed):
        # Avoid division by zero
        mask = d_orig > 1e-10
        return np.sum((d_orig[mask] - d_embed[mask])**2 / d_orig[mask]) / np.sum(d_orig[mask])

    def raw_stress(d_orig, d_embed):
        return np.mean((d_orig - d_embed)**2)

    print(f"Raw stress (pyppur objective):")
    print(f"  MDS: {raw_stress(d_orig, d_mds):.4f}")
    print(f"  pyppur: {raw_stress(d_orig, d_pp):.4f}")

    print(f"\nSammon stress (normalized by distance):")
    print(f"  MDS: {sammon_stress(d_orig, d_mds):.6f}")
    print(f"  pyppur: {sammon_stress(d_orig, d_pp):.6f}")

    print(f"\nCorrelation:")
    print(f"  MDS: {pearsonr(d_orig, d_mds)[0]:.4f}")
    print(f"  pyppur: {pearsonr(d_orig, d_pp)[0]:.4f}")


def main():
    # Load test datasets
    digits = load_digits()
    X_digits = digits.data[:500]

    X_swiss, _ = make_swiss_roll(n_samples=300, noise=0.5, random_state=42)

    X_blobs, _ = make_blobs(n_samples=200, n_features=20, centers=4, random_state=42)

    print("\n" + "#"*60)
    print("# DEEP INVESTIGATION INTO PYPPUR FAILURE MODES")
    print("#"*60)

    # 1. Scale mismatch diagnosis
    print("\n\n" + "="*60)
    print("PART 1: SCALE MISMATCH DIAGNOSIS")
    print("="*60)
    diagnose_scale_mismatch(X_digits, "Digits")
    diagnose_scale_mismatch(X_swiss, "Swiss Roll")

    # 2. Test if scaling fixes things
    print("\n\n" + "="*60)
    print("PART 2: DOES SCALING FIX THE PROBLEM?")
    print("="*60)
    test_scaled_objective(X_digits, "Digits")
    test_scaled_objective(X_swiss, "Swiss Roll")

    # 3. Alpha sensitivity
    print("\n\n" + "="*60)
    print("PART 3: ALPHA SENSITIVITY")
    print("="*60)
    test_alpha_sensitivity(X_digits, "Digits")

    # 4. Optimization quality
    print("\n\n" + "="*60)
    print("PART 4: OPTIMIZATION CONVERGENCE")
    print("="*60)
    test_optimization_quality(X_digits, "Digits")

    # 5. Outlier robustness - potential niche
    print("\n\n" + "="*60)
    print("PART 5: OUTLIER ROBUSTNESS (POTENTIAL NICHE)")
    print("="*60)
    test_outlier_robustness()

    # 6. Cluster separation
    print("\n\n" + "="*60)
    print("PART 6: CLUSTER SEPARATION")
    print("="*60)
    test_cluster_separation()

    # 7. Sammon-style analysis
    print("\n\n" + "="*60)
    print("PART 7: SAMMON-STYLE OBJECTIVE ANALYSIS")
    print("="*60)
    test_sammon_style_objective(X_digits, "Digits")

    # Summary
    print("\n\n" + "#"*60)
    print("# INVESTIGATION SUMMARY")
    print("#"*60)
    print("""
Key findings to analyze:
1. Scale mismatch: Check if embedded distances are much smaller than original
2. Correlation vs MSE: Check if correlation is good even when MSE is bad
3. Alpha sensitivity: Check if there's an optimal alpha that helps
4. Optimization: Check if more initializations help
5. Outlier robustness: Check if tanh provides any advantage
6. Cluster separation: Check if pyppur preserves cluster structure
7. Sammon stress: Check if a different objective formulation would help
""")


if __name__ == "__main__":
    main()
