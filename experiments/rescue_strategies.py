"""
Test concrete rescue strategies for pyppur.

Strategies:
1. Use very small alpha (near-linear) - already shown to help
2. Implement correlation-based objective (scale-invariant)
3. Implement Sammon-style normalized stress
4. Market for outlier robustness use case
5. Add distance scaling post-hoc
"""

import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr, spearmanr
from scipy.optimize import minimize
from sklearn.datasets import load_digits, make_swiss_roll, make_blobs
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

from pyppur import ProjectionPursuit, Objective


def correlation_objective(a_flat, X, k, d_orig, alpha=0.01):
    """
    Alternative objective: maximize distance correlation instead of minimize MSE.
    Returns negative correlation (for minimization).
    """
    n_features = X.shape[1]
    A = a_flat.reshape(k, n_features)

    # Normalize directions
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / norms

    # Project with weak nonlinearity
    Z = np.tanh(alpha * (X @ A.T))

    # Compute embedded distances
    d_embed = pdist(Z, metric="euclidean")

    # Return negative correlation (we want to maximize correlation)
    corr, _ = pearsonr(d_orig, d_embed)
    return -corr


def sammon_objective(a_flat, X, k, d_orig, alpha=0.01):
    """
    Sammon mapping objective: sum((d_X - d_Z)^2 / d_X)
    This normalizes by original distance, handling scale naturally.
    """
    n_features = X.shape[1]
    A = a_flat.reshape(k, n_features)

    # Normalize directions
    norms = np.linalg.norm(A, axis=1, keepdims=True)
    A = A / norms

    # Project with weak nonlinearity
    Z = np.tanh(alpha * (X @ A.T))

    # Compute embedded distances
    d_embed = pdist(Z, metric="euclidean")

    # Sammon stress
    mask = d_orig > 1e-10
    stress = np.sum((d_orig[mask] - d_embed[mask])**2 / d_orig[mask])
    return stress


def test_alternative_objectives(X, name):
    """Test correlation and Sammon objectives."""
    print(f"\n{'='*60}")
    print(f"ALTERNATIVE OBJECTIVES TEST: {name}")
    print("="*60)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    d_orig = pdist(X_scaled, metric="euclidean")

    n_samples, n_features = X_scaled.shape
    k = 2

    # Initialize with PCA
    pca = PCA(n_components=k)
    pca.fit(X_scaled)
    a0 = pca.components_.flatten()

    results = {}

    # 1. Standard pyppur (for comparison)
    pp_standard = ProjectionPursuit(
        n_components=k,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=1.0,
        random_state=42,
        n_init=3,
    )
    X_pp_std = pp_standard.fit_transform(X_scaled)
    d_pp_std = pdist(X_pp_std, metric="euclidean")
    results["pyppur (std, α=1.0)"] = pearsonr(d_orig, d_pp_std)[0]

    # 2. pyppur with very low alpha
    pp_low_alpha = ProjectionPursuit(
        n_components=k,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=0.01,
        random_state=42,
        n_init=3,
    )
    X_pp_low = pp_low_alpha.fit_transform(X_scaled)
    d_pp_low = pdist(X_pp_low, metric="euclidean")
    results["pyppur (α=0.01)"] = pearsonr(d_orig, d_pp_low)[0]

    # 3. Correlation objective (custom optimization)
    print("  Running correlation objective optimization...")
    res_corr = minimize(
        correlation_objective,
        a0,
        args=(X_scaled, k, d_orig, 0.1),
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    A_corr = res_corr.x.reshape(k, n_features)
    A_corr = A_corr / np.linalg.norm(A_corr, axis=1, keepdims=True)
    Z_corr = np.tanh(0.1 * (X_scaled @ A_corr.T))
    d_corr = pdist(Z_corr, metric="euclidean")
    results["Correlation obj (α=0.1)"] = pearsonr(d_orig, d_corr)[0]

    # 4. Sammon objective
    print("  Running Sammon objective optimization...")
    res_sammon = minimize(
        sammon_objective,
        a0,
        args=(X_scaled, k, d_orig, 0.1),
        method="L-BFGS-B",
        options={"maxiter": 500},
    )
    A_sammon = res_sammon.x.reshape(k, n_features)
    A_sammon = A_sammon / np.linalg.norm(A_sammon, axis=1, keepdims=True)
    Z_sammon = np.tanh(0.1 * (X_scaled @ A_sammon.T))
    d_sammon = pdist(Z_sammon, metric="euclidean")
    results["Sammon obj (α=0.1)"] = pearsonr(d_orig, d_sammon)[0]

    # 5. Baselines
    pca_full = PCA(n_components=k)
    X_pca = pca_full.fit_transform(X_scaled)
    d_pca = pdist(X_pca, metric="euclidean")
    results["PCA"] = pearsonr(d_orig, d_pca)[0]

    mds = MDS(n_components=k, normalized_stress="auto", random_state=42, max_iter=300)
    X_mds = mds.fit_transform(X_scaled)
    d_mds = pdist(X_mds, metric="euclidean")
    results["MDS"] = pearsonr(d_orig, d_mds)[0]

    print("\nDistance correlation results (higher = better):")
    print("-" * 50)
    for method, corr in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:25s}: {corr:.4f}")

    return results


def test_outlier_use_case():
    """
    Test the outlier robustness use case in a realistic scenario.
    Use case: Classification with outlier-contaminated training data.
    """
    print(f"\n{'='*60}")
    print(f"OUTLIER ROBUSTNESS USE CASE")
    print("="*60)

    # Create classification dataset
    np.random.seed(42)
    n_samples = 300
    n_features = 20

    X, y = make_blobs(n_samples=n_samples, n_features=n_features,
                      centers=3, cluster_std=2.0, random_state=42)

    # Add outliers to training data (10%)
    n_outliers = n_samples // 10
    outlier_idx = np.random.choice(n_samples, n_outliers, replace=False)
    X_contaminated = X.copy()
    X_contaminated[outlier_idx] += np.random.randn(n_outliers, n_features) * 20

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_contaminated, y, test_size=0.3, random_state=42, stratify=y
    )

    # Note: test data is also contaminated, but in real scenario might be clean
    # Let's create clean test set for fair evaluation
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_test = X_test_c  # Use clean test set
    y_test = y_test_c

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # PCA
    pca = PCA(n_components=2)
    pca.fit(X_train_scaled)
    X_train_pca = pca.transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pca, y_train)
    results["PCA"] = knn.score(X_test_pca, y_test)

    # pyppur with tanh (should be robust)
    pp_tanh = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=1.0,
        random_state=42,
        n_init=3,
    )
    pp_tanh.fit(X_train_scaled)
    X_train_pp = pp_tanh.transform(X_train_scaled)
    X_test_pp = pp_tanh.transform(X_test_scaled)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pp, y_train)
    results["pyppur (tanh, α=1.0)"] = knn.score(X_test_pp, y_test)

    # pyppur with low alpha
    pp_low = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=0.1,
        random_state=42,
        n_init=3,
    )
    pp_low.fit(X_train_scaled)
    X_train_pp_low = pp_low.transform(X_train_scaled)
    X_test_pp_low = pp_low.transform(X_test_scaled)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train_pp_low, y_train)
    results["pyppur (tanh, α=0.1)"] = knn.score(X_test_pp_low, y_test)

    print("\nClassification accuracy on clean test set (trained on contaminated data):")
    print("-" * 50)
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:25s}: {acc:.4f}")

    # Compare with clean training data
    print("\n--- Control: Same experiment WITHOUT outliers ---")
    scaler_clean = StandardScaler()
    X_train_clean = scaler_clean.fit_transform(X_train_c)
    X_test_clean = scaler_clean.transform(X_test_c)

    results_clean = {}

    pca_clean = PCA(n_components=2)
    pca_clean.fit(X_train_clean)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(pca_clean.transform(X_train_clean), y_train_c)
    results_clean["PCA (clean)"] = knn.score(pca_clean.transform(X_test_clean), y_test_c)

    pp_clean = ProjectionPursuit(
        n_components=2,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=1.0,
        random_state=42,
        n_init=3,
    )
    pp_clean.fit(X_train_clean)
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(pp_clean.transform(X_train_clean), y_train_c)
    results_clean["pyppur (clean)"] = knn.score(pp_clean.transform(X_test_clean), y_test_c)

    print("\nClassification accuracy WITHOUT outliers (baseline):")
    for method, acc in results_clean.items():
        print(f"  {method:25s}: {acc:.4f}")

    print("\nDegradation due to outliers:")
    print(f"  PCA: {results_clean['PCA (clean)']:.4f} → {results['PCA']:.4f} (Δ={results['PCA'] - results_clean['PCA (clean)']:+.4f})")
    print(f"  pyppur: {results_clean['pyppur (clean)']:.4f} → {results['pyppur (tanh, α=1.0)']:.4f} (Δ={results['pyppur (tanh, α=1.0)'] - results_clean['pyppur (clean)']:+.4f})")


def test_bounded_representation_use_case():
    """
    Test use case where bounded representations are desirable.
    Example: Features for neural network input (want bounded activations).
    """
    print(f"\n{'='*60}")
    print(f"BOUNDED REPRESENTATION USE CASE")
    print("="*60)

    # Load data
    digits = load_digits()
    X, y = digits.data[:500], digits.target[:500]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.3, random_state=42, stratify=y
    )

    # Get embeddings
    pca = PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)

    pp = ProjectionPursuit(
        n_components=10,
        objective=Objective.DISTANCE_DISTORTION,
        alpha=2.0,  # Higher alpha for more bounded output
        random_state=42,
        n_init=3,
    )
    X_train_pp = pp.fit_transform(X_train)
    X_test_pp = pp.transform(X_test)

    print("\nEmbedding statistics:")
    print(f"  PCA - range: [{X_train_pca.min():.2f}, {X_train_pca.max():.2f}], std: {X_train_pca.std():.2f}")
    print(f"  pyppur - range: [{X_train_pp.min():.2f}, {X_train_pp.max():.2f}], std: {X_train_pp.std():.2f}")

    # Classification performance
    from sklearn.linear_model import LogisticRegression

    lr_pca = LogisticRegression(max_iter=1000, random_state=42)
    lr_pca.fit(X_train_pca, y_train)

    lr_pp = LogisticRegression(max_iter=1000, random_state=42)
    lr_pp.fit(X_train_pp, y_train)

    print(f"\nLogistic regression accuracy:")
    print(f"  PCA: {lr_pca.score(X_test_pca, y_test):.4f}")
    print(f"  pyppur: {lr_pp.score(X_test_pp, y_test):.4f}")


def propose_fixes():
    """Summarize proposed fixes for pyppur."""
    print("\n" + "="*60)
    print("PROPOSED FIXES FOR PYPPUR")
    print("="*60)
    print("""
Based on our investigation, here are concrete fixes:

1. CHANGE DEFAULT ALPHA
   - Current default: alpha=1.0
   - Proposed default: alpha=0.1 or make it adaptive
   - Rationale: Low alpha gives 2x better distance correlation

2. USE CORRELATION-BASED OBJECTIVE (new option)
   - Instead of minimizing MSE of distances, maximize correlation
   - This is scale-invariant and handles the tanh bounding naturally
   - Add parameter: objective_metric='mse' | 'correlation' | 'sammon'

3. ADD POST-HOC SCALING
   - After fitting, optionally scale embedded distances to match original
   - Simple fix that preserves relative structure

4. MARKET FOR OUTLIER ROBUSTNESS
   - This is a REAL advantage (30x better than PCA/MDS)
   - Document this use case prominently
   - Test on more outlier scenarios

5. CONSIDER LEARNABLE ALPHA
   - Make alpha a learnable parameter
   - Or use different alpha per component

6. ADD RANK-BASED METRICS
   - Report Spearman correlation alongside distance distortion
   - This better captures relative ordering preservation
""")


def main():
    # Load test datasets
    digits = load_digits()
    X_digits = digits.data[:500]

    X_swiss, _ = make_swiss_roll(n_samples=300, noise=0.5, random_state=42)

    print("\n" + "#"*60)
    print("# PYPPUR RESCUE STRATEGIES")
    print("#"*60)

    # Test alternative objectives
    test_alternative_objectives(X_digits, "Digits")
    test_alternative_objectives(X_swiss, "Swiss Roll")

    # Test outlier use case
    test_outlier_use_case()

    # Test bounded representation use case
    test_bounded_representation_use_case()

    # Propose fixes
    propose_fixes()


if __name__ == "__main__":
    main()
