"""
Benchmark experiments to evaluate pyppur's value proposition.

Tests:
1. Distance correlation - does pyppur preserve distances better than baselines?
2. k-NN classification accuracy - does the embedding preserve class structure?
3. Reconstruction error - can pyppur reconstruct held-out data?
4. Ablation - does the tanh nonlinearity help?
"""

import numpy as np
from scipy.spatial.distance import pdist
from scipy.stats import pearsonr, spearmanr
from sklearn.datasets import load_digits, load_iris, load_wine, make_swiss_roll
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from pyppur import ProjectionPursuit, Objective


def distance_correlation(X_orig, X_embed):
    """Compute Pearson correlation between pairwise distances."""
    d_orig = pdist(X_orig, metric="euclidean")
    d_embed = pdist(X_embed, metric="euclidean")
    corr, _ = pearsonr(d_orig, d_embed)
    return corr


def run_distance_correlation_test(X, name, n_components=2):
    """Test 2: Distance correlation comparison."""
    print(f"\n{'='*60}")
    print(f"DISTANCE CORRELATION TEST: {name}")
    print(f"Data shape: {X.shape}, embedding to {n_components}D")
    print("="*60)

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    results = {}

    # PCA
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    results["PCA"] = distance_correlation(X_scaled, X_pca)

    # MDS
    mds = MDS(n_components=n_components, normalized_stress="auto", random_state=42, max_iter=300)
    X_mds = mds.fit_transform(X_scaled)
    results["MDS"] = distance_correlation(X_scaled, X_mds)

    # pyppur with distance objective (with tanh)
    pp_dist = ProjectionPursuit(
        n_components=n_components,
        objective=Objective.DISTANCE_DISTORTION,
        use_nonlinearity_in_distance=True,
        random_state=42,
        n_init=3,
        max_iter=500,
    )
    X_pp_dist = pp_dist.fit_transform(X_scaled)
    results["pyppur (dist+tanh)"] = distance_correlation(X_scaled, X_pp_dist)

    # pyppur with distance objective (without tanh - linear)
    pp_dist_linear = ProjectionPursuit(
        n_components=n_components,
        objective=Objective.DISTANCE_DISTORTION,
        use_nonlinearity_in_distance=False,
        random_state=42,
        n_init=3,
        max_iter=500,
    )
    X_pp_dist_linear = pp_dist_linear.fit_transform(X_scaled)
    results["pyppur (dist+linear)"] = distance_correlation(X_scaled, X_pp_dist_linear)

    # pyppur with reconstruction objective
    pp_recon = ProjectionPursuit(
        n_components=n_components,
        objective=Objective.RECONSTRUCTION,
        random_state=42,
        n_init=3,
        max_iter=500,
    )
    X_pp_recon = pp_recon.fit_transform(X_scaled)
    results["pyppur (recon)"] = distance_correlation(X_scaled, X_pp_recon)

    # Print results sorted by correlation
    print("\nResults (higher = better distance preservation):")
    print("-" * 40)
    for method, corr in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:25s}: {corr:.4f}")

    return results


def run_knn_classification_test(X, y, name, n_components=2):
    """Test 1: k-NN classification accuracy on embedding."""
    print(f"\n{'='*60}")
    print(f"k-NN CLASSIFICATION TEST: {name}")
    print(f"Data shape: {X.shape}, classes: {len(np.unique(y))}")
    print("="*60)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    methods = {
        "PCA": PCA(n_components=n_components),
        "MDS": MDS(n_components=n_components, normalized_stress="auto", random_state=42, max_iter=300),
    }

    # Fit sklearn methods
    for method_name, method in methods.items():
        if method_name == "MDS":
            # MDS doesn't have transform, need to fit on all data
            X_all = np.vstack([X_train_scaled, X_test_scaled])
            X_embed = method.fit_transform(X_all)
            X_train_embed = X_embed[:len(X_train_scaled)]
            X_test_embed = X_embed[len(X_train_scaled):]
        else:
            X_train_embed = method.fit_transform(X_train_scaled)
            X_test_embed = method.transform(X_test_scaled)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_embed, y_train)
        acc = knn.score(X_test_embed, y_test)
        results[method_name] = acc

    # pyppur methods
    pp_configs = {
        "pyppur (dist+tanh)": {
            "objective": Objective.DISTANCE_DISTORTION,
            "use_nonlinearity_in_distance": True,
        },
        "pyppur (dist+linear)": {
            "objective": Objective.DISTANCE_DISTORTION,
            "use_nonlinearity_in_distance": False,
        },
        "pyppur (recon)": {
            "objective": Objective.RECONSTRUCTION,
        },
    }

    for method_name, config in pp_configs.items():
        pp = ProjectionPursuit(
            n_components=n_components,
            random_state=42,
            n_init=3,
            max_iter=500,
            **config,
        )
        X_train_embed = pp.fit_transform(X_train_scaled)
        X_test_embed = pp.transform(X_test_scaled)

        knn = KNeighborsClassifier(n_neighbors=5)
        knn.fit(X_train_embed, y_train)
        acc = knn.score(X_test_embed, y_test)
        results[method_name] = acc

    # Print results sorted by accuracy
    print("\nResults (higher = better class separation):")
    print("-" * 40)
    for method, acc in sorted(results.items(), key=lambda x: x[1], reverse=True):
        print(f"  {method:25s}: {acc:.4f}")

    return results


def run_reconstruction_test(X, name, n_components=2):
    """Test 3: Reconstruction error on held-out data."""
    print(f"\n{'='*60}")
    print(f"RECONSTRUCTION TEST: {name}")
    print(f"Data shape: {X.shape}, embedding to {n_components}D")
    print("="*60)

    # Split data
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

    # Standardize
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results = {}

    # PCA reconstruction
    pca = PCA(n_components=n_components)
    pca.fit(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    X_test_recon_pca = pca.inverse_transform(X_test_pca)
    mse_pca = np.mean((X_test_scaled - X_test_recon_pca) ** 2)
    results["PCA"] = mse_pca

    # pyppur reconstruction (tied weights)
    pp_tied = ProjectionPursuit(
        n_components=n_components,
        objective=Objective.RECONSTRUCTION,
        tied_weights=True,
        random_state=42,
        n_init=3,
        max_iter=500,
    )
    pp_tied.fit(X_train_scaled)
    X_test_recon_pp = pp_tied.reconstruct(X_test_scaled)
    mse_pp_tied = np.mean((X_test_scaled - X_test_recon_pp) ** 2)
    results["pyppur (recon, tied)"] = mse_pp_tied

    # pyppur reconstruction (untied weights)
    pp_untied = ProjectionPursuit(
        n_components=n_components,
        objective=Objective.RECONSTRUCTION,
        tied_weights=False,
        random_state=42,
        n_init=3,
        max_iter=500,
    )
    pp_untied.fit(X_train_scaled)
    X_test_recon_pp_untied = pp_untied.reconstruct(X_test_scaled)
    mse_pp_untied = np.mean((X_test_scaled - X_test_recon_pp_untied) ** 2)
    results["pyppur (recon, untied)"] = mse_pp_untied

    # Print results sorted by MSE (lower is better)
    print("\nResults (lower = better reconstruction):")
    print("-" * 40)
    for method, mse in sorted(results.items(), key=lambda x: x[1]):
        print(f"  {method:25s}: {mse:.4f}")

    return results


def main():
    print("\n" + "="*60)
    print("PYPPUR VALUE PROPOSITION BENCHMARK")
    print("="*60)

    # Load datasets
    digits = load_digits()
    X_digits, y_digits = digits.data, digits.target
    # Use subset for speed
    idx = np.random.RandomState(42).choice(len(X_digits), 500, replace=False)
    X_digits_sub, y_digits_sub = X_digits[idx], y_digits[idx]

    iris = load_iris()
    X_iris, y_iris = iris.data, iris.target

    wine = load_wine()
    X_wine, y_wine = wine.data, wine.target

    X_swiss, t_swiss = make_swiss_roll(n_samples=500, noise=0.5, random_state=42)
    # Discretize t for classification
    y_swiss = np.digitize(t_swiss, bins=np.linspace(t_swiss.min(), t_swiss.max(), 5))

    all_results = {
        "distance_correlation": {},
        "knn_classification": {},
        "reconstruction": {},
    }

    # Test 2: Distance Correlation (the key test)
    print("\n" + "#"*60)
    print("# TEST 2: DISTANCE CORRELATION (KEY TEST)")
    print("#"*60)

    all_results["distance_correlation"]["digits"] = run_distance_correlation_test(
        X_digits_sub, "Digits (500 samples)"
    )
    all_results["distance_correlation"]["swiss_roll"] = run_distance_correlation_test(
        X_swiss, "Swiss Roll"
    )
    all_results["distance_correlation"]["iris"] = run_distance_correlation_test(
        X_iris, "Iris"
    )

    # Test 1: k-NN Classification
    print("\n" + "#"*60)
    print("# TEST 1: k-NN CLASSIFICATION")
    print("#"*60)

    all_results["knn_classification"]["digits"] = run_knn_classification_test(
        X_digits_sub, y_digits_sub, "Digits (500 samples)"
    )
    all_results["knn_classification"]["iris"] = run_knn_classification_test(
        X_iris, y_iris, "Iris"
    )
    all_results["knn_classification"]["wine"] = run_knn_classification_test(
        X_wine, y_wine, "Wine"
    )

    # Test 3: Reconstruction
    print("\n" + "#"*60)
    print("# TEST 3: RECONSTRUCTION")
    print("#"*60)

    all_results["reconstruction"]["digits"] = run_reconstruction_test(
        X_digits_sub, "Digits (500 samples)"
    )
    all_results["reconstruction"]["iris"] = run_reconstruction_test(
        X_iris, "Iris"
    )

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print("\n1. DISTANCE CORRELATION (higher = better):")
    print("   Key question: Does pyppur beat MDS at distance preservation?")
    for dataset, results in all_results["distance_correlation"].items():
        mds_score = results.get("MDS", 0)
        pp_dist_score = results.get("pyppur (dist+tanh)", 0)
        pp_linear_score = results.get("pyppur (dist+linear)", 0)
        winner = "pyppur" if max(pp_dist_score, pp_linear_score) > mds_score else "MDS"
        print(f"   {dataset}: MDS={mds_score:.3f}, pyppur(tanh)={pp_dist_score:.3f}, pyppur(linear)={pp_linear_score:.3f} -> {winner}")

    print("\n2. k-NN CLASSIFICATION (higher = better):")
    print("   Key question: Does pyppur preserve class structure?")
    for dataset, results in all_results["knn_classification"].items():
        best_method = max(results.items(), key=lambda x: x[1])
        pca_score = results.get("PCA", 0)
        print(f"   {dataset}: Best={best_method[0]} ({best_method[1]:.3f}), PCA={pca_score:.3f}")

    print("\n3. RECONSTRUCTION (lower = better):")
    print("   Key question: Does pyppur beat PCA at reconstruction?")
    for dataset, results in all_results["reconstruction"].items():
        pca_score = results.get("PCA", float("inf"))
        pp_tied = results.get("pyppur (recon, tied)", float("inf"))
        pp_untied = results.get("pyppur (recon, untied)", float("inf"))
        winner = "pyppur" if min(pp_tied, pp_untied) < pca_score else "PCA"
        print(f"   {dataset}: PCA={pca_score:.3f}, pyppur(tied)={pp_tied:.3f}, pyppur(untied)={pp_untied:.3f} -> {winner}")

    print("\n4. ABLATION - Does tanh help?")
    for dataset, results in all_results["distance_correlation"].items():
        tanh = results.get("pyppur (dist+tanh)", 0)
        linear = results.get("pyppur (dist+linear)", 0)
        verdict = "tanh helps" if tanh > linear else "tanh hurts"
        print(f"   {dataset}: tanh={tanh:.3f}, linear={linear:.3f} -> {verdict}")


if __name__ == "__main__":
    main()
