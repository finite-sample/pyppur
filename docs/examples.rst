Examples
========

This section provides comprehensive examples of using pyppur for different scenarios.

Basic Usage Examples
-------------------

Distance Distortion Example
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.datasets import load_digits
   from sklearn.preprocessing import StandardScaler
   from pyppur import ProjectionPursuit, Objective

   # Load and prepare data
   digits = load_digits()
   X, y = digits.data, digits.target
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

   # Distance distortion with nonlinearity (default)
   pp_nonlinear = ProjectionPursuit(
       n_components=2,
       objective=Objective.DISTANCE_DISTORTION,
       alpha=1.5,
       use_nonlinearity_in_distance=True,
       n_init=5,
       verbose=True
   )

   X_nl = pp_nonlinear.fit_transform(X_scaled)

   # Distance distortion without nonlinearity (linear)
   pp_linear = ProjectionPursuit(
       n_components=2,
       objective=Objective.DISTANCE_DISTORTION,
       alpha=1.5,
       use_nonlinearity_in_distance=False,
       n_init=5,
       verbose=True
   )

   X_linear = pp_linear.fit_transform(X_scaled)

   # Compare results
   print("Nonlinear distance distortion:", pp_nonlinear.distance_distortion(X_scaled))
   print("Linear distance distortion:", pp_linear.distance_distortion(X_scaled))

   # Plot results
   fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

   scatter1 = ax1.scatter(X_nl[:, 0], X_nl[:, 1], c=y, cmap='tab10', alpha=0.7)
   ax1.set_title('Distance Distortion (Nonlinear)')
   ax1.set_xlabel('Component 1')
   ax1.set_ylabel('Component 2')

   scatter2 = ax2.scatter(X_linear[:, 0], X_linear[:, 1], c=y, cmap='tab10', alpha=0.7)
   ax2.set_title('Distance Distortion (Linear)')
   ax2.set_xlabel('Component 1')
   ax2.set_ylabel('Component 2')

   plt.tight_layout()
   plt.show()

Reconstruction Examples
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Reconstruction with tied weights (default)
   pp_tied = ProjectionPursuit(
       n_components=3,
       objective=Objective.RECONSTRUCTION,
       alpha=1.0,
       tied_weights=True,
       n_init=3,
       verbose=True
   )

   X_tied = pp_tied.fit_transform(X_scaled)
   tied_error = pp_tied.reconstruction_error(X_scaled)

   # Reconstruction with untied weights
   pp_untied = ProjectionPursuit(
       n_components=3,
       objective=Objective.RECONSTRUCTION,
       alpha=1.0,
       tied_weights=False,
       l2_reg=0.01,
       n_init=3,
       verbose=True
   )

   X_untied = pp_untied.fit_transform(X_scaled)
   untied_error = pp_untied.reconstruction_error(X_scaled)

   print(f"Tied weights reconstruction error: {tied_error:.6f}")
   print(f"Untied weights reconstruction error: {untied_error:.6f}")
   print(f"Improvement: {((tied_error - untied_error) / tied_error * 100):.1f}%")

   # Access decoder weights
   print("Tied decoder weights:", pp_tied.decoder_weights_)  # None
   print("Untied decoder shape:", pp_untied.decoder_weights_.shape)

Advanced Examples
----------------

Parameter Sensitivity Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.datasets import make_swiss_roll
   import pandas as pd

   # Generate Swiss roll data
   X_swiss, color = make_swiss_roll(n_samples=1000, noise=0.1, random_state=42)
   scaler = StandardScaler()
   X_swiss_scaled = scaler.fit_transform(X_swiss)

   # Test different alpha values
   alphas = [0.1, 0.5, 1.0, 2.0, 5.0]
   results = []

   for alpha in alphas:
       pp = ProjectionPursuit(
           n_components=2,
           objective=Objective.RECONSTRUCTION,
           alpha=alpha,
           tied_weights=False,
           l2_reg=0.01,
           max_iter=100,
           random_state=42
       )
       
       X_proj = pp.fit_transform(X_swiss_scaled)
       recon_error = pp.reconstruction_error(X_swiss_scaled)
       
       results.append({
           'alpha': alpha,
           'reconstruction_error': recon_error,
           'fit_time': pp.fit_time_
       })

   results_df = pd.DataFrame(results)
   print(results_df)

   # Plot reconstruction error vs alpha
   plt.figure(figsize=(8, 6))
   plt.plot(results_df['alpha'], results_df['reconstruction_error'], 'bo-')
   plt.xlabel('Alpha (Ridge Function Steepness)')
   plt.ylabel('Reconstruction Error')
   plt.title('Reconstruction Error vs Alpha Parameter')
   plt.xscale('log')
   plt.grid(True, alpha=0.3)
   plt.show()

Comparison with Other Methods
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from sklearn.decomposition import PCA
   from sklearn.manifold import TSNE
   import time

   # Prepare data
   X_sample = X_scaled[:500]  # Use subset for t-SNE speed
   y_sample = y[:500]

   methods = {}
   times = {}

   # PCA
   start_time = time.time()
   pca = PCA(n_components=2, random_state=42)
   X_pca = pca.fit_transform(X_sample)
   times['PCA'] = time.time() - start_time
   methods['PCA'] = X_pca

   # t-SNE
   start_time = time.time()
   tsne = TSNE(n_components=2, random_state=42, perplexity=30)
   X_tsne = tsne.fit_transform(X_sample)
   times['t-SNE'] = time.time() - start_time
   methods['t-SNE'] = X_tsne

   # pyppur (Distance Distortion)
   start_time = time.time()
   pp_dist = ProjectionPursuit(
       n_components=2,
       objective=Objective.DISTANCE_DISTORTION,
       alpha=1.5,
       n_init=3,
       random_state=42
   )
   X_pp_dist = pp_dist.fit_transform(X_sample)
   times['pyppur (Distance)'] = time.time() - start_time
   methods['pyppur (Distance)'] = X_pp_dist

   # pyppur (Reconstruction)
   start_time = time.time()
   pp_recon = ProjectionPursuit(
       n_components=2,
       objective=Objective.RECONSTRUCTION,
       tied_weights=False,
       alpha=1.0,
       n_init=3,
       random_state=42
   )
   X_pp_recon = pp_recon.fit_transform(X_sample)
   times['pyppur (Reconstruction)'] = time.time() - start_time
   methods['pyppur (Reconstruction)'] = X_pp_recon

   # Plot comparison
   fig, axes = plt.subplots(2, 2, figsize=(12, 10))
   axes = axes.ravel()

   for i, (method_name, X_proj) in enumerate(methods.items()):
       scatter = axes[i].scatter(X_proj[:, 0], X_proj[:, 1], c=y_sample, 
                                cmap='tab10', alpha=0.7, s=20)
       axes[i].set_title(f'{method_name} (Time: {times[method_name]:.2f}s)')
       axes[i].set_xlabel('Component 1')
       axes[i].set_ylabel('Component 2')

   plt.tight_layout()
   plt.show()

   # Print timing comparison
   print("\nTiming Comparison:")
   for method, time_taken in times.items():
       print(f"{method}: {time_taken:.3f} seconds")

Evaluation and Metrics
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   from pyppur.utils.metrics import evaluate_embedding

   # Comprehensive evaluation
   pp = ProjectionPursuit(
       n_components=2,
       objective=Objective.DISTANCE_DISTORTION,
       alpha=1.5,
       n_init=5,
       random_state=42
   )

   X_proj = pp.fit_transform(X_scaled)

   # Built-in evaluation
   metrics = pp.evaluate(X_scaled, y, n_neighbors=10)
   print("Built-in evaluation:")
   for metric, value in metrics.items():
       print(f"  {metric}: {value:.4f}")

   # Manual evaluation using utils
   manual_metrics = evaluate_embedding(X_scaled, X_proj, y, n_neighbors=10)
   print("\nManual evaluation:")
   for metric, value in manual_metrics.items():
       print(f"  {metric}: {value:.4f}")

   # Additional metrics
   print(f"\nAdditional metrics:")
   print(f"Distance distortion: {pp.distance_distortion(X_scaled):.6f}")
   print(f"Reconstruction error: {pp.reconstruction_error(X_scaled):.6f}")
   print(f"Trustworthiness (k=5): {pp.compute_trustworthiness(X_scaled, 5):.4f}")
   print(f"Trustworthiness (k=15): {pp.compute_trustworthiness(X_scaled, 15):.4f}")

Working with Large Datasets
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For large datasets, consider these strategies:

   # 1. Reduce the number of initializations
   pp_fast = ProjectionPursuit(
       n_components=2,
       objective=Objective.RECONSTRUCTION,
       n_init=1,  # Fewer initializations
       max_iter=50,  # Fewer iterations
       alpha=1.0
   )

   # 2. Use reconstruction objective (more memory efficient than distance)
   # Distance distortion requires O(n²) memory for distance matrices
   # Reconstruction requires O(nk) memory

   # 3. For distance distortion with large n, consider subsampling
   if X_scaled.shape[0] > 5000:
       print("Large dataset detected, using reconstruction objective")
       pp_large = ProjectionPursuit(
           n_components=2,
           objective=Objective.RECONSTRUCTION,
           tied_weights=True,  # Faster than untied
           alpha=1.0,
           n_init=1,
           max_iter=100
       )
   else:
       pp_large = ProjectionPursuit(
           n_components=2,
           objective=Objective.DISTANCE_DISTORTION,
           alpha=1.5,
           n_init=3
       )

   X_large_proj = pp_large.fit_transform(X_scaled)
   print(f"Processed {X_scaled.shape[0]} samples in {pp_large.fit_time_:.2f} seconds")

Custom Workflows
~~~~~~~~~~~~~~~

.. code-block:: python

   # Custom preprocessing and postprocessing pipeline
   from sklearn.pipeline import Pipeline
   from sklearn.preprocessing import StandardScaler, RobustScaler

   # Custom pipeline
   pipeline = Pipeline([
       ('robust_scaler', RobustScaler()),  # More robust to outliers
       ('projection_pursuit', ProjectionPursuit(
           n_components=2,
           objective=Objective.RECONSTRUCTION,
           tied_weights=False,
           l2_reg=0.05,
           alpha=1.2,
           n_init=5,
           verbose=True
       ))
   ])

   # Fit and transform
   X_pipeline = pipeline.fit_transform(X)

   # Access the fitted pyppur model
   pp_model = pipeline.named_steps['projection_pursuit']
   print(f"Final loss: {pp_model.best_loss_:.6f}")
   print(f"Optimization info: {pp_model.optimizer_info_}")

   # Visualize results
   plt.figure(figsize=(8, 6))
   scatter = plt.scatter(X_pipeline[:, 0], X_pipeline[:, 1], c=y, cmap='tab10', alpha=0.7)
   plt.colorbar(scatter)
   plt.title('pyppur with Robust Scaling Pipeline')
   plt.xlabel('Component 1')
   plt.ylabel('Component 2')
   plt.show()

This examples section demonstrates the flexibility and power of pyppur for various dimensionality reduction tasks.