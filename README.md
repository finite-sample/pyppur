### 🪈 pyppur: **P**ython **P**rojection **P**ursuit **U**nsupervised **R**eduction

[![PyPI](https://img.shields.io/pypi/v/pyppur.svg)](https://pypi.org/project/pyppur/)
[![PyPI Downloads](https://static.pepy.tech/badge/pyppur)](https://pepy.tech/projects/pyppur)
[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://gojiplus.github.io/pyppur/)

## Overview

`pyppur` is a Python package that implements projection pursuit methods for dimensionality reduction. Unlike traditional methods such as PCA, `pyppur` focuses on finding interesting non-linear projections by minimizing either reconstruction loss or distance distortion.

## Installation

```bash
pip install pyppur
```

## Features

- Two optimization objectives:
  - **Distance Distortion**: Preserves pairwise distances between data points
  - **Reconstruction**: Minimizes reconstruction error using ridge functions
- Multiple initialization strategies (PCA-based and random)
- Full scikit-learn compatible API
- Supports standardization and custom weighting

## Usage

### Basic Example

```python
import numpy as np
from pyppur import ProjectionPursuit, Objective
from sklearn.datasets import load_digits

# Load data
digits = load_digits()
X = digits.data
y = digits.target

# Projection pursuit with distance distortion
pp_dist = ProjectionPursuit(
    n_components=2,
    objective=Objective.DISTANCE_DISTORTION,
    alpha=1.5,  # Steepness of the ridge function
    n_init=3,   # Number of random initializations
    verbose=True
)

# Fit and transform
X_transformed = pp_dist.fit_transform(X)

# Projection pursuit with reconstruction loss (tied weights)
pp_recon_tied = ProjectionPursuit(
    n_components=2,
    objective=Objective.RECONSTRUCTION,
    alpha=1.0,
    tied_weights=True
)

# Projection pursuit with reconstruction loss (free decoder)
pp_recon_free = ProjectionPursuit(
    n_components=2,
    objective=Objective.RECONSTRUCTION,
    alpha=1.0,
    tied_weights=False,
    l2_reg=0.01
)

# Fit and transform
X_transformed_recon_tied = pp_recon_tied.fit_transform(X)
X_transformed_recon_free = pp_recon_free.fit_transform(X)

# Evaluate the methods
dist_metrics = pp_dist.evaluate(X, y)
recon_tied_metrics = pp_recon_tied.evaluate(X, y)
recon_free_metrics = pp_recon_free.evaluate(X, y)

print("Distance distortion method:")
print(f"  Trustworthiness: {dist_metrics['trustworthiness']:.4f}")
print(f"  Silhouette: {dist_metrics['silhouette']:.4f}")
print(f"  Distance distortion: {dist_metrics['distance_distortion']:.4f}")
print(f"  Reconstruction error: {dist_metrics['reconstruction_error']:.4f}")

print("\nReconstruction method (tied weights):")
print(f"  Trustworthiness: {recon_tied_metrics['trustworthiness']:.4f}")
print(f"  Silhouette: {recon_tied_metrics['silhouette']:.4f}")
print(f"  Distance distortion: {recon_tied_metrics['distance_distortion']:.4f}")
print(f"  Reconstruction error: {recon_tied_metrics['reconstruction_error']:.4f}")

print("\nReconstruction method (free decoder):")
print(f"  Trustworthiness: {recon_free_metrics['trustworthiness']:.4f}")
print(f"  Silhouette: {recon_free_metrics['silhouette']:.4f}")
print(f"  Distance distortion: {recon_free_metrics['distance_distortion']:.4f}")
print(f"  Reconstruction error: {recon_free_metrics['reconstruction_error']:.4f}")
```


## API Reference

The main class in `pyppur` is `ProjectionPursuit`, which provides the following methods:

- `fit(X)`: Fit the model to data
- `transform(X)`: Apply dimensionality reduction to new data
- `fit_transform(X)`: Fit the model and transform data
- `reconstruct(X)`: Reconstruct data from projections
- `reconstruction_error(X)`: Compute reconstruction error
- `distance_distortion(X)`: Compute distance distortion
- `compute_trustworthiness(X, n_neighbors)`: Measure how well local structure is preserved
- `compute_silhouette(X, labels)`: Measure how well clusters are separated
- `evaluate(X, labels, n_neighbors)`: Compute all evaluation metrics at once

## Theory

Projection pursuit finds interesting low-dimensional projections of multivariate data. When used for dimensionality reduction, it aims to optimize an "interestingness" index which can be:

1. **Distance Distortion**: Minimizes the difference between pairwise distances in original and projected spaces (optionally with nonlinearity)
2. **Reconstruction Error**: Minimizes the error when reconstructing the data using ridge functions

### Mathematical Formulations

#### Tied-Weights Ridge Autoencoder (Default)
```
Z = g(X A^T)
X̂ = Z A
```

#### Free Decoder Ridge Autoencoder (Available with tied_weights=False)
```
Z = g(X A^T)  
X̂ = Z B
```

Where:
- `X` is the input data matrix (n_samples × n_features)
- `A` are the encoder projection directions (n_components × n_features)
- `B` are the decoder weights (n_components × n_features, when untied)
- `g(z) = tanh(α * z)` is the ridge function with steepness parameter α
- `Z` is the projected data (n_samples × n_components)
- `X̂` is the reconstructed data

#### Distance Distortion Options
- **With nonlinearity**: Compares distances between original space and `g(X A^T)`
- **Without nonlinearity**: Compares distances between original space and linear projections `X A^T`

## Requirements

- Python 3.8+
- NumPy (>=1.20.0)
- SciPy (>=1.7.0)  
- scikit-learn (>=1.0.0)
- matplotlib (>=3.3.0)

## License

MIT

## Citation

If you use `pyppur` in your research, please cite it as:

```
@software{pyppur,
  author = {Gaurav Sood},
  title = {pyppur: Python Projection Pursuit Unsupervised Reduction},
  url = {https://github.com/gojiplus/pyppur},
  version = {0.2.0},
  year = {2025},
}
```

## 🔗 Adjacent Repositories

- [gojiplus/get-weather-data](https://github.com/gojiplus/get-weather-data) — Get weather data for a list of zip codes for a range of dates
- [gojiplus/text-as-data](https://github.com/gojiplus/text-as-data) — Pipeline for Analyzing Text Data: Acquire, Preprocess, Analyze
- [gojiplus/calibre](https://github.com/gojiplus/calibre) — Advanced Calibration Models
- [gojiplus/skiplist_join](https://github.com/gojiplus/skiplist_join)
- [gojiplus/rmcp](https://github.com/gojiplus/rmcp) — R MCP Server
