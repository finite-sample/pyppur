### 🪈 pyppur: **P**ython **P**rojection **P**ursuit **U**nsupervised **R**eduction

[![PyPI](https://img.shields.io/pypi/v/pyppur.svg)](https://pypi.org/project/pyppur/)
[![PyPI Downloads](https://static.pepy.tech/badge/pyppur)](https://pepy.tech/projects/pyppur)

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

# Projection pursuit with reconstruction loss
pp_recon = ProjectionPursuit(
    n_components=2,
    objective=Objective.RECONSTRUCTION,
    alpha=1.0
)

# Fit and transform
X_transformed_recon = pp_recon.fit_transform(X)

# Evaluate the methods
dist_metrics = pp_dist.evaluate(X, y)
recon_metrics = pp_recon.evaluate(X, y)

print("Distance distortion method:")
print(f"  Trustworthiness: {dist_metrics['trustworthiness']:.4f}")
print(f"  Silhouette: {dist_metrics['silhouette']:.4f}")
print(f"  Distance distortion: {dist_metrics['distance_distortion']:.4f}")
print(f"  Reconstruction error: {dist_metrics['reconstruction_error']:.4f}")

print("\nReconstruction method:")
print(f"  Trustworthiness: {recon_metrics['trustworthiness']:.4f}")
print(f"  Silhouette: {recon_metrics['silhouette']:.4f}")
print(f"  Distance distortion: {recon_metrics['distance_distortion']:.4f}")
print(f"  Reconstruction error: {recon_metrics['reconstruction_error']:.4f}")
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

1. **Distance Distortion**: Minimizes the difference between pairwise distances in original and projected spaces
2. **Reconstruction Error**: Minimizes the error when reconstructing the data using ridge functions

The mathematical formulation for the ridge function autoencoder is:

```
z_i = a_j^T x_i
x̂_i = ∑_j g(z_i) a_j
```

Where:
- `x_i` is the input data point
- `a_j` are the projection directions
- `g(z)` is the ridge function (tanh in our implementation)
- `x̂_i` is the reconstructed data point

## Requirements

- Python 3.10+
- NumPy
- SciPy
- scikit-learn

## License

MIT

## Citation

If you use `pyppur` in your research, please cite it as:

```
@software{pyppur,
  author = {Gaurav Sood},
  title = {pyppur: Python Projection Pursuit Unsupervised Reduction},
  url = {https://github.com/gojiplus/pyppur},
  version = {0.1.0},
  year = {2025},
}
```

## 🔗 Adjacent Repositories

- [gojiplus/get-weather-data](https://github.com/gojiplus/get-weather-data) — Get weather data for a list of zip codes for a range of dates
- [gojiplus/text-as-data](https://github.com/gojiplus/text-as-data) — Pipeline for Analyzing Text Data: Acquire, Preprocess, Analyze
- [gojiplus/calibre](https://github.com/gojiplus/calibre) — Advanced Calibration Models
- [gojiplus/skiplist_join](https://github.com/gojiplus/skiplist_join)
- [gojiplus/rmcp](https://github.com/gojiplus/rmcp) — R MCP Server
