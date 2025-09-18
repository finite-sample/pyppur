pyppur: Python Projection Pursuit Unsupervised Reduction
=========================================================

**pyppur** is a Python package that implements projection pursuit methods for dimensionality reduction. 
Unlike traditional methods such as PCA, pyppur focuses on finding interesting non-linear projections by 
minimizing either reconstruction loss or distance distortion.

.. image:: https://img.shields.io/pypi/v/pyppur.svg
   :target: https://pypi.org/project/pyppur/
   :alt: PyPI Version

.. image:: https://static.pepy.tech/badge/pyppur
   :target: https://pepy.tech/projects/pyppur
   :alt: PyPI Downloads

Features
--------

* **Two optimization objectives:**
  
  * **Distance Distortion**: Preserves pairwise distances between data points (with optional nonlinearity)
  * **Reconstruction**: Minimizes reconstruction error using ridge functions (tied or untied weights)

* **Multiple initialization strategies** (PCA-based and random)
* **Full scikit-learn compatible API**
* **Supports standardization and custom weighting**
* **Mathematical flexibility** with configurable ridge functions and regularization

Quick Start
-----------

Installation
~~~~~~~~~~~~

.. code-block:: bash

   pip install pyppur

Basic Usage
~~~~~~~~~~~

.. code-block:: python

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
       alpha=1.5,
       n_init=3
   )

   # Fit and transform
   X_transformed = pp_dist.fit_transform(X)

   # Evaluate the embedding
   metrics = pp_dist.evaluate(X, y)
   print(f"Trustworthiness: {metrics['trustworthiness']:.3f}")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   getting_started
   mathematical_theory
   api_reference
   examples
   changelog

.. toctree::
   :maxdepth: 1
   :caption: API Reference:

   pyppur

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`