Getting Started
===============

Installation
------------

pyppur can be installed from PyPI using pip:

.. code-block:: bash

   pip install pyppur

For development, you can install from source with development dependencies:

.. code-block:: bash

   git clone https://github.com/gojiplus/pyppur.git
   cd pyppur
   pip install -e .[dev]

Requirements
~~~~~~~~~~~~

* Python 3.10+
* NumPy (>=1.20.0)
* SciPy (>=1.7.0)  
* scikit-learn (>=1.0.0)
* matplotlib (>=3.3.0)

Quick Example
-------------

Here's a simple example using the digits dataset:

.. code-block:: python

   import numpy as np
   from pyppur import ProjectionPursuit, Objective
   from sklearn.datasets import load_digits
   from sklearn.preprocessing import StandardScaler

   # Load and standardize data
   digits = load_digits()
   X, y = digits.data, digits.target
   scaler = StandardScaler()
   X_scaled = scaler.fit_transform(X)

   # Create and fit the model
   pp = ProjectionPursuit(
       n_components=2,
       objective=Objective.DISTANCE_DISTORTION,
       alpha=1.5,
       n_init=3,
       verbose=True
   )

   # Transform the data
   X_transformed = pp.fit_transform(X_scaled)
   print(f"Original shape: {X.shape}")
   print(f"Transformed shape: {X_transformed.shape}")

   # Evaluate the embedding
   metrics = pp.evaluate(X_scaled, y)
   for metric, value in metrics.items():
       print(f"{metric}: {value:.4f}")

Key Concepts
------------

Objectives
~~~~~~~~~~

pyppur supports two main objectives:

1. **Distance Distortion** (``Objective.DISTANCE_DISTORTION``):
   Minimizes the difference between pairwise distances in the original and projected spaces.
   
   * With nonlinearity (default): Compares distances after applying tanh transformation
   * Without nonlinearity: Compares linear projection distances

2. **Reconstruction** (``Objective.RECONSTRUCTION``):
   Minimizes reconstruction error using ridge functions.
   
   * Tied weights (default): Encoder and decoder share the same weights
   * Untied weights: Separate encoder and decoder with optional regularization

Ridge Functions
~~~~~~~~~~~~~~~

pyppur uses tanh as the ridge function: :math:`g(z) = \tanh(\alpha \cdot z)`

The steepness parameter :math:`\alpha` controls the nonlinearity:

* Small :math:`\alpha`: Nearly linear behavior
* Large :math:`\alpha`: Strong nonlinearity with saturation

Configuration Options
~~~~~~~~~~~~~~~~~~~~~

Key parameters you can adjust:

* ``n_components``: Number of output dimensions
* ``alpha``: Ridge function steepness
* ``tied_weights``: Whether to use tied encoder/decoder weights (reconstruction only)
* ``use_nonlinearity_in_distance``: Whether to apply nonlinearity in distance objective
* ``l2_reg``: L2 regularization for decoder weights (untied weights only)
* ``n_init``: Number of random initializations
* ``max_iter``: Maximum optimization iterations

Comparison with Other Methods
-----------------------------

pyppur vs PCA
~~~~~~~~~~~~~

* **PCA**: Finds linear projections that maximize variance
* **pyppur**: Finds nonlinear projections that optimize specific objectives (distance/reconstruction)

pyppur vs t-SNE/UMAP
~~~~~~~~~~~~~~~~~~~~

* **t-SNE/UMAP**: Focus on local neighborhood preservation
* **pyppur**: Optimizes global objectives with mathematical interpretability

Next Steps
----------

* Read the :doc:`mathematical_theory` to understand the algorithms
* Explore the :doc:`examples` for more detailed use cases
* Check the :doc:`api_reference` for complete parameter documentation