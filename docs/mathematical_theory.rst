Mathematical Theory
===================

Overview
--------

Projection pursuit finds interesting low-dimensional projections of multivariate data by optimizing 
an "interestingness" index. pyppur implements two main approaches for dimensionality reduction:

1. **Distance Distortion**: Preserves pairwise distances between data points
2. **Reconstruction Error**: Minimizes reconstruction error using ridge functions

Ridge Functions
---------------

pyppur uses the hyperbolic tangent as the ridge function:

.. math::
   g(z) = \tanh(\alpha \cdot z)

where :math:`\alpha` is the steepness parameter that controls the degree of nonlinearity.

The gradient of the ridge function is:

.. math::
   g'(z) = \alpha \cdot \operatorname{sech}^2(\alpha \cdot z) = \alpha \cdot (1 - \tanh^2(\alpha \cdot z))

Reconstruction Objectives
-------------------------

Tied-Weights Ridge Autoencoder (Default)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default reconstruction approach uses tied weights where the encoder and decoder share the same parameters:

.. math::
   \begin{align}
   Z &= g(X A^T) \\
   \hat{X} &= Z A
   \end{align}

The objective is to minimize the reconstruction error:

.. math::
   L_{\text{tied}} = \frac{1}{n} \|X - \hat{X}\|_F^2 = \frac{1}{n} \|X - g(X A^T) A\|_F^2

where:

* :math:`X \in \mathbb{R}^{n \times p}` is the input data matrix
* :math:`A \in \mathbb{R}^{k \times p}` are the encoder/decoder weights  
* :math:`Z \in \mathbb{R}^{n \times k}` is the projected data
* :math:`g(\cdot)` is applied element-wise
* :math:`n` is the number of samples, :math:`p` is the number of features, :math:`k` is the number of components

Free Decoder Ridge Autoencoder
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``tied_weights=False``, the encoder and decoder have separate parameters:

.. math::
   \begin{align}
   Z &= g(X A^T) \\
   \hat{X} &= Z B
   \end{align}

The objective includes optional L2 regularization on the decoder:

.. math::
   L_{\text{untied}} = \frac{1}{n} \|X - Z B\|_F^2 + \lambda \|B\|_F^2

where:

* :math:`A \in \mathbb{R}^{k \times p}` are the encoder weights
* :math:`B \in \mathbb{R}^{k \times p}` are the decoder weights
* :math:`\lambda \geq 0` is the L2 regularization parameter

Distance Distortion Objectives
------------------------------

Distance Metrics
~~~~~~~~~~~~~~~~

pyppur supports three distance metrics for measuring how well pairwise distances are preserved.
The choice of metric significantly affects optimization performance due to scale considerations.

**Correlation Metric (Default, Recommended)**

The correlation metric maximizes Pearson correlation between distance matrices:

.. math::
   L_{\text{corr}} = -\rho(d_X, d_Z)

where :math:`\rho` is the Pearson correlation coefficient computed over the upper triangular
elements of the distance matrices (excluding diagonal).

This metric is **scale-invariant**, making it robust to the scale mismatch between original
distances (which can be large) and embedded distances (bounded by tanh to approximately 2.83 max).

**Spearman Metric**

The Spearman metric uses rank correlation:

.. math::
   L_{\text{spearman}} = -\rho_s(d_X, d_Z)

where :math:`\rho_s` is the Spearman rank correlation. This is useful when only the relative
ordering of distances matters, not their exact values.

**MSE Metric (Original Behavior)**

The MSE metric directly minimizes squared differences:

.. math::
   L_{\text{mse}} = \frac{1}{n^2} \sum_{i,j} (d_{X}(i,j) - d_{Z}(i,j))^2

**Warning**: This metric is scale-sensitive. When using tanh nonlinearity, embedded distances
are bounded while original distances may be large, causing optimization difficulties.

Distance Distortion with Nonlinearity
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The default distance objective compares pairwise distances between the original space and
the nonlinearly transformed projection:

.. math::
   Z = g(X A^T)

where:

* :math:`d_X(i,j) = \|x_i - x_j\|_2` are pairwise distances in the original space
* :math:`d_Z(i,j) = \|z_i - z_j\|_2` are pairwise distances in the transformed space
* :math:`g(\cdot) = \tanh(\alpha \cdot)` is the ridge function

Linear Distance Distortion
~~~~~~~~~~~~~~~~~~~~~~~~~~

With ``use_nonlinearity_in_distance=False``, distances are compared in the linear projection space:

.. math::
   Y = X A^T

where :math:`d_Y(i,j) = \|y_i - y_j\|_2` are distances in the linear projection space.
This reduces to classical multidimensional scaling when :math:`k = p`.

Weighted Distance Distortion
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

When ``weight_by_distance=True``, the objective is weighted to emphasize preservation of small distances:

.. math::
   L_{\text{weighted}} = \sum_{i,j} w_{ij} (d_{X}(i,j) - d_{Z}(i,j))^2

where the weights are:

.. math::
   w_{ij} = \frac{1}{d_X(i,j) + \epsilon}

with :math:`\epsilon = 0.1` to avoid division by zero, and weights are normalized to sum to 1.

Note: Weighted distortion is only used with the MSE metric.

Optimization
------------

Constraint Handling
~~~~~~~~~~~~~~~~~~~

All optimization is performed under the constraint that encoder directions have unit norm:

.. math::
   \|a_j\|_2 = 1 \quad \text{for all } j = 1, \ldots, k

This constraint is enforced by:

1. Normalizing the encoder directions after each optimization step
2. Using a projected gradient approach within the L-BFGS-B optimizer

Initialization Strategies
~~~~~~~~~~~~~~~~~~~~~~~~~

pyppur uses multiple initialization strategies:

1. **PCA Initialization**: Use the first :math:`k` principal components as starting points
2. **Random Initialization**: Sample :math:`n_{\text{init}}` random orthonormal matrices

For untied weights, the decoder is initialized with small random values scaled by 0.1.

The best result across all initializations is retained.

Multi-Start Optimization
~~~~~~~~~~~~~~~~~~~~~~~~

The optimization procedure:

1. Try PCA initialization
2. Try ``n_init`` random initializations  
3. Select the result with the lowest objective value
4. Return the optimal encoder (and decoder if applicable)

Computational Complexity
------------------------

Time Complexity
~~~~~~~~~~~~~~~

For :math:`n` samples, :math:`p` features, and :math:`k` components:

* **Distance distortion**: :math:`O(n^2 k + n p k)` per iteration (dominated by distance computation)
* **Reconstruction**: :math:`O(n p k)` per iteration
* **Overall**: :math:`O(T \cdot (\text{per-iteration cost}))` where :math:`T` is the number of iterations

Space Complexity
~~~~~~~~~~~~~~~~

* **Distance distortion**: :math:`O(n^2)` for storing distance matrices
* **Reconstruction**: :math:`O(n k + p k)` for intermediate computations
* **Parameters**: :math:`O(p k)` for tied weights, :math:`O(2 p k)` for untied weights

Theoretical Properties
----------------------

Convergence
~~~~~~~~~~~

The optimization uses L-BFGS-B, which has the following properties:

* **Local convergence**: Guaranteed to converge to a local minimum under regularity conditions
* **Global optimum**: Not guaranteed due to non-convexity of the objectives
* **Multi-start**: Helps find better local optima

Expressiveness
~~~~~~~~~~~~~~

The ridge function autoencoder can represent a rich class of nonlinear transformations:

* **Universal approximation**: With sufficient components, can approximate any continuous function
* **Regularization**: The :math:`\tanh` nonlinearity provides natural bounded outputs
* **Interpretability**: Each component corresponds to a direction in the original space

Comparison to Related Methods
------------------------------

vs. Linear PCA
~~~~~~~~~~~~~~

* **PCA**: :math:`Z = X A^T` (linear projection)
* **pyppur**: :math:`Z = g(X A^T)` (nonlinear projection with optimized directions)

vs. Kernel PCA
~~~~~~~~~~~~~~

* **Kernel PCA**: Implicit nonlinear mapping via kernel trick
* **pyppur**: Explicit nonlinear mapping with learnable projections

vs. Autoencoders
~~~~~~~~~~~~~~~~

* **Neural autoencoders**: Multiple hidden layers with arbitrary activations
* **pyppur**: Single hidden layer with tanh activation and structured constraints

Outlier Robustness
~~~~~~~~~~~~~~~~~~

A key advantage of pyppur over linear methods (PCA, MDS) is outlier robustness.
The tanh nonlinearity bounds all outputs to :math:`[-1, 1]`, preventing extreme
values from dominating the embedding:

* **PCA/MDS**: Outliers can cause variance ratios >100x between outlier and non-outlier points
* **pyppur**: Bounded outputs limit variance ratios to ~5x even with extreme outliers

This makes pyppur particularly suitable for:

1. Data with natural outliers (financial data, sensor readings)
2. Preprocessing for neural networks (stable, bounded inputs)
3. Robust distance-preserving embeddings