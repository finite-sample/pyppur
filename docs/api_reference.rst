API Reference
=============

This page provides detailed documentation for all pyppur classes and functions.

Main Classes
------------

ProjectionPursuit
~~~~~~~~~~~~~~~~~

.. autoclass:: pyppur.ProjectionPursuit
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

   .. rubric:: Methods

   .. autosummary::
      :toctree: generated/

      ~pyppur.ProjectionPursuit.fit
      ~pyppur.ProjectionPursuit.transform
      ~pyppur.ProjectionPursuit.fit_transform
      ~pyppur.ProjectionPursuit.reconstruct
      ~pyppur.ProjectionPursuit.reconstruction_error
      ~pyppur.ProjectionPursuit.distance_distortion
      ~pyppur.ProjectionPursuit.compute_trustworthiness
      ~pyppur.ProjectionPursuit.compute_silhouette
      ~pyppur.ProjectionPursuit.evaluate

   .. rubric:: Properties

   .. autosummary::
      :toctree: generated/

      ~pyppur.ProjectionPursuit.x_loadings_
      ~pyppur.ProjectionPursuit.decoder_weights_
      ~pyppur.ProjectionPursuit.loss_curve_
      ~pyppur.ProjectionPursuit.best_loss_
      ~pyppur.ProjectionPursuit.fit_time_
      ~pyppur.ProjectionPursuit.optimizer_info_

Objective Types
~~~~~~~~~~~~~~~

.. autoclass:: pyppur.Objective
   :members:
   :undoc-members:
   :show-inheritance:

Objective Functions
------------------

Base Objective
~~~~~~~~~~~~~

.. autoclass:: pyppur.objectives.BaseObjective
   :members:
   :undoc-members:
   :special-members: __init__, __call__
   :show-inheritance:

Distance Objective
~~~~~~~~~~~~~~~~~

.. autoclass:: pyppur.objectives.DistanceObjective
   :members:
   :undoc-members:
   :special-members: __init__, __call__
   :show-inheritance:

Reconstruction Objective
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyppur.objectives.ReconstructionObjective
   :members:
   :undoc-members:
   :special-members: __init__, __call__
   :show-inheritance:

Optimizers
----------

Base Optimizer
~~~~~~~~~~~~~

.. autoclass:: pyppur.optimizers.BaseOptimizer
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

SciPy Optimizer
~~~~~~~~~~~~~~

.. autoclass:: pyppur.optimizers.ScipyOptimizer
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Grid Optimizer
~~~~~~~~~~~~~

.. autoclass:: pyppur.optimizers.GridOptimizer
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Utility Functions
----------------

Metrics
~~~~~~~

.. automodule:: pyppur.utils.metrics
   :members:
   :undoc-members:
   :show-inheritance:

.. autosummary::
   :toctree: generated/

   pyppur.utils.metrics.compute_trustworthiness
   pyppur.utils.metrics.compute_silhouette
   pyppur.utils.metrics.compute_distance_distortion
   pyppur.utils.metrics.evaluate_embedding

Preprocessing
~~~~~~~~~~~~

.. automodule:: pyppur.utils.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
~~~~~~~~~~~~

.. automodule:: pyppur.utils.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
---------------

Normalization
~~~~~~~~~~~~

.. autofunction:: pyppur.optimizers.scipy_optimizer.normalize_projection_directions