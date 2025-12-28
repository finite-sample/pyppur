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

Objective Types
~~~~~~~~~~~~~~~

.. autoclass:: pyppur.Objective
   :members:
   :undoc-members:
   :show-inheritance:

Objective Functions
-------------------

Base Objective
~~~~~~~~~~~~~~

.. autoclass:: pyppur.objectives.BaseObjective
   :members:
   :undoc-members:
   :special-members: __init__, __call__
   :show-inheritance:

Distance Objective
~~~~~~~~~~~~~~~~~~

.. autoclass:: pyppur.objectives.DistanceObjective
   :members:
   :undoc-members:
   :special-members: __init__, __call__
   :show-inheritance:

Reconstruction Objective
~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pyppur.objectives.ReconstructionObjective
   :members:
   :undoc-members:
   :special-members: __init__, __call__
   :show-inheritance:

Optimizers
----------

SciPy Optimizer
~~~~~~~~~~~~~~~

.. autoclass:: pyppur.optimizers.ScipyOptimizer
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Grid Optimizer
~~~~~~~~~~~~~~

.. autoclass:: pyppur.optimizers.GridOptimizer
   :members:
   :undoc-members:
   :special-members: __init__
   :show-inheritance:

Utility Functions
-----------------

Metrics
~~~~~~~

.. automodule:: pyppur.utils.metrics
   :members:
   :undoc-members:
   :show-inheritance:

Preprocessing
~~~~~~~~~~~~~

.. automodule:: pyppur.utils.preprocessing
   :members:
   :undoc-members:
   :show-inheritance:

Visualization
~~~~~~~~~~~~~

.. automodule:: pyppur.utils.visualization
   :members:
   :undoc-members:
   :show-inheritance:

Helper Functions
----------------

Normalization
~~~~~~~~~~~~~

.. autofunction:: pyppur.optimizers.scipy_optimizer.normalize_projection_directions