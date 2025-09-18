Changelog
=========

Version 0.3.0 (Current)
-----------------------

**Major Features**

* Added free decoder option for reconstruction objective
* Added option to disable nonlinearity in distance distortion objective  
* Fixed normalization handling in optimization
* Improved mathematical correctness and documentation

**New Parameters**

* ``tied_weights`` (bool): Whether to use tied weights for reconstruction (default: True)
* ``l2_reg`` (float): L2 regularization strength for decoder weights (default: 0.0) 
* ``use_nonlinearity_in_distance`` (bool): Whether to apply ridge function before computing distances (default: True)

**New Properties**

* ``decoder_weights_``: Access to decoder weights for untied reconstruction models

**API Improvements**

* Maintained full backward compatibility
* Enhanced parameter validation and error messages
* Improved optimization convergence through better normalization handling

**Documentation**

* Added comprehensive Sphinx documentation
* Clarified mathematical formulations in README
* Added detailed examples and API reference
* Fixed mathematical notation inconsistencies

**Testing**

* Added comprehensive test suite for new features
* Verified mathematical correctness of implementations
* Added performance and convergence tests

Version 0.1.x (Previous)
------------------------

* Initial implementation of projection pursuit
* Basic distance distortion and reconstruction objectives
* PCA and random initialization strategies
* Integration with scikit-learn API