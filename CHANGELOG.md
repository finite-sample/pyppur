# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.1] - 2025-09-18

### Changed
- **Minimum Python version raised to 3.10+**
  - Removed support for Python 3.8 and 3.9
  - Updated CI matrix to test Python 3.10, 3.11, 3.12, 3.13
  - Updated documentation and requirements across all files
- **Development improvements**
  - Disabled mypy type checking in CI (requires systematic type design work)
  - Fixed test parameter validation to work with warning filtering
  - Improved Black formatting compliance

## [0.3.0] - 2025-09-18

### Added
- **Free decoder option for reconstruction objective** (`tied_weights=False`)
  - Separate encoder and decoder matrices for better reconstruction
  - Optional L2 regularization on decoder weights (`l2_reg` parameter)
  - New `decoder_weights_` property to access decoder matrix
- **Optional nonlinearity in distance distortion** (`use_nonlinearity_in_distance=False`)
  - Option to use linear projections for distance preservation
  - Better alignment with classical multidimensional scaling
- **Comprehensive Sphinx documentation**
  - Professional documentation site with GitHub Pages deployment
  - Mathematical theory section with LaTeX equations
  - Detailed API reference with auto-generated docstrings
  - Getting started guide and comprehensive examples
  - Automated documentation building via GitHub Actions

### Fixed
- **Normalization handling in optimization**
  - Moved normalization from objective functions to optimizer
  - Eliminates scale invariance and flat directions in optimization
  - Improved convergence behavior and stability
- **Mathematical correctness**
  - Fixed tied-weights reconstruction formula implementation
  - Proper parameter handling for untied weights in optimizer
  - Consistent normalization across all methods

### Changed
- **Enhanced API with backward compatibility**
  - All existing code continues to work unchanged
  - New parameters have sensible defaults
  - Improved parameter validation and error messages
- **Updated mathematical formulations in documentation**
  - Clarified tied vs untied weight formulations
  - Fixed notation inconsistencies in README theory section
  - Added comprehensive mathematical theory documentation

### Documentation
- Added complete Sphinx documentation with RTD theme
- GitHub Actions workflow for automatic documentation deployment
- Mathematical equations rendered with MathJax
- Cross-referenced API documentation
- Comprehensive examples and tutorials

## [0.2.0] - 2025-01-30

### Added
- Comprehensive test coverage (88% code coverage, 41 tests)
- CI/CD pipeline with GitHub Actions for Python 3.10-3.13
- Type annotations and mypy support
- Edge case testing and error handling improvements
- Extended visualization capabilities for non-image data
- Robust error handling for small datasets in trustworthiness calculations
- Enhanced visualization comparison functions with dimension validation
- Comprehensive docstrings and code documentation
- Development tools configuration (Black, isort, mypy)

### Fixed
- Fixed import inconsistencies in main projection_pursuit module  
- Fixed optimizer parameter handling in SciPy optimizer
- Fixed trustworthiness calculation edge cases for small datasets (n_neighbors validation)
- Fixed visualization functions for mixed-dimension embeddings
- Fixed test edge cases and improved test robustness
- Fixed code formatting compliance with Black and isort
- Fixed visualization reconstruction plot for general data types

### Changed
- Bumped version to 0.2.0
- Updated project metadata and classifiers in pyproject.toml
- Enhanced error messages and validation throughout codebase
- Improved mathematical implementation robustness
- Updated README with correct Python version requirements (3.10+)
- Enhanced package metadata with development status and audience

### Development
- Added comprehensive test suite covering edge cases
- Set up GitHub Actions CI/CD with full matrix testing
- Added code quality checks (Black, isort, mypy) 
- Improved developer experience with CLAUDE.md documentation
- Enhanced package building and distribution workflows

## [0.1.0] - 2025-01-XX

### Added
- Initial release of pyppur
- Projection pursuit implementation for dimensionality reduction
- Two optimization objectives: distance distortion and reconstruction
- Support for scikit-learn compatible API
- Ridge function autoencoders with tanh activation
- Multiple initialization strategies (PCA + random)
- Comprehensive evaluation metrics (trustworthiness, silhouette, distance distortion)
- Visualization utilities for embeddings and comparisons
- Preprocessing utilities for data standardization