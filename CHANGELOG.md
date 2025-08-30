# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-30

### Added
- Comprehensive test coverage (88% code coverage, 41 tests)
- CI/CD pipeline with GitHub Actions for Python 3.8-3.11
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
- Updated README with correct Python version requirements (3.8+)
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