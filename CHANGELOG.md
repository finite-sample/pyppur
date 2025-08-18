# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.2.0] - 2025-01-XX

### Added
- Comprehensive test coverage (87%+ code coverage)
- CI/CD pipeline with GitHub Actions
- Type annotations and mypy support
- Edge case testing and error handling
- Extensive visualization testing
- Extended metrics testing
- Better documentation

### Fixed
- Fixed filename typo in utils/__init__.py
- Fixed type annotation issues throughout codebase
- Fixed code formatting and import sorting with Black and isort
- Improved error handling and edge case coverage
- Fixed reproducibility issues with random state handling

### Changed
- Bumped version to 0.2.0
- Updated dependencies (matplotlib>=3.3.0 added)
- Updated development dependencies
- Enhanced code quality with comprehensive linting
- Improved project structure and organization

### Development
- Added comprehensive test suite
- Set up GitHub Actions CI/CD
- Added code quality checks (Black, isort, mypy)
- Improved developer experience

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