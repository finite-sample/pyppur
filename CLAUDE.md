# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

### Testing
```bash
pytest                    # Run all tests
pytest tests/test_objectives.py  # Run specific test file
pytest -v                 # Verbose output
pytest --cov=pyppur --cov-report=term-missing # Run tests with coverage
```

### Code Quality
```bash
ruff check pyppur/ tests/ # Lint code with Ruff
ruff check --fix pyppur/ tests/  # Fix linting issues automatically
ruff format pyppur/ tests/  # Format code with Ruff
ruff format --check pyppur/ tests/  # Check formatting without changing
# Note: mypy is disabled in CI (requires systematic type design work)
```

### Building and Installation
```bash
pip install -e .          # Install in development mode
pip install -e .[dev]     # Install with dev dependencies
pip install -e .[docs]    # Install with documentation dependencies
python -m build           # Build distribution packages
python -m twine check dist/*  # Check package integrity
pip install dist/pyppur-*.whl  # Install built package
```

### Documentation
```bash
cd docs && make html      # Build documentation locally
pip install -e .[docs]    # Install doc dependencies from pyproject.toml
# All dependencies managed in pyproject.toml (no requirements.txt files)
```

### CI/CD
- GitHub Actions workflow runs on push/PR to main
- Tests run on Python 3.10, 3.11, 3.12, 3.13
- Code quality checks (Ruff for linting and formatting)
- mypy disabled (requires systematic type design work)
- Coverage reporting in terminal
- Package build verification
- Documentation builds with GitHub Pages deployment

## Code Architecture

### Core Components

**Main API (`pyppur/projection_pursuit.py`)**
- `ProjectionPursuit`: Primary class implementing scikit-learn compatible API
- Supports two optimization objectives: distance distortion and reconstruction loss
- Uses ridge functions (tanh) for non-linear projections
- Provides PCA + random initialization strategies with multiple restarts

**Objective Functions (`pyppur/objectives/`)**
- `BaseObjective`: Abstract base class defining the ridge function interface
- `DistanceDistortionObjective`: Minimizes pairwise distance differences between original and projected spaces
- `ReconstructionObjective`: Minimizes reconstruction error using ridge function autoencoders
- Ridge function: g(z) = tanh(alpha * z) where alpha controls steepness

**Optimizers (`pyppur/optimizers/`)**
- `BaseOptimizer`: Abstract base for optimization methods
- `ScipyOptimizer`: Wrapper for scipy.optimize methods (default: L-BFGS-B)
- `GridOptimizer`: Grid search optimizer for parameter exploration

**Utilities (`pyppur/utils/`)**
- `metrics.py`: Evaluation metrics (trustworthiness, silhouette, distance distortion)
- `preprocessing.py`: Data standardization utilities  
- `visualization.py`: Plotting utilities for embeddings

### Key Design Patterns

**Mathematical Foundation**
- Ridge function autoencoder: x̂ = Σ g(a_j^T x) a_j where a_j are projection directions
- Distance distortion: minimizes ||d_X - d_Z||² between distance matrices
- Both objectives use the same tanh ridge function for consistency

**Optimization Strategy**
- Multiple initialization: PCA-based + n_init random starts
- Best result selection across all initializations
- Scipy L-BFGS-B as default optimizer (gradient-based)

**Data Processing Pipeline**
1. Optional standardization (center/scale)
2. Multiple initialization attempts
3. Optimization with chosen objective  
4. Ridge function transformation for final embedding

### Configuration Details

**Python Requirements**: 3.10+ (CI tests on 3.10, 3.11, 3.12, 3.13)
**Key Dependencies**: numpy, scipy, scikit-learn
**Code Style**: Ruff for linting and formatting (88 char line length)
**Dependency Management**: All dependencies in pyproject.toml (no requirements.txt files)
**Documentation**: Sphinx with importlib.metadata (no version duplication)
**Type Checking**: mypy disabled in CI (requires systematic type design work)