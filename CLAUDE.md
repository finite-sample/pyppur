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
black pyppur/ tests/      # Format code with Black
black --check pyppur/ tests/  # Check formatting without changing
isort pyppur/ tests/      # Sort imports
isort --check-only pyppur/ tests/  # Check import sorting
mypy pyppur/ --ignore-missing-imports # Type checking
```

### Building and Installation
```bash
pip install -e .          # Install in development mode
pip install -e .[dev]     # Install with dev dependencies
python -m build           # Build distribution packages
python -m twine check dist/*  # Check package integrity
pip install dist/pyppur-*.whl  # Install built package
```

### CI/CD
- GitHub Actions workflow runs on push/PR to main
- Tests run on Python 3.8, 3.9, 3.10, 3.11
- Code quality checks (Black, isort, mypy)
- Coverage reporting with Codecov
- Package build verification

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

**Python Requirements**: 3.8+ (configured for 3.10)
**Key Dependencies**: numpy, scipy, scikit-learn
**Code Style**: Black formatter (88 char line length), isort for imports
**Type Checking**: mypy with strict settings (disallow_untyped_defs = true)