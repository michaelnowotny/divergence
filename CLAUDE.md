# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Divergence is a Python package for computing statistical measures of entropy and divergence from probability distributions and samples. It supports entropy, cross entropy, KL divergence, Jensen-Shannon divergence, joint entropy, conditional entropy, and mutual information for both discrete and continuous distributions.

## Development Environment Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Build & Install

```bash
uv pip install -e ".[dev]"     # Editable install with dev deps
python -m build                 # Build distribution
```

Uses `pyproject.toml` with setuptools backend and setuptools-scm for versioning from git tags.

## Running Tests

```bash
pytest                                          # All tests
pytest tests/test_discrete.py                   # Discrete measure tests only
pytest tests/test_continuous.py                 # Continuous tests (slow, ~8min due to numerical integration)
pytest tests/test_discrete.py::test_discrete_entropy -v  # Single test
```

Tests use fixed random seeds (42) for reproducibility. Continuous tests validate against scipy.stats ground truth via KDE estimation.

## Linting & Formatting

```bash
ruff check src/ tests/          # Lint
ruff check --fix src/ tests/    # Lint with auto-fix
ruff format src/ tests/         # Format
```

## Architecture

### Module Layout (src/ layout)

- **`src/divergence/__init__.py`** — Public API. Provides unified wrapper functions (e.g., `entropy_from_samples()`) that dispatch to discrete or continuous implementations based on `discrete=True/False` parameter. Exports all public functions via explicit `__all__`.
- **`src/divergence/base.py`** — Shared utility: log function selection (`log`, `log2`, `log10`) based on `base` parameter, plus Numba `Logarithm` jitclass for use in JIT-compiled code.
- **`src/divergence/discrete.py`** — Discrete distribution measures. Uses Numba `@njit` for performance-critical frequency counting and internal computation functions.
- **`src/divergence/continuous.py`** — Continuous distribution measures. Uses `cubature` for N-dimensional numerical integration and `statsmodels` for kernel density estimation.
- **`tests/`** — Top-level test directory (outside the package).

### Key Design Patterns

- **Discrete functions** take sample arrays, internally construct frequency distributions, then compute information-theoretic measures. Numba JIT is used on hot paths.
- **Continuous functions** estimate densities from samples using KDE (statsmodels `KDEUnivariate` or scipy `gaussian_kde` for 2D), then numerically integrate over the support using `cubature`. All cubature integrands must return scalars (use `.item()` to extract from arrays).
- **`base` parameter** controls logarithm base throughout (e=nats, 2=bits, 10=hartleys).

### Dependencies

Core: numpy, scipy, numba, statsmodels, cubature. Dev: pytest, hypothesis, pytest-cov, ruff.
