# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Divergence is a Python package for computing statistical measures of entropy and divergence from probability distributions and samples. It supports Shannon measures (entropy, cross entropy, KL divergence, Jensen-Shannon divergence, joint entropy, conditional entropy, mutual information), f-divergences (total variation, Hellinger, chi-squared, Jeffreys, Cressie-Read), Rényi entropy/divergence, integral probability metrics (MMD, energy distance, Wasserstein, sliced Wasserstein), kNN estimators (Kozachenko-Leonenko entropy, KSG mutual information, kNN KL divergence), and two-sample testing — for both discrete and continuous distributions.

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
- **`src/divergence/f_divergences.py`** — General f-divergence engine + TV, Hellinger, chi-squared, Jeffreys, Cressie-Read. Both discrete (frequency-based) and continuous (KDE grid integration) paths.
- **`src/divergence/renyi.py`** — Rényi entropy and divergence parameterized by order alpha, with limit cases.
- **`src/divergence/ipms.py`** — Integral probability metrics: energy distance, Wasserstein-p, MMD, sliced Wasserstein.
- **`src/divergence/knn.py`** — kNN-based estimators: Kozachenko-Leonenko entropy, KSG mutual information, kNN KL divergence. Uses scipy cKDTree.
- **`src/divergence/testing.py`** — Two-sample permutation tests with MMD/energy/kNN methods.
- **`src/divergence/bayesian.py`** — ArviZ integration for Bayesian diagnostics (information gain, chain divergence, uncertainty decomposition, Bayesian surprise).
- **`src/divergence/_types.py`** — Shared NamedTuple types (TestResult).
- **`tests/`** — Top-level test directory (outside the package).

### Key Design Patterns

- **Discrete functions** take sample arrays, internally construct frequency distributions, then compute information-theoretic measures. Numba JIT is used on hot paths.
- **Continuous functions** estimate densities from samples using KDE (statsmodels `KDEUnivariate` or scipy `gaussian_kde` for 2D), then numerically integrate over the support using `cubature`. All cubature integrands must return scalars (use `.item()` to extract from arrays).
- **`base` parameter** controls logarithm base throughout (e=nats, 2=bits, 10=hartleys).

### Dependencies

Core: numpy, scipy, numba, statsmodels, cubature. Dev: pytest, hypothesis, pytest-cov, ruff. Optional: arviz (for Bayesian diagnostics), xarray.

### ArviZ Integration (arviz >= 1.0.0)

**IMPORTANT**: We target ArviZ 1.0.0+ which uses `xarray.DataTree` (not the legacy `InferenceData` class from ArviZ 0.x). The API changed substantially:

- `az.from_dict()` takes a **single nested dict** with group names as keys: `az.from_dict({"posterior": {"mu": arr}, "prior": {"mu": arr}})`
- The return type is `xarray.DataTree`, not `arviz.InferenceData`
- Groups are accessed as children: `idata["posterior"]` returns a DataTree node
- Variables within groups: `idata["posterior"]["mu"].values` returns numpy array with shape `(chain, draw, ...)`
- Check group existence: `"posterior" in idata.children`
- List variables: `list(idata["posterior"].ds.data_vars)`
- `az.extract(idata, group="posterior", combined=True)` flattens chains/draws into a single sample dimension

ArviZ is an **optional dependency** — all bayesian.py functions use lazy imports and raise ImportError with installation instructions if arviz is not installed.
