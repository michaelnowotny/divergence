# Installation

## From PyPI

```bash
pip install divergence
```

## Optional Dependencies

Divergence has optional extras for specific use cases:

```bash
# Bayesian diagnostics (ArviZ integration)
pip install "divergence[bayesian]"

# Development (testing, linting)
pip install "divergence[dev]"

# Integration testing (emcee)
pip install "divergence[integration]"
```

## From Source

```bash
git clone https://github.com/michaelnowotny/divergence.git
cd divergence
pip install -e ".[dev]"
```

## Requirements

- Python >= 3.12
- NumPy >= 1.24
- SciPy >= 1.10
- Numba
- statsmodels
- cubature

For Bayesian diagnostics: ArviZ >= 1.0, xarray >= 2024.1.
