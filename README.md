# Divergence

[![Tests](https://github.com/michaelnowotny/divergence/actions/workflows/test.yml/badge.svg)](https://github.com/michaelnowotny/divergence/actions/workflows/test.yml)
[![PyPI](https://img.shields.io/pypi/v/divergence)](https://pypi.org/project/divergence/)
[![Python](https://img.shields.io/pypi/pyversions/divergence)](https://pypi.org/project/divergence/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Divergence is a Python package to compute statistical measures of entropy and divergence from probability distributions and samples.

The following functionality is provided:
* (Information) Entropy [1], [2]
* Cross Entropy [3]
* Relative Entropy or Kullback-Leibler (KL-) Divergence [4], [5]
* Jensen-Shannon Divergence [6]
* Joint Entropy [7]
* Conditional Entropy [8]
* Mutual Information [9]

The units in which these entropy and divergence measures are calculated can be specified by the user.
This is achieved by setting the argument `base`, to `2.0`, `10.0`, or `np.e`.

In a Bayesian context, relative entropy can be used as a measure of the information gained by moving
from a prior distribution `q` to a posterior distribution `p`.

## Installation

```bash
pip install divergence
```

## Getting Started

### Prerequisites

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) (a fast Python package manager):

```bash
# macOS / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Setup

```bash
git clone https://github.com/michaelnowotny/divergence.git
cd divergence

# Create an isolated virtual environment and install everything
uv venv .venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Run the Notebook

The quickest way to explore the package is through the interactive notebook:

```bash
./scripts/lab
```

This script ensures Jupyter Lab runs inside the project's virtual environment with
all dependencies correctly installed — no version conflicts, no wrong Python.

Alternatively, if you prefer to launch Jupyter manually:

```bash
source .venv/bin/activate
uv pip install jupyterlab matplotlib seaborn
jupyter lab notebooks/
```

> **Important:** Always launch Jupyter from within the activated `.venv` environment.
> Running `jupyter lab` from a system Python or different environment will fail to
> find the `divergence` package.

### Run Tests and Linting

```bash
source .venv/bin/activate
pytest                              # Run all tests
pytest tests/test_discrete.py       # Discrete tests only (fast, ~2s)
pytest tests/test_continuous.py     # Continuous tests (slow, ~8min)

ruff check src/ tests/              # Lint
ruff format src/ tests/             # Format
```

## Examples

See the Jupyter notebook [Divergence](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Divergence.ipynb).

## References

1. https://en.wikipedia.org/wiki/Entropy_(information_theory)
2. Shannon, Claude Elwood (July 1948). "A Mathematical Theory of Communication". Bell System Technical Journal. 27 (3): 379-423
3. https://en.wikipedia.org/wiki/Cross_entropy
4. https://en.wikipedia.org/wiki/Kullback-Leibler_divergence
5. Kullback, S.; Leibler, R.A. (1951). "On information and sufficiency". Annals of Mathematical Statistics. 22 (1): 79-86
6. https://en.wikipedia.org/wiki/Jensen-Shannon_divergence
7. https://en.wikipedia.org/wiki/Joint_entropy
8. https://en.wikipedia.org/wiki/Conditional_entropy
9. https://en.wikipedia.org/wiki/Mutual_information
