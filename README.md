<p align="center">
  <img src="divergence-logo.jpg" alt="Divergence" width="400">
</p>

<h1 align="center">Divergence</h1>
<p align="center"><em>The Dissolution of Uncertainty — One Bit at a Time</em></p>

<p align="center">
  <a href="https://github.com/michaelnowotny/divergence/actions/workflows/test.yml"><img src="https://github.com/michaelnowotny/divergence/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/divergence/"><img src="https://img.shields.io/pypi/v/divergence" alt="PyPI"></a>
  <a href="https://pypi.org/project/divergence/"><img src="https://img.shields.io/pypi/pyversions/divergence" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
</p>

---

Divergence is a Python package for computing information-theoretic measures of entropy and divergence from probability distributions and samples.

In 1948, Claude Shannon published ["A Mathematical Theory of Communication"](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf), laying the foundation for information theory and giving us a rigorous way to quantify uncertainty. Divergence puts Shannon's toolkit — and the extensions built upon it over the following decades — into your hands.

## What You Can Compute

| Measure | Discrete | Continuous | What it tells you |
|---------|:--------:|:----------:|-------------------|
| **Entropy** | yes | yes | How much uncertainty a distribution carries |
| **Cross Entropy** | yes | yes | The cost of encoding one distribution using another's code |
| **KL Divergence** | yes | yes | How much information is lost when approximating one distribution with another |
| **Jensen-Shannon Divergence** | yes | yes | A symmetric, bounded measure of distributional difference |
| **Mutual Information** | yes | yes | How much knowing one variable tells you about another |
| **Joint Entropy** | yes | yes | The total uncertainty in a pair of variables |
| **Conditional Entropy** | yes | yes | The remaining uncertainty in one variable after observing another |

All measures support configurable logarithm bases: `base=np.e` (nats, default), `base=2` (bits), `base=10` (hartleys).

In a Bayesian context, relative entropy can be used as a measure of the information gained by moving from a prior distribution *q* to a posterior distribution *p*.

## Installation

```bash
pip install divergence
```

## Quick Start

```python
import numpy as np
from divergence import entropy_from_samples, relative_entropy_from_samples

# How much uncertainty does this distribution carry?
sample = np.random.normal(0, 1, size=10000)
h = entropy_from_samples(sample, discrete=False)

# How different are these two distributions?
p = np.random.normal(0, 1, size=10000)
q = np.random.normal(0.5, 1.2, size=10000)
kl = relative_entropy_from_samples(p, q, discrete=False)
```

## Explore the Notebook

The interactive [Divergence notebook](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Divergence.ipynb) walks through every measure with explanations, historical context, and worked examples — a self-contained introduction to information theory through code.

## Development Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/) (a fast Python package manager):

```bash
git clone https://github.com/michaelnowotny/divergence.git
cd divergence

# Create an isolated virtual environment and install everything
uv venv .venv --python 3.12
source .venv/bin/activate          # Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

### Run the Notebook

```bash
./scripts/lab
```

This script ensures Jupyter Lab runs inside the project's virtual environment with all dependencies correctly installed — no version conflicts, no wrong Python.

> **Tip:** Always launch Jupyter from within the activated `.venv` environment.
> Running `jupyter lab` from a system Python or different environment will fail to find the `divergence` package.

### Run Tests and Linting

```bash
pytest                              # All 122 tests (~18s)
pytest tests/test_discrete.py       # Discrete tests only (fast, ~2s)

ruff check src/ tests/              # Lint
ruff format src/ tests/             # Format
```

## References

1. Shannon, C. E. (1948). ["A Mathematical Theory of Communication."](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) *Bell System Technical Journal*, 27(3), 379-423.
2. Kullback, S. & Leibler, R. A. (1951). "On Information and Sufficiency." *Annals of Mathematical Statistics*, 22(1), 79-86.
3. Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd edition. Wiley.
4. [Entropy (information theory)](https://en.wikipedia.org/wiki/Entropy_(information_theory)) — Wikipedia
5. [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) — Wikipedia
6. [Jensen-Shannon divergence](https://en.wikipedia.org/wiki/Jensen%E2%80%93Shannon_divergence) — Wikipedia
7. [Mutual information](https://en.wikipedia.org/wiki/Mutual_information) — Wikipedia

## License

MIT
