<p align="center">
  <img src="https://raw.githubusercontent.com/michaelnowotny/divergence/master/divergence-logo.jpg" alt="Divergence" width="400">
</p>

<h1 align="center">Divergence</h1>
<p align="center"><em>The Dissolution of Uncertainty — One Bit at a Time</em></p>

<p align="center">
  <a href="https://github.com/michaelnowotny/divergence/actions/workflows/test.yml"><img src="https://github.com/michaelnowotny/divergence/actions/workflows/test.yml/badge.svg" alt="Tests"></a>
  <a href="https://pypi.org/project/divergence/"><img src="https://img.shields.io/pypi/v/divergence" alt="PyPI"></a>
  <a href="https://pypi.org/project/divergence/"><img src="https://img.shields.io/pypi/pyversions/divergence" alt="Python"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://michaelnowotny.github.io/divergence/"><img src="https://img.shields.io/badge/docs-mkdocs-blue" alt="Docs"></a>
</p>

---

## Why Divergence?

In the summer of 1948, a 32-year-old mathematician at Bell Labs named **Claude Shannon** published a paper that quietly changed the world. ["A Mathematical Theory of Communication"](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) gave humanity something it had never had before: a *rigorous, mathematical definition of information*. Shannon showed that uncertainty could be measured, that communication had fundamental limits, and that a single quantity — **entropy** — sat at the heart of it all.

What followed was an explosion of ideas. In 1951, **Solomon Kullback** and **Richard Leibler** — working as cryptanalysts at the NSA — formalized the concept of *relative entropy*, measuring how one probability distribution diverges from another. In 1961, **Alfréd Rényi** generalized Shannon's entropy into a family parameterized by a single number, revealing that information has not one but infinitely many faces. In the decades since, mathematicians, statisticians, and computer scientists have built an extraordinary edifice on Shannon's foundation: f-divergences, optimal transport distances, kernel methods, score-based measures — each offering a different lens on the same fundamental question: *how different are these two probability distributions?*

**Divergence** puts this entire toolkit into your hands.

It is the most comprehensive information-theoretic package for Python — spanning Shannon measures, f-divergences, Rényi families, integral probability metrics, kNN estimators, score-based divergences, optimal transport, and Bayesian MCMC diagnostics in a single, unified API. Whether you are a Bayesian statistician checking MCMC convergence, a machine learning researcher comparing generative models, a neuroscientist measuring information flow between brain regions, or a student encountering entropy for the first time, Divergence provides the tools you need.

### Who is this for?

- **Bayesian practitioners** using PyMC, Stan, NumPyro, PyJAGS, or emcee — Divergence integrates directly with ArviZ to provide information-theoretic diagnostics that go far beyond R-hat: information gain, chain convergence testing, Bayesian surprise, uncertainty decomposition, and the only diagnostic that tests convergence to the *correct target* (kernel Stein discrepancy)

- **Machine learning researchers** — compare distributions with energy distance, MMD, Wasserstein, or Sinkhorn divergence; run formal two-sample tests; measure feature dependence with mutual information and total correlation

- **Scientists and engineers** — detect causal information flow with transfer entropy, measure multivariate dependence, quantify distributional shift in monitoring systems

- **Students and educators** — six interactive notebooks with historical narrative, from Shannon's foundations through modern optimal transport, including an end-to-end Bayesian detective story using real hydrological data

### Everything in one place

Divergence brings together measures that are otherwise scattered across different packages, subfields, and textbooks — **Shannon measures, f-divergences, Rényi families, integral probability metrics, kNN estimators, score-based divergences, optimal transport, and Bayesian diagnostics** — in a single, coherent API. Discrete and continuous, sample-based and density-based, with configurable units (nats/bits/hartleys), Numba-accelerated performance at scale, and seamless ArviZ integration for Bayesian workflows.

---

## What You Can Compute

### Shannon Measures — The Foundation

*Claude Shannon (1948), Solomon Kullback & Richard Leibler (1951)*

| Measure | Function | What it tells you |
|---------|----------|-------------------|
| Entropy | `entropy(sample)` | How much uncertainty a distribution carries |
| Cross Entropy | `cross_entropy(p, q)` | The cost of encoding P using Q's code |
| KL Divergence | `kl_divergence(p, q)` | Information lost when approximating P with Q |
| Jensen-Shannon | `jensen_shannon_divergence(p, q)` | Symmetric, bounded distributional difference |
| Mutual Information | `mutual_information(x, y)` | How much knowing X tells you about Y |
| Joint Entropy | `joint_entropy(x, y)` | Total uncertainty in a pair of variables |
| Conditional Entropy | `conditional_entropy(x, y)` | Remaining uncertainty after observing the other |

All support `discrete=True/False` and `base=np.e` (nats) / `2` (bits) / `10` (hartleys).

### f-Divergences — The Unifying Family

*Imre Csiszár (1963), Shun-ichi Amari (1985)*

| Measure | Function | Properties |
|---------|----------|------------|
| Total Variation | `total_variation_distance(p, q)` | Symmetric, bounded [0, 1], true metric |
| Squared Hellinger | `squared_hellinger_distance(p, q)` | Symmetric, bounded [0, 2], robust to outliers |
| Chi-Squared | `chi_squared_divergence(p, q)` | Asymmetric, unbounded, classical goodness-of-fit |
| Jeffreys | `jeffreys_divergence(p, q)` | Symmetric KL (sum of both directions) |
| Cressie-Read | `cressie_read_divergence(p, q, lambda_param)` | Parameterized family unifying KL, chi², Hellinger |
| General f-divergence | `f_divergence(p, q, f=...)` | Any convex generator function |

### Rényi Family — The Alpha Telescope

*Alfréd Rényi (1961)*

| Measure | Function | Special cases |
|---------|----------|---------------|
| Rényi Entropy | `renyi_entropy(x, alpha)` | α→0: Hartley, α→1: Shannon, α=2: collision, α→∞: min-entropy |
| Rényi Divergence | `renyi_divergence(p, q, alpha)` | α→1: KL divergence, monotonically non-decreasing in α |

### Integral Probability Metrics — Geometry on Distributions

*Leonid Kantorovich (1942), Gábor Székely (2004), Arthur Gretton (2006)*

| Measure | Function | Key advantage |
|---------|----------|---------------|
| Energy Distance | `energy_distance(p, q)` | No hyperparameters, works in any dimension |
| Wasserstein | `wasserstein_distance(p, q, p=1)` | True metric, interpretable units |
| Sliced Wasserstein | `sliced_wasserstein_distance(p, q)` | Scales to high dimensions via random projections |
| MMD | `maximum_mean_discrepancy(p, q)` | Kernel-based, consistent against all alternatives |

### kNN Estimators — Density-Free Information Theory

*Kozachenko & Leonenko (1987), Kraskov, Stögbauer & Grassberger (2004)*

| Measure | Function | Key advantage |
|---------|----------|---------------|
| kNN Entropy | `knn_entropy(x, k=5)` | Scales gracefully to high dimensions |
| kNN KL Divergence | `knn_kl_divergence(p, q, k=5)` | No density estimation needed |
| KSG Mutual Information | `ksg_mutual_information(x, y, k=5)` | Detects all dependence, linear and nonlinear |

### Multivariate Dependence — Beyond Pairwise

*Satosi Watanabe (1960), Marina Meilă (2003)*

| Measure | Function | What it measures |
|---------|----------|-----------------|
| Total Correlation | `total_correlation(samples)` | Total redundancy among d ≥ 2 variables |
| Normalized MI | `normalized_mutual_information(x, y)` | MI on a [0, 1] scale for comparison |
| Variation of Information | `variation_of_information(x, y)` | True metric on partitions (triangle inequality) |

### Causal and Temporal — The Arrow of Information

*Thomas Schreiber (2000)*

| Measure | Function | What it detects |
|---------|----------|----------------|
| Transfer Entropy | `transfer_entropy(source, target)` | Directed information flow between time series |

### Score-Based Measures — Slopes Instead of Heights

*R. A. Fisher (1925), Qiang Liu, Jason Lee & Michael Jordan (2016), Jackson Gorham & Lester Mackey (2017)*

| Measure | Function | Key advantage |
|---------|----------|---------------|
| Fisher Divergence | `fisher_divergence(p, score_q)` | Compares score functions, no normalizing constant |
| Kernel Stein Discrepancy | `kernel_stein_discrepancy(x, score)` | Goodness-of-fit without computing Z (RBF + IMQ kernels) |

### Optimal Transport — The Cost of Rearrangement

*Marco Cuturi (2013), Aude Genevay (2018)*

| Measure | Function | Key advantage |
|---------|----------|---------------|
| Sinkhorn Divergence | `sinkhorn_divergence(p, q)` | Fast, differentiable optimal transport |

### Two-Sample Testing — Is the Difference Real?

*Ronald Fisher (1930s), Arthur Gretton (2012)*

| Function | What it does |
|----------|-------------|
| `two_sample_test(p, q, method="mmd")` | Permutation test with calibrated p-values (MMD, energy, kNN methods) |

### Bayesian MCMC Diagnostics — The ArviZ Companion

*Dennis Lindley (1956), Andrew Gelman & Donald Rubin (1992)*

| Function | What it answers |
|----------|----------------|
| `information_gain(idata)` | How much did the data update our beliefs? |
| `chain_divergence(idata)` | Are chains sampling the same distribution? |
| `chain_ksd(idata, score_fn)` | Have chains converged to the *correct* target? |
| `chain_two_sample_test(idata)` | Formal p-values for chain homogeneity |
| `mixing_diagnostic(idata)` | Has each chain reached stationarity? |
| `bayesian_surprise(idata)` | Which observations are most unexpected? |
| `uncertainty_decomposition(idata)` | How much is noise vs. parameter uncertainty? |
| `prior_sensitivity(idata, ref)` | Does the conclusion depend on the prior? |
| `model_divergence(idata1, idata2)` | How different are two models' predictions? |

Works with **PyMC, Stan, NumPyro, PyJAGS, emcee** — any package that produces ArviZ `InferenceData`.

---

## Performance

For large-scale computations (n ≥ 5,000), energy distance, MMD, and KSD automatically use **Numba JIT-compiled kernels** with O(1) memory and multicore parallelism — 10-18x faster than vectorized implementations and enabling n=50,000+ (which previously exhausted memory).

| Function | n=5K | n=20K | n=50K |
|----------|------|-------|-------|
| Energy distance | 21ms | 276ms | 1.4s |
| MMD | 136ms | 1.9s | 12s |
| KSD | 114ms | 1.4s | 8.6s |

---

## Installation

```bash
pip install divergence
```

For Bayesian diagnostics with ArviZ:

```bash
pip install "divergence[bayesian]"
```

## Quick Start

```python
import numpy as np
from divergence import entropy, kl_divergence, two_sample_test

rng = np.random.default_rng(42)
p = rng.normal(0, 1, 5000)
q = rng.normal(0.5, 1.2, 5000)

# How much uncertainty?
h = entropy(p)

# How different are these distributions?
kl = kl_divergence(p, q)

# Is the difference statistically significant?
result = two_sample_test(p, q, method="energy", n_permutations=500)
print(f"p-value: {result.p_value:.4f}")
```

## Tutorials

Six interactive notebooks form a progressive learning path, with historical narrative and visualizations throughout:

| # | Notebook | Topics |
|---|----------|--------|
| 1 | [Shannon's Foundations](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Divergence.ipynb) | Entropy, KL divergence, mutual information, the information-theoretic web |
| 2 | [Beyond KL](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Beyond_KL.ipynb) | f-divergences, Cressie-Read continuum, Rényi family |
| 3 | [Distances & Testing](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Distances_and_Testing.ipynb) | Wasserstein, energy, MMD, kNN estimators, permutation tests |
| 4 | [Dependence & Causality](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Dependence_and_Causality.ipynb) | Total correlation, variation of information, transfer entropy |
| 5 | [Scores & Transport](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Scores_and_Transport.ipynb) | Fisher divergence, kernel Stein discrepancy, Sinkhorn |
| 6 | [The Nile's Secret](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Bayesian_Diagnostics.ipynb) | End-to-end Bayesian inference with emcee — a detective story |

## Documentation

Full API reference and rendered tutorials at **[michaelnowotny.github.io/divergence](https://michaelnowotny.github.io/divergence/)**.

## Development

```bash
git clone https://github.com/michaelnowotny/divergence.git
cd divergence
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

make test          # Run 345 tests
make lint          # Ruff check + format
make docs-serve    # Live documentation preview
```

## References

1. Shannon, C. E. (1948). ["A Mathematical Theory of Communication."](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) *Bell System Technical Journal*, 27(3), 379-423.
2. Kullback, S. & Leibler, R. A. (1951). "On Information and Sufficiency." *Annals of Mathematical Statistics*, 22(1), 79-86.
3. Rényi, A. (1961). "On Measures of Entropy and Information." *Proc. 4th Berkeley Symposium*, 1, 547-561.
4. Csiszár, I. (1963). "Eine informationstheoretische Ungleichung und ihre Anwendung auf den Beweis der Ergodizitat von Markoffschen Ketten." *Magyar Tud. Akad. Mat. Kutato Int. Kozl.*, 8, 85-108.
5. Gretton, A. et al. (2012). "A Kernel Two-Sample Test." *JMLR*, 13, 723-773.
6. Kraskov, A., Stögbauer, H. & Grassberger, P. (2004). "Estimating Mutual Information." *Physical Review E*, 69(6), 066138.
7. Gorham, J. & Mackey, L. (2017). "Measuring Sample Quality with Kernels." *ICML*.
8. Peyré, G. & Cuturi, M. (2019). *Computational Optimal Transport.* Foundations and Trends in Machine Learning.
9. Cover, T. M. & Thomas, J. A. (2006). *Elements of Information Theory*, 2nd edition. Wiley.

## License

MIT
