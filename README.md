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

In 1948, **Claude Shannon's** ["A Mathematical Theory of Communication"](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf) gave information a precise definition. Entropy, measured in bits, became the unit of uncertainty.

Three years later, **Solomon Kullback** and **Richard Leibler** — cryptanalysts at the NSA — defined *relative entropy*: a way to say how much one distribution differs from another. In 1961, **Alfréd Rényi** generalised Shannon's entropy into a one-parameter family. The decades since produced f-divergences, optimal-transport distances, kernel methods, and score-based measures — variations on the same question: *how different are these two distributions?*

Divergence is a Python library that implements that toolkit in one place: Shannon measures, f-divergences, Rényi, integral probability metrics, kNN estimators, score-based measures, optimal transport, and Bayesian MCMC diagnostics. Discrete or continuous, sample-based or density-based, with Numba acceleration on the hot paths and ArviZ integration for MCMC workflows.

### Who uses it

If you run NUTS or HMC in NumPyro, PyMC, Stan, PyJAGS, or emcee, `chain_ksd` answers a question R-hat can't: did your chains converge to the *correct* target distribution? `chain_divergence` and `chain_two_sample_test` complement it for chain-by-chain agreement, and `information_gain` quantifies how much the data updated your prior.

If you compare distributions for a living — generative-model evaluation, dataset shift detection, two-sample tests, feature-dependence screening — energy distance, MMD, Wasserstein, Sinkhorn, and KSG mutual information are all here, with permutation tests built in.

If you're learning information theory, the nine notebooks walk through the field's history with worked examples, from Shannon and Kullback-Leibler through Csiszár, Rényi, Watanabe, Schreiber, Cuturi, and Gorham-Mackey.

---

## What You Can Compute

### Shannon measures

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

### f-divergences

*Imre Csiszár (1963), Shun-ichi Amari (1985)*

| Measure | Function | Properties |
|---------|----------|------------|
| Total Variation | `total_variation_distance(p, q)` | Symmetric, bounded [0, 1], true metric |
| Squared Hellinger | `squared_hellinger_distance(p, q)` | Symmetric, bounded [0, 2], robust to outliers |
| Chi-Squared | `chi_squared_divergence(p, q)` | Asymmetric, unbounded, classical goodness-of-fit |
| Jeffreys | `jeffreys_divergence(p, q)` | Symmetric KL (sum of both directions) |
| Cressie-Read | `cressie_read_divergence(p, q, lambda_param)` | Parameterized family unifying KL, chi², Hellinger |
| General f-divergence | `f_divergence(p, q, f=...)` | Any convex generator function |

### Rényi family

*Alfréd Rényi (1961)*

| Measure | Function | Special cases |
|---------|----------|---------------|
| Rényi Entropy | `renyi_entropy(x, alpha)` | α→0: Hartley, α→1: Shannon, α=2: collision, α→∞: min-entropy |
| Rényi Divergence | `renyi_divergence(p, q, alpha)` | α→1: KL divergence, monotonically non-decreasing in α |

### Integral probability metrics

*Leonid Kantorovich (1942), Gábor Székely (2004), Arthur Gretton (2006)*

| Measure | Function | Key advantage |
|---------|----------|---------------|
| Energy Distance | `energy_distance(p, q)` | No hyperparameters, works in any dimension |
| Wasserstein | `wasserstein_distance(p, q, p=1)` | True metric, interpretable units |
| Sliced Wasserstein | `sliced_wasserstein_distance(p, q)` | Scales to high dimensions via random projections |
| MMD | `maximum_mean_discrepancy(p, q)` | Kernel-based, consistent against all alternatives |

### kNN estimators

*Kozachenko & Leonenko (1987), Kraskov, Stögbauer & Grassberger (2004)*

| Measure | Function | Key advantage |
|---------|----------|---------------|
| kNN Entropy | `knn_entropy(x, k=5)` | Scales gracefully to high dimensions |
| kNN KL Divergence | `knn_kl_divergence(p, q, k=5)` | No density estimation needed |
| KSG Mutual Information | `ksg_mutual_information(x, y, k=5)` | Detects all dependence, linear and nonlinear |

### Multivariate dependence

*Satosi Watanabe (1960), Marina Meilă (2003)*

| Measure | Function | What it measures |
|---------|----------|-----------------|
| Total Correlation | `total_correlation(samples)` | Total redundancy among d ≥ 2 variables |
| Normalized MI | `normalized_mutual_information(x, y)` | MI on a [0, 1] scale; pass a list of normalizations to compute several at once |
| Variation of Information | `variation_of_information(x, y)` | True metric on partitions (triangle inequality) |

### Causal and temporal — the arrow of information

*Thomas Schreiber (2000)*

| Measure | Function | What it detects |
|---------|----------|----------------|
| Transfer Entropy | `transfer_entropy(source, target)` | Directed information flow between time series |

### Score-based measures — slopes instead of heights

*R. A. Fisher (1925), Qiang Liu, Jason Lee & Michael Jordan (2016), Jackson Gorham & Lester Mackey (2017)*

| Measure | Function | Key advantage |
|---------|----------|---------------|
| Fisher Divergence | `fisher_divergence(p, score_q)` | Compares score functions, no normalizing constant |
| Kernel Stein Discrepancy | `kernel_stein_discrepancy(x, score)` | Goodness-of-fit without computing Z (RBF + IMQ kernels) |

### Optimal transport

*Marco Cuturi (2013), Aude Genevay (2018)*

| Measure | Function | Key advantage |
|---------|----------|---------------|
| Sinkhorn Divergence | `sinkhorn_divergence(p, q)` | Fast, differentiable optimal transport |

### Two-sample testing — is the difference real?

*Ronald Fisher (1930s), Arthur Gretton (2012)*

| Function | What it does |
|----------|-------------|
| `two_sample_test(p, q, method="mmd")` | Permutation test with calibrated p-values (MMD, energy, kNN methods) |

### Bayesian MCMC diagnostics

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

The hot paths use Numba JIT kernels, dispatched automatically by input size.

Energy distance has a 1D sort-based kernel (n=3000 runs in ~30 μs) and a multi-D streaming kernel that handles n=50,000+ without exhausting RAM. MMD JITs at n ≥ 500; n=2000 runs in ~43 ms. The MMD permutation test in `two_sample_test` precomputes the full kernel matrix once and uses the identity `S_PQ = (K_total - K_PP - K_QQ) / 2` to skip one block sum per permutation. Sinkhorn's log-domain iterations are inlined in Numba (~4× faster than the SciPy reference); there is no Python fallback. KSD has a streaming Stein-kernel sum for both the RBF and IMQ choices, dispatched at n ≥ 500.

For large-scale two-sample testing, 1D energy distance is the fastest choice: n=3000 per group with 500 permutations runs in ~0.11 s end-to-end.

A GPU backend (JAX, energy distance only at the moment) is available via `backend="gpu"` or the `DIVERGENCE_BACKEND=gpu` environment variable.

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

Nine notebooks form a progressive learning path. The first four build the toolbox; the next two apply it; the last three are the climax (goodness-of-fit via KSD) and an applied showcase.

| # | Notebook | Topics |
|---|----------|--------|
| 1 | [Shannon's Foundations](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Divergence.ipynb) | Entropy, KL divergence, mutual information, joint and conditional entropy |
| 2 | [Beyond KL](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Beyond_KL.ipynb) | f-divergences, Cressie-Read continuum, Rényi family |
| 3 | [Distances & Testing](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Distances_and_Testing.ipynb) | Wasserstein, energy, MMD, Sinkhorn, kNN estimators, permutation tests |
| 4 | [Dependence & Causality](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Dependence_and_Causality.ipynb) | Total correlation, variation of information, transfer entropy |
| 5 | [Bayesian Diagnostics — The Nile](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Bayesian_Diagnostics.ipynb) | End-to-end Bayesian change-point analysis with emcee |
| 6 | [Real-World Applications](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Real_World_Applications.ipynb) | Stock-market contagion, crop yields, Phillips Curve diagnostics |
| 7 | [Score-Based Divergences: Fisher and Stein](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Scores_and_Transport.ipynb) | Fisher divergence, kernel Stein discrepancy, the 250-year journey from Bayes to Stein |
| 8 | [Did My Sampler Find the Truth?](https://github.com/michaelnowotny/divergence/blob/master/notebooks/NumPyro_KSD.ipynb) | KSD as convergence diagnostic with NumPyro: NUTS vs VI vs wrong samples |
| 9 | [Phillips Curve TVP](https://github.com/michaelnowotny/divergence/blob/master/notebooks/Phillips_Curve_TVP.ipynb) | Time-varying Phillips Curve via PyJAGS Gibbs sampling — stagflation as a structural break |

## Documentation

Full API reference and rendered tutorials at **[michaelnowotny.github.io/divergence](https://michaelnowotny.github.io/divergence/)**.

## Development

```bash
git clone https://github.com/michaelnowotny/divergence.git
cd divergence
uv venv .venv --python 3.12 && source .venv/bin/activate
uv pip install -e ".[dev]"

make test          # Run the test suite (391 tests)
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
