# Divergence

**The information-theoretic toolkit for Python.**

Divergence computes statistical measures of entropy, divergence, and dependence from probability distributions and samples. It provides a unified API spanning Shannon measures, f-divergences, Rényi families, integral probability metrics, kNN estimators, score-based measures, optimal transport, and Bayesian MCMC diagnostics.

## Why Divergence?

| Need | Divergence provides |
|------|---------------------|
| Compare two distributions | KL, Jensen-Shannon, Hellinger, TV, energy, MMD, Wasserstein, Sinkhorn |
| Measure uncertainty | Shannon entropy, Rényi entropy, kNN entropy |
| Detect dependence | Mutual information, total correlation, NMI, variation of information |
| Detect causality | Transfer entropy |
| Test H₀: P = Q | Permutation tests with MMD, energy, kNN statistics |
| Assess MCMC convergence | Chain divergence, KSD, two-sample tests, mixing diagnostics |
| Bayesian diagnostics | Information gain, surprise, uncertainty decomposition, prior sensitivity |
| Goodness-of-fit without Z | Kernel Stein discrepancy (RBF + IMQ kernels) |

## Quick Example

```python
import numpy as np
from divergence import entropy_from_samples, knn_kl_divergence, two_sample_test

rng = np.random.default_rng(42)
p = rng.normal(0, 1, 5000)
q = rng.normal(0.5, 1.2, 5000)

# Entropy
h = entropy_from_samples(p)

# KL divergence (kNN-based, no density estimation)
kl = knn_kl_divergence(p, q)

# Formal two-sample test
result = two_sample_test(p, q, method="energy", n_permutations=500)
print(f"p-value: {result.p_value:.4f}")
```

## Installation

```bash
pip install divergence
```

For Bayesian diagnostics with ArviZ:

```bash
pip install "divergence[bayesian]"
```

## Learn More

Start with the [tutorials](notebooks/Divergence.ipynb) — six interactive notebooks that build from Shannon's foundations to end-to-end Bayesian inference, with historical context and visualizations throughout.

For the complete function reference, see the [API documentation](api/index.md).
