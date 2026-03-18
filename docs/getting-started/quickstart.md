# Quick Start

## Comparing Two Distributions

```python
import numpy as np
from divergence import (
    knn_kl_divergence,
    energy_distance,
    jensen_shannon_divergence_from_samples,
    two_sample_test,
)

rng = np.random.default_rng(42)
p = rng.normal(0, 1, 3000)
q = rng.normal(0.5, 1.2, 3000)

# KL divergence (kNN-based, works in any dimension)
kl = knn_kl_divergence(p, q)

# Energy distance (no hyperparameters)
ed = energy_distance(p, q)

# Jensen-Shannon divergence (symmetric, bounded)
jsd = jensen_shannon_divergence_from_samples(p, q)

# Formal hypothesis test: H0: P = Q
result = two_sample_test(p, q, method="energy", n_permutations=500, seed=42)
print(f"Reject H0? {result.p_value < 0.05} (p = {result.p_value:.4f})")
```

## Measuring Entropy

```python
from divergence import entropy_from_samples, knn_entropy

# KDE-based entropy (1D)
h_kde = entropy_from_samples(p)

# kNN-based entropy (scales to high dimensions)
h_knn = knn_entropy(p, k=5)
```

## Discrete Distributions

```python
from divergence import discrete_entropy, discrete_mutual_information

labels_x = rng.integers(0, 5, 1000)
labels_y = rng.integers(0, 5, 1000)

h = discrete_entropy(labels_x, base=2)  # in bits
mi = discrete_mutual_information(labels_x, labels_y, base=2)
```

## Bayesian Diagnostics (ArviZ)

```python
import arviz as az
from divergence import information_gain, chain_divergence

# After running MCMC...
idata = az.from_dict({
    "posterior": {"mu": rng.normal(5, 0.5, (4, 1000))},
    "prior": {"mu": rng.normal(0, 10, (4, 1000))},
})

# How much did the data update our beliefs?
ig = information_gain(idata)

# Are the chains sampling the same distribution?
cd = chain_divergence(idata)
```

## Next Steps

Explore the [tutorials](../notebooks/Divergence.ipynb) for in-depth coverage with visualizations and historical context, or dive into the [API reference](../api/index.md) for the full function catalog.
