# API Reference

Divergence provides 72 public functions and 4 result types organized into thematic modules.

## Modules

| Module | Description |
|--------|-------------|
| [Shannon Measures](shannon.md) | Entropy, cross entropy, KL divergence, Jensen-Shannon, mutual information, joint and conditional entropy — both discrete and continuous |
| [f-Divergences](f_divergences.md) | General f-divergence engine plus TV, Hellinger, chi-squared, Jeffreys, and Cressie-Read |
| [Rényi Family](renyi.md) | Rényi entropy and divergence parameterized by order alpha |
| [Integral Probability Metrics](ipms.md) | Energy distance, Wasserstein, MMD, sliced Wasserstein |
| [kNN Estimators](knn.md) | Kozachenko-Leonenko entropy, KSG mutual information, kNN KL divergence |
| [Multivariate Dependence](multivariate.md) | Total correlation, normalized MI, variation of information |
| [Causal / Temporal](causal.md) | Transfer entropy for directed information flow |
| [Score-Based Measures](score_based.md) | Fisher divergence and kernel Stein discrepancy (RBF + IMQ) |
| [Sinkhorn Divergence](sinkhorn.md) | Debiased entropy-regularized optimal transport |
| [Two-Sample Testing](testing.md) | Permutation tests with MMD, energy, and kNN statistics |
| [Bayesian Diagnostics](bayesian.md) | ArviZ integration for MCMC convergence and inference diagnostics |
| [Result Types](types.md) | Named tuples returned by testing and diagnostic functions |

## Common Parameters

Most functions accept these parameters:

- **`base`** — Logarithm base controlling the unit of measurement: `np.e` (nats, default), `2` (bits), `10` (hartleys)
- **`discrete`** — If `True`, use discrete estimators; if `False` (default), use continuous estimators
- **`k`** — Number of nearest neighbors for kNN-based methods (default: 5)
