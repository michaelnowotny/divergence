# Integral Probability Metrics

Sample-based distance measures that do not require density estimation. These operate directly on point clouds and are true metrics (or pseudo-metrics) on the space of probability distributions.

!!! note "Performance"
    `energy_distance` and `maximum_mean_discrepancy` dispatch to Numba
    JIT kernels above small thresholds. For 1D inputs, `energy_distance`
    uses a sort-based O(n log n) closed form once n ≥ 200; the multi-D
    streaming kernel takes over at n ≥ 5,000. `maximum_mean_discrepancy`
    switches to the JIT path at n ≥ 500 — much earlier than energy
    distance because the vectorized fallback materializes three full
    kernel matrices.

    `two_sample_test(method="mmd")` precomputes the RBF kernel matrix
    once outside the permutation loop and uses the symmetry identity
    `S_PQ = (K_total - K_PP - K_QQ) / 2` to avoid one block sum per
    permutation. The 1D energy permutation test calls the sort-based
    kernel under each permutation and runs n=3000 per group with 200
    permutations in well under a second.

::: divergence.energy_distance

::: divergence.wasserstein_distance

::: divergence.maximum_mean_discrepancy

::: divergence.sliced_wasserstein_distance
