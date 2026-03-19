# Integral Probability Metrics

Sample-based distance measures that do not require density estimation. These operate directly on point clouds and are true metrics (or pseudo-metrics) on the space of probability distributions.

!!! note "Large-scale performance"
    For n >= 5,000 samples, `energy_distance` and `maximum_mean_discrepancy`
    automatically use Numba JIT-compiled kernels that run 10-18x faster and
    use O(1) memory instead of O(n^2). This makes computations feasible at
    n=50,000+ (which would previously exhaust memory).

::: divergence.energy_distance

::: divergence.wasserstein_distance

::: divergence.maximum_mean_discrepancy

::: divergence.sliced_wasserstein_distance
