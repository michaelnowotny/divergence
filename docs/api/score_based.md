# Score-Based Measures

Divergences that use score functions (gradients of log-densities) rather than densities themselves. These have the key advantage of not requiring the normalizing constant.

!!! note "Large-scale performance"
    For n >= 5,000 samples, `kernel_stein_discrepancy` automatically uses
    a Numba JIT-compiled kernel that runs 14-18x faster and uses O(n)
    memory (for pre-evaluated scores) instead of O(n^2). This makes KSD
    feasible at n=50,000+.

::: divergence.fisher_divergence

::: divergence.kernel_stein_discrepancy
