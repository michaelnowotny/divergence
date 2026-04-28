# Score-Based Measures

Divergences that use score functions (gradients of log-densities) rather than densities themselves. These have the key advantage of not requiring the normalizing constant.

!!! note "Performance"
    `kernel_stein_discrepancy` dispatches to a Numba JIT kernel at
    n ≥ 500. The vectorized path materializes (n, n, d) diff arrays
    plus several (n, n) intermediates; the JIT path streams with
    O(n) memory (for the pre-evaluated scores). At n=3000 the JIT
    path is roughly 17× faster.

    `fisher_divergence` with an estimated `score_p` uses a kernel
    density gradient that is now expressed via a linearity identity
    rather than an (m, n, d) intermediate, and the median-bandwidth
    helper used by both `fisher_divergence` and
    `kernel_stein_discrepancy` switches to a subsampling kernel at
    n ≥ 500.

::: divergence.fisher_divergence

::: divergence.kernel_stein_discrepancy
