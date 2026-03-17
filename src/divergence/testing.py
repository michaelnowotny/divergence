"""Two-sample hypothesis testing via permutation tests.

Provides a unified interface for testing the null hypothesis H0: P = Q
against the alternative H1: P != Q using various divergence and distance
measures as test statistics.

The permutation test is exact under H0: since the labels (P vs Q) are
exchangeable, we can construct the null distribution by randomly shuffling
the combined samples and recomputing the test statistic.

References
----------
.. [1] Gretton, A. et al. (2012). "A Kernel Two-Sample Test."
       JMLR, 13, 723-773.
.. [2] Szekely, G. J. & Rizzo, M. L. (2004). "Testing for Equal Distributions
       in High Dimension." InterStat, 5.
"""

import typing as tp

import numpy as np

from divergence._types import TestResult


def _permutation_test(
    combined: np.ndarray,
    n_p: int,
    statistic_fn: tp.Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build null distribution via permutation.

    Parameters
    ----------
    combined : np.ndarray
        Pooled samples, shape (n_p + n_q, ...).
    n_p : int
        Number of samples from P.
    statistic_fn : callable
        Function(samples_p, samples_q) -> float.
    n_permutations : int
        Number of permutations.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of null statistics, shape (n_permutations,).
    """
    null_stats = np.empty(n_permutations)
    n_total = len(combined)
    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        null_stats[i] = statistic_fn(combined[perm[:n_p]], combined[perm[n_p:]])
    return null_stats


def two_sample_test(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    method: str = "mmd",
    n_permutations: int = 1000,
    seed: int | None = None,
    **kwargs: tp.Any,
) -> TestResult:
    r"""Two-sample hypothesis test via permutation.

    Tests H0: P = Q against H1: P != Q by computing a test statistic and
    comparing it to a null distribution obtained by permuting the combined
    samples.

    Parameters
    ----------
    samples_p : np.ndarray
        Samples from distribution P.
    samples_q : np.ndarray
        Samples from distribution Q.
    method : str
        Test statistic to use:

        - ``"mmd"``: Maximum Mean Discrepancy (default). Good general-purpose
          choice with strong theoretical properties.
        - ``"energy"``: Energy distance. Works well in arbitrary dimensions
          without kernel bandwidth selection.
        - ``"kl_knn"``: kNN KL divergence estimator. Sensitive to density
          ratio differences.
    n_permutations : int
        Number of permutations for the null distribution (default 1000).
        Higher values give more precise p-values but take longer.
    seed : int or None
        Random seed for reproducibility.
    **kwargs
        Additional arguments passed to the test statistic function.
        For ``"mmd"``: ``kernel``, ``bandwidth``.
        For ``"kl_knn"``: ``k``.

    Returns
    -------
    TestResult
        Named tuple with fields:

        - ``statistic``: float — the observed test statistic
        - ``p_value``: float — permutation p-value
        - ``null_distribution``: np.ndarray — null statistics from permutations

    Notes
    -----
    The p-value is computed as:

        p = (1 + #{b : T_b >= T_obs}) / (1 + B)

    where T_obs is the observed statistic, T_b are the null statistics,
    and B is the number of permutations. The +1 in numerator and denominator
    ensures the p-value is never exactly 0 and accounts for the observed
    statistic itself.

    The permutation test is:

    - **Exact** under H0 (finite-sample valid)
    - **Non-parametric** (no distributional assumptions)
    - **Consistent** against all alternatives (for MMD with characteristic kernel)

    Examples
    --------
    >>> import numpy as np
    >>> from divergence import two_sample_test
    >>> rng = np.random.default_rng(42)
    >>> p = rng.normal(0, 1, 200)
    >>> q = rng.normal(1, 1, 200)
    >>> result = two_sample_test(p, q, method="energy", n_permutations=500, seed=42)
    >>> result.p_value < 0.05
    True

    References
    ----------
    .. [1] Gretton, A. et al. (2012). "A Kernel Two-Sample Test."
           JMLR, 13, 723-773.
    .. [2] Szekely, G. J. & Rizzo, M. L. (2004). "Testing for Equal
           Distributions in High Dimension." InterStat, 5.
    """
    rng = np.random.default_rng(seed)

    # Select test statistic function
    if method == "mmd":
        from divergence.ipms import maximum_mean_discrepancy

        def stat_fn(p: np.ndarray, q: np.ndarray) -> float:
            return maximum_mean_discrepancy(p, q, **kwargs)

    elif method == "energy":
        from divergence.ipms import energy_distance

        def stat_fn(p: np.ndarray, q: np.ndarray) -> float:
            return energy_distance(p, q)

    elif method == "kl_knn":
        from divergence.knn import knn_kl_divergence

        def stat_fn(p: np.ndarray, q: np.ndarray) -> float:
            return knn_kl_divergence(p, q, **kwargs)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: 'mmd', 'energy', 'kl_knn'."
        )

    # Compute observed statistic
    observed = stat_fn(samples_p, samples_q)

    # Build null distribution via permutation
    combined = np.concatenate([samples_p, samples_q], axis=0)
    null_dist = _permutation_test(
        combined, len(samples_p), stat_fn, n_permutations, rng
    )

    # Compute p-value (with continuity correction)
    p_value = (1 + np.sum(null_dist >= observed)) / (1 + n_permutations)

    return TestResult(
        statistic=float(observed),
        p_value=float(p_value),
        null_distribution=null_dist,
    )
