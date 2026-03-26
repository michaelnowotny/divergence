"""Two-sample hypothesis testing via permutation tests.

Provides a unified interface for testing the null hypothesis H0: P = Q
against the alternative H1: P != Q using various divergence and distance
measures as test statistics.

The permutation test is exact under H0: since the labels (P vs Q) are
exchangeable, we can construct the null distribution by randomly shuffling
the combined samples and recomputing the test statistic.

For the ``"energy"`` and ``"mmd"`` methods, two strategies are available:

- **Precomputed matrix** (default for small N): Precompute the full NxN
  distance matrix once and reindex for each permutation.  Fast but
  requires O(N²) memory.
- **Low-memory** (automatic for large N, or ``low_memory=True``):
  Recompute the test statistic from scratch each permutation using
  Numba JIT kernels with O(N) memory.  Slower per permutation but
  enables N > 50K without exhausting RAM.

References
----------
.. [1] Gretton, A. et al. (2012). "A Kernel Two-Sample Test."
       JMLR, 13, 723-773.
.. [2] Szekely, G. J. & Rizzo, M. L. (2004). "Testing for Equal Distributions
       in High Dimension." InterStat, 5.
"""

import typing as tp

import numpy as np
from scipy.spatial.distance import cdist

from divergence._types import TestResult

# Distance matrix size (bytes) above which we auto-switch to low-memory mode.
# Default: 1 GiB.  A 50K x 50K float64 matrix is ~20 GiB, so the threshold
# kicks in well before that.
_LOW_MEMORY_BYTES_THRESHOLD = 1 * 1024**3


# ---------------------------------------------------------------------------
# Fast permutation helpers using precomputed distance matrices
# ---------------------------------------------------------------------------
def _energy_from_distance_matrix(
    D: np.ndarray, idx_p: np.ndarray, idx_q: np.ndarray
) -> float:
    """Compute energy distance from a precomputed Euclidean distance matrix.

    Uses the U-statistic estimator (excludes self-distances on the
    diagonal), consistent with ``_energy_distance_jit``.

    Parameters
    ----------
    D : np.ndarray
        Full pairwise Euclidean distance matrix, shape ``(N, N)``.
    idx_p : np.ndarray
        Indices of the P-samples in the combined array.
    idx_q : np.ndarray
        Indices of the Q-samples in the combined array.

    Returns
    -------
    float
        Energy distance (U-statistic).
    """
    n_p = len(idx_p)
    n_q = len(idx_q)

    d_pq = D[np.ix_(idx_p, idx_q)]
    mean_pq = np.sum(d_pq) / (n_p * n_q)

    # Within-group: exclude diagonal (self-distances = 0) for U-statistic
    d_pp = D[np.ix_(idx_p, idx_p)]
    mean_pp = np.sum(d_pp) / (n_p * (n_p - 1)) if n_p > 1 else 0.0

    d_qq = D[np.ix_(idx_q, idx_q)]
    mean_qq = np.sum(d_qq) / (n_q * (n_q - 1)) if n_q > 1 else 0.0

    return float(2.0 * mean_pq - mean_pp - mean_qq)


def _mmd_from_sq_distance_matrix(
    D_sq: np.ndarray,
    idx_p: np.ndarray,
    idx_q: np.ndarray,
    gamma: float,
) -> float:
    """Compute MMD² from a precomputed squared Euclidean distance matrix.

    Parameters
    ----------
    D_sq : np.ndarray
        Full pairwise squared Euclidean distance matrix, shape ``(N, N)``.
    idx_p : np.ndarray
        Indices of the P-samples.
    idx_q : np.ndarray
        Indices of the Q-samples.
    gamma : float
        RBF kernel parameter: ``1 / (2 * bandwidth²)``.

    Returns
    -------
    float
        Squared MMD (U-statistic).
    """
    m = len(idx_p)
    n = len(idx_q)

    k_pp = np.exp(-gamma * D_sq[np.ix_(idx_p, idx_p)])
    k_qq = np.exp(-gamma * D_sq[np.ix_(idx_q, idx_q)])
    k_pq = np.exp(-gamma * D_sq[np.ix_(idx_p, idx_q)])

    np.fill_diagonal(k_pp, 0.0)
    np.fill_diagonal(k_qq, 0.0)

    term_pp = np.sum(k_pp) / (m * (m - 1))
    term_qq = np.sum(k_qq) / (n * (n - 1))
    term_pq = np.sum(k_pq) / (m * n)

    return float(term_pp - 2.0 * term_pq + term_qq)


def _fast_permutation_test(
    n_total: int,
    n_p: int,
    stat_from_indices: tp.Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build null distribution by permuting index arrays.

    Instead of recomputing distances each iteration, the caller provides
    a function that computes the statistic from index arrays into a
    precomputed distance matrix.

    Parameters
    ----------
    n_total : int
        Total number of combined samples.
    n_p : int
        Number of P-samples.
    stat_from_indices : callable
        Function ``(idx_p, idx_q) -> float`` that computes the test
        statistic from index arrays.
    n_permutations : int
        Number of permutations.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of null statistics, shape ``(n_permutations,)``.
    """
    null_stats = np.empty(n_permutations)
    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        null_stats[i] = stat_from_indices(perm[:n_p], perm[n_p:])
    return null_stats


# ---------------------------------------------------------------------------
# Generic (slow) permutation fallback
# ---------------------------------------------------------------------------
def _permutation_test(
    combined: np.ndarray,
    n_p: int,
    statistic_fn: tp.Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build null distribution via permutation (generic fallback).

    Recomputes the statistic from scratch on each permutation. Used for
    methods (like ``"kl_knn"``) where precomputed distance matrices do
    not apply.

    Parameters
    ----------
    combined : np.ndarray
        Pooled samples, shape ``(n_p + n_q, ...)``.
    n_p : int
        Number of samples from P.
    statistic_fn : callable
        ``Function(samples_p, samples_q) -> float``.
    n_permutations : int
        Number of permutations.
    rng : np.random.Generator
        Random number generator.

    Returns
    -------
    np.ndarray
        Array of null statistics, shape ``(n_permutations,)``.
    """
    null_stats = np.empty(n_permutations)
    n_total = len(combined)
    for i in range(n_permutations):
        perm = rng.permutation(n_total)
        null_stats[i] = statistic_fn(combined[perm[:n_p]], combined[perm[n_p:]])
    return null_stats


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def _should_use_low_memory(n_total: int, low_memory: bool | None) -> bool:
    """Decide whether to use low-memory mode.

    Returns ``True`` if explicitly requested or if the NxN distance
    matrix would exceed ``_LOW_MEMORY_BYTES_THRESHOLD``.
    """
    if low_memory is not None:
        return low_memory
    matrix_bytes = n_total * n_total * 8  # float64
    return matrix_bytes > _LOW_MEMORY_BYTES_THRESHOLD


def two_sample_test(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    method: str = "mmd",
    n_permutations: int = 1000,
    seed: int | None = None,
    low_memory: bool | None = None,
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
    low_memory : bool or None
        Memory strategy for ``"energy"`` and ``"mmd"`` methods:

        - ``None`` (default): auto-detect. Uses low-memory mode when
          the NxN distance matrix would exceed ~1 GiB.
        - ``True``: force low-memory mode. Uses Numba JIT kernels to
          recompute the statistic from scratch each permutation with
          O(N) memory. Enables N > 50K without exhausting RAM.
        - ``False``: force precomputed matrix. Faster per permutation
          but requires O(N²) memory.
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

    samples_p = np.asarray(samples_p, dtype=float)
    samples_q = np.asarray(samples_q, dtype=float)
    if samples_p.ndim == 1:
        samples_p = samples_p.reshape(-1, 1)
    if samples_q.ndim == 1:
        samples_q = samples_q.reshape(-1, 1)

    n_p = len(samples_p)
    n_q = len(samples_q)
    combined = np.concatenate([samples_p, samples_q], axis=0)
    n_total = n_p + n_q
    idx_p_orig = np.arange(n_p)
    idx_q_orig = np.arange(n_p, n_total)

    use_low_mem = _should_use_low_memory(n_total, low_memory)

    # GPU dispatch: use GPU kernels when available and beneficial
    backend = kwargs.pop("backend", None)
    if backend != "cpu" and method in ("energy", "mmd") and use_low_mem:
        try:
            from divergence._backend import get_backend

            resolved = get_backend(backend)
        except Exception:
            resolved = "cpu"
        if resolved == "gpu" and method == "energy":
            from divergence._gpu_kernels import energy_permutation_test_gpu

            observed, null_dist = energy_permutation_test_gpu(
                combined,
                n_p,
                n_permutations,
                seed=seed if seed is not None else 42,
            )
            p_value = (1 + np.sum(null_dist >= observed)) / (1 + n_permutations)
            return TestResult(
                statistic=float(observed),
                p_value=float(p_value),
                null_distribution=null_dist,
            )

    if method == "energy":
        if use_low_mem:
            # Low-memory path: recompute energy distance from scratch
            # each permutation using Numba JIT kernels.  O(N) memory.
            from divergence._numba_kernels import _energy_distance_jit

            observed = float(
                _energy_distance_jit(
                    np.ascontiguousarray(samples_p),
                    np.ascontiguousarray(samples_q),
                )
            )

            def _energy_from_samples(ip, iq):
                return float(
                    _energy_distance_jit(
                        np.ascontiguousarray(combined[ip]),
                        np.ascontiguousarray(combined[iq]),
                    )
                )

            null_dist = _fast_permutation_test(
                n_total,
                n_p,
                _energy_from_samples,
                n_permutations,
                rng,
            )
        else:
            # Precomputed matrix path: O(N²) memory, fast reindexing.
            D = cdist(combined, combined, metric="euclidean")
            observed = _energy_from_distance_matrix(D, idx_p_orig, idx_q_orig)
            null_dist = _fast_permutation_test(
                n_total,
                n_p,
                lambda ip, iq: _energy_from_distance_matrix(D, ip, iq),
                n_permutations,
                rng,
            )

    elif method == "mmd":
        # Compute bandwidth from pooled data (needed for both paths).
        bandwidth = kwargs.get("bandwidth")
        if bandwidth is None:
            if use_low_mem:
                from divergence._numba_kernels import _median_bandwidth_jit

                bandwidth = float(_median_bandwidth_jit(np.ascontiguousarray(combined)))
            else:
                D_sq = cdist(combined, combined, metric="sqeuclidean")
                D_euc = np.sqrt(D_sq)
                triu = D_euc[np.triu_indices_from(D_euc, k=1)]
                bandwidth = float(np.median(triu))
            if bandwidth == 0.0:
                bandwidth = 1.0

        gamma = 1.0 / (2.0 * bandwidth**2)

        if use_low_mem:
            from divergence._numba_kernels import _mmd_squared_jit

            observed = float(
                _mmd_squared_jit(
                    np.ascontiguousarray(samples_p),
                    np.ascontiguousarray(samples_q),
                    gamma,
                )
            )

            def _mmd_from_samples(ip, iq):
                return float(
                    _mmd_squared_jit(
                        np.ascontiguousarray(combined[ip]),
                        np.ascontiguousarray(combined[iq]),
                        gamma,
                    )
                )

            null_dist = _fast_permutation_test(
                n_total,
                n_p,
                _mmd_from_samples,
                n_permutations,
                rng,
            )
        else:
            D_sq = cdist(combined, combined, metric="sqeuclidean")
            observed = _mmd_from_sq_distance_matrix(D_sq, idx_p_orig, idx_q_orig, gamma)
            null_dist = _fast_permutation_test(
                n_total,
                n_p,
                lambda ip, iq: _mmd_from_sq_distance_matrix(D_sq, ip, iq, gamma),
                n_permutations,
                rng,
            )

    elif method == "kl_knn":
        # kNN requires tree rebuilds each permutation — use generic fallback
        from divergence.knn import knn_kl_divergence

        def stat_fn(p: np.ndarray, q: np.ndarray) -> float:
            return knn_kl_divergence(p, q, **kwargs)

        observed = stat_fn(samples_p, samples_q)
        null_dist = _permutation_test(combined, n_p, stat_fn, n_permutations, rng)

    else:
        raise ValueError(
            f"Unknown method '{method}'. Supported methods: 'mmd', 'energy', 'kl_knn'."
        )

    # Compute p-value (with continuity correction)
    p_value = (1 + np.sum(null_dist >= observed)) / (1 + n_permutations)

    return TestResult(
        statistic=float(observed),
        p_value=float(p_value),
        null_distribution=null_dist,
    )
