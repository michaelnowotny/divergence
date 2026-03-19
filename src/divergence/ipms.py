"""Integral Probability Metrics (IPMs) for comparing probability distributions.

This module implements sample-based estimators for several integral probability
metrics: energy distance, Wasserstein distance, maximum mean discrepancy (MMD),
and sliced Wasserstein distance. Unlike information-theoretic divergences, these
metrics operate directly on sample point clouds and do not require density
estimation or logarithm base selection.

References
----------
.. [1] A. Muller, "Integral probability metrics and their generating classes of
   functions," Advances in Applied Probability, 29(2), 429-443, 1997.
.. [2] B. K. Sriperumbudur, K. Fukumizu, A. Gretton, B. Scholkopf, G. R. G.
   Lanckriet, "On the empirical estimation of integral probability metrics,"
   Electronic Journal of Statistics, 6, 1550-1599, 2012.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.stats import wasserstein_distance as _scipy_wasserstein

# Threshold above which JIT kernels are used (avoids compilation overhead
# for small problems while preventing O(n^2) memory allocation for large ones)
_JIT_THRESHOLD = 5000


def _ensure_2d(x: np.ndarray) -> np.ndarray:
    """Reshape 1D array to column vector for consistent pairwise distance computation."""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return x.reshape(-1, 1)
    return x


def energy_distance(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
) -> float:
    r"""Compute the energy distance between two sample sets.

    Parameters
    ----------
    samples_p : np.ndarray
        Samples from distribution P. Shape ``(n,)`` for 1D or ``(n, d)`` for
        d-dimensional data.
    samples_q : np.ndarray
        Samples from distribution Q. Shape ``(m,)`` for 1D or ``(m, d)`` for
        d-dimensional data.

    Returns
    -------
    float
        The energy distance, a non-negative value.

    Notes
    -----
    The energy distance is defined as

    .. math::

        E(P, Q) = 2\,\mathbb{E}\|X - Y\|
                   - \mathbb{E}\|X - X'\|
                   - \mathbb{E}\|Y - Y'\|

    where :math:`X, X' \sim P` and :math:`Y, Y' \sim Q` are independent.

    Properties:

    * Non-negative: :math:`E(P, Q) \geq 0`.
    * Symmetric: :math:`E(P, Q) = E(Q, P)`.
    * :math:`E(P, Q) = 0` if and only if :math:`P = Q`.

    For univariate data, Euclidean distances reduce to absolute differences.

    Examples
    --------
    >>> import numpy as np
    >>> from divergence.ipms import energy_distance
    >>> rng = np.random.default_rng(42)
    >>> p = rng.normal(0, 1, 500)
    >>> q = rng.normal(1, 1, 500)
    >>> energy_distance(p, q) > 0
    True

    References
    ----------
    .. [1] G. J. Szekely and M. L. Rizzo, "Testing for equal distributions in
       high dimension," InterStat, 2004.
    .. [2] G. J. Szekely and M. L. Rizzo, "Energy statistics: A class of
       statistics based on distances," Journal of Statistical Planning and
       Inference, 2013.
    """
    x = _ensure_2d(samples_p)
    y = _ensure_2d(samples_q)

    if max(len(x), len(y)) >= _JIT_THRESHOLD:
        from divergence._numba_kernels import _energy_distance_jit

        return float(
            _energy_distance_jit(np.ascontiguousarray(x), np.ascontiguousarray(y))
        )

    # Vectorized path for small n (avoids JIT compilation overhead)
    n = len(x)
    m = len(y)
    d_xy = cdist(x, y, metric="euclidean")
    d_xx = cdist(x, x, metric="euclidean")
    d_yy = cdist(y, y, metric="euclidean")

    # Use U-statistic: exclude diagonal (self-pairs) from E[||X-X'||]
    mean_xy = np.sum(d_xy) / (n * m)
    mean_xx = np.sum(d_xx) / (n * (n - 1)) if n > 1 else 0.0
    mean_yy = np.sum(d_yy) / (m * (m - 1)) if m > 1 else 0.0

    return float(2.0 * mean_xy - mean_xx - mean_yy)


def wasserstein_distance(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    p: int = 1,
) -> float:
    r"""Compute the p-Wasserstein distance between two 1D sample sets.

    Parameters
    ----------
    samples_p : np.ndarray
        1D samples from distribution P. Shape ``(n,)``.
    samples_q : np.ndarray
        1D samples from distribution Q. Shape ``(m,)``.
    p : int, optional
        Order of the Wasserstein distance. Must be >= 1. Default is 1.

    Returns
    -------
    float
        The p-Wasserstein distance, a non-negative value.

    Notes
    -----
    For ``p=1``, this delegates to :func:`scipy.stats.wasserstein_distance`.

    For ``p >= 2`` the computation proceeds as follows:

    * Sort both sample arrays.
    * If the sample sizes are equal, pair sorted elements directly:

      .. math::

          W_p = \left(\frac{1}{n}\sum_{i=1}^{n}
                |x_{(i)} - y_{(i)}|^p\right)^{1/p}

    * If the sample sizes differ, interpolate both empirical quantile
      functions onto a common grid of quantile levels and integrate:

      .. math::

          W_p = \left(\int_0^1 |F_P^{-1}(t) - F_Q^{-1}(t)|^p\,dt
                \right)^{1/p}

    Properties:

    * Non-negative: :math:`W_p \geq 0`.
    * Symmetric: :math:`W_p(P, Q) = W_p(Q, P)`.
    * Satisfies the triangle inequality (it is a true metric).

    Examples
    --------
    >>> import numpy as np
    >>> from divergence.ipms import wasserstein_distance
    >>> rng = np.random.default_rng(42)
    >>> p_samples = rng.normal(0, 1, 1000)
    >>> q_samples = rng.normal(2, 1, 1000)
    >>> wasserstein_distance(p_samples, q_samples, p=1) > 0
    True

    References
    ----------
    .. [1] C. Villani, "Optimal Transport: Old and New," Springer, 2008.
    .. [2] M. Arjovsky, S. Chintala, L. Bottou, "Wasserstein Generative
       Adversarial Networks," ICML, 2017.
    """
    samples_p = np.asarray(samples_p, dtype=float).ravel()
    samples_q = np.asarray(samples_q, dtype=float).ravel()

    if p < 1:
        raise ValueError(f"p must be >= 1, got {p}")

    if p == 1:
        return float(_scipy_wasserstein(samples_p, samples_q))

    sorted_p = np.sort(samples_p)
    sorted_q = np.sort(samples_q)

    n = len(sorted_p)
    m = len(sorted_q)

    if n == m:
        # Equal sizes: direct pairing of sorted elements
        return float(np.mean(np.abs(sorted_p - sorted_q) ** p) ** (1.0 / p))

    # Unequal sizes: interpolate to common quantile grid
    n_grid = max(n, m)
    quantiles = np.linspace(0.0, 1.0, n_grid)

    # Empirical quantile functions via interpolation
    cdf_p = np.linspace(0.0, 1.0, n)
    cdf_q = np.linspace(0.0, 1.0, m)
    qf_p = np.interp(quantiles, cdf_p, sorted_p)
    qf_q = np.interp(quantiles, cdf_q, sorted_q)

    return float(np.mean(np.abs(qf_p - qf_q) ** p) ** (1.0 / p))


def maximum_mean_discrepancy(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    kernel: str = "rbf",
    bandwidth: float | None = None,
) -> float:
    r"""Compute the squared maximum mean discrepancy (MMD\ :sup:`2`) via the
    unbiased U-statistic estimator.

    Parameters
    ----------
    samples_p : np.ndarray
        Samples from distribution P. Shape ``(n,)`` for 1D or ``(n, d)`` for
        d-dimensional data.
    samples_q : np.ndarray
        Samples from distribution Q. Shape ``(m,)`` for 1D or ``(m, d)`` for
        d-dimensional data.
    kernel : str, optional
        Kernel function. Currently only ``"rbf"`` (radial basis function /
        Gaussian kernel) is supported. Default is ``"rbf"``.
    bandwidth : float or None, optional
        Bandwidth parameter :math:`\sigma` for the RBF kernel. If ``None``,
        the median heuristic is used: :math:`\sigma` is set to the median of
        all pairwise Euclidean distances in the pooled sample. Default is
        ``None``.

    Returns
    -------
    float
        The squared MMD (MMD\ :sup:`2`). The unbiased U-statistic estimator
        can produce slightly negative values.

    Notes
    -----
    The squared MMD with RBF kernel :math:`k(x, y) = \exp(-\|x-y\|^2 /
    (2\sigma^2))` is estimated via the unbiased U-statistic:

    .. math::

        \widehat{\mathrm{MMD}}^2_u =
          \frac{1}{m(m-1)}\sum_{i \neq j} k(x_i, x_j)
        - \frac{2}{mn}\sum_{i,j} k(x_i, y_j)
        + \frac{1}{n(n-1)}\sum_{i \neq j} k(y_i, y_j)

    The median heuristic sets :math:`\sigma` to the median of all pairwise
    distances in :math:`\{x_1, \ldots, x_m\} \cup \{y_1, \ldots, y_n\}`.

    Properties:

    * :math:`\mathrm{MMD}^2(P, Q) \geq 0` in expectation for characteristic
      kernels (the unbiased estimator can be slightly negative).
    * Symmetric: :math:`\mathrm{MMD}(P, Q) = \mathrm{MMD}(Q, P)`.

    Examples
    --------
    >>> import numpy as np
    >>> from divergence.ipms import maximum_mean_discrepancy
    >>> rng = np.random.default_rng(42)
    >>> p = rng.normal(0, 1, 500)
    >>> q = rng.normal(1, 1, 500)
    >>> maximum_mean_discrepancy(p, q) > 0
    True

    References
    ----------
    .. [1] A. Gretton, K. M. Borgwardt, M. J. Rasch, B. Scholkopf, A. Smola,
       "A kernel two-sample test," JMLR, 2012.
    """
    if kernel != "rbf":
        raise ValueError(f"Only 'rbf' kernel is supported, got '{kernel}'")

    x = _ensure_2d(samples_p)
    y = _ensure_2d(samples_q)

    m = x.shape[0]
    n = y.shape[0]

    if m < 2 or n < 2:
        raise ValueError(
            f"Need at least 2 samples from each distribution, got m={m}, n={n}"
        )

    # Bandwidth: median heuristic if not provided
    if bandwidth is None:
        if max(m, n) >= _JIT_THRESHOLD:
            from divergence._numba_kernels import _median_bandwidth_jit

            pooled = np.ascontiguousarray(np.concatenate([x, y], axis=0))
            bandwidth = float(_median_bandwidth_jit(pooled))
        else:
            pooled = np.concatenate([x, y], axis=0)
            d_pooled = cdist(pooled, pooled, metric="euclidean")
            triu_indices = np.triu_indices_from(d_pooled, k=1)
            bandwidth = float(np.median(d_pooled[triu_indices]))
        if bandwidth == 0.0:
            bandwidth = 1.0

    gamma = 1.0 / (2.0 * bandwidth**2)

    if max(m, n) >= _JIT_THRESHOLD:
        from divergence._numba_kernels import _mmd_squared_jit

        return float(
            _mmd_squared_jit(np.ascontiguousarray(x), np.ascontiguousarray(y), gamma)
        )

    # Vectorized path for small n
    d_xx_sq = cdist(x, x, metric="sqeuclidean")
    d_yy_sq = cdist(y, y, metric="sqeuclidean")
    d_xy_sq = cdist(x, y, metric="sqeuclidean")

    k_xx = np.exp(-gamma * d_xx_sq)
    k_yy = np.exp(-gamma * d_yy_sq)
    k_xy = np.exp(-gamma * d_xy_sq)

    np.fill_diagonal(k_xx, 0.0)
    np.fill_diagonal(k_yy, 0.0)

    term_xx = np.sum(k_xx) / (m * (m - 1))
    term_yy = np.sum(k_yy) / (n * (n - 1))
    term_xy = np.sum(k_xy) / (m * n)

    return float(term_xx - 2.0 * term_xy + term_yy)


def sliced_wasserstein_distance(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    n_projections: int = 100,
    p: int = 2,
    seed: int | None = None,
) -> float:
    r"""Compute the sliced Wasserstein distance between two sample sets.

    Parameters
    ----------
    samples_p : np.ndarray
        Samples from distribution P. Shape ``(n,)`` for 1D or ``(n, d)`` for
        d-dimensional data.
    samples_q : np.ndarray
        Samples from distribution Q. Shape ``(m,)`` for 1D or ``(m, d)`` for
        d-dimensional data.
    n_projections : int, optional
        Number of random projections to average over. Ignored for 1D inputs.
        Default is 100.
    p : int, optional
        Order of the Wasserstein distance for each 1D projection. Default is 2.
    seed : int or None, optional
        Random seed for reproducible projection directions. Default is ``None``.

    Returns
    -------
    float
        The sliced Wasserstein distance, a non-negative value.

    Notes
    -----
    For d-dimensional samples, the sliced Wasserstein distance is defined as

    .. math::

        SW_p(P, Q) = \left(\mathbb{E}_{\theta \sim \mathrm{Uniform}(S^{d-1})}
                      \left[W_p^p(\theta^\top_\# P,\,
                      \theta^\top_\# Q)\right]\right)^{1/p}

    where :math:`\theta^\top_\# P` denotes the pushforward (projection) of
    :math:`P` onto direction :math:`\theta`, and the expectation is
    approximated by averaging over ``n_projections`` random unit vectors.

    For 1D input, the sliced Wasserstein distance reduces to the ordinary
    Wasserstein distance and projections are skipped.

    Properties:

    * Non-negative: :math:`SW_p \geq 0`.
    * Symmetric: :math:`SW_p(P, Q) = SW_p(Q, P)`.
    * :math:`SW_p(P, Q) = 0` if and only if :math:`P = Q`.

    Examples
    --------
    >>> import numpy as np
    >>> from divergence.ipms import sliced_wasserstein_distance
    >>> rng = np.random.default_rng(42)
    >>> p_samples = rng.normal(0, 1, (500, 3))
    >>> q_samples = rng.normal(1, 1, (500, 3))
    >>> sliced_wasserstein_distance(p_samples, q_samples, seed=0) > 0
    True

    References
    ----------
    .. [1] M. Rabin, J. Delon, Y. Gousseau, "Wasserstein Barycenter and Its
       Application to Texture Mixing," SSVM, 2011.
    .. [2] S. Kolouri, K. Nadjahi, U. Simsekli, R. Badeau, G. Rohde,
       "Generalized Sliced Wasserstein Distances," NeurIPS, 2019.
    """
    samples_p = np.asarray(samples_p, dtype=float)
    samples_q = np.asarray(samples_q, dtype=float)

    # For 1D input, just compute regular Wasserstein distance
    if samples_p.ndim == 1 and samples_q.ndim == 1:
        return wasserstein_distance(samples_p, samples_q, p=p)

    x = _ensure_2d(samples_p)
    y = _ensure_2d(samples_q)

    d = x.shape[1]
    if y.shape[1] != d:
        raise ValueError(
            f"Dimension mismatch: samples_p has {d} features, "
            f"samples_q has {y.shape[1]}"
        )

    rng = np.random.default_rng(seed)

    # Draw random directions on the unit sphere
    directions = rng.standard_normal((n_projections, d))
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)

    # Compute 1D Wasserstein distances for each projection
    wp_values = np.empty(n_projections)
    for i in range(n_projections):
        proj_p = x @ directions[i]
        proj_q = y @ directions[i]
        w = wasserstein_distance(proj_p, proj_q, p=p)
        wp_values[i] = w**p

    return float(np.mean(wp_values) ** (1.0 / p))
