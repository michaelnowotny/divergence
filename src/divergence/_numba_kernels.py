"""Numba JIT-compiled kernels for O(n^2) computations.

These kernels compute pairwise statistics using O(1) memory instead of
O(n^2) by accumulating running sums rather than materializing full
distance matrices.  This enables computations at n=50K+ that would
otherwise exhaust available RAM.

All functions use ``@numba.njit(parallel=True)`` with ``prange`` for
automatic multicore parallelism on the outer loop.
"""

import numba
import numpy as np


@numba.njit(parallel=True, cache=True)
def _sum_block_jit(D: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray) -> float:
    """Sum the block ``D[idx_a, idx_b]`` without materializing a submatrix.

    Equivalent to ``D[np.ix_(idx_a, idx_b)].sum()`` but avoids allocating
    an ``(len(idx_a), len(idx_b))`` intermediate.  Used inside the
    permutation-test hot loop where the same large matrix ``D`` is
    re-partitioned on every permutation — the fancy-indexing allocation
    dominates the per-permutation cost without this kernel.

    Parameters
    ----------
    D : np.ndarray
        Square or rectangular matrix of values.
    idx_a : np.ndarray
        1D integer array of row indices.
    idx_b : np.ndarray
        1D integer array of column indices.

    Returns
    -------
    float
        Sum of ``D[a, b]`` for ``a in idx_a`` and ``b in idx_b``.
    """
    n_a = idx_a.shape[0]
    n_b = idx_b.shape[0]
    total = 0.0
    for i in numba.prange(n_a):
        ia = idx_a[i]
        row_sum = 0.0
        for j in range(n_b):
            row_sum += D[ia, idx_b[j]]
        total += row_sum
    return total


@numba.njit(parallel=True, cache=True)
def _sum_block_rbf_jit(
    D_sq: np.ndarray, idx_a: np.ndarray, idx_b: np.ndarray, gamma: float
) -> float:
    """Sum the RBF kernel ``exp(-gamma * D_sq[idx_a, idx_b])`` over a block.

    Like ``_sum_block_jit`` but applies the RBF nonlinearity elementwise
    before summing.  Used by the MMD permutation test.  Avoids allocating
    both the submatrix and the exponentiated copy.

    Parameters
    ----------
    D_sq : np.ndarray
        Squared-Euclidean distance matrix.
    idx_a : np.ndarray
        Row indices.
    idx_b : np.ndarray
        Column indices.
    gamma : float
        RBF kernel parameter.

    Returns
    -------
    float
        Sum of ``exp(-gamma * D_sq[a, b])`` over the block.
    """
    n_a = idx_a.shape[0]
    n_b = idx_b.shape[0]
    total = 0.0
    for i in numba.prange(n_a):
        ia = idx_a[i]
        row_sum = 0.0
        for j in range(n_b):
            row_sum += np.exp(-gamma * D_sq[ia, idx_b[j]])
        total += row_sum
    return total


@numba.njit(cache=True)
def _energy_distance_1d_jit(x: np.ndarray, y: np.ndarray) -> float:
    """Energy distance for 1D samples in O((n + m) log(n + m)) time.

    For one-dimensional data, the sums of pairwise absolute differences
    that appear in the energy-distance U-statistic admit closed-form
    expressions in terms of the sorted order statistics.  These
    expressions turn the problem from O((n + m)^2) into O((n + m) log
    (n + m)) — roughly two orders of magnitude faster at n = 3000.

    Parameters
    ----------
    x : np.ndarray
        Samples from P, 1D array of length ``n``.
    y : np.ndarray
        Samples from Q, 1D array of length ``m``.

    Returns
    -------
    float
        Energy distance (U-statistic estimator), identical to
        ``_energy_distance_jit`` applied to the same data reshaped
        as ``(n, 1)`` / ``(m, 1)``.

    Notes
    -----
    Within-sample term: for sorted ``x_(0) <= ... <= x_(n-1)``,

        sum_{i < j} (x_(j) - x_(i)) = sum_k x_(k) * (2k - n + 1)

    where ``k`` is 0-indexed.  The same holds for y.

    Cross-sample term: for each ``x_i``, let ``k_i`` be the number of
    ``y_j`` with ``y_j <= x_i``.  Then

        sum_j |x_i - y_j| = (2 k_i - m) * x_i + S_y - 2 * S_y(<= x_i)

    where ``S_y`` is the total sum of y and ``S_y(<= x_i)`` is the
    prefix sum up to ``y_j <= x_i``.
    """
    n = x.shape[0]
    m = y.shape[0]
    if n < 2 or m < 2:
        return 0.0

    xs = np.sort(x.astype(np.float64))
    ys = np.sort(y.astype(np.float64))

    # Within-sample sum for x: sum_{i<j} |x_i - x_j|
    sum_xx_lt = 0.0
    for k in range(n):
        sum_xx_lt += xs[k] * (2 * k - n + 1)

    # Within-sample sum for y
    sum_yy_lt = 0.0
    for k in range(m):
        sum_yy_lt += ys[k] * (2 * k - m + 1)

    # Prefix sums of sorted y (length m + 1; y_prefix[k] = sum of ys[0:k])
    y_prefix = np.empty(m + 1, dtype=np.float64)
    y_prefix[0] = 0.0
    for k in range(m):
        y_prefix[k + 1] = y_prefix[k] + ys[k]
    y_total = y_prefix[m]

    # Cross-sample sum: sum_{i,j} |x_i - y_j|
    sum_xy = 0.0
    for i in range(n):
        xi = xs[i]
        # Binary search for the first index where ys[mid] > xi (i.e. count
        # of ys at or below xi).  This is equivalent to np.searchsorted
        # with side='right'.
        lo = 0
        hi = m
        while lo < hi:
            mid = (lo + hi) // 2
            if ys[mid] <= xi:
                lo = mid + 1
            else:
                hi = mid
        k = lo
        s_below = y_prefix[k]
        sum_xy += (2 * k - m) * xi + y_total - 2.0 * s_below

    # Assemble the U-statistic.  The within-sample sums are over i < j
    # (half the i != j sum), which cancels the factor of 2 in
    # sum_{i != j} = 2 * sum_{i < j}, giving mean = 2 * sum_lt / (n * (n - 1)).
    mean_xy = sum_xy / (n * m)
    mean_xx = 2.0 * sum_xx_lt / (n * (n - 1))
    mean_yy = 2.0 * sum_yy_lt / (m * (m - 1))

    return 2.0 * mean_xy - mean_xx - mean_yy


@numba.njit(parallel=True)
def _energy_distance_jit(x: np.ndarray, y: np.ndarray) -> float:
    """Compute energy distance with O(1) memory.

    Parameters
    ----------
    x : np.ndarray
        Samples from P, shape ``(n, d)``.
    y : np.ndarray
        Samples from Q, shape ``(m, d)``.

    Returns
    -------
    float
        Energy distance.
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    # E[||X - Y||]
    sum_xy = 0.0
    for i in numba.prange(n):
        s = 0.0
        for j in range(m):
            dist_sq = 0.0
            for k in range(d):
                diff = x[i, k] - y[j, k]
                dist_sq += diff * diff
            s += np.sqrt(dist_sq)
        sum_xy += s

    # E[||X - X'||]
    sum_xx = 0.0
    for i in numba.prange(n):
        s = 0.0
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(d):
                diff = x[i, k] - x[j, k]
                dist_sq += diff * diff
            s += np.sqrt(dist_sq)
        sum_xx += s

    # E[||Y - Y'||]
    sum_yy = 0.0
    for i in numba.prange(m):
        s = 0.0
        for j in range(i + 1, m):
            dist_sq = 0.0
            for k in range(d):
                diff = y[i, k] - y[j, k]
                dist_sq += diff * diff
            s += np.sqrt(dist_sq)
        sum_yy += s

    mean_xy = sum_xy / (n * m)
    mean_xx = 2.0 * sum_xx / (n * (n - 1)) if n > 1 else 0.0
    mean_yy = 2.0 * sum_yy / (m * (m - 1)) if m > 1 else 0.0

    return 2.0 * mean_xy - mean_xx - mean_yy


@numba.njit(parallel=True)
def _mmd_squared_jit(x: np.ndarray, y: np.ndarray, gamma: float) -> float:
    """Compute MMD^2 (U-statistic) with RBF kernel using O(1) memory.

    Parameters
    ----------
    x : np.ndarray
        Samples from P, shape ``(n, d)``.
    y : np.ndarray
        Samples from Q, shape ``(m, d)``.
    gamma : float
        RBF kernel parameter: ``1 / (2 * bandwidth^2)``.

    Returns
    -------
    float
        Squared MMD (U-statistic).
    """
    n = x.shape[0]
    m = y.shape[0]
    d = x.shape[1]

    # k(x_i, x_j) for i != j
    sum_xx = 0.0
    for i in numba.prange(n):
        s = 0.0
        for j in range(i + 1, n):
            dist_sq = 0.0
            for k in range(d):
                diff = x[i, k] - x[j, k]
                dist_sq += diff * diff
            s += np.exp(-gamma * dist_sq)
        sum_xx += s

    # k(y_i, y_j) for i != j
    sum_yy = 0.0
    for i in numba.prange(m):
        s = 0.0
        for j in range(i + 1, m):
            dist_sq = 0.0
            for k in range(d):
                diff = y[i, k] - y[j, k]
                dist_sq += diff * diff
            s += np.exp(-gamma * dist_sq)
        sum_yy += s

    # k(x_i, y_j)
    sum_xy = 0.0
    for i in numba.prange(n):
        s = 0.0
        for j in range(m):
            dist_sq = 0.0
            for k in range(d):
                diff = x[i, k] - y[j, k]
                dist_sq += diff * diff
            s += np.exp(-gamma * dist_sq)
        sum_xy += s

    term_xx = 2.0 * sum_xx / (n * (n - 1))
    term_yy = 2.0 * sum_yy / (m * (m - 1))
    term_xy = sum_xy / (n * m)

    return term_xx - 2.0 * term_xy + term_yy


@numba.njit(parallel=True)
def _median_bandwidth_jit(x: np.ndarray) -> float:
    """Compute median heuristic bandwidth with O(1) memory.

    Parameters
    ----------
    x : np.ndarray
        Pooled samples, shape ``(N, d)``.

    Returns
    -------
    float
        Median of pairwise Euclidean distances.
    """
    n = x.shape[0]
    d = x.shape[1]
    n_pairs = n * (n - 1) // 2

    # Collect all pairwise distances — this is O(n^2) memory for the
    # distances array, but each element is a scalar float, not a row.
    # For n=50K this is 50K*49999/2 ≈ 1.25 billion pairs — too many.
    # Instead, use a reservoir sampling approach or approximate.
    # For practical purposes, subsample pairs.
    max_pairs = min(n_pairs, 500_000)

    if n_pairs <= max_pairs:
        # Small enough to compute all
        dists = np.empty(n_pairs)
        idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = 0.0
                for k in range(d):
                    diff = x[i, k] - x[j, k]
                    dist_sq += diff * diff
                dists[idx] = np.sqrt(dist_sq)
                idx += 1
        return np.median(dists)
    else:
        # Subsample pairs deterministically
        step = max(1, n_pairs // max_pairs)
        dists = np.empty(max_pairs)
        count = 0
        pair_idx = 0
        for i in range(n):
            for j in range(i + 1, n):
                if pair_idx % step == 0 and count < max_pairs:
                    dist_sq = 0.0
                    for k in range(d):
                        diff = x[i, k] - x[j, k]
                        dist_sq += diff * diff
                    dists[count] = np.sqrt(dist_sq)
                    count += 1
                pair_idx += 1
        return np.median(dists[:count])


@numba.njit(parallel=True)
def _ksd_stein_kernel_sum_jit(
    x: np.ndarray,
    scores: np.ndarray,
    sq_bandwidth: float,
    kernel_type: int,
) -> float:
    """Compute the KSD U-statistic sum with O(1) memory.

    Parameters
    ----------
    x : np.ndarray
        Sample points, shape ``(n, d)``.
    scores : np.ndarray
        Pre-evaluated score function values, shape ``(n, d)``.
    sq_bandwidth : float
        Squared bandwidth (sigma^2 for RBF, c^2 for IMQ).
    kernel_type : int
        0 for RBF, 1 for IMQ.

    Returns
    -------
    float
        Sum of Stein kernel over all pairs i != j.
    """
    n = x.shape[0]
    d = x.shape[1]
    total = 0.0

    for i in numba.prange(n):
        local_sum = 0.0
        for j in range(i + 1, n):
            # Pairwise squared distance
            r_sq = 0.0
            for k in range(d):
                diff = x[i, k] - x[j, k]
                r_sq += diff * diff

            # s_i . s_j
            ss = 0.0
            for k in range(d):
                ss += scores[i, k] * scores[j, k]

            # s_i . (x_i - x_j) and s_j . (x_i - x_j)
            sd_i = 0.0
            sd_j = 0.0
            for k in range(d):
                diff = x[i, k] - x[j, k]
                sd_i += scores[i, k] * diff
                sd_j += scores[j, k] * diff

            if kernel_type == 0:
                # RBF kernel
                gamma = 0.5 / sq_bandwidth
                K = np.exp(-gamma * r_sq)
                t1 = ss * K
                t2 = 2.0 * gamma * sd_i * K
                t3 = -2.0 * gamma * sd_j * K
                t4 = (2.0 * gamma * d - 4.0 * gamma * gamma * r_sq) * K
            else:
                # IMQ kernel: k(x,y) = (c^2 + r^2)^beta, beta = -0.5
                beta = -0.5
                base = sq_bandwidth + r_sq
                K = base**beta
                K_bm1 = base ** (beta - 1.0)
                K_bm2 = base ** (beta - 2.0)
                t1 = ss * K
                t2 = -2.0 * beta * sd_i * K_bm1
                t3 = 2.0 * beta * sd_j * K_bm1
                t4 = -2.0 * beta * K_bm2 * (d * base + 2.0 * (beta - 1.0) * r_sq)

            local_sum += t1 + t2 + t3 + t4
        total += local_sum

    return 2.0 * total  # each pair counted once, U-stat needs both (i,j) and (j,i)
