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
