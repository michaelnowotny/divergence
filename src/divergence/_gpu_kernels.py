"""GPU-accelerated kernels for pairwise computations via JAX.

Two computation strategies:

- **1D fast path (O(N log N))**: For univariate data, uses sorted arrays
  and prefix sums to compute energy distance without any pairwise loops.
  Permutation tests are fully vectorized via ``jax.vmap``, running all B
  permutations in parallel on GPU.
- **General tiled path (O(N²))**: For multivariate data, uses tiled
  reductions on GPU with O(N * tile_size) memory.

Performance (RTX 4090 Laptop, 16 GB):

    N=200K, B=500 permutations, 1D:  ~0.8 seconds
    N=200K, single energy distance:  ~1.6 ms (after JIT compile)
"""

from functools import partial

import numpy as np

# Lazy imports — JAX is optional
_jax = None
_jnp = None


def _ensure_jax():
    """Import JAX lazily and verify GPU is available."""
    global _jax, _jnp
    if _jax is None:
        import jax
        import jax.numpy as jnp

        if not any(d.platform == "gpu" for d in jax.devices()):
            raise RuntimeError("No GPU device found by JAX")
        jax.config.update("jax_enable_x64", True)
        _jax = jax
        _jnp = jnp
    return _jax, _jnp


def gpu_available() -> bool:
    """Return True if JAX can see a GPU."""
    try:
        _ensure_jax()
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# 1D fast path: O(N log N) energy distance via sorted prefix sums
# ---------------------------------------------------------------------------


def _get_energy_1d_sorted():
    """Return JIT-compiled 1D energy distance function (lazy compile)."""
    jax, jnp = _ensure_jax()

    @jax.jit
    def energy_1d_sorted(x_sorted, y_sorted, n_f, m_f):
        """Energy distance from pre-sorted 1D arrays."""
        # Cross term: sum |x_i - y_j| using searchsorted + prefix sums
        y_cumsum = jnp.cumsum(y_sorted)
        y_total = y_cumsum[-1]
        idx = jnp.searchsorted(y_sorted, x_sorted)
        left_cs = jnp.where(idx > 0, y_cumsum[jnp.clip(idx - 1, 0)], 0.0)
        left_n = idx.astype(jnp.float64)
        right_cs = y_total - left_cs
        right_n = m_f - left_n
        sum_xy = jnp.sum(left_n * x_sorted - left_cs + right_cs - right_n * x_sorted)

        # Self terms: for sorted a, sum_{i<j} |a_i-a_j| = sum_i a_i*(2i-n+1)
        ix = jnp.arange(x_sorted.shape[0], dtype=jnp.float64)
        sum_xx = jnp.sum(x_sorted * (2.0 * ix - n_f + 1.0))
        iy = jnp.arange(y_sorted.shape[0], dtype=jnp.float64)
        sum_yy = jnp.sum(y_sorted * (2.0 * iy - m_f + 1.0))

        mean_xy = sum_xy / (n_f * m_f)
        mean_xx = 2.0 * sum_xx / (n_f * (n_f - 1.0))
        mean_yy = 2.0 * sum_yy / (m_f * (m_f - 1.0))
        return 2.0 * mean_xy - mean_xx - mean_yy

    return energy_1d_sorted


# Cache the compiled function
_energy_1d_sorted_fn = None


def _get_cached_energy_fn():
    global _energy_1d_sorted_fn
    if _energy_1d_sorted_fn is None:
        _energy_1d_sorted_fn = _get_energy_1d_sorted()
    return _energy_1d_sorted_fn


# ---------------------------------------------------------------------------
# General tiled path: O(N²) for multivariate data
# ---------------------------------------------------------------------------


def _tiled_pairwise_sum(x, y, tile_size=2048):
    """Sum of ||x_i - y_j|| over all (i, j) using double tiling."""
    _, jnp = _ensure_jax()
    total = jnp.float64(0.0)
    for i in range(0, x.shape[0], tile_size):
        x_tile = x[i : i + tile_size]
        for j in range(0, y.shape[0], tile_size):
            y_tile = y[j : j + tile_size]
            diffs = x_tile[:, None, :] - y_tile[None, :, :]
            dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1))
            total = total + jnp.sum(dists)
    return total


def _tiled_pairwise_sum_self(x, tile_size=2048):
    """Sum of ||x_i - x_j|| for i < j using double tiling."""
    _, jnp = _ensure_jax()
    total = jnp.float64(0.0)
    n = x.shape[0]
    for i in range(0, n, tile_size):
        x_tile_i = x[i : i + tile_size]
        ti = x_tile_i.shape[0]
        for j in range(i + tile_size, n, tile_size):
            x_tile_j = x[j : j + tile_size]
            diffs = x_tile_i[:, None, :] - x_tile_j[None, :, :]
            dists = jnp.sqrt(jnp.sum(diffs**2, axis=-1))
            total = total + jnp.sum(dists)
        if ti > 1:
            diffs_self = x_tile_i[:, None, :] - x_tile_i[None, :, :]
            dists_self = jnp.sqrt(jnp.sum(diffs_self**2, axis=-1))
            mask = jnp.triu(jnp.ones((ti, ti), dtype=bool), k=1)
            total = total + jnp.sum(jnp.where(mask, dists_self, 0.0))
    return total


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def energy_distance_gpu(
    x: np.ndarray,
    y: np.ndarray,
    tile_size: int = 2048,
) -> float:
    """Compute energy distance on GPU.

    For 1D data, uses an O(N log N) algorithm based on sorted prefix
    sums.  For multivariate data, uses tiled O(N²) reductions.

    Parameters
    ----------
    x : np.ndarray
        Samples from P, shape ``(n,)`` or ``(n, d)``.
    y : np.ndarray
        Samples from Q, shape ``(m,)`` or ``(m, d)``.
    tile_size : int
        Tile size for multivariate path.

    Returns
    -------
    float
        Energy distance (U-statistic).
    """
    _, jnp = _ensure_jax()

    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)

    if x.ndim == 1 or (x.ndim == 2 and x.shape[1] == 1):
        # 1D fast path: O(N log N)
        x_flat = jnp.asarray(x.ravel(), dtype=jnp.float64)
        y_flat = jnp.asarray(y.ravel(), dtype=jnp.float64)
        n = x_flat.shape[0]
        m = y_flat.shape[0]
        fn = _get_cached_energy_fn()
        result = fn(
            jnp.sort(x_flat),
            jnp.sort(y_flat),
            jnp.float64(n),
            jnp.float64(m),
        )
        return float(result)
    else:
        # Multivariate: tiled O(N²)
        x_gpu = jnp.asarray(x, dtype=jnp.float32)
        y_gpu = jnp.asarray(y, dtype=jnp.float32)
        n, m = x_gpu.shape[0], y_gpu.shape[0]
        sum_xy = _tiled_pairwise_sum(x_gpu, y_gpu, tile_size)
        sum_xx = _tiled_pairwise_sum_self(x_gpu, tile_size)
        sum_yy = _tiled_pairwise_sum_self(y_gpu, tile_size)
        mean_xy = sum_xy / (jnp.float64(n) * jnp.float64(m))
        mean_xx = 2.0 * sum_xx / (jnp.float64(n) * jnp.float64(n - 1)) if n > 1 else 0.0
        mean_yy = 2.0 * sum_yy / (jnp.float64(m) * jnp.float64(m - 1)) if m > 1 else 0.0
        return float(2.0 * mean_xy - mean_xx - mean_yy)


def energy_permutation_test_gpu(
    combined: np.ndarray,
    n_p: int,
    n_permutations: int,
    seed: int = 42,
    tile_size: int = 2048,
) -> tuple[float, np.ndarray]:
    """GPU-accelerated energy distance permutation test.

    For 1D data, all permutations are vectorized via ``jax.vmap`` and
    run in parallel on GPU using the O(N log N) sorted prefix-sum
    algorithm.  For N=200K with B=500 permutations, this completes
    in under 1 second on an RTX 4090.

    Parameters
    ----------
    combined : np.ndarray
        Pooled samples, shape ``(N,)`` or ``(N, d)``.
    n_p : int
        Number of samples from P.
    n_permutations : int
        Number of permutations.
    seed : int
        Random seed.
    tile_size : int
        Tile size for multivariate path.

    Returns
    -------
    observed : float
        Energy distance on the original split.
    null_distribution : np.ndarray
        Null statistics, shape ``(n_permutations,)``.
    """
    jax, jnp = _ensure_jax()

    combined = np.asarray(combined, dtype=np.float64)
    is_1d = combined.ndim == 1 or (combined.ndim == 2 and combined.shape[1] == 1)

    combined_2d = combined.reshape(-1, 1) if combined.ndim == 1 else combined

    n_total = combined_2d.shape[0]

    if is_1d:
        # Vectorized 1D path: all permutations in parallel on GPU
        Z_gpu = jnp.asarray(combined.ravel(), dtype=jnp.float64)
        fn = _get_cached_energy_fn()

        @partial(jax.jit, static_argnums=(2, 3))
        def _batch(Z, perms, _n_p, _n_total):
            def single(perm):
                x = jnp.sort(Z[perm[:_n_p]])
                y = jnp.sort(Z[perm[_n_p:]])
                return fn(x, y, jnp.float64(_n_p), jnp.float64(_n_total - _n_p))

            return jax.vmap(single)(perms)

        # Generate permutations on CPU, transfer to GPU
        rng = np.random.default_rng(seed)
        # Include the original ordering as the first "permutation"
        orig = np.arange(n_total)
        perms = [orig] + [rng.permutation(n_total) for _ in range(n_permutations)]
        perms_gpu = jnp.asarray(np.array(perms))

        # Run all permutations (including observed) in one batch
        all_stats = _batch(Z_gpu, perms_gpu, n_p, n_total)
        all_stats = np.asarray(all_stats)

        observed = float(all_stats[0])
        null_distribution = all_stats[1:]

    else:
        # Multivariate: sequential permutations with tiled GPU kernels
        observed = energy_distance_gpu(combined_2d[:n_p], combined_2d[n_p:], tile_size)
        rng = np.random.default_rng(seed)
        null_distribution = np.empty(n_permutations)
        for b in range(n_permutations):
            perm = rng.permutation(n_total)
            null_distribution[b] = energy_distance_gpu(
                combined_2d[perm[:n_p]],
                combined_2d[perm[n_p:]],
                tile_size,
            )

    return observed, null_distribution
