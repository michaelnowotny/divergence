# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Divergence is a Python package for computing statistical measures of entropy and divergence from probability distributions and samples. It supports Shannon measures (entropy, cross entropy, KL divergence, Jensen-Shannon divergence, joint entropy, conditional entropy, mutual information), f-divergences (total variation, Hellinger, chi-squared, Jeffreys, Cressie-Read), Rényi entropy/divergence, integral probability metrics (MMD, energy distance, Wasserstein, sliced Wasserstein), kNN estimators (Kozachenko-Leonenko entropy, KSG mutual information, kNN KL divergence), and two-sample testing — for both discrete and continuous distributions.

## Development Environment Setup

Requires [uv](https://docs.astral.sh/uv/getting-started/installation/):

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
uv pip install -e ".[dev]"
```

## Build & Install

```bash
uv pip install -e ".[dev]"     # Editable install with dev deps
python -m build                 # Build distribution
```

Uses `pyproject.toml` with setuptools backend and setuptools-scm for versioning from git tags.

## Running Tests

```bash
pytest                                          # All tests
pytest tests/test_discrete.py                   # Discrete measure tests only
pytest tests/test_continuous.py                 # Continuous tests (slow, ~8min due to numerical integration)
pytest tests/test_discrete.py::test_discrete_entropy -v  # Single test
```

Tests use fixed random seeds (42) for reproducibility. Continuous tests validate against scipy.stats ground truth via KDE estimation.

## Linting & Formatting

```bash
ruff check src/ tests/          # Lint
ruff check --fix src/ tests/    # Lint with auto-fix
ruff format src/ tests/         # Format
```

## Architecture

### Module Layout (src/ layout)

- **`src/divergence/__init__.py`** — Public API. Provides unified wrapper functions (e.g., `entropy_from_samples()`) that dispatch to discrete or continuous implementations based on `discrete=True/False` parameter. Exports all public functions via explicit `__all__`.
- **`src/divergence/base.py`** — Shared utility: log function selection (`log`, `log2`, `log10`) based on `base` parameter, plus Numba `Logarithm` jitclass for use in JIT-compiled code.
- **`src/divergence/discrete.py`** — Discrete distribution measures. Uses Numba `@njit` for performance-critical frequency counting and internal computation functions.
- **`src/divergence/continuous.py`** — Continuous distribution measures. Uses `cubature` for N-dimensional numerical integration and `statsmodels` for kernel density estimation.
- **`src/divergence/f_divergences.py`** — General f-divergence engine + TV, Hellinger, chi-squared, Jeffreys, Cressie-Read. Both discrete (frequency-based) and continuous (KDE grid integration) paths.
- **`src/divergence/renyi.py`** — Rényi entropy and divergence parameterized by order alpha, with limit cases.
- **`src/divergence/ipms.py`** — Integral probability metrics: energy distance, Wasserstein-p, MMD, sliced Wasserstein.
- **`src/divergence/knn.py`** — kNN-based estimators: Kozachenko-Leonenko entropy, KSG mutual information, kNN KL divergence. Uses scipy cKDTree.
- **`src/divergence/testing.py`** — Two-sample permutation tests with MMD/energy/kNN methods.
- **`src/divergence/bayesian.py`** — ArviZ integration for Bayesian diagnostics (information gain, chain divergence, uncertainty decomposition, Bayesian surprise).
- **`src/divergence/_types.py`** — Shared NamedTuple types (TestResult).
- **`tests/`** — Top-level test directory (outside the package).

### Key Design Patterns

- **Discrete functions** take sample arrays, internally construct frequency distributions, then compute information-theoretic measures. Numba JIT is used on hot paths.
- **Continuous functions** estimate densities from samples using KDE (statsmodels `KDEUnivariate` or scipy `gaussian_kde` for 2D), then numerically integrate over the support using `cubature`. All cubature integrands must return scalars (use `.item()` to extract from arrays).
- **`base` parameter** controls logarithm base throughout (e=nats, 2=bits, 10=hartleys).

### Performance architecture

Hot paths are JIT-compiled via Numba and dispatched automatically based on input size and dimensionality. The dispatch rules are per-function — there is no global "fast mode." Callers do not need to opt in; they get the fast path when it's faster.

**Numba kernels** (`src/divergence/_numba_kernels.py`):

- `_energy_distance_1d_jit(x, y)` — O((n + m) log(n + m)) sort-based kernel using closed-form sums of order statistics. **Dispatched by `energy_distance` when input is 1D and `max(n, m) >= 200`.** Also dispatched by `two_sample_test(method='energy')` for 1D data, which is the common case for scalar MCMC parameters.
- `_energy_distance_jit(x, y)` — O(n·m) multi-dimensional kernel with O(1) memory. Dispatched when `max(n, m) >= _JIT_THRESHOLD` (5000).
- `_mmd_squared_jit(x, y, gamma)` — O(n·m) multi-dimensional MMD² with RBF kernel, O(1) memory. Dispatched when `max(n, m) >= _MMD_JIT_THRESHOLD` (500; the threshold is lower than energy distance's because the vectorized MMD path must materialize three full kernel matrices).
- `_sinkhorn_cost_jit(C, epsilon, max_iter, tol)` — inlined log-domain Sinkhorn iterations. **Always** used by `sinkhorn_divergence` (no Python fallback); ~4× faster than the SciPy-based reference at n=500 that it replaced.
- `_sum_block_jit(D, idx_a, idx_b)` — sums the block `D[idx_a, idx_b]` in place without materializing a submatrix. Used by the precomputed-matrix path in `two_sample_test` to avoid per-permutation `np.ix_()` allocations. The block sums exploit the symmetry identity `D.sum() = S_PP + S_QQ + 2·S_PQ` to compute only two blocks per permutation instead of three.
- `_sum_block_rbf_jit(D_sq, idx_a, idx_b, gamma)` — same idea with RBF applied. Currently exposed via the `_mmd_from_sq_distance_matrix` helper but the MMD permutation loop in `two_sample_test` uses a precomputed-kernel-matrix approach that bypasses it for speed.
- `_median_bandwidth_jit(x)` — median pairwise distance for RBF bandwidth; uses subsampling for n ≥ 1000.
- `_ksd_stein_kernel_sum_jit(...)` — Stein kernel evaluation used by `kernel_stein_discrepancy`.

**GPU backend** (`src/divergence/_gpu_kernels.py`, optional):
- JAX-based kernels, currently for energy distance only. Activated by `DIVERGENCE_BACKEND=gpu` or by passing `backend="gpu"` to `two_sample_test`. Auto-detection via `_backend.py`.

**Functions without dedicated JIT kernels** (and why):
- `wasserstein_distance`, `sliced_wasserstein_distance` — scipy's 1D Wasserstein is already sort-based O(n log n).
- `knn_entropy`, `knn_kl_divergence`, `ksg_mutual_information` — scipy `cKDTree` is C-optimized; Numba can't improve neighbor queries.  Tree-reuse across permutations was investigated and rejected: tree construction is only ~15% of per-permutation cost, so the ceiling is too low to justify the added complexity.
- `total_correlation`, `variation_of_information`, `normalized_mutual_information` — compose kNN entropy; as fast as the underlying kNN.
- `transfer_entropy` — composes kNN mutual information.
- f-divergences (continuous path) — use fixed-grid trapezoidal integration over pooled KDE support; already fast (<1ms at n=500).
- Rényi, discrete measures, most Bayesian diagnostics — operate on small arrays or delegate.

**Bayesian chain diagnostics** (`bayesian.py`):
- `chain_divergence` with `method='mmd'` computes the RBF bandwidth once from the pooled samples across all chains and passes it explicitly to each pairwise `maximum_mean_discrepancy` call.  The per-pair median-heuristic call dominated the naive loop; eliminating it gives a ~1.7–2× speedup (e.g., 8 chains × 1000 draws: 1.1 s → 0.6 s; 16 chains: 5.3 s → 2.7 s).  The matrix-based amortization variant (one big kernel matrix + block sums) was benchmarked and rejected: the JIT streaming MMD is already so fast per-call that the O((C·m)²) exp work in the amortized path cancels the savings.

**Performance benchmarks** (dev machine, Linux x86_64):
- `energy_distance` at n=3000 (1D): ~30 μs
- `maximum_mean_discrepancy` at n=2000: ~43 ms
- `sinkhorn_divergence` at n=500: ~900 ms
- `two_sample_test(energy, n_permutations=500)` at n=3000 per group (1D): ~0.11 s
- `two_sample_test(mmd, n_permutations=500)` at n=2000 per group: ~7 s
- `chain_divergence(mmd)` at 8 chains × 1000 draws: ~0.6 s (down from ~1.1 s)

### Dependencies

Core: numpy, scipy, numba, statsmodels, cubature. Dev: pytest, hypothesis, pytest-cov, ruff, ipywidgets. Optional: arviz (for Bayesian diagnostics), xarray.

### ArviZ Integration (arviz >= 1.0.0)

**IMPORTANT**: We target ArviZ 1.0.0+ which uses `xarray.DataTree` (not the legacy `InferenceData` class from ArviZ 0.x). The API changed substantially:

- `az.from_dict()` takes a **single nested dict** with group names as keys: `az.from_dict({"posterior": {"mu": arr}, "prior": {"mu": arr}})`
- The return type is `xarray.DataTree`, not `arviz.InferenceData`
- Groups are accessed as children: `idata["posterior"]` returns a DataTree node
- Variables within groups: `idata["posterior"]["mu"].values` returns numpy array with shape `(chain, draw, ...)`
- Check group existence: `"posterior" in idata.children`
- List variables: `list(idata["posterior"].ds.data_vars)`
- `az.extract(idata, group="posterior", combined=True)` flattens chains/draws into a single sample dimension

ArviZ is an **optional dependency** — all bayesian.py functions use lazy imports and raise ImportError with installation instructions if arviz is not installed.
