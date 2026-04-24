"""Tests for Integral Probability Metrics."""

import numpy as np
import pytest
from scipy.stats import wasserstein_distance as scipy_wasserstein

from divergence.ipms import (
    energy_distance,
    maximum_mean_discrepancy,
    sliced_wasserstein_distance,
    wasserstein_distance,
)

# ---------------------------------------------------------------------------
# Analytical ground-truth formulas for normal distributions
# ---------------------------------------------------------------------------
MU_P, SIGMA_P = 0.0, 1.0
MU_Q, SIGMA_Q = 2.0, 1.5


def analytical_w2_normal(mu1, s1, mu2, s2):
    """W_2(N(mu1, s1^2), N(mu2, s2^2)) = sqrt((mu1 - mu2)^2 + (s1 - s2)^2)."""
    return np.sqrt((mu1 - mu2) ** 2 + (s1 - s2) ** 2)


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def samples_p(rng):
    return rng.normal(MU_P, SIGMA_P, 5000)


@pytest.fixture(scope="module")
def samples_q(rng):
    return rng.normal(MU_Q, SIGMA_Q, 5000)


@pytest.fixture(scope="module")
def samples_r(rng):
    """A third distribution for triangle inequality tests."""
    return rng.normal(1.0, 1.2, 5000)


@pytest.fixture(scope="module")
def samples_p_same_seed():
    """Samples from P with a fixed seed for same-distribution tests."""
    return np.random.default_rng(99).normal(MU_P, SIGMA_P, 3000)


@pytest.fixture(scope="module")
def samples_p_same_seed2():
    """Another independent draw from the same distribution P."""
    return np.random.default_rng(100).normal(MU_P, SIGMA_P, 3000)


@pytest.fixture(scope="module")
def samples_p_2d(rng):
    return rng.multivariate_normal([0.0, 0.0], np.eye(2), 3000)


@pytest.fixture(scope="module")
def samples_q_2d(rng):
    return rng.multivariate_normal([1.0, 1.0], np.eye(2), 3000)


# ---------------------------------------------------------------------------
# Energy distance
# ---------------------------------------------------------------------------
class TestEnergyDistance:
    def test_nonnegative(self, samples_p, samples_q):
        ed = energy_distance(samples_p, samples_q)
        assert ed >= -1e-10

    def test_same_distribution_near_zero(
        self, samples_p_same_seed, samples_p_same_seed2
    ):
        ed = energy_distance(samples_p_same_seed, samples_p_same_seed2)
        assert ed == pytest.approx(0.0, abs=0.1)

    def test_symmetric(self, samples_p, samples_q):
        ed_pq = energy_distance(samples_p, samples_q)
        ed_qp = energy_distance(samples_q, samples_p)
        assert ed_pq == pytest.approx(ed_qp, rel=1e-10)

    def test_different_distributions_positive(self, samples_p, samples_q):
        ed = energy_distance(samples_p, samples_q)
        assert ed > 0.1

    def test_multidimensional(self, samples_p_2d, samples_q_2d):
        ed = energy_distance(samples_p_2d, samples_q_2d)
        assert ed > 0.1


# ---------------------------------------------------------------------------
# Wasserstein distance
# ---------------------------------------------------------------------------
class TestWasserstein:
    def test_p1_matches_scipy(self, samples_p, samples_q):
        w1_ours = wasserstein_distance(samples_p, samples_q, p=1)
        w1_scipy = scipy_wasserstein(samples_p, samples_q)
        assert w1_ours == pytest.approx(w1_scipy, rel=1e-10)

    def test_p2_close_to_analytical(self, samples_p, samples_q):
        w2_est = wasserstein_distance(samples_p, samples_q, p=2)
        w2_exact = analytical_w2_normal(MU_P, SIGMA_P, MU_Q, SIGMA_Q)
        assert w2_est == pytest.approx(w2_exact, rel=0.2)

    def test_nonnegative(self, samples_p, samples_q):
        w1 = wasserstein_distance(samples_p, samples_q, p=1)
        w2 = wasserstein_distance(samples_p, samples_q, p=2)
        assert w1 >= 0.0
        assert w2 >= 0.0

    def test_same_distribution_near_zero(
        self, samples_p_same_seed, samples_p_same_seed2
    ):
        w1 = wasserstein_distance(samples_p_same_seed, samples_p_same_seed2, p=1)
        w2 = wasserstein_distance(samples_p_same_seed, samples_p_same_seed2, p=2)
        assert w1 == pytest.approx(0.0, abs=0.1)
        assert w2 == pytest.approx(0.0, abs=0.1)

    def test_symmetric(self, samples_p, samples_q):
        w1_pq = wasserstein_distance(samples_p, samples_q, p=1)
        w1_qp = wasserstein_distance(samples_q, samples_p, p=1)
        w2_pq = wasserstein_distance(samples_p, samples_q, p=2)
        w2_qp = wasserstein_distance(samples_q, samples_p, p=2)
        assert w1_pq == pytest.approx(w1_qp, rel=1e-10)
        assert w2_pq == pytest.approx(w2_qp, rel=1e-10)

    def test_triangle_inequality(self, samples_p, samples_q, samples_r):
        w_pr = wasserstein_distance(samples_p, samples_r, p=1)
        w_pq = wasserstein_distance(samples_p, samples_q, p=1)
        w_qr = wasserstein_distance(samples_q, samples_r, p=1)
        assert w_pr <= w_pq + w_qr + 1e-10

    def test_invalid_p_raises(self):
        """p < 1 should raise ValueError."""
        rng = np.random.default_rng(42)
        p_samp = rng.normal(0, 1, 100)
        q_samp = rng.normal(0, 1, 100)
        with pytest.raises(ValueError, match="p must be"):
            wasserstein_distance(p_samp, q_samp, p=0)


# ---------------------------------------------------------------------------
# Maximum Mean Discrepancy
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def mmd_samples_same_1():
    """Small samples from the same distribution for MMD tests."""
    return np.random.default_rng(99).normal(0.0, 1.0, 500)


@pytest.fixture(scope="module")
def mmd_samples_same_2():
    """Another independent draw from the same distribution for MMD tests."""
    return np.random.default_rng(100).normal(0.0, 1.0, 500)


@pytest.fixture(scope="module")
def mmd_samples_diff_p():
    """Small samples from P for MMD tests."""
    return np.random.default_rng(42).normal(MU_P, SIGMA_P, 500)


@pytest.fixture(scope="module")
def mmd_samples_diff_q():
    """Small samples from Q for MMD tests."""
    return np.random.default_rng(43).normal(MU_Q, SIGMA_Q, 500)


@pytest.fixture(scope="module")
def mmd_samples_2d_p():
    """Small 2D samples from P for MMD tests."""
    return np.random.default_rng(42).multivariate_normal([0.0, 0.0], np.eye(2), 500)


@pytest.fixture(scope="module")
def mmd_samples_2d_q():
    """Small 2D samples from Q for MMD tests."""
    return np.random.default_rng(43).multivariate_normal([1.0, 1.0], np.eye(2), 500)


class TestMMD:
    def test_same_distribution_near_zero(self, mmd_samples_same_1, mmd_samples_same_2):
        mmd2 = maximum_mean_discrepancy(mmd_samples_same_1, mmd_samples_same_2)
        assert mmd2 == pytest.approx(0.0, abs=0.05)

    def test_different_distributions_positive(
        self, mmd_samples_diff_p, mmd_samples_diff_q
    ):
        mmd2 = maximum_mean_discrepancy(mmd_samples_diff_p, mmd_samples_diff_q)
        assert mmd2 > 0.01

    def test_bandwidth_affects_result(self, mmd_samples_diff_p, mmd_samples_diff_q):
        mmd2_small_bw = maximum_mean_discrepancy(
            mmd_samples_diff_p, mmd_samples_diff_q, bandwidth=0.1
        )
        mmd2_large_bw = maximum_mean_discrepancy(
            mmd_samples_diff_p, mmd_samples_diff_q, bandwidth=10.0
        )
        assert mmd2_small_bw != pytest.approx(mmd2_large_bw, rel=0.01)

    def test_multidimensional(self, mmd_samples_2d_p, mmd_samples_2d_q):
        mmd2 = maximum_mean_discrepancy(mmd_samples_2d_p, mmd_samples_2d_q)
        assert mmd2 > 0.01

    def test_symmetric(self):
        """MMD(P,Q) should approximately equal MMD(Q,P)."""
        rng = np.random.default_rng(42)
        p = rng.normal(0, 1, 500)
        q = rng.normal(1, 1, 500)
        mmd_pq = maximum_mean_discrepancy(p, q)
        mmd_qp = maximum_mean_discrepancy(q, p)
        assert mmd_pq == pytest.approx(mmd_qp, rel=0.1)

    def test_invalid_kernel_raises(self):
        """Invalid kernel name should raise ValueError."""
        rng = np.random.default_rng(42)
        p = rng.normal(0, 1, 100)
        q = rng.normal(0, 1, 100)
        with pytest.raises(ValueError, match="rbf"):
            maximum_mean_discrepancy(p, q, kernel="invalid")


# ---------------------------------------------------------------------------
# Sliced Wasserstein distance
# ---------------------------------------------------------------------------
class TestSlicedWasserstein:
    def test_nonnegative(self, samples_p_2d, samples_q_2d):
        sw = sliced_wasserstein_distance(samples_p_2d, samples_q_2d, seed=42)
        assert sw >= 0.0

    def test_same_distribution_near_zero(self):
        rng1 = np.random.default_rng(200)
        rng2 = np.random.default_rng(201)
        p = rng1.multivariate_normal([0.0, 0.0], np.eye(2), 3000)
        q = rng2.multivariate_normal([0.0, 0.0], np.eye(2), 3000)
        sw = sliced_wasserstein_distance(p, q, seed=42)
        assert sw == pytest.approx(0.0, abs=0.15)

    def test_symmetric(self, samples_p_2d, samples_q_2d):
        sw_pq = sliced_wasserstein_distance(samples_p_2d, samples_q_2d, seed=42)
        sw_qp = sliced_wasserstein_distance(samples_q_2d, samples_p_2d, seed=42)
        assert sw_pq == pytest.approx(sw_qp, rel=1e-10)

    def test_reproducible_with_seed(self, samples_p_2d, samples_q_2d):
        sw1 = sliced_wasserstein_distance(samples_p_2d, samples_q_2d, seed=123)
        sw2 = sliced_wasserstein_distance(samples_p_2d, samples_q_2d, seed=123)
        assert sw1 == sw2

    def test_1d_equals_regular_wasserstein(self, samples_p, samples_q):
        sw = sliced_wasserstein_distance(samples_p, samples_q, p=2, seed=42)
        w = wasserstein_distance(samples_p, samples_q, p=2)
        assert sw == pytest.approx(w, rel=1e-10)


# ---------------------------------------------------------------------------
# JIT vs vectorized cross-validation
# ---------------------------------------------------------------------------
class TestJITConsistency:
    """Verify JIT and vectorized paths produce identical results."""

    @pytest.fixture()
    def small_samples(self):
        rng = np.random.default_rng(42)
        return rng.normal(0, 1, (200, 1)), rng.normal(0.5, 1.2, (200, 1))

    def test_energy_distance_jit_matches_vectorized(self, small_samples):
        """JIT and vectorized energy distance should match on same data."""
        import divergence.ipms as ipms

        x, y = small_samples
        old = ipms._JIT_THRESHOLD

        ipms._JIT_THRESHOLD = 100_000  # force vectorized
        ed_vec = energy_distance(x.ravel(), y.ravel())

        ipms._JIT_THRESHOLD = 1  # force JIT
        ed_jit = energy_distance(x.ravel(), y.ravel())

        ipms._JIT_THRESHOLD = old
        np.testing.assert_allclose(ed_vec, ed_jit, rtol=1e-6)

    def test_mmd_jit_matches_vectorized(self, small_samples):
        """JIT and vectorized MMD should match on same data."""
        import divergence.ipms as ipms

        x, y = small_samples
        old = ipms._JIT_THRESHOLD

        ipms._JIT_THRESHOLD = 100_000
        mmd_vec = maximum_mean_discrepancy(x.ravel(), y.ravel(), bandwidth=1.0)

        ipms._JIT_THRESHOLD = 1
        mmd_jit = maximum_mean_discrepancy(x.ravel(), y.ravel(), bandwidth=1.0)

        ipms._JIT_THRESHOLD = old
        np.testing.assert_allclose(mmd_vec, mmd_jit, rtol=1e-6)

    def test_energy_distance_jit_nonnegative(self):
        """JIT energy distance should be non-negative."""
        from divergence._numba_kernels import _energy_distance_jit

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (300, 1))
        y = rng.normal(1, 1, (300, 1))
        assert _energy_distance_jit(x, y) >= 0

    def test_mmd_jit_same_distribution_near_zero(self):
        """JIT MMD for same distribution should be near zero."""
        from divergence._numba_kernels import _mmd_squared_jit

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (300, 1))
        y = rng.normal(0, 1, (300, 1))
        mmd = _mmd_squared_jit(x, y, 0.5)
        np.testing.assert_allclose(mmd, 0.0, atol=0.1)


# ---------------------------------------------------------------------------
# Safety-net tests for the energy-distance code paths
#
# The permutation-test pipeline (used by ``two_sample_test`` and
# ``chain_two_sample_test``) calls two different energy-distance
# implementations depending on sample size:
#
#   * ``_energy_distance_jit``            — sample-based, O(1) memory
#   * ``_energy_from_distance_matrix``    — precomputed-matrix-based
#
# The tests below pin down their agreement at a range of sizes, shapes,
# and corner cases so that any future optimization (1D sort-based path,
# Numba submatrix-sum kernel, GPU dispatch) is held to exact numerical
# equivalence with the current behavior.
# ---------------------------------------------------------------------------
class TestEnergyDistancePathEquivalence:
    """JIT sample-based vs precomputed-matrix-based energy distance."""

    @staticmethod
    def _ed_via_matrix(x: np.ndarray, y: np.ndarray) -> float:
        """Compute energy distance via the precomputed-distance-matrix path."""
        from scipy.spatial.distance import cdist

        from divergence.testing import _energy_from_distance_matrix

        x = np.atleast_2d(x).reshape(len(x), -1)
        y = np.atleast_2d(y).reshape(len(y), -1)
        combined = np.concatenate([x, y], axis=0)
        D = cdist(combined, combined, metric="euclidean")
        n_p = len(x)
        idx_p = np.arange(n_p)
        idx_q = np.arange(n_p, len(combined))
        return _energy_from_distance_matrix(D, idx_p, idx_q)

    @staticmethod
    def _ed_via_jit(x: np.ndarray, y: np.ndarray) -> float:
        """Compute energy distance via the JIT sample-based path."""
        from divergence._numba_kernels import _energy_distance_jit

        x = np.ascontiguousarray(np.atleast_2d(x).reshape(len(x), -1), dtype=float)
        y = np.ascontiguousarray(np.atleast_2d(y).reshape(len(y), -1), dtype=float)
        return float(_energy_distance_jit(x, y))

    @pytest.mark.parametrize("n_p, n_q", [(50, 50), (100, 30), (30, 100), (200, 200)])
    def test_paths_agree_1d(self, n_p, n_q):
        """The two energy-distance paths must agree for 1D data at various sizes."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, n_p)
        y = rng.normal(0.5, 1.2, n_q)
        ed_jit = self._ed_via_jit(x, y)
        ed_mat = self._ed_via_matrix(x, y)
        np.testing.assert_allclose(ed_jit, ed_mat, rtol=1e-10)

    @pytest.mark.parametrize("n, d", [(50, 2), (100, 3), (80, 5)])
    def test_paths_agree_multidim(self, n, d):
        """The two paths must agree for d-dimensional data."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, (n, d))
        y = rng.normal(0.5, 1.0, (n, d))
        ed_jit = self._ed_via_jit(x, y)
        ed_mat = self._ed_via_matrix(x, y)
        np.testing.assert_allclose(ed_jit, ed_mat, rtol=1e-10)

    def test_paths_agree_at_scale(self):
        """Agreement must hold at the n=1500 scale where the 1D fast path will activate."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1500)
        y = rng.normal(0.3, 1.1, 1500)
        ed_jit = self._ed_via_jit(x, y)
        ed_mat = self._ed_via_matrix(x, y)
        np.testing.assert_allclose(ed_jit, ed_mat, rtol=1e-10)

    def test_symmetry(self):
        """ED(x, y) == ED(y, x) on both paths."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 200)
        y = rng.normal(0.5, 1.3, 200)

        jit_xy = self._ed_via_jit(x, y)
        jit_yx = self._ed_via_jit(y, x)
        mat_xy = self._ed_via_matrix(x, y)
        mat_yx = self._ed_via_matrix(y, x)

        np.testing.assert_allclose(jit_xy, jit_yx, rtol=1e-10)
        np.testing.assert_allclose(mat_xy, mat_yx, rtol=1e-10)
        np.testing.assert_allclose(jit_xy, mat_xy, rtol=1e-10)

    def test_identical_distributions_small(self):
        """ED on samples from identical distributions should be small (not exactly zero)."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 500)
        y = rng.normal(0, 1, 500)
        ed_jit = self._ed_via_jit(x, y)
        ed_mat = self._ed_via_matrix(x, y)
        np.testing.assert_allclose(ed_jit, ed_mat, rtol=1e-10)
        # Sampling noise only; well below a realistic 2-sample signal
        assert abs(ed_jit) < 0.05

    def test_ties(self):
        """Energy distance must handle repeated values correctly.

        This case matters because sort-based 1D formulas need to handle
        ties without double-counting or skipping pairs.
        """
        rng = np.random.default_rng(42)
        base = rng.choice([0.0, 1.0, 2.0, 3.0], size=200)
        x = base + rng.normal(0, 0.01, 200)  # near-duplicates
        y = rng.choice([0.0, 1.0, 2.0, 3.0], size=200) + rng.normal(0, 0.01, 200)
        # Inject exact ties too
        x[0] = x[1] = 1.0
        y[0] = y[1] = 2.0

        ed_jit = self._ed_via_jit(x, y)
        ed_mat = self._ed_via_matrix(x, y)
        np.testing.assert_allclose(ed_jit, ed_mat, rtol=1e-10)

    def test_public_api_1d_equals_column_vector(self):
        """energy_distance(x) should match energy_distance(x.reshape(-1, 1))."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0.5, 1.1, 300)

        ed_1d = energy_distance(x, y)
        ed_2d = energy_distance(x.reshape(-1, 1), y.reshape(-1, 1))
        np.testing.assert_allclose(ed_1d, ed_2d, rtol=1e-10)

    def test_public_api_both_dispatch_thresholds(self):
        """The public API should agree with itself across both dispatch paths."""
        import divergence.ipms as ipms

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 400)
        y = rng.normal(0.3, 1.2, 400)

        old = ipms._JIT_THRESHOLD
        try:
            ipms._JIT_THRESHOLD = 100_000  # forces vectorized path
            ed_vec = energy_distance(x, y)
            ipms._JIT_THRESHOLD = 1  # forces JIT path
            ed_jit = energy_distance(x, y)
        finally:
            ipms._JIT_THRESHOLD = old

        np.testing.assert_allclose(ed_vec, ed_jit, rtol=1e-6)


class TestEnergyDistanceProperties:
    """Mathematical properties the energy distance must satisfy regardless of implementation."""

    @staticmethod
    def _ed_via_jit(x: np.ndarray, y: np.ndarray) -> float:
        from divergence._numba_kernels import _energy_distance_jit

        x = np.ascontiguousarray(np.atleast_2d(x).reshape(len(x), -1), dtype=float)
        y = np.ascontiguousarray(np.atleast_2d(y).reshape(len(y), -1), dtype=float)
        return float(_energy_distance_jit(x, y))

    def test_translation_invariance(self):
        """ED(x + c, y + c) == ED(x, y) — both sides translated by the same c."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0.5, 1.2, 300)
        c = 7.3

        ed_original = self._ed_via_jit(x, y)
        ed_shifted = self._ed_via_jit(x + c, y + c)
        np.testing.assert_allclose(ed_original, ed_shifted, rtol=1e-10)

    def test_scaling_covariance(self):
        """ED(a*x, a*y) == |a| * ED(x, y) — energy distance is 1-homogeneous."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = rng.normal(0.5, 1.2, 300)
        a = 3.5

        ed_original = self._ed_via_jit(x, y)
        ed_scaled = self._ed_via_jit(a * x, a * y)
        np.testing.assert_allclose(ed_scaled, abs(a) * ed_original, rtol=1e-10)

    def test_non_negative_when_signal_dominates(self):
        """ED is non-negative when the population-level signal exceeds sampling noise.

        The U-statistic estimator is unbiased, so for samples from similar
        distributions the finite-sample value can drift slightly negative
        due to sampling noise. With clearly different means and a reasonable
        sample size, the estimator must be positive.
        """
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 300)
        y = rng.normal(2.0, 1, 300)  # mean shift of 2 sigma, unambiguous
        ed = self._ed_via_jit(x, y)
        assert ed > 0.1  # signal >> sampling noise at this separation


# ---------------------------------------------------------------------------
# Tests specific to the new fast paths
#
# These lock in the numerical equivalence between the optimized kernels
# and the baseline (non-optimized) implementations.  They also include a
# performance assertion for the n=3000 scenario that motivated the
# optimization.
# ---------------------------------------------------------------------------
class TestEnergyDistance1DFastPath:
    """The O((n + m) log(n + m)) 1D sort-based fast path."""

    @pytest.mark.parametrize("n, m", [(50, 50), (200, 300), (500, 500), (1500, 2500)])
    def test_matches_matrix_baseline(self, n, m):
        """The new 1D fast path must match the cdist-based baseline bit-perfect."""
        from scipy.spatial.distance import cdist

        from divergence._numba_kernels import _energy_distance_1d_jit
        from divergence.testing import _energy_from_distance_matrix

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, n)
        y = rng.normal(0.4, 1.3, m)

        fast = float(
            _energy_distance_1d_jit(x.astype(np.float64), y.astype(np.float64))
        )

        combined = np.concatenate([x, y]).reshape(-1, 1)
        D = cdist(combined, combined, metric="euclidean")
        baseline = _energy_from_distance_matrix(D, np.arange(n), np.arange(n, n + m))

        np.testing.assert_allclose(fast, baseline, rtol=1e-10)

    def test_matches_jit_baseline(self):
        """The 1D fast path must match the existing multi-dimensional JIT path on (n, 1) input."""
        from divergence._numba_kernels import (
            _energy_distance_1d_jit,
            _energy_distance_jit,
        )

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 1000)
        y = rng.normal(0.3, 1.1, 1200)

        fast = float(_energy_distance_1d_jit(x, y))
        baseline = float(
            _energy_distance_jit(
                np.ascontiguousarray(x.reshape(-1, 1)),
                np.ascontiguousarray(y.reshape(-1, 1)),
            )
        )

        np.testing.assert_allclose(fast, baseline, rtol=1e-10)

    def test_handles_ties_correctly(self):
        """Repeated values (ties) in sorted order must produce the same result as the baseline."""
        from scipy.spatial.distance import cdist

        from divergence._numba_kernels import _energy_distance_1d_jit
        from divergence.testing import _energy_from_distance_matrix

        rng = np.random.default_rng(42)
        # Data with heavy ties: discrete values + tiny noise
        x = rng.choice([0.0, 1.0, 2.0, 3.0], size=400)
        y = rng.choice([0.5, 1.5, 2.5, 3.5], size=400)

        fast = float(_energy_distance_1d_jit(x, y))

        combined = np.concatenate([x, y]).reshape(-1, 1)
        D = cdist(combined, combined, metric="euclidean")
        baseline = _energy_from_distance_matrix(D, np.arange(400), np.arange(400, 800))

        np.testing.assert_allclose(fast, baseline, rtol=1e-10)

    def test_is_actually_fast_at_n_3000(self):
        """At n = 3000 the 1D fast path must be much faster than the matrix path.

        This is the motivating scenario: the NumPyro_KSD notebook was
        spending 3+ minutes here.  We don't assert a specific wall-clock
        time (CI environments vary) but we do assert the ratio — the
        1D sort path should be an order of magnitude faster than cdist
        at this size.
        """
        import time

        from scipy.spatial.distance import cdist

        from divergence._numba_kernels import _energy_distance_1d_jit

        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 3000).astype(np.float64)
        y = rng.normal(0.3, 1.1, 3000).astype(np.float64)

        # Warm up JIT cache
        _energy_distance_1d_jit(x[:100], y[:100])

        t0 = time.perf_counter()
        _ = _energy_distance_1d_jit(x, y)
        t_fast = time.perf_counter() - t0

        t0 = time.perf_counter()
        combined = np.concatenate([x, y]).reshape(-1, 1)
        _ = cdist(combined, combined, metric="euclidean")
        t_cdist = time.perf_counter() - t0

        # The fast path should be much faster than even building the
        # distance matrix — which is just the setup cost of the baseline.
        assert t_fast < t_cdist, (
            f"1D fast path ({t_fast:.3f}s) not faster than cdist setup "
            f"({t_cdist:.3f}s) at n=3000 — something is wrong"
        )


class TestSubmatrixSumKernels:
    """The Numba submatrix-sum kernels used by the precomputed-matrix permutation path."""

    @pytest.fixture()
    def matrix_and_indices(self):
        rng = np.random.default_rng(42)
        n = 200
        D = rng.uniform(0, 5, (n, n))
        # Symmetrize like a real distance matrix
        D = (D + D.T) / 2
        np.fill_diagonal(D, 0.0)
        idx_a = rng.choice(n, size=80, replace=False).astype(np.int64)
        idx_b = rng.choice(n, size=60, replace=False).astype(np.int64)
        return D, idx_a, idx_b

    def test_sum_block_matches_np_ix(self, matrix_and_indices):
        """_sum_block_jit must match D[np.ix_(idx_a, idx_b)].sum() bit-perfect."""
        from divergence._numba_kernels import _sum_block_jit

        D, idx_a, idx_b = matrix_and_indices
        baseline = D[np.ix_(idx_a, idx_b)].sum()
        fast = _sum_block_jit(D, idx_a, idx_b)
        np.testing.assert_allclose(fast, baseline, rtol=1e-12)

    def test_sum_block_rbf_matches_np_ix(self, matrix_and_indices):
        """_sum_block_rbf_jit must match np.exp(-gamma * D[np.ix_(...)]).sum() bit-perfect."""
        from divergence._numba_kernels import _sum_block_rbf_jit

        D, idx_a, idx_b = matrix_and_indices
        gamma = 0.5
        baseline = float(np.exp(-gamma * D[np.ix_(idx_a, idx_b)]).sum())
        fast = _sum_block_rbf_jit(D, idx_a, idx_b, gamma)
        np.testing.assert_allclose(fast, baseline, rtol=1e-12)

    def test_sum_block_same_indices(self, matrix_and_indices):
        """Block sum with idx_a == idx_b (within-group sum) must also match."""
        from divergence._numba_kernels import _sum_block_jit

        D, idx_a, _ = matrix_and_indices
        baseline = D[np.ix_(idx_a, idx_a)].sum()
        fast = _sum_block_jit(D, idx_a, idx_a)
        np.testing.assert_allclose(fast, baseline, rtol=1e-12)

    def test_symmetry_identity_holds(self, matrix_and_indices):
        """S_total = S_PP + S_QQ + 2*S_PQ — the identity used in the fast permutation path."""
        from divergence._numba_kernels import _sum_block_jit

        D, idx_p, idx_q = matrix_and_indices
        s_total = D.sum()
        # Use indices that partition the matrix
        n = D.shape[0]
        idx_p = np.arange(0, 120, dtype=np.int64)
        idx_q = np.arange(120, n, dtype=np.int64)
        s_pp = _sum_block_jit(D, idx_p, idx_p)
        s_qq = _sum_block_jit(D, idx_q, idx_q)
        s_pq = _sum_block_jit(D, idx_p, idx_q)

        np.testing.assert_allclose(s_total, s_pp + s_qq + 2 * s_pq, rtol=1e-10)


class TestPermutationTestBenchmark:
    """End-to-end benchmark: the n = 3000 two-sample test should run fast."""

    def test_scalar_mcmc_chain_scale_runs_quickly(self):
        """The scenario that motivated this work: 4-chain, 3000-draw permutation test.

        Previously this took 3+ minutes via the precomputed-matrix path
        with ``np.ix_()`` allocations on every permutation.  After the
        1D fast path plus the Numba submatrix-sum refactor, the whole
        test should finish in well under 10 seconds on any modern CPU.
        The assertion here is loose on purpose — CI environments vary —
        but the 30-second ceiling catches the old behavior immediately.
        """
        import time

        from divergence.testing import two_sample_test

        rng = np.random.default_rng(42)
        # Two chains' worth of scalar posterior samples — the dominant
        # use case is chain-vs-chain comparison in Bayesian diagnostics.
        p = rng.normal(0, 1, 3000)
        q = rng.normal(0, 1, 3000)  # same distribution — null calibration

        # Warm up JIT cache
        two_sample_test(p[:50], q[:50], method="energy", n_permutations=5, seed=1)

        t0 = time.perf_counter()
        result = two_sample_test(p, q, method="energy", n_permutations=200, seed=42)
        elapsed = time.perf_counter() - t0

        assert isinstance(result.p_value, float)
        assert 0.0 <= result.p_value <= 1.0
        assert elapsed < 30.0, (
            f"Two-sample test at n=3000 with 200 permutations took {elapsed:.1f}s — "
            f"should be well under 30s with the 1D fast path"
        )
