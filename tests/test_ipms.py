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
