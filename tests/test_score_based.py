"""Tests for score-based divergence measures.

Validates Fisher divergence and kernel Stein discrepancy using standard
normal score functions where analytical values are known.
"""

import numpy as np
import pytest

from divergence.score_based import fisher_divergence, kernel_stein_discrepancy


def standard_normal_score(x: np.ndarray) -> np.ndarray:
    """Score function for N(0, I): nabla log p(x) = -x."""
    return -x


def shifted_normal_score(x: np.ndarray, mu: float = 2.0) -> np.ndarray:
    """Score function for N(mu, I): nabla log p(x) = -(x - mu)."""
    return -(x - mu)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def n_samples():
    return 2000


@pytest.fixture(scope="module")
def normal_01(rng, n_samples):
    """N(0,1) samples."""
    return rng.standard_normal(n_samples)


@pytest.fixture(scope="module")
def normal_01_2d(rng, n_samples):
    """N(0, I_2) samples."""
    return rng.standard_normal((n_samples, 2))


@pytest.fixture(scope="module")
def normal_shifted(rng, n_samples):
    """N(2, 1) samples."""
    return rng.normal(2.0, 1.0, n_samples)


# ===========================================================================
# Fisher Divergence
# ===========================================================================
class TestFisherDivergence:
    """Tests for fisher_divergence."""

    def test_same_distribution_near_zero(self, normal_01):
        """D_F(P||P) should be near 0 when both score functions match."""
        fd = fisher_divergence(
            normal_01, standard_normal_score, score_p=standard_normal_score
        )
        np.testing.assert_allclose(fd, 0.0, atol=0.01)

    def test_different_distributions_positive(self, normal_01):
        """D_F(P||Q) should be positive when P and Q differ."""
        fd = fisher_divergence(
            normal_01,
            lambda x: shifted_normal_score(x, mu=2.0),
            score_p=standard_normal_score,
        )
        assert fd > 1.0, f"Expected large Fisher divergence, got {fd}"

    def test_known_value_normal_shift(self, normal_01):
        """D_F(N(0,1)||N(mu,1)) = E[||(-x) - (-(x-mu))||^2] = mu^2."""
        mu = 2.0
        fd = fisher_divergence(
            normal_01,
            lambda x: shifted_normal_score(x, mu=mu),
            score_p=standard_normal_score,
        )
        # Should be close to mu^2 = 4.0
        np.testing.assert_allclose(fd, mu**2, rtol=0.1)

    def test_estimated_score_p(self, normal_01):
        """Using estimated score_p should still give reasonable results."""
        fd = fisher_divergence(
            normal_01,
            lambda x: shifted_normal_score(x, mu=2.0),
        )
        assert fd > 1.0, (
            f"Expected positive Fisher divergence with estimated score, got {fd}"
        )

    def test_nonnegative(self, normal_01):
        """Fisher divergence should be non-negative."""
        fd = fisher_divergence(normal_01, standard_normal_score)
        assert fd >= -0.01

    def test_2d(self, normal_01_2d):
        """Fisher divergence should work in 2D."""
        fd = fisher_divergence(
            normal_01_2d, standard_normal_score, score_p=standard_normal_score
        )
        np.testing.assert_allclose(fd, 0.0, atol=0.01)


# ===========================================================================
# Kernel Stein Discrepancy
# ===========================================================================
class TestKernelSteinDiscrepancy:
    """Tests for kernel_stein_discrepancy."""

    def test_matching_distribution_near_zero(self, normal_01):
        """KSD should be near 0 when samples come from the target."""
        ksd = kernel_stein_discrepancy(normal_01, standard_normal_score)
        np.testing.assert_allclose(ksd, 0.0, atol=0.05)

    def test_mismatched_distribution_positive(self, normal_shifted):
        """KSD should be positive when samples don't match the target score."""
        ksd = kernel_stein_discrepancy(normal_shifted, standard_normal_score)
        assert ksd > 0.05, f"Expected positive KSD for mismatched dist, got {ksd}"

    def test_2d_matching(self, normal_01_2d):
        """KSD in 2D should be near 0 for matching samples."""
        ksd = kernel_stein_discrepancy(normal_01_2d, standard_normal_score)
        np.testing.assert_allclose(ksd, 0.0, atol=0.1)

    def test_invalid_kernel(self, normal_01):
        """Unknown kernel should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported kernel"):
            kernel_stein_discrepancy(normal_01, standard_normal_score, kernel="laplace")

    def test_too_few_samples(self):
        """Single sample should raise ValueError."""
        with pytest.raises(ValueError, match="at least 2"):
            kernel_stein_discrepancy(np.array([1.0]), standard_normal_score)

    def test_nonnegative_for_large_mismatch(self):
        """KSD should be clearly positive for a big distribution mismatch."""
        rng = np.random.default_rng(42)
        samples = rng.normal(5.0, 0.5, 500)
        ksd = kernel_stein_discrepancy(samples, standard_normal_score)
        assert ksd > 0.1


# ===========================================================================
# TestIMQKernelSteinDiscrepancy
# ===========================================================================
class TestIMQKernelSteinDiscrepancy:
    """Tests for kernel_stein_discrepancy with kernel='imq'."""

    def test_matching_near_zero(self, normal_01):
        """KSD(IMQ) should be near 0 when samples match the target."""
        ksd = kernel_stein_discrepancy(normal_01, standard_normal_score, kernel="imq")
        np.testing.assert_allclose(ksd, 0.0, atol=0.05)

    def test_mismatched_positive(self, normal_shifted):
        """KSD(IMQ) should be positive for mismatched distribution."""
        ksd = kernel_stein_discrepancy(
            normal_shifted, standard_normal_score, kernel="imq"
        )
        assert ksd > 0.05

    def test_2d_matching(self, normal_01_2d):
        """IMQ KSD in 2D should be near 0 for matching samples."""
        ksd = kernel_stein_discrepancy(
            normal_01_2d, standard_normal_score, kernel="imq"
        )
        np.testing.assert_allclose(ksd, 0.0, atol=0.1)

    def test_custom_bandwidth(self, normal_01):
        """Custom bandwidth should produce finite result."""
        ksd = kernel_stein_discrepancy(
            normal_01, standard_normal_score, kernel="imq", bandwidth=1.0
        )
        assert np.isfinite(ksd)

    def test_detects_shift(self):
        """IMQ should detect distribution mismatch."""
        rng = np.random.default_rng(42)
        samples = rng.normal(3.0, 1.0, 1000)
        ksd_imq = kernel_stein_discrepancy(samples, standard_normal_score, kernel="imq")
        assert ksd_imq > 0.5


# ===========================================================================
# TestKSDJITConsistency
# ===========================================================================
class TestKSDJITConsistency:
    """Verify JIT and vectorized KSD paths produce identical results."""

    def test_rbf_jit_matches_vectorized(self, normal_01):
        """JIT and vectorized RBF KSD should match."""
        from divergence._numba_kernels import _ksd_stein_kernel_sum_jit

        x = normal_01.reshape(-1, 1).copy()
        s = standard_normal_score(x)
        bw = 1.0

        # Vectorized
        ksd_vec = kernel_stein_discrepancy(
            normal_01, standard_normal_score, kernel="rbf", bandwidth=bw
        )

        # JIT
        n = len(x)
        total = _ksd_stein_kernel_sum_jit(
            np.ascontiguousarray(x),
            np.ascontiguousarray(s),
            bw**2,
            0,  # RBF
        )
        ksd_jit = total / (n * (n - 1))

        np.testing.assert_allclose(ksd_vec, ksd_jit, rtol=1e-6)

    def test_imq_jit_matches_vectorized(self, normal_01):
        """JIT and vectorized IMQ KSD should match."""
        from divergence._numba_kernels import _ksd_stein_kernel_sum_jit

        x = normal_01.reshape(-1, 1).copy()
        s = standard_normal_score(x)
        bw = 1.0

        ksd_vec = kernel_stein_discrepancy(
            normal_01, standard_normal_score, kernel="imq", bandwidth=bw
        )

        n = len(x)
        total = _ksd_stein_kernel_sum_jit(
            np.ascontiguousarray(x),
            np.ascontiguousarray(s),
            bw**2,
            1,  # IMQ
        )
        ksd_jit = total / (n * (n - 1))

        np.testing.assert_allclose(ksd_vec, ksd_jit, rtol=1e-6)
