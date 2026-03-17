"""Tests for Sinkhorn divergence.

Validates the debiased Sinkhorn divergence for basic properties: identity,
symmetry, non-negativity, and positivity for different distributions.
"""

import numpy as np
import pytest

from divergence.sinkhorn import sinkhorn_divergence


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def n_samples():
    return 500


@pytest.fixture(scope="module")
def normal_01(rng, n_samples):
    """N(0,1) samples."""
    return rng.standard_normal(n_samples)


@pytest.fixture(scope="module")
def normal_21(rng, n_samples):
    """N(2,1) samples."""
    return rng.normal(2.0, 1.0, n_samples)


@pytest.fixture(scope="module")
def normal_2d_a(rng, n_samples):
    """2D standard normal samples."""
    return rng.standard_normal((n_samples, 2))


@pytest.fixture(scope="module")
def normal_2d_b(rng, n_samples):
    """2D shifted normal samples."""
    return rng.normal(1.0, 1.0, (n_samples, 2))


# ===========================================================================
# Sinkhorn Divergence
# ===========================================================================
class TestSinkhornDivergence:
    """Tests for sinkhorn_divergence."""

    def test_identity_near_zero(self, normal_01):
        """S(P, P) should be close to 0 (debiased)."""
        s = sinkhorn_divergence(normal_01, normal_01, epsilon=0.1)
        np.testing.assert_allclose(s, 0.0, atol=0.01)

    def test_different_distributions_positive(self, normal_01, normal_21):
        """S(P, Q) should be positive for different distributions."""
        s = sinkhorn_divergence(normal_01, normal_21, epsilon=0.1)
        assert s > 0.1, f"Expected positive Sinkhorn divergence, got {s}"

    def test_symmetric(self, normal_01, normal_21):
        """S(P, Q) should equal S(Q, P)."""
        s_pq = sinkhorn_divergence(normal_01, normal_21, epsilon=0.1)
        s_qp = sinkhorn_divergence(normal_21, normal_01, epsilon=0.1)
        np.testing.assert_allclose(s_pq, s_qp, rtol=0.05)

    def test_nonnegative(self, normal_01, normal_21):
        """Sinkhorn divergence should be non-negative."""
        s = sinkhorn_divergence(normal_01, normal_21, epsilon=0.1)
        assert s >= -0.01, f"Expected non-negative, got {s}"

    def test_2d_positive(self, normal_2d_a, normal_2d_b):
        """2D Sinkhorn divergence should be positive for shifted distributions."""
        s = sinkhorn_divergence(normal_2d_a, normal_2d_b, epsilon=0.5)
        assert s > 0.05

    def test_2d_identity(self, normal_2d_a):
        """2D S(P, P) should be near 0."""
        s = sinkhorn_divergence(normal_2d_a, normal_2d_a, epsilon=0.5)
        np.testing.assert_allclose(s, 0.0, atol=0.05)

    def test_epsilon_effect(self, normal_01, normal_21):
        """Different epsilon values should all give positive divergence."""
        s_small = sinkhorn_divergence(normal_01, normal_21, epsilon=0.01)
        s_large = sinkhorn_divergence(normal_01, normal_21, epsilon=1.0)
        # Both should be positive for different distributions
        assert s_small > 0
        assert s_large > 0

    def test_increases_with_separation(self, rng, n_samples):
        """Divergence should increase as distributions move apart."""
        p = rng.standard_normal(n_samples)
        q1 = rng.normal(0.5, 1.0, n_samples)
        q2 = rng.normal(2.0, 1.0, n_samples)
        s1 = sinkhorn_divergence(p, q1, epsilon=0.1)
        s2 = sinkhorn_divergence(p, q2, epsilon=0.1)
        assert s2 > s1, f"Expected s2 > s1, got {s2:.3f} vs {s1:.3f}"
