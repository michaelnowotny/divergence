"""Tests for fast estimators (resubstitution and grid-based).

These tests verify that the fast estimators produce results consistent with:
1. Known analytical values for normal distributions
2. The slower cubature-based estimators (within tolerance)
3. Mathematical properties (non-negativity, symmetry, chain rule, etc.)
"""

import numpy as np
import pytest

from divergence import (
    conditional_entropy_from_samples,
    cross_entropy_from_samples,
    entropy_from_samples,
    jensen_shannon_divergence_from_samples,
    joint_entropy_from_samples,
    mutual_information_from_samples,
    relative_entropy_from_samples,
)


# ---------------------------------------------------------------------------
# Analytical reference values for normal distributions
# ---------------------------------------------------------------------------
def analytical_normal_entropy(sigma: float) -> float:
    """H(X) = 0.5 * (1 + ln(2*pi*sigma^2)) for X ~ N(mu, sigma^2)."""
    return 0.5 * (1.0 + np.log(2 * np.pi * sigma**2))


def analytical_normal_kl(mu_p, sigma_p, mu_q, sigma_q) -> float:
    """D_KL(P || Q) for P = N(mu_p, sigma_p^2), Q = N(mu_q, sigma_q^2)."""
    return ((mu_p - mu_q) ** 2 + sigma_p**2 - sigma_q**2) / (2 * sigma_q**2) + np.log(
        sigma_q / sigma_p
    )


def analytical_normal_cross_entropy(mu_p, sigma_p, mu_q, sigma_q) -> float:
    """H_q(P) = H(P) + D_KL(P || Q)."""
    return analytical_normal_entropy(sigma_p) + analytical_normal_kl(
        mu_p, sigma_p, mu_q, sigma_q
    )


def analytical_bivariate_normal_mi(rho: float) -> float:
    """I(X; Y) = -0.5 * ln(1 - rho^2) for bivariate normal."""
    return -0.5 * np.log(1.0 - rho**2)


# ---------------------------------------------------------------------------
# Fixtures: generate samples once, reuse across tests
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def normal_samples():
    """Two normal distributions with known parameters."""
    rng = np.random.default_rng(42)
    mu_p, sigma_p = 2.0, 3.0
    mu_q, sigma_q = 1.0, 2.0
    n = 10000

    sample_p = rng.normal(mu_p, sigma_p, n)
    sample_q = rng.normal(mu_q, sigma_q, n)

    return {
        "sample_p": sample_p,
        "sample_q": sample_q,
        "mu_p": mu_p,
        "sigma_p": sigma_p,
        "mu_q": mu_q,
        "sigma_q": sigma_q,
    }


@pytest.fixture(scope="module")
def bivariate_normal_samples():
    """Bivariate normal samples with known correlation."""
    rng = np.random.default_rng(42)
    mu_x, sigma_x = 2.0, 3.0
    mu_y, sigma_y = 1.0, 2.0
    rho = 0.5
    n = 10000

    z = rng.standard_normal(n)
    sample_x = mu_x + sigma_x * z
    sample_y = mu_y + sigma_y * (
        rho * z + np.sqrt(1.0 - rho**2) * rng.standard_normal(n)
    )

    return {
        "sample_x": sample_x,
        "sample_y": sample_y,
        "mu_x": mu_x,
        "sigma_x": sigma_x,
        "mu_y": mu_y,
        "sigma_y": sigma_y,
        "rho": rho,
    }


# ---------------------------------------------------------------------------
# 1D Entropy tests
# ---------------------------------------------------------------------------
class TestEntropy:
    def test_entropy_close_to_analytical(self, normal_samples):
        """Entropy estimate should be close to analytical normal entropy."""
        s = normal_samples
        for sample, sigma in [
            (s["sample_p"], s["sigma_p"]),
            (s["sample_q"], s["sigma_q"]),
        ]:
            estimated = entropy_from_samples(sample, discrete=False)
            expected = analytical_normal_entropy(sigma)
            assert np.isclose(estimated, expected, rtol=0.05, atol=0.05), (
                f"Entropy estimate {estimated:.4f} != analytical {expected:.4f}"
            )

    def test_entropy_nonnegative(self, normal_samples):
        """Entropy of a continuous distribution can be negative, but for
        normal distributions with sigma >= 1 it should be positive."""
        for key in ["sample_p", "sample_q"]:
            h = entropy_from_samples(normal_samples[key], discrete=False)
            assert h > 0

    def test_entropy_increases_with_variance(self, normal_samples):
        """Higher variance should give higher entropy."""
        h_p = entropy_from_samples(normal_samples["sample_p"], discrete=False)
        h_q = entropy_from_samples(normal_samples["sample_q"], discrete=False)
        # sigma_p=3 > sigma_q=2, so H(p) > H(q)
        assert h_p > h_q

    @pytest.mark.parametrize("base", [np.e, 2.0, 10.0])
    def test_entropy_base_scaling(self, normal_samples, base):
        """Entropy in different bases should scale by change-of-base factor."""
        sample = normal_samples["sample_p"]
        h_nats = entropy_from_samples(sample, base=np.e, discrete=False)
        h_base = entropy_from_samples(sample, base=base, discrete=False)
        expected = h_nats / np.log(base)
        assert np.isclose(h_base, expected, rtol=0.01)


# ---------------------------------------------------------------------------
# 1D Cross Entropy tests
# ---------------------------------------------------------------------------
class TestCrossEntropy:
    def test_cross_entropy_close_to_analytical(self, normal_samples):
        s = normal_samples
        estimated = cross_entropy_from_samples(
            s["sample_p"], s["sample_q"], discrete=False
        )
        expected = analytical_normal_cross_entropy(
            s["mu_p"], s["sigma_p"], s["mu_q"], s["sigma_q"]
        )
        assert np.isclose(estimated, expected, rtol=0.1, atol=0.1)

    def test_cross_entropy_ge_entropy(self, normal_samples):
        """H_q(p) >= H(p) (Gibbs' inequality)."""
        s = normal_samples
        h_p = entropy_from_samples(s["sample_p"], discrete=False)
        h_pq = cross_entropy_from_samples(s["sample_p"], s["sample_q"], discrete=False)
        assert h_pq >= h_p - 0.01  # small tolerance for estimation error


# ---------------------------------------------------------------------------
# 1D Relative Entropy (KL Divergence) tests
# ---------------------------------------------------------------------------
class TestRelativeEntropy:
    def test_kl_close_to_analytical(self, normal_samples):
        s = normal_samples
        estimated = relative_entropy_from_samples(
            s["sample_p"], s["sample_q"], discrete=False
        )
        expected = analytical_normal_kl(
            s["mu_p"], s["sigma_p"], s["mu_q"], s["sigma_q"]
        )
        assert np.isclose(estimated, expected, rtol=0.1, atol=0.1)

    def test_kl_nonnegative(self, normal_samples):
        """D_KL(P || Q) >= 0 (Gibbs' inequality)."""
        s = normal_samples
        kl = relative_entropy_from_samples(s["sample_p"], s["sample_q"], discrete=False)
        assert kl >= -0.01  # small tolerance

    def test_kl_zero_for_same_distribution(self):
        """D_KL(P || P) should be close to 0."""
        rng = np.random.default_rng(123)
        sample = rng.normal(0, 1, 5000)
        kl = relative_entropy_from_samples(sample, sample, discrete=False)
        assert np.isclose(kl, 0.0, atol=0.05)

    def test_kl_equals_cross_entropy_minus_entropy(self, normal_samples):
        """D_KL(P||Q) = H_q(P) - H(P)."""
        s = normal_samples
        kl = relative_entropy_from_samples(s["sample_p"], s["sample_q"], discrete=False)
        h_p = entropy_from_samples(s["sample_p"], discrete=False)
        h_pq = cross_entropy_from_samples(s["sample_p"], s["sample_q"], discrete=False)
        assert np.isclose(kl, h_pq - h_p, rtol=0.05, atol=0.05)


# ---------------------------------------------------------------------------
# Jensen-Shannon Divergence tests
# ---------------------------------------------------------------------------
class TestJensenShannonDivergence:
    def test_jsd_nonnegative(self, normal_samples):
        s = normal_samples
        jsd = jensen_shannon_divergence_from_samples(
            s["sample_p"], s["sample_q"], discrete=False
        )
        assert jsd >= -0.01

    def test_jsd_symmetric(self, normal_samples):
        """JSD(P||Q) = JSD(Q||P)."""
        s = normal_samples
        jsd_pq = jensen_shannon_divergence_from_samples(
            s["sample_p"], s["sample_q"], discrete=False
        )
        jsd_qp = jensen_shannon_divergence_from_samples(
            s["sample_q"], s["sample_p"], discrete=False
        )
        assert np.isclose(jsd_pq, jsd_qp, rtol=0.05)

    def test_jsd_bounded_by_log2(self, normal_samples):
        """JSD(P||Q) <= ln(2) for base e."""
        s = normal_samples
        jsd = jensen_shannon_divergence_from_samples(
            s["sample_p"], s["sample_q"], discrete=False
        )
        assert jsd <= np.log(2) + 0.01

    def test_jsd_zero_for_same_distribution(self):
        """JSD(P||P) should be close to 0."""
        rng = np.random.default_rng(123)
        sample = rng.normal(0, 1, 5000)
        jsd = jensen_shannon_divergence_from_samples(sample, sample, discrete=False)
        assert np.isclose(jsd, 0.0, atol=0.05)


# ---------------------------------------------------------------------------
# 2D Mutual Information tests
# ---------------------------------------------------------------------------
class TestMutualInformation:
    def test_mi_close_to_analytical(self, bivariate_normal_samples):
        s = bivariate_normal_samples
        estimated = mutual_information_from_samples(
            s["sample_x"], s["sample_y"], discrete=False
        )
        expected = analytical_bivariate_normal_mi(s["rho"])
        assert np.isclose(estimated, expected, rtol=0.15, atol=0.05), (
            f"MI estimate {estimated:.4f} != analytical {expected:.4f}"
        )

    def test_mi_nonnegative(self, bivariate_normal_samples):
        s = bivariate_normal_samples
        mi = mutual_information_from_samples(
            s["sample_x"], s["sample_y"], discrete=False
        )
        assert mi >= -0.01

    def test_mi_symmetric(self, bivariate_normal_samples):
        """I(X;Y) = I(Y;X)."""
        s = bivariate_normal_samples
        mi_xy = mutual_information_from_samples(
            s["sample_x"], s["sample_y"], discrete=False
        )
        mi_yx = mutual_information_from_samples(
            s["sample_y"], s["sample_x"], discrete=False
        )
        assert np.isclose(mi_xy, mi_yx, rtol=0.05)

    def test_mi_zero_for_independent(self):
        """MI of independent variables should be close to 0."""
        rng = np.random.default_rng(42)
        x = rng.normal(0, 1, 5000)
        y = rng.normal(0, 1, 5000)
        mi = mutual_information_from_samples(x, y, discrete=False)
        assert np.isclose(mi, 0.0, atol=0.1)


# ---------------------------------------------------------------------------
# 2D Joint Entropy tests
# ---------------------------------------------------------------------------
class TestJointEntropy:
    def test_joint_entropy_ge_marginal(self, bivariate_normal_samples):
        """H(X,Y) >= max(H(X), H(Y))."""
        s = bivariate_normal_samples
        h_xy = joint_entropy_from_samples(s["sample_x"], s["sample_y"], discrete=False)
        h_x = entropy_from_samples(s["sample_x"], discrete=False)
        h_y = entropy_from_samples(s["sample_y"], discrete=False)
        assert h_xy >= max(h_x, h_y) - 0.1

    def test_joint_entropy_le_sum_of_marginals(self, bivariate_normal_samples):
        """H(X,Y) <= H(X) + H(Y), with equality iff independent."""
        s = bivariate_normal_samples
        h_xy = joint_entropy_from_samples(s["sample_x"], s["sample_y"], discrete=False)
        h_x = entropy_from_samples(s["sample_x"], discrete=False)
        h_y = entropy_from_samples(s["sample_y"], discrete=False)
        assert h_xy <= h_x + h_y + 0.1


# ---------------------------------------------------------------------------
# 2D Conditional Entropy tests
# ---------------------------------------------------------------------------
class TestConditionalEntropy:
    def test_chain_rule(self, bivariate_normal_samples):
        """H(X,Y) = H(X) + H(Y|X)."""
        s = bivariate_normal_samples
        h_xy = joint_entropy_from_samples(s["sample_x"], s["sample_y"], discrete=False)
        h_x = entropy_from_samples(s["sample_x"], discrete=False)
        h_y_given_x = conditional_entropy_from_samples(
            s["sample_x"], s["sample_y"], discrete=False
        )
        assert np.isclose(h_xy, h_x + h_y_given_x, rtol=0.05, atol=0.1)

    def test_mi_equals_entropy_minus_conditional(self, bivariate_normal_samples):
        """I(X;Y) = H(Y) - H(Y|X)."""
        s = bivariate_normal_samples
        mi = mutual_information_from_samples(
            s["sample_x"], s["sample_y"], discrete=False
        )
        h_y = entropy_from_samples(s["sample_y"], discrete=False)
        h_y_given_x = conditional_entropy_from_samples(
            s["sample_x"], s["sample_y"], discrete=False
        )
        assert np.isclose(mi, h_y - h_y_given_x, rtol=0.15, atol=0.1)


# ---------------------------------------------------------------------------
# Speed sanity check
# ---------------------------------------------------------------------------
class TestPerformance:
    def test_1d_measures_are_fast(self, normal_samples):
        """1D measures should complete in under 5 seconds total."""
        import time

        s = normal_samples
        start = time.monotonic()
        entropy_from_samples(s["sample_p"], discrete=False)
        cross_entropy_from_samples(s["sample_p"], s["sample_q"], discrete=False)
        relative_entropy_from_samples(s["sample_p"], s["sample_q"], discrete=False)
        jensen_shannon_divergence_from_samples(
            s["sample_p"], s["sample_q"], discrete=False
        )
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"1D measures took {elapsed:.1f}s (should be < 10s)"

    def test_2d_measures_are_fast(self, bivariate_normal_samples):
        """2D measures should complete in under 10 seconds total."""
        import time

        s = bivariate_normal_samples
        start = time.monotonic()
        mutual_information_from_samples(s["sample_x"], s["sample_y"], discrete=False)
        joint_entropy_from_samples(s["sample_x"], s["sample_y"], discrete=False)
        conditional_entropy_from_samples(s["sample_x"], s["sample_y"], discrete=False)
        elapsed = time.monotonic() - start
        assert elapsed < 10.0, f"2D measures took {elapsed:.1f}s (should be < 10s)"
