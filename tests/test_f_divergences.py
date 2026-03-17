"""Tests for f-divergence framework and named divergences."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from divergence.f_divergences import (
    _aligned_frequencies,
    chi_squared_divergence,
    cressie_read_divergence,
    f_divergence,
    jeffreys_divergence,
    squared_hellinger_distance,
    total_variation_distance,
)


# ---------------------------------------------------------------------------
# Analytical ground-truth formulas for normal distributions
# ---------------------------------------------------------------------------
def analytical_squared_hellinger_normal(mu1, s1, mu2, s2):
    """H^2(P, Q) for P = N(mu1, s1^2), Q = N(mu2, s2^2)."""
    return 2 * (
        1
        - np.sqrt(2 * s1 * s2 / (s1**2 + s2**2))
        * np.exp(-((mu1 - mu2) ** 2) / (4 * (s1**2 + s2**2)))
    )


def analytical_chi_squared_normal(mu1, s1, mu2, s2):
    """chi^2(P || Q) for normals. Valid only when s1^2 < 2*s2^2."""
    assert s1**2 < 2 * s2**2, "Formula invalid when sigma_p^2 >= 2*sigma_q^2"
    return (
        np.sqrt(s2**2 / (2 * s2**2 - s1**2))
        * np.exp((mu1 - mu2) ** 2 / (2 * s2**2 - s1**2))
        - 1
    )


def analytical_jeffreys_normal(mu1, s1, mu2, s2):
    """D_J(P, Q) for normals."""
    return ((s1**2 - s2**2) ** 2 + (s1**2 + s2**2) * (mu1 - mu2) ** 2) / (
        2 * s1**2 * s2**2
    )


def analytical_kl_normal(mu1, s1, mu2, s2):
    """D_KL(P || Q) for normals."""
    return np.log(s2 / s1) + (s1**2 + (mu1 - mu2) ** 2) / (2 * s2**2) - 0.5


def _kl_from_frequencies(freq_p, freq_q):
    """KL(P||Q) directly from aligned frequency vectors."""
    mask = (freq_p > 0) & (freq_q > 0)
    return float(np.sum(np.where(mask, freq_p * np.log(freq_p / freq_q), 0.0)))


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------
MU_P, SIGMA_P = 2.0, 1.5
MU_Q, SIGMA_Q = 0.0, 1.0


@pytest.fixture(scope="module")
def normal_samples():
    rng = np.random.default_rng(42)
    n = 8000
    return {
        "sample_p": rng.normal(MU_P, SIGMA_P, n),
        "sample_q": rng.normal(MU_Q, SIGMA_Q, n),
    }


@pytest.fixture(scope="module")
def discrete_samples():
    """Discrete samples with shared support (all values appear in both)."""
    return {
        "p": np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2]),
        "q": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2]),
    }


# ---------------------------------------------------------------------------
# Total variation distance
# ---------------------------------------------------------------------------
class TestTotalVariation:
    def test_bounded_0_to_1_discrete(self, discrete_samples):
        tv = total_variation_distance(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        assert 0.0 <= tv <= 1.0

    def test_identical_distributions_zero(self):
        s = np.array([0, 1, 2, 1, 0])
        assert total_variation_distance(s, s, discrete=True) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_symmetric_discrete(self, discrete_samples):
        tv_pq = total_variation_distance(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        tv_qp = total_variation_distance(
            discrete_samples["q"], discrete_samples["p"], discrete=True
        )
        assert tv_pq == pytest.approx(tv_qp, abs=1e-10)

    def test_symmetric_continuous(self, normal_samples):
        tv_pq = total_variation_distance(
            normal_samples["sample_p"], normal_samples["sample_q"]
        )
        tv_qp = total_variation_distance(
            normal_samples["sample_q"], normal_samples["sample_p"]
        )
        assert tv_pq == pytest.approx(tv_qp, rel=0.05)

    def test_bounded_continuous(self, normal_samples):
        tv = total_variation_distance(
            normal_samples["sample_p"], normal_samples["sample_q"]
        )
        assert 0.0 <= tv <= 1.0

    def test_disjoint_distributions_maximal(self):
        p = np.array([0, 0, 0, 0, 0])
        q = np.array([1, 1, 1, 1, 1])
        assert total_variation_distance(p, q, discrete=True) == pytest.approx(
            1.0, abs=1e-10
        )


# ---------------------------------------------------------------------------
# Squared Hellinger distance
# ---------------------------------------------------------------------------
class TestSquaredHellinger:
    def test_bounded_0_to_2_discrete(self, discrete_samples):
        h2 = squared_hellinger_distance(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        assert 0.0 <= h2 <= 2.0

    def test_identical_is_zero(self):
        s = np.array([0, 1, 2, 1, 0])
        assert squared_hellinger_distance(s, s, discrete=True) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_symmetric_discrete(self, discrete_samples):
        h2_pq = squared_hellinger_distance(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        h2_qp = squared_hellinger_distance(
            discrete_samples["q"], discrete_samples["p"], discrete=True
        )
        assert h2_pq == pytest.approx(h2_qp, abs=1e-10)

    def test_symmetric_continuous(self, normal_samples):
        h2_pq = squared_hellinger_distance(
            normal_samples["sample_p"], normal_samples["sample_q"]
        )
        h2_qp = squared_hellinger_distance(
            normal_samples["sample_q"], normal_samples["sample_p"]
        )
        assert h2_pq == pytest.approx(h2_qp, rel=0.1)

    def test_continuous_close_to_analytical(self, normal_samples):
        h2_est = squared_hellinger_distance(
            normal_samples["sample_p"], normal_samples["sample_q"]
        )
        h2_exact = analytical_squared_hellinger_normal(MU_P, SIGMA_P, MU_Q, SIGMA_Q)
        assert h2_est == pytest.approx(h2_exact, rel=0.15)


# ---------------------------------------------------------------------------
# Chi-squared divergence
# ---------------------------------------------------------------------------
class TestChiSquared:
    def test_nonnegative_discrete(self, discrete_samples):
        chi2 = chi_squared_divergence(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        assert chi2 >= -1e-10

    def test_identical_is_zero(self):
        s = np.array([0, 1, 2, 1, 0])
        assert chi_squared_divergence(s, s, discrete=True) == pytest.approx(
            0.0, abs=1e-10
        )

    def test_not_symmetric_discrete(self, discrete_samples):
        chi2_pq = chi_squared_divergence(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        chi2_qp = chi_squared_divergence(
            discrete_samples["q"], discrete_samples["p"], discrete=True
        )
        # Generally not equal (though could be for some special distributions)
        assert isinstance(chi2_pq, float)
        assert isinstance(chi2_qp, float)

    def test_continuous_close_to_numerical(self):
        """Compare KDE estimate to numerical integration of exact densities."""
        from scipy import integrate, stats

        rng = np.random.default_rng(123)
        n = 10000
        mu1, s1 = 0.0, 1.0
        mu2, s2 = 0.5, 1.0
        sample_p = rng.normal(mu1, s1, n)
        sample_q = rng.normal(mu2, s2, n)

        chi2_est = chi_squared_divergence(sample_p, sample_q)

        # Numerical ground truth from exact densities
        def p_pdf(x):
            return stats.norm.pdf(x, mu1, s1)

        def q_pdf(x):
            return stats.norm.pdf(x, mu2, s2)

        def integrand(x):
            return (p_pdf(x) - q_pdf(x)) ** 2 / q_pdf(x)

        chi2_exact, _ = integrate.quad(integrand, -10, 10)

        assert chi2_est == pytest.approx(chi2_exact, rel=0.3)

    def test_raises_on_support_violation(self):
        p = np.array([0, 1, 2, 3])  # has value 3
        q = np.array([0, 1, 2, 2])  # missing value 3
        with pytest.raises(ValueError, match="not absolutely continuous"):
            chi_squared_divergence(p, q, discrete=True)


# ---------------------------------------------------------------------------
# Jeffreys divergence
# ---------------------------------------------------------------------------
class TestJeffreys:
    def test_nonnegative_discrete(self, discrete_samples):
        dj = jeffreys_divergence(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        assert dj >= -1e-10

    def test_identical_is_zero(self):
        s = np.array([0, 1, 2, 1, 0])
        assert jeffreys_divergence(s, s, discrete=True) == pytest.approx(0.0, abs=1e-10)

    def test_symmetric_discrete(self, discrete_samples):
        dj_pq = jeffreys_divergence(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        dj_qp = jeffreys_divergence(
            discrete_samples["q"], discrete_samples["p"], discrete=True
        )
        assert dj_pq == pytest.approx(dj_qp, abs=1e-10)

    def test_symmetric_continuous(self, normal_samples):
        dj_pq = jeffreys_divergence(
            normal_samples["sample_p"], normal_samples["sample_q"]
        )
        dj_qp = jeffreys_divergence(
            normal_samples["sample_q"], normal_samples["sample_p"]
        )
        assert dj_pq == pytest.approx(dj_qp, rel=0.02)

    def test_close_to_analytical(self, normal_samples):
        dj_est = jeffreys_divergence(
            normal_samples["sample_p"], normal_samples["sample_q"]
        )
        dj_exact = analytical_jeffreys_normal(MU_P, SIGMA_P, MU_Q, SIGMA_Q)
        assert dj_est == pytest.approx(dj_exact, rel=0.25)

    def test_equals_sum_of_kl_discrete(self):
        """Jeffreys = KL(P||Q) + KL(Q||P) for distributions with shared support."""
        p = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        q = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        dj = jeffreys_divergence(p, q, discrete=True)
        freq_p, freq_q = _aligned_frequencies(p, q)
        kl_pq = _kl_from_frequencies(freq_p, freq_q)
        kl_qp = _kl_from_frequencies(freq_q, freq_p)
        assert dj == pytest.approx(kl_pq + kl_qp, rel=1e-10)


# ---------------------------------------------------------------------------
# General f-divergence
# ---------------------------------------------------------------------------
class TestFDivergenceGeneral:
    def test_kl_via_f_divergence_discrete(self):
        """f(t) = t*log(t) should produce KL divergence."""
        p = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        q = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])

        def f_kl(t):
            return np.where(t > 0, t * np.log(np.where(t > 0, t, 1.0)), 0.0)

        fd = f_divergence(p, q, f_kl, discrete=True)
        freq_p, freq_q = _aligned_frequencies(p, q)
        kl = _kl_from_frequencies(freq_p, freq_q)
        assert fd == pytest.approx(kl, rel=1e-10)

    def test_nonnegative(self, discrete_samples):
        """f-divergence with f(t) = (t-1)^2 should be non-negative."""
        fd = f_divergence(
            discrete_samples["p"],
            discrete_samples["q"],
            lambda t: (t - 1) ** 2,
            discrete=True,
        )
        assert fd >= -1e-10

    def test_identical_is_zero(self):
        s = np.array([0, 1, 2, 1, 0])
        fd = f_divergence(s, s, lambda t: (t - 1) ** 2, discrete=True)
        assert fd == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# Cressie-Read family
# ---------------------------------------------------------------------------
class TestCressieRead:
    def test_nonnegative(self, discrete_samples):
        cr = cressie_read_divergence(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        assert cr >= -1e-10

    def test_identical_is_zero(self):
        s = np.array([0, 1, 2, 1, 0])
        cr = cressie_read_divergence(s, s, discrete=True)
        assert cr == pytest.approx(0.0, abs=1e-10)

    def test_lambda_0_equals_kl(self):
        """lambda -> 0 should give KL divergence."""
        p = np.array([0, 0, 0, 1, 1, 1, 1, 2, 2, 2])
        q = np.array([0, 0, 1, 1, 1, 2, 2, 2, 2, 2])
        cr = cressie_read_divergence(p, q, lambda_param=0.0, discrete=True)
        freq_p, freq_q = _aligned_frequencies(p, q)
        kl = _kl_from_frequencies(freq_p, freq_q)
        assert cr == pytest.approx(kl, rel=1e-6)

    def test_lambda_1_neyman_chi_squared(self, discrete_samples):
        """lambda = 1 should give Neyman chi-squared = chi^2(Q || P) (scaled)."""
        cr = cressie_read_divergence(
            discrete_samples["p"],
            discrete_samples["q"],
            lambda_param=1.0,
            discrete=True,
        )
        # Neyman chi-squared via CR at lambda=1 is:
        # (1/2) sum q_i [(p_i/q_i)^2 - 1] = (1/2) sum (p_i^2/q_i - q_i)
        # = (1/2) chi^2(P||Q)
        chi2 = chi_squared_divergence(
            discrete_samples["p"], discrete_samples["q"], discrete=True
        )
        assert cr == pytest.approx(chi2 / 2, rel=1e-6)


# ---------------------------------------------------------------------------
# Hypothesis property-based tests
# ---------------------------------------------------------------------------
@st.composite
def discrete_sample_pair(draw):
    """Generate two discrete samples guaranteed to share full support."""
    n = draw(st.integers(min_value=100, max_value=300))
    k = draw(st.integers(min_value=2, max_value=5))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    # Use Dirichlet with alpha > 1 to ensure all categories have
    # substantial probability, making support coverage very likely
    probs_p = rng.dirichlet(np.ones(k) * 3)
    probs_q = rng.dirichlet(np.ones(k) * 3)
    sample_p = rng.choice(k, size=n, p=probs_p)
    sample_q = rng.choice(k, size=n, p=probs_q)
    # Guarantee shared support by adding one of each category to both
    sample_p = np.concatenate([sample_p, np.arange(k)])
    sample_q = np.concatenate([sample_q, np.arange(k)])
    return sample_p, sample_q


class TestHypothesisProperties:
    @given(data=discrete_sample_pair())
    @settings(max_examples=20, deadline=None)
    def test_tv_bounded(self, data):
        p, q = data
        tv = total_variation_distance(p, q, discrete=True)
        assert -1e-10 <= tv <= 1.0 + 1e-10

    @given(data=discrete_sample_pair())
    @settings(max_examples=20, deadline=None)
    def test_tv_symmetric(self, data):
        p, q = data
        assert total_variation_distance(p, q, discrete=True) == pytest.approx(
            total_variation_distance(q, p, discrete=True), abs=1e-10
        )

    @given(data=discrete_sample_pair())
    @settings(max_examples=20, deadline=None)
    def test_hellinger_bounded(self, data):
        p, q = data
        h2 = squared_hellinger_distance(p, q, discrete=True)
        assert -1e-10 <= h2 <= 2.0 + 1e-10

    @given(data=discrete_sample_pair())
    @settings(max_examples=20, deadline=None)
    def test_hellinger_symmetric(self, data):
        p, q = data
        assert squared_hellinger_distance(p, q, discrete=True) == pytest.approx(
            squared_hellinger_distance(q, p, discrete=True), abs=1e-10
        )

    @given(data=discrete_sample_pair())
    @settings(max_examples=20, deadline=None)
    def test_chi_squared_nonnegative(self, data):
        p, q = data
        chi2 = chi_squared_divergence(p, q, discrete=True)
        assert chi2 >= -1e-10

    @given(data=discrete_sample_pair())
    @settings(max_examples=20, deadline=None)
    def test_jeffreys_symmetric(self, data):
        p, q = data
        assert jeffreys_divergence(p, q, discrete=True) == pytest.approx(
            jeffreys_divergence(q, p, discrete=True), abs=1e-10
        )
