"""Tests for Renyi entropy and divergence estimators."""

import numpy as np
import pytest

from divergence.discrete import discrete_entropy
from divergence.f_divergences import _aligned_frequencies
from divergence.renyi import renyi_divergence, renyi_entropy


# ---------------------------------------------------------------------------
# Analytical formulas for normal distributions
# ---------------------------------------------------------------------------
def analytical_renyi_entropy_normal(mu, sigma, alpha, base=np.e):
    r"""Renyi entropy of a normal distribution N(mu, sigma^2).

    H_alpha(N(mu, sigma^2)) = 0.5 * log(2 * pi * sigma^2) + log(alpha) / (2 * (alpha - 1))

    This formula is valid for alpha > 0 and alpha != 1. The log is taken
    in the specified base.
    """
    log_fun = np.log  # compute in nats, then convert
    result = 0.5 * log_fun(2 * np.pi * sigma**2) + log_fun(alpha) / (2 * (alpha - 1))
    # Convert from nats to desired base
    return result / log_fun(base)


def analytical_renyi_divergence_normal(mu1, s1, mu2, s2, alpha, base=np.e):
    r"""Renyi divergence between two normal distributions.

    D_alpha(N(mu1, s1^2) || N(mu2, s2^2)) =
        alpha * (mu1 - mu2)^2 / (2 * ((1-alpha)*s1^2 + alpha*s2^2))
        - 1 / (2*(alpha-1)) * log(((1-alpha)*s1^2 + alpha*s2^2)
                                   / (s1^(2*(1-alpha)) * s2^(2*alpha)))

    Valid for alpha > 0, alpha != 1, and (1-alpha)*s1^2 + alpha*s2^2 > 0.
    """
    log_fun = np.log  # compute in nats, then convert
    sigma_mix = (1 - alpha) * s1**2 + alpha * s2**2

    if sigma_mix <= 0:
        return np.inf

    term1 = alpha * (mu1 - mu2) ** 2 / (2 * sigma_mix)
    term2 = (
        -1.0
        / (2 * (alpha - 1))
        * log_fun(sigma_mix / (s1 ** (2 * (1 - alpha)) * s2 ** (2 * alpha)))
    )

    result = term1 + term2
    return result / log_fun(base)


# ---------------------------------------------------------------------------
# Test distribution parameters
# ---------------------------------------------------------------------------
MU_P, SIGMA_P = 2.0, 1.5
MU_Q, SIGMA_Q = 0.0, 1.0


# ---------------------------------------------------------------------------
# Module-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def continuous_samples():
    """Large continuous samples from two normal distributions."""
    rng = np.random.default_rng(42)
    n = 10000
    return {
        "sample_p": rng.normal(MU_P, SIGMA_P, n),
        "sample_q": rng.normal(MU_Q, SIGMA_Q, n),
    }


@pytest.fixture(scope="module")
def discrete_samples():
    """Discrete samples with known probabilities."""
    rng = np.random.default_rng(42)
    n = 50000
    probs_p = np.array([0.2, 0.3, 0.5])
    probs_q = np.array([0.3, 0.3, 0.4])
    return {
        "sample_p": rng.choice([0, 1, 2], size=n, p=probs_p),
        "sample_q": rng.choice([0, 1, 2], size=n, p=probs_q),
        "probs_p": probs_p,
        "probs_q": probs_q,
    }


# ---------------------------------------------------------------------------
# TestRenyiEntropy
# ---------------------------------------------------------------------------
class TestRenyiEntropy:
    def test_alpha_1_equals_shannon_discrete(self, discrete_samples):
        """At alpha -> 1, Renyi entropy equals Shannon entropy (discrete)."""
        renyi = renyi_entropy(
            discrete_samples["sample_p"], alpha=1.0, base=np.e, discrete=True
        )
        shannon = discrete_entropy(sample=discrete_samples["sample_p"], base=np.e)
        assert renyi == pytest.approx(shannon, rel=1e-6)

    def test_alpha_1_equals_shannon_continuous(self, continuous_samples):
        """At alpha -> 1, Renyi entropy delegates to Shannon entropy (continuous)."""
        from divergence.continuous import continuous_entropy_from_sample

        renyi = renyi_entropy(
            continuous_samples["sample_p"],
            alpha=1.0,
            base=np.e,
            discrete=False,
        )
        shannon = continuous_entropy_from_sample(
            sample=continuous_samples["sample_p"], base=np.e
        )
        assert renyi == pytest.approx(shannon, rel=1e-6)

    def test_alpha_0_equals_hartley(self, discrete_samples):
        """At alpha = 0, Renyi entropy equals Hartley entropy = log(|support|)."""
        sample = discrete_samples["sample_p"]
        renyi = renyi_entropy(sample, alpha=0, base=np.e, discrete=True)
        support_size = len(np.unique(sample))
        hartley = np.log(support_size)
        assert renyi == pytest.approx(hartley, rel=1e-6)

    def test_alpha_inf_equals_min_entropy(self, discrete_samples):
        """At alpha = +inf, Renyi entropy equals min-entropy = -log(max p_i)."""
        sample = discrete_samples["sample_p"]
        renyi = renyi_entropy(sample, alpha=np.inf, base=np.e, discrete=True)
        _, counts = np.unique(sample, return_counts=True)
        frequencies = counts / len(sample)
        min_entropy = -np.log(np.max(frequencies))
        assert renyi == pytest.approx(min_entropy, rel=1e-6)

    def test_alpha_2_collision_entropy(self):
        """Collision entropy for a known distribution [1/2, 1/4, 1/4]."""
        # For p = [1/2, 1/4, 1/4]:
        # H_2 = -log(sum p_i^2) = -log(1/4 + 1/16 + 1/16)
        #      = -log(6/16) = -log(3/8) = log(8/3)
        n = 100000
        rng = np.random.default_rng(42)
        sample = rng.choice([0, 1, 2], size=n, p=[0.5, 0.25, 0.25])
        renyi = renyi_entropy(sample, alpha=2, base=np.e, discrete=True)
        expected = np.log(8.0 / 3.0)
        assert renyi == pytest.approx(expected, rel=1e-2)

    def test_monotonically_decreasing_in_alpha(self, discrete_samples):
        """H_alpha1 >= H_alpha2 when alpha1 < alpha2 (discrete)."""
        sample = discrete_samples["sample_p"]
        alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 10.0]
        entropies = [
            renyi_entropy(sample, alpha=a, base=np.e, discrete=True) for a in alphas
        ]
        for i in range(len(entropies) - 1):
            assert entropies[i] >= entropies[i + 1] - 1e-10, (
                f"H_{alphas[i]} = {entropies[i]} < H_{alphas[i + 1]} = "
                f"{entropies[i + 1]}"
            )

    def test_close_to_analytical_normal(self, continuous_samples):
        """Continuous Renyi entropy estimate is close to the analytical formula."""
        alpha = 2.0
        renyi = renyi_entropy(
            continuous_samples["sample_p"],
            alpha=alpha,
            base=np.e,
            discrete=False,
        )
        expected = analytical_renyi_entropy_normal(MU_P, SIGMA_P, alpha, base=np.e)
        assert renyi == pytest.approx(expected, rel=0.15)

    def test_base_scaling(self, discrete_samples):
        """H_alpha in bits equals H_alpha in nats / ln(2)."""
        sample = discrete_samples["sample_p"]
        alpha = 2.0
        h_nats = renyi_entropy(sample, alpha=alpha, base=np.e, discrete=True)
        h_bits = renyi_entropy(sample, alpha=alpha, base=2, discrete=True)
        assert h_bits == pytest.approx(h_nats / np.log(2), rel=1e-6)

    def test_nonnegative_discrete(self, discrete_samples):
        """Renyi entropy is non-negative for discrete distributions."""
        sample = discrete_samples["sample_p"]
        for alpha in [0.5, 1.0, 2.0, 5.0, np.inf]:
            h = renyi_entropy(sample, alpha=alpha, base=np.e, discrete=True)
            assert h >= -1e-10, f"H_{alpha} = {h} is negative"


# ---------------------------------------------------------------------------
# TestRenyiDivergence
# ---------------------------------------------------------------------------
class TestRenyiDivergence:
    def test_alpha_1_equals_kl(self, discrete_samples):
        """At alpha -> 1, Renyi divergence equals KL divergence (discrete)."""
        renyi = renyi_divergence(
            discrete_samples["sample_p"],
            discrete_samples["sample_q"],
            alpha=1.0,
            base=np.e,
            discrete=True,
        )
        # Compute KL via aligned frequencies (avoids bug in discrete_relative_entropy)
        freq_p, freq_q = _aligned_frequencies(
            discrete_samples["sample_p"], discrete_samples["sample_q"]
        )
        mask = (freq_p > 0) & (freq_q > 0)
        kl = float(np.sum(np.where(mask, freq_p * np.log(freq_p / freq_q), 0.0)))
        assert renyi == pytest.approx(kl, rel=1e-6)

    def test_nonnegative(self, discrete_samples):
        """Renyi divergence is non-negative (discrete)."""
        for alpha in [0.5, 1.0, 2.0, 5.0]:
            d = renyi_divergence(
                discrete_samples["sample_p"],
                discrete_samples["sample_q"],
                alpha=alpha,
                base=np.e,
                discrete=True,
            )
            assert d >= -1e-10, f"D_{alpha} = {d} is negative"

    def test_identical_is_zero(self):
        """D_alpha(P, P) = 0 for any alpha (discrete)."""
        rng = np.random.default_rng(42)
        sample = rng.choice([0, 1, 2], size=10000, p=[0.2, 0.3, 0.5])
        for alpha in [0.5, 2.0, 5.0]:
            d = renyi_divergence(sample, sample, alpha=alpha, base=np.e, discrete=True)
            assert d == pytest.approx(0.0, abs=1e-10)

    def test_close_to_analytical_normal(self):
        """Continuous Renyi divergence estimate close to analytical formula."""
        # Use alpha=0.5 because alpha=2 with sigma_p > sigma_q makes
        # (1-alpha)*sigma_p^2 + alpha*sigma_q^2 < 0 (formula diverges)
        rng = np.random.default_rng(42)
        n = 10000
        mu1, s1 = 0.0, 1.0
        mu2, s2 = 0.5, 1.0
        alpha = 0.5
        renyi = renyi_divergence(
            rng.normal(mu1, s1, n),
            rng.normal(mu2, s2, n),
            alpha=alpha,
            base=np.e,
            discrete=False,
        )
        expected = analytical_renyi_divergence_normal(
            mu1, s1, mu2, s2, alpha, base=np.e
        )
        assert expected > 0, f"Expected positive divergence, got {expected}"
        assert renyi == pytest.approx(expected, rel=0.25)

    def test_monotonically_nondecreasing_in_alpha(self, discrete_samples):
        """D_alpha1 <= D_alpha2 when alpha1 < alpha2 (discrete)."""
        alphas = [0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
        divergences = [
            renyi_divergence(
                discrete_samples["sample_p"],
                discrete_samples["sample_q"],
                alpha=a,
                base=np.e,
                discrete=True,
            )
            for a in alphas
        ]
        for i in range(len(divergences) - 1):
            assert divergences[i] <= divergences[i + 1] + 1e-10, (
                f"D_{alphas[i]} = {divergences[i]} > D_{alphas[i + 1]} = "
                f"{divergences[i + 1]}"
            )
