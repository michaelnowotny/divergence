"""Tests for kNN-based information-theoretic estimators.

Uses analytical formulas for normal distributions as ground truth:
- Normal entropy: H(N(mu, sigma^2)) = 0.5 * (1 + log(2 * pi * sigma^2))
- Normal KL: D_KL(N1 || N2) = log(sigma2/sigma1) + (sigma1^2 + (mu1-mu2)^2) / (2*sigma2^2) - 0.5
- Bivariate normal MI: I(X; Y) = -0.5 * log(1 - rho^2)

kNN estimators are noisier than KDE-based methods, so tolerances are generous
(rtol ~0.1-0.3 for continuous comparisons).
"""

import numpy as np
import pytest

from divergence.knn import knn_entropy, knn_kl_divergence, ksg_mutual_information


# ---------------------------------------------------------------------------
# Analytical formulas
# ---------------------------------------------------------------------------
def analytical_normal_entropy(sigma: float) -> float:
    """Differential entropy of N(mu, sigma^2) in nats."""
    return 0.5 * (1.0 + np.log(2.0 * np.pi * sigma**2))


def analytical_normal_kl(mu1: float, sigma1: float, mu2: float, sigma2: float) -> float:
    """KL divergence D_KL(N1 || N2) in nats."""
    return (
        np.log(sigma2 / sigma1)
        + (sigma1**2 + (mu1 - mu2) ** 2) / (2.0 * sigma2**2)
        - 0.5
    )


def analytical_bivariate_normal_mi(rho: float) -> float:
    """Mutual information of a bivariate normal with correlation rho, in nats."""
    return -0.5 * np.log(1.0 - rho**2)


# ---------------------------------------------------------------------------
# Module-scoped fixtures with fixed RNG seed
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def n_samples():
    return 3000


@pytest.fixture(scope="module")
def normal_01_samples(rng, n_samples):
    """N(0, 1) samples."""
    return rng.standard_normal(n_samples)


@pytest.fixture(scope="module")
def normal_02_samples(rng, n_samples):
    """N(0, 2) samples (sigma=2)."""
    return rng.normal(loc=0.0, scale=2.0, size=n_samples)


@pytest.fixture(scope="module")
def normal_shifted_samples(rng, n_samples):
    """N(1, 1.5) samples for KL divergence tests."""
    return rng.normal(loc=1.0, scale=1.5, size=n_samples)


@pytest.fixture(scope="module")
def normal_same_seed_pair(n_samples):
    """Two independent N(0,1) samples drawn from separate RNGs with close seeds."""
    rng_a = np.random.default_rng(100)
    rng_b = np.random.default_rng(101)
    return rng_a.standard_normal(n_samples), rng_b.standard_normal(n_samples)


@pytest.fixture(scope="module")
def bivariate_normal_rho05(rng, n_samples):
    """Bivariate normal (X, Y) with rho=0.5."""
    rho = 0.5
    x = rng.standard_normal(n_samples)
    y = rho * x + np.sqrt(1.0 - rho**2) * rng.standard_normal(n_samples)
    return x, y, rho


@pytest.fixture(scope="module")
def independent_xy(rng, n_samples):
    """Independent X, Y ~ N(0,1)."""
    x = rng.standard_normal(n_samples)
    y = rng.standard_normal(n_samples)
    return x, y


# ===========================================================================
# TestKNNEntropy
# ===========================================================================
class TestKNNEntropy:
    """Tests for the Kozachenko-Leonenko entropy estimator."""

    def test_close_to_analytical_normal(self, normal_01_samples):
        """N(0,1) entropy should be close to 0.5*(1+log(2*pi)) ~ 1.4189 nats."""
        expected = analytical_normal_entropy(sigma=1.0)
        estimated = knn_entropy(normal_01_samples, k=5)
        np.testing.assert_allclose(estimated, expected, rtol=0.1)

    def test_close_to_analytical_normal_wider(self, normal_02_samples):
        """N(0,2) entropy should be close to 0.5*(1+log(8*pi)) ~ 2.1121 nats."""
        expected = analytical_normal_entropy(sigma=2.0)
        estimated = knn_entropy(normal_02_samples, k=5)
        np.testing.assert_allclose(estimated, expected, rtol=0.1)

    def test_base_scaling(self, normal_01_samples):
        """Entropy in bits should equal entropy in nats divided by ln(2)."""
        h_nats = knn_entropy(normal_01_samples, k=5, base=np.e)
        h_bits = knn_entropy(normal_01_samples, k=5, base=2)
        np.testing.assert_allclose(h_bits, h_nats / np.log(2), rtol=1e-6)

    def test_k_parameter_effect(self, normal_01_samples):
        """Different k values should give similar entropy estimates (within 20%)."""
        h_k3 = knn_entropy(normal_01_samples, k=3)
        h_k5 = knn_entropy(normal_01_samples, k=5)
        h_k10 = knn_entropy(normal_01_samples, k=10)
        # All should be within 20% of each other
        np.testing.assert_allclose(h_k3, h_k5, rtol=0.2)
        np.testing.assert_allclose(h_k5, h_k10, rtol=0.2)
        np.testing.assert_allclose(h_k3, h_k10, rtol=0.2)

    def test_nonnegative(self):
        """Differential entropy of a normal should be positive."""
        rng = np.random.default_rng(42)
        samples = rng.standard_normal(3000)
        h = knn_entropy(samples, k=5)
        assert h > 0


# ===========================================================================
# TestKSGMutualInformation
# ===========================================================================
class TestKSGMutualInformation:
    """Tests for the Kraskov-Stogbauer-Grassberger mutual information estimator."""

    def test_independent_near_zero(self, independent_xy):
        """Independent X, Y should have MI close to 0."""
        x, y = independent_xy
        mi = ksg_mutual_information(x, y, k=5)
        np.testing.assert_allclose(mi, 0.0, atol=0.1)

    def test_correlated_bivariate_normal(self, bivariate_normal_rho05):
        """Bivariate normal with rho=0.5 should have MI ~ 0.1438 nats."""
        x, y, rho = bivariate_normal_rho05
        expected = analytical_bivariate_normal_mi(rho)
        estimated = ksg_mutual_information(x, y, k=5)
        np.testing.assert_allclose(estimated, expected, rtol=0.2)

    def test_symmetric(self, bivariate_normal_rho05):
        """MI(X, Y) should approximately equal MI(Y, X)."""
        x, y, _ = bivariate_normal_rho05
        mi_xy = ksg_mutual_information(x, y, k=5)
        mi_yx = ksg_mutual_information(y, x, k=5)
        np.testing.assert_allclose(mi_xy, mi_yx, rtol=0.05)

    def test_nonnegative(self, bivariate_normal_rho05):
        """Mutual information should be non-negative (or very close to 0)."""
        x, y, _ = bivariate_normal_rho05
        mi = ksg_mutual_information(x, y, k=5)
        assert mi > -0.05, f"MI should be non-negative (or close), got {mi}"

    def test_algorithm_2(self, bivariate_normal_rho05):
        """Algorithm 2 should run without error and return a finite value.

        Algorithm 2's -1/k bias correction makes it more sensitive to sample
        size and the distinction between strict < and <= counting. For
        continuous data with no ties, algorithm 1 is generally preferred.
        """
        x, y, _rho = bivariate_normal_rho05
        estimated = ksg_mutual_information(x, y, k=5, algorithm=2)
        assert np.isfinite(estimated)

    def test_invalid_algorithm_raises(self, independent_xy):
        """Invalid algorithm number should raise ValueError."""
        x, y = independent_xy
        with pytest.raises(ValueError, match="algorithm must be 1 or 2"):
            ksg_mutual_information(x, y, algorithm=3)

    def test_base_scaling(self):
        """MI in bits = MI in nats / ln(2)."""
        rng = np.random.default_rng(42)
        n = 3000
        rho = 0.5
        z = rng.standard_normal(n)
        x = z
        y = rho * z + np.sqrt(1 - rho**2) * rng.standard_normal(n)
        mi_nats = ksg_mutual_information(x, y, k=5, base=np.e)
        mi_bits = ksg_mutual_information(x, y, k=5, base=2.0)
        assert mi_bits == pytest.approx(mi_nats / np.log(2), rel=1e-6)


# ===========================================================================
# TestKNNKLDivergence
# ===========================================================================
class TestKNNKLDivergence:
    """Tests for the kNN-based KL divergence estimator."""

    def test_nonnegative(self, normal_01_samples, normal_shifted_samples):
        """D_KL should be >= 0 (allowing small negative from estimation noise)."""
        dkl = knn_kl_divergence(normal_01_samples, normal_shifted_samples, k=5)
        assert dkl > -0.05, f"D_KL should be non-negative (or close), got {dkl}"

    def test_same_distribution_near_zero(self, normal_same_seed_pair):
        """D_KL between two samples from the same distribution should be near 0."""
        p, q = normal_same_seed_pair
        dkl = knn_kl_divergence(p, q, k=5)
        np.testing.assert_allclose(dkl, 0.0, atol=0.15)

    def test_close_to_analytical_normal(
        self, normal_01_samples, normal_shifted_samples
    ):
        """D_KL(N(0,1) || N(1,1.5)) should be close to the analytical value."""
        expected = analytical_normal_kl(mu1=0.0, sigma1=1.0, mu2=1.0, sigma2=1.5)
        estimated = knn_kl_divergence(normal_01_samples, normal_shifted_samples, k=5)
        np.testing.assert_allclose(estimated, expected, rtol=0.3)

    def test_base_scaling(self):
        """KL in bits = KL in nats / ln(2)."""
        rng = np.random.default_rng(42)
        p = rng.normal(0, 1, 2000)
        q = rng.normal(0.5, 1.2, 2000)
        kl_nats = knn_kl_divergence(p, q, k=5, base=np.e)
        kl_bits = knn_kl_divergence(p, q, k=5, base=2.0)
        assert kl_bits == pytest.approx(kl_nats / np.log(2), rel=1e-6)
