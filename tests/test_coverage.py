"""Tests targeting uncovered code paths for coverage.

These exercise the cubature-based *_from_densities_with_support() functions,
base selection paths, and discrete internal functions.
"""

import numpy as np
import pytest
import scipy as sp

from divergence import (
    conditional_entropy_from_densities_with_support,
    # Unified API with discrete=True
    conditional_entropy_from_samples,
    cross_entropy_from_densities_with_support,
    cross_entropy_from_samples,
    entropy_from_density_with_support,
    entropy_from_kde,
    entropy_from_samples,
    jensen_shannon_divergence_from_densities_with_support,
    jensen_shannon_divergence_from_samples,
    joint_entropy_from_densities_with_support,
    joint_entropy_from_samples,
    mutual_information_from_densities_with_support,
    mutual_information_from_samples,
    relative_entropy_from_densities_with_support,
    relative_entropy_from_samples,
)
from divergence.base import _select_vectorized_log_fun_for_base


# ---------------------------------------------------------------------------
# Base module coverage
# ---------------------------------------------------------------------------
class TestBaseLogSelection:
    def test_log_base_e(self):
        f = _select_vectorized_log_fun_for_base(np.e)
        assert f is np.log

    def test_log_base_2(self):
        f = _select_vectorized_log_fun_for_base(2)
        assert f is np.log2

    def test_log_base_10(self):
        f = _select_vectorized_log_fun_for_base(10)
        assert f is np.log10

    def test_unsupported_base(self):
        with pytest.raises(ValueError, match="base not supported"):
            _select_vectorized_log_fun_for_base(3.0)


# ---------------------------------------------------------------------------
# Unified API dispatch (discrete=True paths in __init__.py)
# ---------------------------------------------------------------------------
class TestUnifiedAPIDiscrete:
    """Covers the discrete=True branches in __init__.py wrapper functions."""

    def setup_method(self):
        self.sample_p = np.array([1, 2, 3, 2, 3, 3, 3, 2, 1, 1])
        self.sample_q = np.array([1, 2, 3, 3, 3, 3, 3, 3, 3, 3])
        self.sample_x = np.array([1, 1, 3, 1, 2, 3])
        self.sample_y = np.array([1, 1, 1, 3, 2, 1])

    def test_entropy_discrete(self):
        h = entropy_from_samples(self.sample_p, discrete=True)
        assert h > 0

    def test_cross_entropy_discrete(self):
        ce = cross_entropy_from_samples(self.sample_p, self.sample_q, discrete=True)
        assert ce > 0

    def test_relative_entropy_discrete(self):
        kl = relative_entropy_from_samples(self.sample_q, self.sample_p, discrete=True)
        assert kl >= 0

    def test_jensen_shannon_discrete(self):
        jsd = jensen_shannon_divergence_from_samples(
            self.sample_p, self.sample_q, discrete=True
        )
        assert 0 <= jsd <= np.log(2) + 0.01

    def test_mutual_information_discrete(self):
        mi = mutual_information_from_samples(
            self.sample_x, self.sample_y, discrete=True
        )
        assert mi >= 0

    def test_joint_entropy_discrete(self):
        h = joint_entropy_from_samples(self.sample_x, self.sample_y, discrete=True)
        assert h > 0

    def test_conditional_entropy_discrete(self):
        h = conditional_entropy_from_samples(
            self.sample_x, self.sample_y, discrete=True
        )
        assert h >= 0


# ---------------------------------------------------------------------------
# Cubature-based *_from_densities_with_support() functions
# ---------------------------------------------------------------------------
class TestCubatureDensityFunctions:
    """Test the cubature path with known analytical density functions."""

    def setup_method(self):
        self.mu_p, self.sigma_p = 0.0, 1.0
        self.mu_q, self.sigma_q = 0.5, 1.5
        self.pdf_p = lambda x: sp.stats.norm.pdf(x, self.mu_p, self.sigma_p)
        self.pdf_q = lambda x: sp.stats.norm.pdf(x, self.mu_q, self.sigma_q)
        self.a, self.b = -6.0, 6.0

    def test_entropy_from_density(self):
        h = entropy_from_density_with_support(self.pdf_p, self.a, self.b)
        expected = 0.5 * (1 + np.log(2 * np.pi * self.sigma_p**2))
        assert np.isclose(h, expected, rtol=0.01)

    def test_entropy_from_density_base2(self):
        h = entropy_from_density_with_support(self.pdf_p, self.a, self.b, base=2.0)
        expected = 0.5 * (1 + np.log(2 * np.pi * self.sigma_p**2)) / np.log(2)
        assert np.isclose(h, expected, rtol=0.01)

    def test_cross_entropy_from_density(self):
        ce = cross_entropy_from_densities_with_support(
            self.pdf_p, self.pdf_q, self.a, self.b
        )
        h_p = 0.5 * (1 + np.log(2 * np.pi * self.sigma_p**2))
        kl = ((self.mu_p - self.mu_q) ** 2 + self.sigma_p**2 - self.sigma_q**2) / (
            2 * self.sigma_q**2
        ) + np.log(self.sigma_q / self.sigma_p)
        assert np.isclose(ce, h_p + kl, rtol=0.02)

    def test_relative_entropy_from_density(self):
        kl = relative_entropy_from_densities_with_support(
            self.pdf_p, self.pdf_q, self.a, self.b
        )
        expected = (
            (self.mu_p - self.mu_q) ** 2 + self.sigma_p**2 - self.sigma_q**2
        ) / (2 * self.sigma_q**2) + np.log(self.sigma_q / self.sigma_p)
        assert np.isclose(kl, expected, rtol=0.02)

    def test_jensen_shannon_from_density(self):
        jsd = jensen_shannon_divergence_from_densities_with_support(
            self.pdf_p, self.pdf_q, self.a, self.b
        )
        assert 0 <= jsd <= np.log(2)

    def test_mutual_information_from_density(self):
        """MI from bivariate normal density functions."""
        rho = 0.5
        mu_x, mu_y = 0.0, 0.0
        sigma_x, sigma_y = 1.0, 1.0

        def pdf_x(x):
            return sp.stats.norm.pdf(x, mu_x, sigma_x)

        def pdf_y(y):
            return sp.stats.norm.pdf(y, mu_y, sigma_y)

        cov = [
            [sigma_x**2, rho * sigma_x * sigma_y],
            [rho * sigma_x * sigma_y, sigma_y**2],
        ]
        mvn = sp.stats.multivariate_normal(mean=[mu_x, mu_y], cov=cov)
        pdf_xy = mvn.pdf

        mi = mutual_information_from_densities_with_support(
            pdf_x, pdf_y, pdf_xy, x_min=-5, x_max=5, y_min=-5, y_max=5
        )
        expected = -0.5 * np.log(1 - rho**2)
        assert np.isclose(mi, expected, rtol=0.05)

    def test_joint_entropy_from_density(self):
        """Joint entropy of bivariate normal."""
        rho = 0.3
        cov = [[1, rho], [rho, 1]]
        mvn = sp.stats.multivariate_normal(mean=[0, 0], cov=cov)

        h_xy = joint_entropy_from_densities_with_support(
            mvn.pdf, x_min=-5, x_max=5, y_min=-5, y_max=5
        )
        # H(X,Y) = 0.5 * ln((2*pi*e)^2 * det(cov))
        expected = 0.5 * np.log((2 * np.pi * np.e) ** 2 * np.linalg.det(cov))
        assert np.isclose(h_xy, expected, rtol=0.05)

    def test_conditional_entropy_from_density(self):
        """H(Y|X) = H(X,Y) - H(X) for bivariate normal."""
        rho = 0.3
        cov = [[1, rho], [rho, 1]]
        mvn = sp.stats.multivariate_normal(mean=[0, 0], cov=cov)

        def pdf_x(x):
            return sp.stats.norm.pdf(x, 0, 1)

        h_y_given_x = conditional_entropy_from_densities_with_support(
            pdf_x, mvn.pdf, x_min=-5, x_max=5, y_min=-5, y_max=5
        )
        # H(Y|X) = H(X,Y) - H(X) = 0.5*ln(2*pi*e*(1-rho^2))
        expected = 0.5 * np.log(2 * np.pi * np.e * (1 - rho**2))
        assert np.isclose(h_y_given_x, expected, rtol=0.05)


# ---------------------------------------------------------------------------
# Entropy from KDE object (grid path)
# ---------------------------------------------------------------------------
class TestEntropyFromKDE:
    def test_entropy_from_kde_with_base2(self):
        import statsmodels.api as sm

        rng = np.random.default_rng(42)
        sample = rng.normal(0, 1, 5000)
        kde = sm.nonparametric.KDEUnivariate(sample)
        kde.fit()

        h_nats = entropy_from_kde(kde, base=np.e)
        h_bits = entropy_from_kde(kde, base=2.0)
        assert np.isclose(h_bits, h_nats / np.log(2), rtol=0.01)
