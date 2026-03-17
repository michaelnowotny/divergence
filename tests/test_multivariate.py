"""Tests for multivariate dependence measures.

Validates total correlation, normalized mutual information, and variation of
information against analytical results for known distributions.
"""

import numpy as np
import pytest

from divergence.knn import ksg_mutual_information
from divergence.multivariate import (
    normalized_mutual_information,
    total_correlation,
    variation_of_information,
)


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def n_samples():
    return 3000


@pytest.fixture(scope="module")
def independent_2d(rng, n_samples):
    """Two independent N(0,1) columns — TC should be ~0."""
    return rng.standard_normal((n_samples, 2))


@pytest.fixture(scope="module")
def correlated_2d(rng, n_samples):
    """Correlated bivariate normal with rho=0.7."""
    rho = 0.7
    x = rng.standard_normal(n_samples)
    y = rho * x + np.sqrt(1 - rho**2) * rng.standard_normal(n_samples)
    return np.column_stack([x, y]), rho


@pytest.fixture(scope="module")
def independent_3d(rng, n_samples):
    """Three independent N(0,1) columns."""
    return rng.standard_normal((n_samples, 3))


@pytest.fixture(scope="module")
def discrete_identical(rng, n_samples):
    """Two identical discrete variables — VI should be 0."""
    x = rng.integers(0, 5, size=n_samples)
    return x, x.copy()


@pytest.fixture(scope="module")
def discrete_independent(rng, n_samples):
    """Two independent discrete uniform variables."""
    x = rng.integers(0, 5, size=n_samples)
    y = rng.integers(0, 5, size=n_samples)
    return x, y


# ===========================================================================
# Total Correlation
# ===========================================================================
class TestTotalCorrelation:
    """Tests for total_correlation."""

    def test_independent_near_zero(self, independent_2d):
        """TC of independent variables should be close to 0."""
        tc = total_correlation(independent_2d, estimator="knn")
        np.testing.assert_allclose(tc, 0.0, atol=0.15)

    def test_correlated_positive(self, correlated_2d):
        """TC of correlated variables should be positive."""
        samples, rho = correlated_2d
        tc = total_correlation(samples, estimator="knn")
        assert tc > 0.05, f"Expected positive TC for rho={rho}, got {tc}"

    def test_two_vars_equals_mi(self, correlated_2d):
        """For d=2, TC should equal MI."""
        samples, _rho = correlated_2d
        tc = total_correlation(samples, estimator="knn")
        mi = ksg_mutual_information(samples[:, 0], samples[:, 1])
        np.testing.assert_allclose(tc, mi, rtol=0.15)

    def test_independent_3d_near_zero(self, independent_3d):
        """TC of 3 independent variables should be ~0."""
        tc = total_correlation(independent_3d, estimator="knn")
        np.testing.assert_allclose(tc, 0.0, atol=0.2)

    def test_nonnegative(self, correlated_2d):
        """TC should be non-negative (or close due to estimation noise)."""
        samples, _ = correlated_2d
        tc = total_correlation(samples, estimator="knn")
        assert tc > -0.1

    def test_discrete(self, discrete_independent):
        """Discrete TC of independent variables should be near 0."""
        x, y = discrete_independent
        samples = np.column_stack([x, y])
        tc = total_correlation(samples, discrete=True)
        np.testing.assert_allclose(tc, 0.0, atol=0.05)

    def test_invalid_shape(self):
        """Should raise ValueError for 1D input."""
        with pytest.raises(ValueError, match="at least 2 columns"):
            total_correlation(np.array([1.0, 2.0, 3.0]))

    def test_base_scaling(self, correlated_2d):
        """TC in bits should equal TC in nats / ln(2)."""
        samples, _ = correlated_2d
        tc_nats = total_correlation(samples, base=np.e, estimator="knn")
        tc_bits = total_correlation(samples, base=2.0, estimator="knn")
        np.testing.assert_allclose(tc_bits, tc_nats / np.log(2), rtol=1e-6)


# ===========================================================================
# Normalized Mutual Information
# ===========================================================================
class TestNormalizedMutualInformation:
    """Tests for normalized_mutual_information."""

    def test_independent_near_zero(self, independent_2d):
        """NMI of independent variables should be near 0."""
        nmi = normalized_mutual_information(independent_2d[:, 0], independent_2d[:, 1])
        np.testing.assert_allclose(nmi, 0.0, atol=0.15)

    def test_correlated_positive(self, correlated_2d):
        """NMI of correlated variables should be positive."""
        samples, _ = correlated_2d
        nmi = normalized_mutual_information(samples[:, 0], samples[:, 1])
        assert nmi > 0.05

    def test_normalizations_consistent(self, correlated_2d):
        """All normalizations should produce positive values for correlated data."""
        samples, _ = correlated_2d
        for norm in ["geometric", "arithmetic", "max", "min", "joint"]:
            nmi = normalized_mutual_information(
                samples[:, 0], samples[:, 1], normalization=norm
            )
            assert nmi > 0, f"NMI with {norm} normalization should be positive"

    def test_discrete_identical(self, discrete_identical):
        """NMI of identical discrete variables should be ~1."""
        x, y = discrete_identical
        nmi = normalized_mutual_information(x, y, discrete=True)
        np.testing.assert_allclose(nmi, 1.0, atol=0.01)

    def test_invalid_normalization(self, independent_2d):
        """Unknown normalization should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown normalization"):
            normalized_mutual_information(
                independent_2d[:, 0], independent_2d[:, 1], normalization="bad"
            )


# ===========================================================================
# Variation of Information
# ===========================================================================
class TestVariationOfInformation:
    """Tests for variation_of_information."""

    def test_discrete_identical_is_zero(self, discrete_identical):
        """VI of identical variables should be 0."""
        x, y = discrete_identical
        vi = variation_of_information(x, y, discrete=True)
        np.testing.assert_allclose(vi, 0.0, atol=0.01)

    def test_discrete_independent_positive(self, discrete_independent):
        """VI of independent discrete variables should be positive."""
        x, y = discrete_independent
        vi = variation_of_information(x, y, discrete=True)
        assert vi > 0.1

    def test_symmetric(self, correlated_2d):
        """VI should be symmetric."""
        samples, _ = correlated_2d
        vi_xy = variation_of_information(samples[:, 0], samples[:, 1])
        vi_yx = variation_of_information(samples[:, 1], samples[:, 0])
        np.testing.assert_allclose(vi_xy, vi_yx, rtol=0.05)

    def test_nonnegative(self, correlated_2d):
        """VI should be non-negative (or close)."""
        samples, _ = correlated_2d
        vi = variation_of_information(samples[:, 0], samples[:, 1])
        assert vi > -0.1
