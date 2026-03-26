"""Tests for two-sample hypothesis testing."""

import numpy as np
import pytest

from divergence._types import TestResult
from divergence.testing import _should_use_low_memory, two_sample_test


@pytest.fixture(scope="module")
def different_distributions():
    """Two clearly different normal distributions."""
    rng = np.random.default_rng(42)
    return {
        "p": rng.normal(0, 1, 300),
        "q": rng.normal(2, 1, 300),
    }


@pytest.fixture(scope="module")
def same_distribution():
    """Two samples from the same distribution (different seeds)."""
    rng = np.random.default_rng(42)
    return {
        "p": rng.normal(0, 1, 300),
        "q": rng.normal(0, 1, 300),
    }


class TestTwoSampleTest:
    def test_different_distributions_rejects_energy(self, different_distributions):
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=200,
            seed=42,
        )
        assert result.p_value < 0.05

    def test_same_distribution_does_not_reject_energy(self, same_distribution):
        result = two_sample_test(
            same_distribution["p"],
            same_distribution["q"],
            method="energy",
            n_permutations=200,
            seed=42,
        )
        assert result.p_value > 0.01

    def test_different_distributions_rejects_mmd(self, different_distributions):
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="mmd",
            n_permutations=200,
            seed=42,
        )
        assert result.p_value < 0.05

    def test_result_type(self, different_distributions):
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=50,
            seed=42,
        )
        assert isinstance(result, TestResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert isinstance(result.null_distribution, np.ndarray)

    def test_null_distribution_shape(self, different_distributions):
        n_perm = 100
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=n_perm,
            seed=42,
        )
        assert len(result.null_distribution) == n_perm

    def test_p_value_bounds(self, different_distributions):
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=100,
            seed=42,
        )
        assert 0.0 <= result.p_value <= 1.0

    def test_reproducible_with_seed(self, different_distributions):
        r1 = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=50,
            seed=123,
        )
        r2 = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=50,
            seed=123,
        )
        assert r1.statistic == r2.statistic
        assert r1.p_value == r2.p_value
        np.testing.assert_array_equal(r1.null_distribution, r2.null_distribution)

    def test_invalid_method(self, different_distributions):
        with pytest.raises(ValueError, match="Unknown method"):
            two_sample_test(
                different_distributions["p"],
                different_distributions["q"],
                method="invalid",
            )

    def test_method_kl_knn(self, different_distributions):
        """kl_knn method should detect different distributions."""
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="kl_knn",
            n_permutations=100,
            seed=42,
            k=3,
        )
        assert result.p_value < 0.05


# ---------------------------------------------------------------------------
# Low-memory mode tests
# ---------------------------------------------------------------------------
class TestLowMemoryAutoDetect:
    """Test the auto-detection logic for low-memory mode."""

    def test_small_n_uses_precomputed(self):
        assert _should_use_low_memory(100, None) is False

    def test_large_n_uses_low_memory(self):
        # 50K combined: 50K x 50K x 8 = 20 GiB >> 1 GiB threshold
        assert _should_use_low_memory(50_000, None) is True

    def test_explicit_true_overrides(self):
        assert _should_use_low_memory(100, True) is True

    def test_explicit_false_overrides(self):
        assert _should_use_low_memory(50_000, False) is False


class TestLowMemoryEnergy:
    """Test that low_memory=True produces correct results for energy distance."""

    def test_matches_precomputed_different(self, different_distributions):
        """Low-memory and precomputed paths should give the same statistic."""
        r_pre = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=50,
            seed=42,
            low_memory=False,
        )
        r_low = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=50,
            seed=42,
            low_memory=True,
        )
        # Observed statistics should be very close (both compute energy distance)
        np.testing.assert_allclose(r_pre.statistic, r_low.statistic, rtol=1e-6)

    def test_matches_precomputed_same(self, same_distribution):
        r_pre = two_sample_test(
            same_distribution["p"],
            same_distribution["q"],
            method="energy",
            n_permutations=50,
            seed=42,
            low_memory=False,
        )
        r_low = two_sample_test(
            same_distribution["p"],
            same_distribution["q"],
            method="energy",
            n_permutations=50,
            seed=42,
            low_memory=True,
        )
        np.testing.assert_allclose(r_pre.statistic, r_low.statistic, rtol=1e-6)

    def test_detects_different_distributions(self, different_distributions):
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=200,
            seed=42,
            low_memory=True,
        )
        assert result.p_value < 0.05

    def test_does_not_reject_same_distribution(self, same_distribution):
        result = two_sample_test(
            same_distribution["p"],
            same_distribution["q"],
            method="energy",
            n_permutations=200,
            seed=42,
            low_memory=True,
        )
        assert result.p_value > 0.01

    def test_multivariate(self):
        """Low-memory energy distance works with multivariate data."""
        rng = np.random.default_rng(42)
        p = rng.normal(0, 1, (200, 3))
        q = rng.normal(1, 1, (200, 3))
        result = two_sample_test(
            p,
            q,
            method="energy",
            n_permutations=100,
            seed=42,
            low_memory=True,
        )
        assert result.p_value < 0.05

    def test_result_type(self, different_distributions):
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="energy",
            n_permutations=50,
            seed=42,
            low_memory=True,
        )
        assert isinstance(result, TestResult)
        assert isinstance(result.statistic, float)
        assert isinstance(result.p_value, float)
        assert len(result.null_distribution) == 50


class TestLowMemoryMMD:
    """Test that low_memory=True produces correct results for MMD."""

    def test_matches_precomputed(self, different_distributions):
        r_pre = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="mmd",
            n_permutations=50,
            seed=42,
            low_memory=False,
        )
        r_low = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="mmd",
            n_permutations=50,
            seed=42,
            low_memory=True,
        )
        # Statistics should be close (both compute MMD with same bandwidth)
        np.testing.assert_allclose(r_pre.statistic, r_low.statistic, rtol=0.05)

    def test_detects_different_distributions(self, different_distributions):
        result = two_sample_test(
            different_distributions["p"],
            different_distributions["q"],
            method="mmd",
            n_permutations=200,
            seed=42,
            low_memory=True,
        )
        assert result.p_value < 0.05

    def test_does_not_reject_same_distribution(self, same_distribution):
        result = two_sample_test(
            same_distribution["p"],
            same_distribution["q"],
            method="mmd",
            n_permutations=200,
            seed=42,
            low_memory=True,
        )
        assert result.p_value > 0.01
