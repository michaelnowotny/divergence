"""Tests for two-sample hypothesis testing."""

import numpy as np
import pytest

from divergence._types import TestResult
from divergence.testing import two_sample_test


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
