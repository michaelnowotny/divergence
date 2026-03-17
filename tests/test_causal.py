"""Tests for causal/temporal information measures.

Validates transfer entropy on coupled time series where the causal direction
is known by construction.
"""

import numpy as np
import pytest

from divergence.causal import transfer_entropy


@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def n_timesteps():
    return 3000


@pytest.fixture(scope="module")
def coupled_xy(rng, n_timesteps):
    """X causes Y: Y_t = 0.8 * X_{t-1} + noise."""
    x = rng.standard_normal(n_timesteps)
    y = np.zeros(n_timesteps)
    y[0] = rng.standard_normal()
    for t in range(1, n_timesteps):
        y[t] = 0.8 * x[t - 1] + 0.2 * rng.standard_normal()
    return x, y


@pytest.fixture(scope="module")
def independent_xy(rng, n_timesteps):
    """Two independent time series."""
    x = rng.standard_normal(n_timesteps)
    y = rng.standard_normal(n_timesteps)
    return x, y


# ===========================================================================
# Transfer Entropy
# ===========================================================================
class TestTransferEntropy:
    """Tests for transfer_entropy."""

    def test_causal_direction_positive(self, coupled_xy):
        """TE_{X->Y} should be positive when X causes Y."""
        x, y = coupled_xy
        te_xy = transfer_entropy(source=x, target=y, k=1, lag=1)
        assert te_xy > 0.1, f"Expected positive TE_{{X->Y}}, got {te_xy}"

    def test_causal_asymmetry(self, coupled_xy):
        """TE_{X->Y} should be much larger than TE_{Y->X}."""
        x, y = coupled_xy
        te_xy = transfer_entropy(source=x, target=y, k=1, lag=1)
        te_yx = transfer_entropy(source=y, target=x, k=1, lag=1)
        assert te_xy > te_yx + 0.1, (
            f"Expected TE_{{X->Y}} >> TE_{{Y->X}}, got {te_xy:.3f} vs {te_yx:.3f}"
        )

    def test_independent_near_zero(self, independent_xy):
        """TE between independent series should be near 0."""
        x, y = independent_xy
        te = transfer_entropy(source=x, target=y, k=1, lag=1)
        np.testing.assert_allclose(te, 0.0, atol=0.15)

    def test_base_scaling(self, coupled_xy):
        """TE in bits should equal TE in nats / ln(2)."""
        x, y = coupled_xy
        te_nats = transfer_entropy(source=x, target=y, base=np.e)
        te_bits = transfer_entropy(source=x, target=y, base=2.0)
        np.testing.assert_allclose(te_bits, te_nats / np.log(2), rtol=1e-6)

    def test_higher_embedding(self, coupled_xy):
        """Higher embedding dimension should still detect causality."""
        x, y = coupled_xy
        te = transfer_entropy(source=x, target=y, k=2, lag=2)
        assert te > 0.05, f"Expected positive TE with k=2,l=2, got {te}"

    def test_mismatched_lengths_raises(self):
        """Different-length series should raise ValueError."""
        with pytest.raises(ValueError, match="same length"):
            transfer_entropy(np.ones(100), np.ones(50))

    def test_too_short_raises(self):
        """Very short series should raise ValueError."""
        with pytest.raises(ValueError, match="too short"):
            transfer_entropy(np.ones(3), np.ones(3), k=2, lag=2)

    def test_symmetric_noise(self, rng, n_timesteps):
        """For symmetric coupling, both TE values should be similar."""
        x = np.zeros(n_timesteps)
        y = np.zeros(n_timesteps)
        x[0] = rng.standard_normal()
        y[0] = rng.standard_normal()
        for t in range(1, n_timesteps):
            x[t] = 0.5 * y[t - 1] + 0.5 * rng.standard_normal()
            y[t] = 0.5 * x[t - 1] + 0.5 * rng.standard_normal()
        te_xy = transfer_entropy(source=x, target=y, k=1, lag=1)
        te_yx = transfer_entropy(source=y, target=x, k=1, lag=1)
        # Both should be positive and roughly equal
        assert te_xy > 0.05
        assert te_yx > 0.05
        np.testing.assert_allclose(te_xy, te_yx, rtol=0.5)
