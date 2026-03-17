"""End-to-end integration tests using emcee as a real MCMC backend.

These tests run actual Bayesian estimation via emcee, convert the chains
to ArviZ format, and then compute divergence measures.  This catches
compatibility issues between divergence and real-world ArviZ objects that
unit tests with synthetic az.from_dict() data miss (e.g., dimension
ordering, InferenceData wrapping, Dataset vs DataTree nodes).

emcee is an optional test dependency — these tests are skipped if emcee
is not installed.
"""

import numpy as np
import pytest

az = pytest.importorskip("arviz")
emcee = pytest.importorskip("emcee")

from divergence.bayesian import (  # noqa: E402
    chain_divergence,
    chain_ksd,
    chain_two_sample_test,
    information_gain,
    mixing_diagnostic,
)


# ---------------------------------------------------------------------------
# Fixtures: run a real MCMC estimation
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def y_obs():
    """Observed data: 30 draws from N(5, 2)."""
    return np.random.default_rng(42).normal(5, 2, 30)


@pytest.fixture(scope="module")
def emcee_idata(y_obs):
    """Run emcee on a simple Gaussian model and return ArviZ DataTree.

    Model: y_i ~ N(mu, sigma^2)
    Priors: mu ~ N(0, 10^2), log_sigma ~ N(0, 2^2)
    """
    n_obs = len(y_obs)

    def log_prob(theta):
        mu, log_sigma = theta
        sigma = np.exp(log_sigma)
        if sigma < 0.01:
            return -np.inf
        ll = -0.5 * np.sum((y_obs - mu) ** 2 / sigma**2) - n_obs * np.log(sigma)
        lp = -0.5 * (mu**2 / 100) - 0.5 * (log_sigma**2 / 4)
        return ll + lp

    rng = np.random.default_rng(42)
    ndim, nwalkers = 2, 8
    p0 = rng.normal(size=(nwalkers, ndim))
    p0[:, 0] += 5  # near true mu
    p0[:, 1] += np.log(2)  # near true log_sigma

    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_prob)
    # Burn-in
    state = sampler.run_mcmc(p0, 200, progress=False)
    sampler.reset()
    # Production
    sampler.run_mcmc(state, 500, progress=False)

    return az.from_emcee(sampler, ["mu", "log_sigma"])


@pytest.fixture(scope="module")
def emcee_idata_with_prior(emcee_idata):
    """Add a synthetic prior group to the emcee idata.

    emcee doesn't produce prior samples natively, so we attach them
    manually using az.from_dict and DataTree merge.
    """
    rng = np.random.default_rng(99)

    # Get the actual posterior shape to match chain/draw structure
    post = emcee_idata["posterior"]
    n_draws = post["mu"].sizes.get("draw", post["mu"].sizes.get("chain", 500))
    n_chains = post["mu"].sizes.get("chain", post["mu"].sizes.get("draw", 8))

    # Create a separate DataTree with prior and merge
    prior_dt = az.from_dict(
        {
            "prior": {
                "mu": rng.normal(0, 10, (n_chains, n_draws)),
                "log_sigma": rng.normal(0, 2, (n_chains, n_draws)),
            }
        }
    )

    # Merge prior into the emcee idata
    import xarray as xr

    merged = xr.DataTree(children={})
    for child_name in emcee_idata.children:
        merged[child_name] = emcee_idata[child_name]
    for child_name in prior_dt.children:
        merged[child_name] = prior_dt[child_name]

    return merged


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------
class TestEmceeChainDivergence:
    """Test chain_divergence with real emcee chains."""

    def test_runs_without_error(self, emcee_idata):
        """chain_divergence should work on emcee output."""
        result = chain_divergence(emcee_idata, var_names=["mu"])
        assert "mu" in result
        assert result["mu"].shape[0] == result["mu"].shape[1]

    def test_diagonal_near_zero(self, emcee_idata):
        """Diagonal should be near zero (chain vs itself)."""
        result = chain_divergence(emcee_idata, var_names=["mu"])
        np.testing.assert_allclose(np.diag(result["mu"]), 0.0, atol=0.01)

    def test_multiple_variables(self, emcee_idata):
        """Should work with all variables."""
        result = chain_divergence(emcee_idata)
        assert "mu" in result
        assert "log_sigma" in result


class TestEmceeInformationGain:
    """Test information_gain with emcee chains + synthetic prior."""

    def test_runs_with_prior(self, emcee_idata_with_prior):
        """information_gain should work with merged emcee + prior data."""
        result = information_gain(emcee_idata_with_prior, var_names=["mu"])
        assert "mu" in result
        assert result["mu"] > 0  # posterior should differ from prior


class TestEmceeChainKSD:
    """Test chain_ksd with real emcee chains."""

    def test_runs_without_error(self, emcee_idata):
        """chain_ksd should work on emcee output with a simple score."""
        # Use a deliberately wrong score — just checking it runs
        result = chain_ksd(
            emcee_idata,
            lambda x: -x,  # N(0,1) score — wrong for this posterior
            var_names=["mu"],
            kernel="imq",
            split=False,
        )
        assert "mu" in result
        assert result["mu"].ksd_per_chain.shape[0] > 0
        assert isinstance(result["mu"].ksd_pooled, float)

    def test_split_mode(self, emcee_idata):
        """Split KSD should produce first/second half arrays."""
        result = chain_ksd(
            emcee_idata,
            lambda x: -x,
            var_names=["mu"],
            split=True,
        )
        r = result["mu"]
        assert r.ksd_split_first is not None
        assert r.ksd_split_second is not None
        assert r.ksd_split_first.shape == r.ksd_per_chain.shape


class TestEmceeChainTwoSampleTest:
    """Test chain_two_sample_test with real emcee chains."""

    def test_runs_without_error(self, emcee_idata):
        """Permutation test should complete on emcee chains."""
        result = chain_two_sample_test(
            emcee_idata,
            var_names=["mu"],
            n_permutations=50,
            seed=42,
        )
        assert "mu" in result
        r = result["mu"]
        n = r.p_value_matrix.shape[0]
        assert r.p_value_matrix.shape == (n, n)
        assert r.statistic_matrix.shape == (n, n)

    def test_diagonal_pvalues_one(self, emcee_idata):
        """Diagonal p-values should be 1.0."""
        result = chain_two_sample_test(
            emcee_idata,
            var_names=["mu"],
            n_permutations=50,
            seed=42,
        )
        np.testing.assert_array_equal(np.diag(result["mu"].p_value_matrix), 1.0)


class TestEmceeMixingDiagnostic:
    """Test mixing_diagnostic with real emcee chains."""

    def test_runs_without_error(self, emcee_idata):
        """Mixing diagnostic should complete on emcee chains."""
        result = mixing_diagnostic(emcee_idata, var_names=["mu"])
        assert "mu" in result
        r = result["mu"]
        assert r.stationarity_te.shape[0] > 0

    def test_output_shapes(self, emcee_idata):
        """Output arrays should have correct shapes."""
        result = mixing_diagnostic(emcee_idata, var_names=["mu"])
        r = result["mu"]
        n_chains = r.stationarity_te.shape[0]
        assert r.cross_chain_te.shape == (max(n_chains - 1, 0),)
