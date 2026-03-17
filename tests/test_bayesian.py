"""Tests for ArviZ Bayesian integration module.

All tests use synthetic InferenceData created with az.from_dict() —
no dependency on PyMC, Stan, or NumPyro.
"""

import numpy as np
import pytest

az = pytest.importorskip("arviz")

from divergence.bayesian import (  # noqa: E402
    bayesian_surprise,
    chain_divergence,
    information_gain,
    model_divergence,
    prior_sensitivity,
    uncertainty_decomposition,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def simple_idata(rng):
    """Clear prior-to-posterior shift: prior N(0,2), posterior N(3,0.5)."""
    return az.from_dict(
        {
            "posterior": {
                "mu": rng.normal(3.0, 0.5, (4, 1000)),
                "sigma": np.abs(rng.normal(0.3, 0.1, (4, 1000))),
            },
            "prior": {
                "mu": rng.normal(0.0, 2.0, (4, 1000)),
                "sigma": np.abs(rng.normal(1.0, 0.5, (4, 1000))),
            },
        }
    )


@pytest.fixture(scope="module")
def vector_param_idata(rng):
    """InferenceData with a vector parameter (theta with 5 components)."""
    return az.from_dict(
        {
            "posterior": {
                "theta": rng.dirichlet(np.ones(5) * 5, size=(4, 500)),
            },
            "prior": {
                "theta": rng.dirichlet(np.ones(5), size=(4, 500)),
            },
        }
    )


@pytest.fixture(scope="module")
def multichain_idata(rng):
    """4 chains: 2 converged around 0, 2 offset to 3."""
    chains = np.empty((4, 1000))
    chains[0] = rng.normal(0, 1, 1000)
    chains[1] = rng.normal(0, 1, 1000)
    chains[2] = rng.normal(3, 1, 1000)  # offset
    chains[3] = rng.normal(3, 1, 1000)  # offset
    return az.from_dict({"posterior": {"mu": chains}})


@pytest.fixture(scope="module")
def predictive_idata(rng):
    """Posterior predictive with 50 observations per draw."""
    return az.from_dict(
        {
            "posterior_predictive": {
                "y": rng.normal(0, 1, (2, 500, 50)),
            },
        }
    )


@pytest.fixture(scope="module")
def log_likelihood_idata(rng):
    """Log-likelihood with one outlier observation (index 0)."""
    ll = rng.standard_normal((2, 500, 100)) * 0.5
    ll[:, :, 0] = -10.0  # outlier: very low log-likelihood
    return az.from_dict({"log_likelihood": {"y": ll}})


# ---------------------------------------------------------------------------
# TestInformationGain
# ---------------------------------------------------------------------------
class TestInformationGain:
    def test_positive_with_shifted_posterior(self, simple_idata):
        result = information_gain(simple_idata)
        assert result["mu"] > 0

    def test_near_zero_with_same_distribution(self, rng):
        samples = rng.normal(0, 1, (4, 1000))
        idata = az.from_dict(
            {
                "posterior": {"mu": samples},
                "prior": {"mu": samples.copy()},
            }
        )
        result = information_gain(idata)
        assert abs(result["mu"]) < 0.5

    def test_var_names_filtering(self, simple_idata):
        result = information_gain(simple_idata, var_names=["mu"])
        assert "mu" in result
        assert "sigma" not in result

    @pytest.mark.parametrize("method", ["kl", "js", "hellinger", "tv", "energy", "mmd"])
    def test_all_methods(self, simple_idata, method):
        result = information_gain(simple_idata, method=method, var_names=["mu"])
        assert isinstance(result["mu"], float)
        assert np.isfinite(result["mu"])

    def test_vector_parameter_componentwise(self, vector_param_idata):
        result = information_gain(vector_param_idata, var_names=["theta"])
        assert isinstance(result["theta"], np.ndarray)
        assert result["theta"].shape == (5,)

    def test_vector_parameter_multivariate(self, vector_param_idata):
        result = information_gain(
            vector_param_idata,
            var_names=["theta"],
            method="mmd",
            multivariate=True,
        )
        assert isinstance(result["theta"], float)

    def test_missing_group_raises(self, rng):
        idata = az.from_dict({"posterior": {"mu": rng.normal(0, 1, (2, 500))}})
        with pytest.raises(ValueError, match="does not contain group 'prior'"):
            information_gain(idata)

    def test_invalid_method_raises(self, simple_idata):
        with pytest.raises(ValueError, match="Unknown divergence method"):
            information_gain(simple_idata, method="invalid")

    def test_missing_var_raises(self, simple_idata):
        with pytest.raises(KeyError, match="nonexistent"):
            information_gain(simple_idata, var_names=["nonexistent"])


# ---------------------------------------------------------------------------
# TestChainDivergence
# ---------------------------------------------------------------------------
class TestChainDivergence:
    def test_output_shape(self, multichain_idata):
        result = chain_divergence(multichain_idata)
        assert result["mu"].shape == (4, 4)

    def test_diagonal_near_zero(self, multichain_idata):
        result = chain_divergence(multichain_idata)
        np.testing.assert_allclose(np.diag(result["mu"]), 0.0, atol=1e-10)

    def test_symmetric_matrix(self, multichain_idata):
        result = chain_divergence(multichain_idata)
        np.testing.assert_allclose(result["mu"], result["mu"].T, rtol=0.01)

    def test_divergent_chains_detected(self, multichain_idata):
        """Chains 0,1 (around 0) vs chains 2,3 (around 3) should show large divergence."""
        result = chain_divergence(multichain_idata)
        # Within-group divergence should be small
        within = result["mu"][0, 1]
        # Between-group divergence should be large
        between = result["mu"][0, 2]
        assert between > within * 2

    def test_converged_chains_small(self, rng):
        """All chains from the same distribution should have small divergence."""
        idata = az.from_dict({"posterior": {"mu": rng.normal(0, 1, (4, 1000))}})
        result = chain_divergence(idata)
        # All off-diagonal should be small
        off_diag = result["mu"][np.triu_indices(4, k=1)]
        assert np.all(off_diag < 0.5)


# ---------------------------------------------------------------------------
# TestBayesianSurprise
# ---------------------------------------------------------------------------
class TestBayesianSurprise:
    def test_outlier_has_highest_surprise(self, log_likelihood_idata):
        result = bayesian_surprise(log_likelihood_idata)
        surprise = result["y"]
        # Observation 0 (the outlier) should have the highest surprise
        assert np.argmax(surprise) == 0

    def test_output_shape(self, log_likelihood_idata):
        result = bayesian_surprise(log_likelihood_idata)
        assert result["y"].shape == (100,)

    def test_all_finite(self, log_likelihood_idata):
        result = bayesian_surprise(log_likelihood_idata)
        assert np.all(np.isfinite(result["y"]))

    def test_missing_group_raises(self, rng):
        idata = az.from_dict({"posterior": {"mu": rng.normal(0, 1, (2, 500))}})
        with pytest.raises(ValueError, match="does not contain group"):
            bayesian_surprise(idata)

    def test_var_name_selection(self, log_likelihood_idata):
        result = bayesian_surprise(log_likelihood_idata, var_name="y")
        assert "y" in result


# ---------------------------------------------------------------------------
# TestUncertaintyDecomposition
# ---------------------------------------------------------------------------
class TestUncertaintyDecomposition:
    def test_returns_three_components(self, predictive_idata):
        result = uncertainty_decomposition(predictive_idata)
        assert set(result["y"].keys()) == {"total", "aleatoric", "epistemic"}

    def test_total_equals_sum(self, predictive_idata):
        result = uncertainty_decomposition(predictive_idata)
        r = result["y"]
        assert r["total"] == pytest.approx(r["aleatoric"] + r["epistemic"], abs=0.01)

    def test_epistemic_nonnegative(self, predictive_idata):
        result = uncertainty_decomposition(predictive_idata)
        # Epistemic should be non-negative (or very close to 0)
        assert result["y"]["epistemic"] >= -0.5

    def test_insufficient_obs_raises(self, rng):
        """Only 3 observations per draw — too few."""
        idata = az.from_dict(
            {"posterior_predictive": {"y": rng.normal(0, 1, (2, 500, 3))}}
        )
        with pytest.raises(ValueError, match="at least 10"):
            uncertainty_decomposition(idata)

    def test_missing_obs_dim_raises(self, rng):
        """Scalar prediction (no obs dim) should raise."""
        idata = az.from_dict(
            {"posterior_predictive": {"y": rng.normal(0, 1, (2, 500))}}
        )
        with pytest.raises(ValueError, match="3 dimensions"):
            uncertainty_decomposition(idata)


# ---------------------------------------------------------------------------
# TestPriorSensitivity
# ---------------------------------------------------------------------------
class TestPriorSensitivity:
    def test_returns_correct_keys(self, simple_idata, rng):
        ref = {"mu": rng.normal(0, 10, 5000)}
        result = prior_sensitivity(simple_idata, ref, var_names=["mu"])
        assert set(result["mu"].keys()) == {"actual", "reference", "sensitivity"}

    def test_sensitivity_nonnegative(self, simple_idata, rng):
        ref = {"mu": rng.normal(0, 10, 5000)}
        result = prior_sensitivity(simple_idata, ref, var_names=["mu"])
        assert result["mu"]["sensitivity"] >= 0

    def test_same_prior_low_sensitivity(self, rng):
        """When reference prior = actual prior, sensitivity should be near 0."""
        prior_samples = rng.normal(0, 1, (2, 1000))
        idata = az.from_dict(
            {
                "posterior": {"mu": rng.normal(1, 0.5, (2, 1000))},
                "prior": {"mu": prior_samples},
            }
        )
        # Use the same samples as reference
        ref = {"mu": prior_samples.ravel()}
        result = prior_sensitivity(idata, ref, var_names=["mu"])
        assert result["mu"]["sensitivity"] < 0.5


# ---------------------------------------------------------------------------
# TestModelDivergence
# ---------------------------------------------------------------------------
class TestModelDivergence:
    def test_same_model_near_zero(self, rng):
        samples = rng.normal(0, 1, (2, 500))
        idata1 = az.from_dict({"posterior_predictive": {"y": samples}})
        idata2 = az.from_dict({"posterior_predictive": {"y": samples.copy()}})
        result = model_divergence(idata1, idata2)
        assert result["y"] < 0.1

    def test_different_models_positive(self, rng):
        idata1 = az.from_dict(
            {"posterior_predictive": {"y": rng.normal(0, 1, (2, 500))}}
        )
        idata2 = az.from_dict(
            {"posterior_predictive": {"y": rng.normal(3, 1, (2, 500))}}
        )
        result = model_divergence(idata1, idata2)
        assert result["y"] > 0.1

    def test_missing_group_raises(self, rng):
        idata1 = az.from_dict({"posterior": {"mu": rng.normal(0, 1, (2, 500))}})
        idata2 = az.from_dict({"posterior": {"mu": rng.normal(0, 1, (2, 500))}})
        with pytest.raises(ValueError, match="does not contain group"):
            model_divergence(idata1, idata2)
