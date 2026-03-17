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
    chain_ksd,
    chain_two_sample_test,
    information_gain,
    mixing_diagnostic,
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


# ---------------------------------------------------------------------------
# Convergence Diagnostic Fixtures
# ---------------------------------------------------------------------------
def _standard_normal_score(x):
    """Score function for N(0, 1): s(x) = -x."""
    return -x


@pytest.fixture(scope="module")
def converged_chains_idata():
    """4 chains all from N(0, 1) — well-mixed, independent RNG."""
    _rng = np.random.default_rng(123)
    return az.from_dict({"posterior": {"mu": _rng.normal(0, 1, (4, 500))}})


@pytest.fixture(scope="module")
def divergent_chains_idata(rng):
    """4 chains: chains 0,1 from N(0,1), chains 2,3 from N(5,1)."""
    chains = np.empty((4, 500))
    chains[0] = rng.normal(0, 1, 500)
    chains[1] = rng.normal(0, 1, 500)
    chains[2] = rng.normal(5, 1, 500)
    chains[3] = rng.normal(5, 1, 500)
    return az.from_dict({"posterior": {"mu": chains}})


@pytest.fixture(scope="module")
def nonstationary_idata(rng):
    """Chains with a linear trend: mean drifts from 0 to 5 over the run."""
    n_draws = 500
    t = np.linspace(0, 5, n_draws)
    chains = np.empty((2, n_draws))
    for c in range(2):
        chains[c] = t + 0.3 * rng.standard_normal(n_draws)
    return az.from_dict({"posterior": {"mu": chains}})


@pytest.fixture(scope="module")
def single_chain_idata(rng):
    """Single chain for edge case testing."""
    return az.from_dict({"posterior": {"mu": rng.normal(0, 1, (1, 500))}})


# ---------------------------------------------------------------------------
# TestChainKSD
# ---------------------------------------------------------------------------
class TestChainKSD:
    """Tests for chain_ksd convergence diagnostic."""

    def test_matching_score_low_ksd(self, converged_chains_idata):
        """Chains from N(0,1) with correct score -> KSD near 0."""
        result = chain_ksd(
            converged_chains_idata, _standard_normal_score, var_names=["mu"]
        )
        for v in result["mu"].ksd_per_chain:
            assert v < 0.15

    def test_wrong_score_positive_ksd(self, converged_chains_idata):
        """N(0,1) chains with N(5,1) score -> positive KSD."""
        wrong_score = lambda x: -(x - 5.0)  # noqa: E731
        result = chain_ksd(converged_chains_idata, wrong_score, var_names=["mu"])
        assert result["mu"].ksd_pooled > 0.5

    def test_split_produces_arrays(self, converged_chains_idata):
        """split=True should fill ksd_split_first and ksd_split_second."""
        result = chain_ksd(converged_chains_idata, _standard_normal_score, split=True)
        r = result["mu"]
        assert r.ksd_split_first is not None
        assert r.ksd_split_second is not None
        assert r.ksd_split_first.shape == (4,)
        assert r.ksd_split_second.shape == (4,)

    def test_no_split(self, converged_chains_idata):
        """split=False should return None for split fields."""
        result = chain_ksd(converged_chains_idata, _standard_normal_score, split=False)
        assert result["mu"].ksd_split_first is None
        assert result["mu"].ksd_split_second is None

    def test_single_chain(self, single_chain_idata):
        """Should work with a single chain."""
        result = chain_ksd(single_chain_idata, _standard_normal_score)
        assert result["mu"].ksd_per_chain.shape == (1,)
        assert isinstance(result["mu"].ksd_pooled, float)

    def test_rbf_kernel(self, converged_chains_idata):
        """RBF kernel should also work."""
        result = chain_ksd(
            converged_chains_idata,
            _standard_normal_score,
            kernel="rbf",
            var_names=["mu"],
        )
        assert all(np.isfinite(result["mu"].ksd_per_chain))


# ---------------------------------------------------------------------------
# TestChainTwoSampleTest
# ---------------------------------------------------------------------------
class TestChainTwoSampleTest:
    """Tests for chain_two_sample_test convergence diagnostic."""

    def test_converged_high_pvalues(self, converged_chains_idata):
        """Well-mixed chains should have high p-values."""
        result = chain_two_sample_test(
            converged_chains_idata,
            var_names=["mu"],
            n_permutations=200,
            seed=42,
        )
        assert result["mu"].min_p_value > 0.01
        assert not result["mu"].any_significant

    def test_divergent_low_pvalues(self, divergent_chains_idata):
        """Divergent chains should have low p-values for cross-group pairs."""
        result = chain_two_sample_test(
            divergent_chains_idata,
            var_names=["mu"],
            n_permutations=200,
            seed=42,
        )
        assert result["mu"].any_significant
        assert result["mu"].min_p_value < 0.05

    def test_matrix_shape(self, converged_chains_idata):
        """Output matrices should be (n_chains, n_chains)."""
        result = chain_two_sample_test(
            converged_chains_idata,
            var_names=["mu"],
            n_permutations=50,
            seed=42,
        )
        assert result["mu"].p_value_matrix.shape == (4, 4)
        assert result["mu"].statistic_matrix.shape == (4, 4)

    def test_matrix_symmetry(self, converged_chains_idata):
        """P-value and statistic matrices should be symmetric."""
        result = chain_two_sample_test(
            converged_chains_idata,
            var_names=["mu"],
            n_permutations=50,
            seed=42,
        )
        np.testing.assert_array_equal(
            result["mu"].p_value_matrix, result["mu"].p_value_matrix.T
        )

    def test_diagonal_pvalues_one(self, converged_chains_idata):
        """Diagonal p-values should be 1.0."""
        result = chain_two_sample_test(
            converged_chains_idata,
            var_names=["mu"],
            n_permutations=50,
            seed=42,
        )
        np.testing.assert_array_equal(np.diag(result["mu"].p_value_matrix), 1.0)

    def test_energy_method(self, divergent_chains_idata):
        """Energy distance method should also detect divergent chains."""
        result = chain_two_sample_test(
            divergent_chains_idata,
            var_names=["mu"],
            method="energy",
            n_permutations=200,
            seed=42,
        )
        assert result["mu"].any_significant


# ---------------------------------------------------------------------------
# TestMixingDiagnostic
# ---------------------------------------------------------------------------
class TestMixingDiagnostic:
    """Tests for mixing_diagnostic convergence diagnostic."""

    def test_stationary_low_te(self, converged_chains_idata):
        """Stationary chains should have low stationarity TE."""
        result = mixing_diagnostic(converged_chains_idata, var_names=["mu"])
        for te in result["mu"].stationarity_te:
            assert te < 0.3

    def test_nonstationary_higher_cross_te(self, nonstationary_idata):
        """Non-stationary chains sharing a trend should have elevated cross-chain TE."""
        result = mixing_diagnostic(nonstationary_idata, var_names=["mu"])
        # Chains sharing a common trend have high cross-chain TE
        assert np.mean(result["mu"].cross_chain_te) > 0.05

    def test_independent_low_cross_te(self, converged_chains_idata):
        """Independent chains should have low cross-chain TE."""
        result = mixing_diagnostic(converged_chains_idata, var_names=["mu"])
        for te in result["mu"].cross_chain_te:
            assert te < 0.3

    def test_output_shapes(self, converged_chains_idata):
        """Output arrays should have correct shapes."""
        result = mixing_diagnostic(converged_chains_idata, var_names=["mu"])
        assert result["mu"].stationarity_te.shape == (4,)
        assert result["mu"].cross_chain_te.shape == (3,)

    def test_single_chain_cross_te(self, single_chain_idata):
        """Single chain should produce empty cross-chain TE."""
        result = mixing_diagnostic(single_chain_idata, var_names=["mu"])
        assert result["mu"].cross_chain_te.shape == (0,)
        assert result["mu"].stationarity_te.shape == (1,)
