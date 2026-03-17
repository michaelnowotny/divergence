"""Property-based tests using Hypothesis.

Tests mathematical properties that must hold for ALL valid inputs,
not just specific examples. This catches edge cases that hand-written
tests miss.
"""

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from divergence import (
    conditional_entropy_from_samples,
    cross_entropy_from_samples,
    discrete_conditional_entropy_of_y_given_x,
    discrete_cross_entropy,
    discrete_entropy,
    discrete_joint_entropy,
    discrete_mutual_information,
    discrete_relative_entropy,
    entropy_from_samples,
    jensen_shannon_divergence_from_samples,
    joint_entropy_from_samples,
    mutual_information_from_samples,
    relative_entropy_from_samples,
)

# ---------------------------------------------------------------------------
# Strategies for generating test data
# ---------------------------------------------------------------------------

# Discrete samples: arrays of integers with at least 2 elements, at least 2 unique values
discrete_sample = st.builds(
    lambda vals, n: np.random.default_rng(abs(hash(tuple(vals))) % 2**32).choice(
        vals, size=n
    ),
    vals=st.lists(st.integers(0, 10), min_size=2, max_size=6, unique=True),
    n=st.integers(min_value=20, max_value=100),
)


# Paired discrete samples with guaranteed shared support
@st.composite
def paired_discrete_samples(draw):
    n = draw(st.integers(min_value=20, max_value=100))
    k = draw(st.integers(min_value=2, max_value=6))
    vals = list(range(k))
    rng = np.random.default_rng(draw(st.integers(0, 2**32 - 1)))
    sample_p = rng.choice(vals, size=n)
    sample_q = rng.choice(vals, size=n)
    return sample_p, sample_q


# Continuous samples from normal distributions
@st.composite
def normal_sample(draw):
    mu = draw(st.floats(min_value=-5, max_value=5))
    sigma = draw(st.floats(min_value=0.5, max_value=5.0))
    n = draw(st.integers(min_value=500, max_value=2000))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    return rng.normal(mu, sigma, n)


@st.composite
def paired_normal_samples(draw):
    n = draw(st.integers(min_value=2000, max_value=5000))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    mu_p = draw(st.floats(min_value=-3, max_value=3))
    sigma_p = draw(st.floats(min_value=1.0, max_value=3.0))
    mu_q = draw(st.floats(min_value=-3, max_value=3))
    sigma_q = draw(st.floats(min_value=1.0, max_value=3.0))
    sample_p = rng.normal(mu_p, sigma_p, n)
    sample_q = rng.normal(mu_q, sigma_q, n)
    return sample_p, sample_q


@st.composite
def correlated_bivariate_samples(draw):
    n = draw(st.integers(min_value=500, max_value=2000))
    seed = draw(st.integers(0, 2**32 - 1))
    rng = np.random.default_rng(seed)
    rho = draw(st.floats(min_value=-0.9, max_value=0.9))
    z = rng.standard_normal(n)
    x = z
    y = rho * z + np.sqrt(1 - rho**2) * rng.standard_normal(n)
    return x, y, rho


# ============================================================================
# DISCRETE PROPERTY TESTS
# ============================================================================
class TestDiscreteEntropy:
    @given(sample=discrete_sample)
    @settings(max_examples=30, deadline=None)
    def test_nonnegative(self, sample):
        """H(X) >= 0 for discrete distributions."""
        assert discrete_entropy(sample) >= -1e-10

    @given(data=st.data())
    @settings(max_examples=20, deadline=None)
    def test_maximum_at_uniform(self, data):
        """H(X) <= log(k) where k is the number of categories."""
        k = data.draw(st.integers(min_value=2, max_value=8))
        n = data.draw(st.integers(min_value=100, max_value=500))
        sample = np.repeat(np.arange(k), n // k)
        h = discrete_entropy(sample)
        assert h <= np.log(k) + 0.01

    @given(data=st.data())
    @settings(max_examples=20, deadline=None)
    def test_uniform_entropy_equals_log_k(self, data):
        """H(uniform over k) = log(k)."""
        k = data.draw(st.integers(min_value=2, max_value=8))
        sample = np.tile(np.arange(k), 100)
        h = discrete_entropy(sample)
        assert np.isclose(h, np.log(k), rtol=1e-10)

    @given(base=st.sampled_from([np.e, 2.0, 10.0]))
    @settings(max_examples=3, deadline=None)
    def test_base_scaling(self, base):
        """Entropy in different bases scales by change-of-base factor."""
        sample = np.array([1, 1, 2, 2, 3, 3, 3, 4])
        h_nats = discrete_entropy(sample, base=np.e)
        h_base = discrete_entropy(sample, base=base)
        assert np.isclose(h_base, h_nats / np.log(base), rtol=1e-10)


class TestDiscreteRelativeEntropy:
    @given(samples=paired_discrete_samples())
    @settings(max_examples=30, deadline=None)
    def test_nonnegative(self, samples):
        """D_KL(P||Q) >= 0 (Gibbs' inequality)."""
        sample_p, sample_q = samples
        try:
            kl = discrete_relative_entropy(sample_p, sample_q)
            assert kl >= -1e-10
        except ValueError:
            pass  # q(x)=0 where p(x)>0 is expected to raise

    @given(sample=discrete_sample)
    @settings(max_examples=20, deadline=None)
    def test_identity(self, sample):
        """D_KL(P||P) = 0."""
        kl = discrete_relative_entropy(sample, sample)
        assert np.isclose(kl, 0.0, atol=1e-10)

    @given(samples=paired_discrete_samples())
    @settings(max_examples=20, deadline=None)
    def test_kl_equals_cross_entropy_minus_entropy(self, samples):
        """D_KL(P||Q) = H_Q(P) - H(P)."""
        sample_p, sample_q = samples
        try:
            kl = discrete_relative_entropy(sample_p, sample_q)
            ce = discrete_cross_entropy(sample_p, sample_q)
            h = discrete_entropy(sample_p)
            assert np.isclose(kl, ce - h, rtol=1e-10)
        except ValueError:
            pass


class TestDiscreteMutualInformation:
    @given(data=st.data())
    @settings(max_examples=20, deadline=None)
    def test_nonnegative(self, data):
        """I(X;Y) >= 0."""
        n = data.draw(st.integers(min_value=20, max_value=100))
        seed = data.draw(st.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        x = rng.choice(4, size=n)
        y = rng.choice(4, size=n)
        mi = discrete_mutual_information(x, y)
        assert mi >= -1e-10

    @given(data=st.data())
    @settings(max_examples=20, deadline=None)
    def test_symmetric(self, data):
        """I(X;Y) = I(Y;X)."""
        n = data.draw(st.integers(min_value=20, max_value=100))
        seed = data.draw(st.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        x = rng.choice(4, size=n)
        y = rng.choice(4, size=n)
        assert np.isclose(
            discrete_mutual_information(x, y),
            discrete_mutual_information(y, x),
            rtol=1e-10,
        )

    @given(sample=discrete_sample)
    @settings(max_examples=20, deadline=None)
    def test_mi_of_self_equals_entropy(self, sample):
        """I(X;X) = H(X)."""
        mi = discrete_mutual_information(sample, sample)
        h = discrete_entropy(sample)
        assert np.isclose(mi, h, rtol=1e-10)


class TestDiscreteChainRule:
    @given(data=st.data())
    @settings(max_examples=20, deadline=None)
    def test_joint_equals_marginal_plus_conditional(self, data):
        """H(X,Y) = H(X) + H(Y|X)."""
        n = data.draw(st.integers(min_value=30, max_value=100))
        seed = data.draw(st.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        x = rng.choice(4, size=n)
        y = rng.choice(4, size=n)
        h_xy = discrete_joint_entropy(x, y)
        h_x = discrete_entropy(x)
        h_y_given_x = discrete_conditional_entropy_of_y_given_x(x, y)
        assert np.isclose(h_xy, h_x + h_y_given_x, rtol=1e-10)

    @given(data=st.data())
    @settings(max_examples=20, deadline=None)
    def test_mi_equals_entropy_minus_conditional(self, data):
        """I(X;Y) = H(Y) - H(Y|X)."""
        n = data.draw(st.integers(min_value=30, max_value=100))
        seed = data.draw(st.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        x = rng.choice(4, size=n)
        y = rng.choice(4, size=n)
        mi = discrete_mutual_information(x, y)
        h_y = discrete_entropy(y)
        h_y_given_x = discrete_conditional_entropy_of_y_given_x(x, y)
        assert np.isclose(mi, h_y - h_y_given_x, rtol=1e-10)


class TestDiscreteAdditivity:
    @given(data=st.data())
    @settings(max_examples=20, deadline=None)
    def test_independent_joint_entropy(self, data):
        """H(X,Y) <= H(X) + H(Y), with equality for independent X, Y."""
        n = data.draw(st.integers(min_value=50, max_value=200))
        seed = data.draw(st.integers(0, 2**32 - 1))
        rng = np.random.default_rng(seed)
        x = rng.choice(4, size=n)
        y = rng.choice(4, size=n)
        h_xy = discrete_joint_entropy(x, y)
        h_x = discrete_entropy(x)
        h_y = discrete_entropy(y)
        assert h_xy <= h_x + h_y + 1e-10


# ============================================================================
# CONTINUOUS PROPERTY TESTS
# ============================================================================
class TestContinuousEntropy:
    @given(sample=normal_sample())
    @settings(max_examples=10, deadline=30000)
    def test_base_scaling(self, sample):
        """Entropy in different bases scales by change-of-base factor."""
        h_nats = entropy_from_samples(sample, base=np.e, discrete=False)
        h_bits = entropy_from_samples(sample, base=2.0, discrete=False)
        assert np.isclose(h_bits, h_nats / np.log(2), rtol=0.01)


class TestContinuousRelativeEntropy:
    @given(samples=paired_normal_samples())
    @settings(max_examples=10, deadline=30000)
    def test_nonnegative(self, samples):
        """D_KL(P||Q) >= 0 (Gibbs' inequality)."""
        sample_p, sample_q = samples
        kl = relative_entropy_from_samples(sample_p, sample_q, discrete=False)
        # KDE-based KL can go slightly negative for very similar distributions
        # with small samples due to interpolation artifacts
        assert kl >= -0.5

    @given(samples=paired_normal_samples())
    @settings(max_examples=10, deadline=30000)
    def test_kl_equals_cross_entropy_minus_entropy(self, samples):
        """D_KL(P||Q) = H_Q(P) - H(P)."""
        sample_p, sample_q = samples
        kl = relative_entropy_from_samples(sample_p, sample_q, discrete=False)
        ce = cross_entropy_from_samples(sample_p, sample_q, discrete=False)
        h = entropy_from_samples(sample_p, discrete=False)
        assert np.isclose(kl, ce - h, rtol=0.05, atol=0.05)


class TestContinuousJensenShannon:
    @given(samples=paired_normal_samples())
    @settings(max_examples=10, deadline=30000)
    def test_symmetric(self, samples):
        """JSD(P||Q) = JSD(Q||P)."""
        sample_p, sample_q = samples
        jsd_pq = jensen_shannon_divergence_from_samples(
            sample_p, sample_q, discrete=False
        )
        jsd_qp = jensen_shannon_divergence_from_samples(
            sample_q, sample_p, discrete=False
        )
        assert np.isclose(jsd_pq, jsd_qp, rtol=0.1)

    @given(samples=paired_normal_samples())
    @settings(max_examples=10, deadline=30000)
    def test_bounded(self, samples):
        """0 <= JSD(P||Q) <= ln(2)."""
        sample_p, sample_q = samples
        jsd = jensen_shannon_divergence_from_samples(sample_p, sample_q, discrete=False)
        assert -0.01 <= jsd <= np.log(2) + 0.01


class TestContinuousMutualInformation:
    @given(data=correlated_bivariate_samples())
    @settings(max_examples=10, deadline=30000)
    def test_nonnegative(self, data):
        """I(X;Y) >= 0."""
        x, y, _rho = data
        mi = mutual_information_from_samples(x, y, discrete=False)
        assert mi >= -0.1

    @given(data=correlated_bivariate_samples())
    @settings(max_examples=10, deadline=30000)
    def test_symmetric(self, data):
        """I(X;Y) = I(Y;X)."""
        x, y, _rho = data
        mi_xy = mutual_information_from_samples(x, y, discrete=False)
        mi_yx = mutual_information_from_samples(y, x, discrete=False)
        assert np.isclose(mi_xy, mi_yx, rtol=0.15, atol=0.05)


class TestContinuousChainRule:
    @given(data=correlated_bivariate_samples())
    @settings(max_examples=5, deadline=60000)
    def test_joint_equals_marginal_plus_conditional(self, data):
        """H(X,Y) = H(X) + H(Y|X)."""
        x, y, _rho = data
        h_xy = joint_entropy_from_samples(x, y, discrete=False)
        h_x = entropy_from_samples(x, discrete=False)
        h_y_given_x = conditional_entropy_from_samples(x, y, discrete=False)
        assert np.isclose(h_xy, h_x + h_y_given_x, rtol=0.1, atol=0.2)
