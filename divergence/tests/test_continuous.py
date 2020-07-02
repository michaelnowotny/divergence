import numpy as np
import pytest
import scipy as sp
import statsmodels.api as sm
import typing as tp

from divergence import *
from divergence.base import _select_vectorized_log_fun_for_base


def entropy_of_normal_distribution(sigma: float,
                                   log_fun: tp.Callable = np.log) \
        -> float:
    return 0.5 * (1.0 + log_fun(2 * np.pi * sigma**2))


def relative_entropy_between_two_normal_distributions(mu_1: float,
                                                      sigma_1: float,
                                                      mu_2: float,
                                                      sigma_2: float,
                                                      log_fun: tp.Callable = np.log) \
        -> float:
    return ((mu_1 - mu_2)**2 + sigma_1**2 - sigma_2**2) / (2 * sigma_2**2) + \
           log_fun(sigma_2/sigma_1)


def cross_entropy_between_two_normal_distributions(mu_1: float,
                                                   sigma_1: float,
                                                   mu_2: float,
                                                   sigma_2: float,
                                                   log_fun: tp.Callable = np.log) \
        -> float:
    return entropy_of_normal_distribution(sigma_1, log_fun=log_fun) + \
           relative_entropy_between_two_normal_distributions(mu_1=mu_1,
                                                             sigma_1=sigma_1,
                                                             mu_2=mu_2,
                                                             sigma_2=sigma_2,
                                                             log_fun=log_fun)


def mutual_information_for_bivariate_normal_distribution(rho: float) -> float:
    return - 0.5 * np.log(1.0 - rho**2)


def generate_normal_sample(mu: float,
                           sigma: float,
                           n: int,
                           antithetic: bool = False) -> np.ndarray:
    z = np.random.randn(n)
    if antithetic:
        z = np.hstack((z, -z))

    return mu + sigma * z


# fix random seed for reproducibility
np.random.seed(42)

# set parameters of the normal distributions p and q
mu_p = 2
sigma_p = 3
mu_q = 1
sigma_q = 2

# draw samples from each normal distribution
n = 10000

sample_p = generate_normal_sample(mu_p, sigma_p, n=n, antithetic=True)
sample_q = generate_normal_sample(mu_q, sigma_q, n=n, antithetic=True)

# fit a non-parametric density estimate for both distributions
kde_p = sm.nonparametric.KDEUnivariate(sample_p)
kde_q = sm.nonparametric.KDEUnivariate(sample_q)
kde_p.fit()
kde_q.fit()

# construct exact normal densities for p and q
pdf_p = lambda x: sp.stats.norm.pdf(x, mu_p, sigma_p)
pdf_q = lambda x: sp.stats.norm.pdf(x, mu_q, sigma_q)

# compute support for kernel density estimates
p_min = min(kde_p.support)
p_max = max(kde_p.support)
q_min = min(kde_q.support)
q_max = max(kde_q.support)
combined_min = min(p_min, q_min)
combined_max = max(p_max, q_max)


@pytest.mark.parametrize("sigma, sample", ((sigma_p, sample_p), (sigma_q, sample_q)))
def test_entropy(sigma: float, sample: np.ndarray, base: float = np.e):
    log_fun = _select_vectorized_log_fun_for_base(base)

    assert np.isclose(entropy_from_samples(sample, base=base, discrete=False),
                      entropy_of_normal_distribution(sigma, log_fun=log_fun),
                      rtol=1e-2,
                      atol=1e-2)


def test_cross_entropy(base: float = np.e):
    log_fun = _select_vectorized_log_fun_for_base(base)

    assert np.isclose(cross_entropy_from_samples(sample_p,
                                                 sample_q,
                                                 base=base,
                                                 discrete=False),
                      cross_entropy_between_two_normal_distributions(mu_p,
                                                                     sigma_p,
                                                                     mu_q,
                                                                     sigma_q,
                                                                     log_fun=log_fun),
                      rtol=1e-1,
                      atol=1e-1)


def test_relative_entropy(base: float = np.e):
    log_fun = _select_vectorized_log_fun_for_base(base)

    assert np.isclose(relative_entropy_from_samples(sample_p,
                                                    sample_q,
                                                    base=base,
                                                    discrete=False),
                      relative_entropy_between_two_normal_distributions(mu_p,
                                                                        sigma_p,
                                                                        mu_q,
                                                                        sigma_q,
                                                                        log_fun=log_fun),
                      rtol=1e-1,
                      atol=1e-1)


# set parameters of the normal distributions x and y
mu_x = 2
sigma_x = 3
mu_y = 1
sigma_y = 2
rho = 0.5

# draw 1000 samples from each normal distribution
n = 10000
z = np.random.randn(n)
sample_x = mu_x + sigma_x * z
sample_y = mu_y + sigma_y * (rho * z + np.sqrt(1.0 - rho**2) * np.random.randn(n))

# fit a non-parametric density estimate for both distributions
kde_x = sm.nonparametric.KDEUnivariate(sample_x)
kde_y = sm.nonparametric.KDEUnivariate(sample_y)
kde_x.fit() # Estimate the densities
kde_y.fit() # Estimate the densities
kde_xy = sp.stats.gaussian_kde([sample_x, sample_y])

# construct exact normal densities for x and y
pdf_x = lambda x: sp.stats.norm.pdf(x, mu_x, sigma_x)
pdf_y = lambda y: sp.stats.norm.pdf(y, mu_y, sigma_y)
pdf_xy = sp.stats.multivariate_normal(mean=[mu_x, mu_y],
                                      cov=[[sigma_x**2, rho * sigma_x * sigma_y],
                                           [rho * sigma_x * sigma_y, sigma_y**2]]).pdf

# # compute support for kernel density estimates
x_min = min(kde_x.support)
x_max = max(kde_x.support)
y_min = min(kde_y.support)
y_max = max(kde_y.support)


@pytest.fixture
def mutual_information_from_bivariate_normal_samples() -> float:
    return continuous_mutual_information_from_samples(sample_x=sample_x,
                                                      sample_y=sample_y)


@pytest.fixture
def joint_entropy_of_x_and_y() -> float:
    return joint_entropy_from_samples(sample_x, sample_y)


@pytest.fixture
def conditional_entropy_of_y_given_x_from_bivariate_normal_samples() -> float:
    return conditional_entropy_from_samples(sample_x, sample_y)


@pytest.fixture
def conditional_entropy_of_x_given_y_from_bivariate_normal_samples() -> float:
    return conditional_entropy_from_samples(sample_y, sample_x)


def test_mutual_information(mutual_information_from_bivariate_normal_samples):
    theoretical_mutual_information = mutual_information_for_bivariate_normal_distribution(rho)

    assert np.isclose(theoretical_mutual_information,
                      mutual_information_from_bivariate_normal_samples,
                      rtol=1e-1,
                      atol=1e-1)


def test_joint_entropy_via_conditional_entropy_of_y_given_x(
        joint_entropy_of_x_and_y,
        conditional_entropy_of_y_given_x_from_bivariate_normal_samples):
    np.isclose(entropy_from_samples(sample_x) +
               conditional_entropy_of_y_given_x_from_bivariate_normal_samples,
               joint_entropy_of_x_and_y,
               rtol=1e-2,
               atol=1e-3)


def test_joint_entropy_via_conditional_entropy_of_x_given_y(
        joint_entropy_of_x_and_y,
        conditional_entropy_of_x_given_y_from_bivariate_normal_samples):
    np.isclose(entropy_from_samples(sample_y) +
               conditional_entropy_of_x_given_y_from_bivariate_normal_samples,
               joint_entropy_of_x_and_y,
               rtol=1e-2,
               atol=1e-3)
