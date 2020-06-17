import numpy as np
import pytest
import scipy as sp
import statsmodels.api as sm
import typing as tp

from divergence import *


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
def test_entropy(sigma: float, sample: np.ndarray, log_fun: tp.Callable = np.log):
    assert np.isclose(entropy_from_samples(sample, log_fun=log_fun, discrete=False),
                      entropy_of_normal_distribution(sigma, log_fun=log_fun),
                      rtol=1e-2,
                      atol=1e-2)


def test_cross_entropy(log_fun: tp.Callable = np.log):
    assert np.isclose(cross_entropy_from_samples(sample_p,
                                                 sample_q,
                                                 log_fun=log_fun,
                                                 discrete=False),
                      cross_entropy_between_two_normal_distributions(mu_p,
                                                                     sigma_p,
                                                                     mu_q,
                                                                     sigma_q,
                                                                     log_fun=log_fun),
                      rtol=1e-1,
                      atol=1e-1)


def test_relative_entropy(log_fun: tp.Callable = np.log):
    assert np.isclose(relative_entropy_from_samples(sample_p,
                                                    sample_q,
                                                    log_fun=log_fun,
                                                    discrete=False),
                      relative_entropy_between_two_normal_distributions(mu_p,
                                                                        sigma_p,
                                                                        mu_q,
                                                                        sigma_q,
                                                                        log_fun=log_fun),
                      rtol=1e-1,
                      atol=1e-1)
