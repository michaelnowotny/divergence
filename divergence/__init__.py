from .continuous import *
from .discrete import (
    discrete_entropy,
    discrete_relative_entropy,
    discrete_cross_entropy,
    discrete_jensen_shannon_divergence
)


def entropy_from_samples(samples: np.ndarray,
                         log_fun: tp.Callable = np.log,
                         discrete: bool = False) -> float:
    if discrete:
        return discrete_entropy(sample=samples,
                                log_fun=log_fun)
    else:
        return continuous_entropy_from_samples(samples=samples,
                                               log_fun=log_fun)


def cross_entropy_from_samples(samples_p: np.ndarray,
                               samples_q: np.ndarray,
                               log_fun: tp.Callable = np.log,
                               discrete: bool = False) -> float:
    if discrete:
        return discrete_cross_entropy(sample_p=samples_p,
                                      sample_q=samples_q,
                                      log_fun=log_fun)
    else:
        return continuous_cross_entropy_from_samples(samples_p=samples_p,
                                                     samples_q=samples_q,
                                                     log_fun=log_fun)


def relative_entropy_from_samples(samples_p: np.ndarray,
                                  samples_q: np.ndarray,
                                  log_fun: tp.Callable = np.log,
                                  discrete: bool = False) -> float:
    if discrete:
        return discrete_relative_entropy(sample_p=samples_p,
                                         sample_q=samples_q,
                                         log_fun=log_fun)
    else:
        return continuous_relative_entropy_from_samples(samples_p=samples_p,
                                                        samples_q=samples_q,
                                                        log_fun=log_fun)


def jensen_shannon_divergence_from_samples(samples_p: np.ndarray,
                                           samples_q: np.ndarray,
                                           log_fun: tp.Callable = np.log,
                                           discrete: bool = False) -> float:
    if discrete:
        return discrete_jensen_shannon_divergence(sample_p=samples_p,
                                                  sample_q=samples_q,
                                                  log_fun=log_fun)
    else:
        return continuous_jensen_shannon_divergence_from_samples(samples_p=samples_p,
                                                                 samples_q=samples_q,
                                                                 log_fun=log_fun)
