from .continuous import *
from .discrete import (
    discrete_entropy,
    discrete_relative_entropy,
    discrete_cross_entropy,
    discrete_jensen_shannon_divergence,
    discrete_mutual_information,
    discrete_joint_entropy,
    discrete_conditional_entropy_of_y_given_x
)


def entropy_from_samples(sample: np.ndarray,
                         log_fun: tp.Callable = np.log,
                         discrete: bool = False) -> float:
    if discrete:
        return discrete_entropy(sample=sample,
                                log_fun=log_fun)
    else:
        return continuous_entropy_from_sample(sample=sample,
                                              log_fun=log_fun)


def cross_entropy_from_samples(sample_p: np.ndarray,
                               sample_q: np.ndarray,
                               log_fun: tp.Callable = np.log,
                               discrete: bool = False) -> float:
    if discrete:
        return discrete_cross_entropy(sample_p=sample_p,
                                      sample_q=sample_q,
                                      log_fun=log_fun)
    else:
        return continuous_cross_entropy_from_sample(sample_p=sample_p,
                                                    sample_q=sample_q,
                                                    log_fun=log_fun)


def relative_entropy_from_samples(sample_p: np.ndarray,
                                  sample_q: np.ndarray,
                                  log_fun: tp.Callable = np.log,
                                  discrete: bool = False) -> float:
    if discrete:
        return discrete_relative_entropy(sample_p=sample_p,
                                         sample_q=sample_q,
                                         log_fun=log_fun)
    else:
        return continuous_relative_entropy_from_sample(sample_p=sample_p,
                                                       sample_q=sample_q,
                                                       log_fun=log_fun)


def jensen_shannon_divergence_from_samples(sample_p: np.ndarray,
                                           sample_q: np.ndarray,
                                           log_fun: tp.Callable = np.log,
                                           discrete: bool = False) -> float:
    if discrete:
        return discrete_jensen_shannon_divergence(sample_p=sample_p,
                                                  sample_q=sample_q,
                                                  log_fun=log_fun)
    else:
        return continuous_jensen_shannon_divergence_from_sample(sample_p=sample_p,
                                                                sample_q=sample_q,
                                                                log_fun=log_fun)
