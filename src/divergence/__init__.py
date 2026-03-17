import numpy as np

from ._types import TestResult
from .continuous import (
    conditional_entropy_from_densities_with_support,
    conditional_entropy_from_kde,
    continuous_conditional_entropy_from_samples,
    continuous_cross_entropy_from_sample,
    continuous_entropy_from_sample,
    continuous_jensen_shannon_divergence_from_sample,
    continuous_joint_entropy_from_samples,
    continuous_mutual_information_from_samples,
    continuous_relative_entropy_from_sample,
    cross_entropy_from_densities_with_support,
    cross_entropy_from_kde,
    entropy_from_density_with_support,
    entropy_from_kde,
    intersection,
    jensen_shannon_divergence_from_densities_with_support,
    jensen_shannon_divergence_from_kde,
    joint_entropy_from_densities_with_support,
    joint_entropy_from_kde,
    mutual_information_from_densities_with_support,
    mutual_information_from_kde,
    relative_entropy_from_densities_with_support,
    relative_entropy_from_kde,
)
from .discrete import (
    discrete_conditional_entropy_of_y_given_x,
    discrete_cross_entropy,
    discrete_entropy,
    discrete_jensen_shannon_divergence,
    discrete_joint_entropy,
    discrete_mutual_information,
    discrete_relative_entropy,
)
from .f_divergences import (
    chi_squared_divergence,
    cressie_read_divergence,
    f_divergence,
    jeffreys_divergence,
    squared_hellinger_distance,
    total_variation_distance,
)
from .ipms import (
    energy_distance,
    maximum_mean_discrepancy,
    sliced_wasserstein_distance,
    wasserstein_distance,
)
from .knn import (
    knn_entropy,
    knn_kl_divergence,
    ksg_mutual_information,
)
from .renyi import (
    renyi_divergence,
    renyi_entropy,
)
from .testing import two_sample_test

__all__ = [
    # Two-sample testing
    "TestResult",
    # f-divergences
    "chi_squared_divergence",
    "conditional_entropy_from_densities_with_support",
    "conditional_entropy_from_kde",
    "conditional_entropy_from_samples",
    # Continuous
    "continuous_conditional_entropy_from_samples",
    "continuous_cross_entropy_from_sample",
    "continuous_entropy_from_sample",
    "continuous_jensen_shannon_divergence_from_sample",
    "continuous_joint_entropy_from_samples",
    "continuous_mutual_information_from_samples",
    "continuous_relative_entropy_from_sample",
    "cressie_read_divergence",
    "cross_entropy_from_densities_with_support",
    "cross_entropy_from_kde",
    "cross_entropy_from_samples",
    # Discrete
    "discrete_conditional_entropy_of_y_given_x",
    "discrete_cross_entropy",
    "discrete_entropy",
    "discrete_jensen_shannon_divergence",
    "discrete_joint_entropy",
    "discrete_mutual_information",
    "discrete_relative_entropy",
    # IPMs
    "energy_distance",
    "entropy_from_density_with_support",
    "entropy_from_kde",
    # Unified API
    "entropy_from_samples",
    "f_divergence",
    "intersection",
    "jeffreys_divergence",
    "jensen_shannon_divergence_from_densities_with_support",
    "jensen_shannon_divergence_from_kde",
    "jensen_shannon_divergence_from_samples",
    "joint_entropy_from_densities_with_support",
    "joint_entropy_from_kde",
    "joint_entropy_from_samples",
    # kNN estimators
    "knn_entropy",
    "knn_kl_divergence",
    "ksg_mutual_information",
    "maximum_mean_discrepancy",
    "mutual_information_from_densities_with_support",
    "mutual_information_from_kde",
    "mutual_information_from_samples",
    "relative_entropy_from_densities_with_support",
    "relative_entropy_from_kde",
    "relative_entropy_from_samples",
    # Rényi family
    "renyi_divergence",
    "renyi_entropy",
    "sliced_wasserstein_distance",
    "squared_hellinger_distance",
    "total_variation_distance",
    "two_sample_test",
    "wasserstein_distance",
]


def entropy_from_samples(
    sample: np.ndarray,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    if discrete:
        return discrete_entropy(sample=sample, base=base)
    else:
        return continuous_entropy_from_sample(sample=sample, base=base)


def cross_entropy_from_samples(
    sample_p: np.ndarray,
    sample_q: np.ndarray,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    if discrete:
        return discrete_cross_entropy(sample_p=sample_p, sample_q=sample_q, base=base)
    else:
        return continuous_cross_entropy_from_sample(
            sample_p=sample_p, sample_q=sample_q, base=base
        )


def relative_entropy_from_samples(
    sample_p: np.ndarray,
    sample_q: np.ndarray,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    if discrete:
        return discrete_relative_entropy(
            sample_p=sample_p, sample_q=sample_q, base=base
        )
    else:
        return continuous_relative_entropy_from_sample(
            sample_p=sample_p, sample_q=sample_q, base=base
        )


def jensen_shannon_divergence_from_samples(
    sample_p: np.ndarray,
    sample_q: np.ndarray,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    if discrete:
        return discrete_jensen_shannon_divergence(
            sample_p=sample_p, sample_q=sample_q, base=base
        )
    else:
        return continuous_jensen_shannon_divergence_from_sample(
            sample_p=sample_p, sample_q=sample_q, base=base
        )


def mutual_information_from_samples(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    if discrete:
        return discrete_mutual_information(
            sample_x=sample_x, sample_y=sample_y, base=base
        )
    else:
        return continuous_mutual_information_from_samples(
            sample_x=sample_x, sample_y=sample_y, base=base
        )


def joint_entropy_from_samples(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    if discrete:
        return discrete_joint_entropy(sample_x=sample_x, sample_y=sample_y, base=base)
    else:
        return continuous_joint_entropy_from_samples(
            sample_x=sample_x, sample_y=sample_y, base=base
        )


def conditional_entropy_from_samples(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    if discrete:
        return discrete_conditional_entropy_of_y_given_x(
            sample_x=sample_x, sample_y=sample_y, base=base
        )
    else:
        return continuous_conditional_entropy_from_samples(
            sample_x=sample_x, sample_y=sample_y, base=base
        )
