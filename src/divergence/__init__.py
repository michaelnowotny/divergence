import numpy as np

from ._types import ChainKSDResult, ChainTestResult, MixingDiagnostic, TestResult
from .bayesian import (
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
from .causal import transfer_entropy
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
from .multivariate import (
    normalized_mutual_information,
    total_correlation,
    variation_of_information,
)
from .renyi import (
    renyi_divergence,
    renyi_entropy,
)
from .score_based import fisher_divergence, kernel_stein_discrepancy
from .sinkhorn import sinkhorn_divergence
from .testing import two_sample_test

__all__ = [
    # Result types
    "ChainKSDResult",
    "ChainTestResult",
    "MixingDiagnostic",
    "TestResult",
    # Bayesian diagnostics (ArviZ integration)
    "bayesian_surprise",
    "chain_divergence",
    # Convergence diagnostics
    "chain_ksd",
    "chain_two_sample_test",
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
    # Score-based measures
    "fisher_divergence",
    "information_gain",
    "intersection",
    "jeffreys_divergence",
    "jensen_shannon_divergence_from_densities_with_support",
    "jensen_shannon_divergence_from_kde",
    "jensen_shannon_divergence_from_samples",
    "joint_entropy_from_densities_with_support",
    "joint_entropy_from_kde",
    "joint_entropy_from_samples",
    # Score-based measures
    "kernel_stein_discrepancy",
    # kNN estimators
    "knn_entropy",
    "knn_kl_divergence",
    "ksg_mutual_information",
    "maximum_mean_discrepancy",
    # Convergence diagnostics
    "mixing_diagnostic",
    "model_divergence",
    "mutual_information_from_densities_with_support",
    "mutual_information_from_kde",
    "mutual_information_from_samples",
    # Multivariate dependence
    "normalized_mutual_information",
    "prior_sensitivity",
    "relative_entropy_from_densities_with_support",
    "relative_entropy_from_kde",
    "relative_entropy_from_samples",
    # Rényi family
    "renyi_divergence",
    "renyi_entropy",
    # Sinkhorn divergence
    "sinkhorn_divergence",
    "sliced_wasserstein_distance",
    "squared_hellinger_distance",
    # Multivariate dependence
    "total_correlation",
    "total_variation_distance",
    # Causal/temporal
    "transfer_entropy",
    "two_sample_test",
    "uncertainty_decomposition",
    # Multivariate dependence
    "variation_of_information",
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
