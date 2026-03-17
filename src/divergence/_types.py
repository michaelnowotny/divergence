"""Shared types for the divergence package."""

import typing as tp

import numpy as np


class TestResult(tp.NamedTuple):
    """Result of a two-sample permutation hypothesis test.

    Returned by :func:`~divergence.two_sample.permutation_test` and related
    testing functions. Fields can be accessed by name or by index.

    Attributes
    ----------
    statistic : float
        The observed test statistic computed on the original (unpermuted)
        samples.
    p_value : float
        The permutation p-value: the fraction of permuted statistics that
        are at least as extreme as the observed statistic.
    null_distribution : np.ndarray
        Array of test-statistic values computed under each permutation of
        the sample labels, representing the null distribution.
    """

    statistic: float
    p_value: float
    null_distribution: np.ndarray


class ChainKSDResult(tp.NamedTuple):
    """Result of per-chain kernel Stein discrepancy analysis.

    Returned by :func:`~divergence.bayesian.chain_ksd`. Each field
    corresponds to KSD values computed on different subsets of the MCMC
    output. Small values indicate that the chain's empirical distribution
    is close to the target.

    Attributes
    ----------
    ksd_per_chain : np.ndarray
        KSD value for each chain, shape ``(n_chains,)``.
    ksd_split_first : np.ndarray or None
        KSD for the first half of each chain, shape ``(n_chains,)``.
        ``None`` if ``split=False`` was specified.
    ksd_split_second : np.ndarray or None
        KSD for the second half of each chain, shape ``(n_chains,)``.
        ``None`` if ``split=False`` was specified.
    ksd_pooled : float
        KSD computed from all chains pooled together.
    """

    ksd_per_chain: np.ndarray
    ksd_split_first: np.ndarray | None
    ksd_split_second: np.ndarray | None
    ksd_pooled: float


class ChainTestResult(tp.NamedTuple):
    """Result of pairwise two-sample tests between MCMC chains.

    Returned by :func:`~divergence.bayesian.chain_two_sample_test`.
    Provides calibrated p-values for the null hypothesis that each pair
    of chains samples from the same distribution.

    Attributes
    ----------
    p_value_matrix : np.ndarray
        Pairwise p-values, shape ``(n_chains, n_chains)``. Diagonal
        entries are ``1.0`` (a chain is identical to itself).
    statistic_matrix : np.ndarray
        Pairwise test statistics, shape ``(n_chains, n_chains)``.
        Diagonal entries are ``0.0``.
    min_p_value : float
        Minimum off-diagonal p-value across all chain pairs. The
        most evidence of distributional discrepancy.
    any_significant : bool
        Whether any chain pair has a p-value below 0.05.
    """

    p_value_matrix: np.ndarray
    statistic_matrix: np.ndarray
    min_p_value: float
    any_significant: bool


class MixingDiagnostic(tp.NamedTuple):
    """Result of transfer-entropy-based mixing diagnostics.

    Returned by :func:`~divergence.bayesian.mixing_diagnostic`. Uses
    transfer entropy to detect non-stationarity within chains and
    spurious dependence between chains.

    Attributes
    ----------
    stationarity_te : np.ndarray
        Transfer entropy from the first half to the second half of each
        chain, shape ``(n_chains,)``. Low values indicate stationarity;
        elevated values suggest the chain has not yet reached its
        stationary distribution.
    cross_chain_te : np.ndarray
        Transfer entropy between consecutive chain pairs, shape
        ``(n_chains - 1,)``. Low values indicate independence between
        chains; elevated values suggest shared non-stationarity or
        coupling artifacts.
    """

    stationarity_te: np.ndarray
    cross_chain_te: np.ndarray
