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
