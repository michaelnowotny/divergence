"""Shared types for the divergence package."""

import typing as tp

import numpy as np


class TestResult(tp.NamedTuple):
    """Result of a two-sample hypothesis test.

    Attributes
    ----------
    statistic : float
        The observed test statistic.
    p_value : float
        The permutation p-value.
    null_distribution : np.ndarray
        The null distribution of the test statistic under permutation.
    """

    statistic: float
    p_value: float
    null_distribution: np.ndarray
