"""Multivariate dependence measures.

This module provides measures of statistical dependence among multiple variables:
total correlation (multi-information), normalized mutual information, and
variation of information. All functions are thin wrappers over the existing
entropy and mutual information estimators.

Functions
---------
total_correlation
    Watanabe's total correlation (multi-information) for d >= 2 variables.
normalized_mutual_information
    Mutual information normalized by a chosen function of the marginal entropies.
variation_of_information
    Meila's variation of information, a true metric on partitions.

References
----------
.. [1] Watanabe, S. (1960). "Information theoretical analysis of multivariate
   correlation." *IBM Journal of Research and Development*, 4(1), 66-82.
.. [2] Strehl, A., & Ghosh, J. (2002). "Cluster ensembles — a knowledge reuse
   framework for combining multiple partitions." *JMLR*, 3, 583-617.
.. [3] Meila, M. (2003). "Comparing clusterings by the variation of
   information." *Learning Theory and Kernel Machines*, 173-187.
"""

import numpy as np

from .discrete import (
    discrete_entropy,
    discrete_joint_entropy,
    discrete_mutual_information,
)
from .knn import knn_entropy, ksg_mutual_information


def total_correlation(
    samples: np.ndarray,
    *,
    base: float = np.e,
    discrete: bool = False,
    estimator: str = "knn",
) -> float:
    r"""Compute the total correlation (multi-information) of a multivariate sample.

    .. math::

        TC(X_1, \ldots, X_d) = \sum_{i=1}^{d} H(X_i) - H(X_1, \ldots, X_d)

    Parameters
    ----------
    samples : np.ndarray
        Sample array of shape ``(n, d)`` with ``d >= 2`` variables.
    base : float, optional
        Logarithm base. Default is ``np.e`` (nats).
    discrete : bool, optional
        If True, use discrete estimators. Default is False.
    estimator : str, optional
        Estimator for continuous data: ``"knn"`` (default) or ``"kde"``.
        Ignored when ``discrete=True``.

    Returns
    -------
    float
        Total correlation, non-negative.

    Raises
    ------
    ValueError
        If ``samples`` does not have at least 2 columns.
    """
    samples = np.asarray(samples, dtype=float)
    if samples.ndim != 2 or samples.shape[1] < 2:
        raise ValueError("samples must be 2D with at least 2 columns (variables)")

    _n, d = samples.shape

    if discrete:
        # Sum of marginal entropies
        sum_marginals = sum(
            discrete_entropy(samples[:, i].astype(int), base=base) for i in range(d)
        )
        # Joint entropy via pairwise for d=2, or via generalized counting for d>2
        if d == 2:
            h_joint = discrete_joint_entropy(
                samples[:, 0].astype(int), samples[:, 1].astype(int), base=base
            )
        else:
            # For d > 2, encode tuples as strings for discrete_entropy
            tuples = np.array(
                [str(tuple(row)) for row in samples.astype(int)], dtype=object
            )
            h_joint = discrete_entropy(tuples, base=base)
        return sum_marginals - h_joint

    # Continuous estimators
    if estimator == "knn":
        sum_marginals = sum(knn_entropy(samples[:, i], base=base) for i in range(d))
        h_joint = knn_entropy(samples, base=base)
        return sum_marginals - h_joint

    if estimator == "kde":
        from .continuous import continuous_entropy_from_sample

        sum_marginals = sum(
            continuous_entropy_from_sample(samples[:, i], base=base) for i in range(d)
        )
        # KDE doesn't scale beyond 1D easily; use kNN for joint
        h_joint = knn_entropy(samples, base=base)
        return sum_marginals - h_joint

    raise ValueError(f"Unknown estimator: {estimator!r}. Use 'knn' or 'kde'.")


def normalized_mutual_information(
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    *,
    normalization: str = "geometric",
    base: float = np.e,
    discrete: bool = False,
) -> float:
    r"""Compute normalized mutual information between two variables.

    .. math::

        \mathrm{NMI}(X, Y) = \frac{I(X; Y)}{\mathrm{norm}(H(X), H(Y))}

    Parameters
    ----------
    samples_x : np.ndarray
        Samples of variable X, shape ``(n,)``.
    samples_y : np.ndarray
        Samples of variable Y, shape ``(n,)``.
    normalization : str, optional
        Normalization method: ``"geometric"`` (default), ``"arithmetic"``,
        ``"max"``, ``"min"``, or ``"joint"``.
    base : float, optional
        Logarithm base. Default is ``np.e``.
    discrete : bool, optional
        If True, use discrete estimators. Default is False.

    Returns
    -------
    float
        Normalized mutual information in [0, 1] (approximately).

    Raises
    ------
    ValueError
        If normalization is unknown.
    """
    if discrete:
        mi = discrete_mutual_information(samples_x, samples_y, base=base)
        h_x = discrete_entropy(samples_x, base=base)
        h_y = discrete_entropy(samples_y, base=base)
    else:
        mi = ksg_mutual_information(samples_x, samples_y, base=base)
        h_x = knn_entropy(samples_x, base=base)
        h_y = knn_entropy(samples_y, base=base)

    if normalization == "geometric":
        denom = np.sqrt(h_x * h_y) if h_x > 0 and h_y > 0 else 0.0
    elif normalization == "arithmetic":
        denom = 0.5 * (h_x + h_y)
    elif normalization == "max":
        denom = max(h_x, h_y)
    elif normalization == "min":
        denom = min(h_x, h_y)
    elif normalization == "joint":
        if discrete:
            denom = discrete_joint_entropy(samples_x, samples_y, base=base)
        else:
            denom = knn_entropy(
                np.column_stack(
                    [
                        np.asarray(samples_x).ravel(),
                        np.asarray(samples_y).ravel(),
                    ]
                ),
                base=base,
            )
    else:
        raise ValueError(
            f"Unknown normalization: {normalization!r}. "
            "Use 'geometric', 'arithmetic', 'max', 'min', or 'joint'."
        )

    if denom == 0.0:
        return 0.0

    return mi / denom


def variation_of_information(
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    *,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    r"""Compute the variation of information between two variables.

    .. math::

        VI(X, Y) = H(X) + H(Y) - 2\,I(X; Y)

    This is a true metric on the space of clusterings/partitions.

    Parameters
    ----------
    samples_x : np.ndarray
        Samples of variable X, shape ``(n,)``.
    samples_y : np.ndarray
        Samples of variable Y, shape ``(n,)``.
    base : float, optional
        Logarithm base. Default is ``np.e``.
    discrete : bool, optional
        If True, use discrete estimators. Default is False.

    Returns
    -------
    float
        Variation of information, non-negative.
    """
    if discrete:
        h_x = discrete_entropy(samples_x, base=base)
        h_y = discrete_entropy(samples_y, base=base)
        mi = discrete_mutual_information(samples_x, samples_y, base=base)
    else:
        h_x = knn_entropy(samples_x, base=base)
        h_y = knn_entropy(samples_y, base=base)
        mi = ksg_mutual_information(samples_x, samples_y, base=base)

    return h_x + h_y - 2.0 * mi
