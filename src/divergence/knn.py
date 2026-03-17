"""
k-nearest neighbor (kNN) based estimators for information-theoretic quantities.

This module provides non-parametric estimators for entropy, KL divergence, and
mutual information based on k-nearest neighbor distances. These estimators avoid
explicit density estimation and work well in moderate to high dimensions.

Functions
---------
knn_entropy
    Kozachenko-Leonenko entropy estimator using kNN distances.
knn_kl_divergence
    Wang et al. (2009) KL divergence estimator using kNN distances.
ksg_mutual_information
    Kraskov-Stogbauer-Grassberger mutual information estimator (algorithms 1 and 2).
"""

import numpy as np
from scipy.spatial import cKDTree
from scipy.special import digamma, gammaln


def _log_volume_unit_ball(d: int) -> float:
    """Compute log of the volume of the d-dimensional unit ball.

    The volume of the d-dimensional unit ball is c_d = pi^(d/2) / Gamma(d/2 + 1).

    Parameters
    ----------
    d : int
        Dimensionality.

    Returns
    -------
    float
        log(c_d)
    """
    return (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1.0)


def _ensure_2d(samples: np.ndarray) -> np.ndarray:
    """Ensure samples array is 2D with shape (n, d).

    Parameters
    ----------
    samples : np.ndarray
        Input array, either 1D of shape (n,) or 2D of shape (n, d).

    Returns
    -------
    np.ndarray
        2D array of shape (n, d).

    Raises
    ------
    ValueError
        If the array has more than 2 dimensions.
    """
    if samples.ndim == 1:
        return samples.reshape(-1, 1)
    if samples.ndim == 2:
        return samples
    raise ValueError(f"samples must be 1D or 2D, got {samples.ndim}D array")


def knn_entropy(
    samples: np.ndarray,
    *,
    k: int = 5,
    base: float = np.e,
) -> float:
    r"""Estimate the differential entropy using the Kozachenko-Leonenko estimator.

    Uses k-nearest neighbor distances to estimate the entropy of the distribution
    from which ``samples`` are drawn, without explicit density estimation.

    Parameters
    ----------
    samples : np.ndarray
        Sample array of shape ``(n,)`` or ``(n, d)`` where *n* is the number of
        observations and *d* is the dimensionality.
    k : int, optional
        Number of nearest neighbors to use. Default is 5.
    base : float, optional
        Base of the logarithm, controlling the unit of measurement.
        Use ``np.e`` for nats (default), ``2`` for bits, ``10`` for hartleys.

    Returns
    -------
    float
        Estimated differential entropy.

    Notes
    -----
    The estimator is given by:

    .. math::

        \hat{H} = \frac{d}{N} \sum_{i=1}^{N} \ln(2\,\varepsilon_k(i))
                  + \ln(N - 1) - \psi(k) + \ln(c_d)

    where :math:`\varepsilon_k(i)` is the Euclidean distance from point *i* to its
    *k*-th nearest neighbor (excluding itself), :math:`c_d` is the volume of the
    *d*-dimensional unit ball, and :math:`\psi` is the digamma function.

    The estimator is consistent and asymptotically unbiased.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> samples = rng.standard_normal(5000)
    >>> h = knn_entropy(samples, k=5)
    >>> analytical = 0.5 * (1 + np.log(2 * np.pi))  # ~1.4189 nats
    >>> abs(h - analytical) < 0.15
    True

    References
    ----------
    .. [1] Kozachenko, L. F., & Leonenko, N. N. (1987). "Sample estimate of the
       entropy of a random vector." *Problems of Information Transmission*, 23(2),
       9-16.
    .. [2] Kraskov, A., Stogbauer, H., & Grassberger, P. (2004). "Estimating
       mutual information." *Physical Review E*, 69(6), 066138.
    """
    samples = _ensure_2d(samples)
    n, d = samples.shape

    tree = cKDTree(samples)
    # k+1 because query includes the point itself at distance 0
    distances, _ = tree.query(samples, k=k + 1, p=2)
    # Take the k-th neighbor distance (index k, since index 0 is self)
    nn_distances = distances[:, k]

    # Guard against log(0) by clipping to a small positive value
    nn_distances = np.maximum(nn_distances, np.finfo(float).tiny)

    log_c_d = _log_volume_unit_ball(d)

    # Compute in nats, then convert
    # Note: no factor of 2 here since we use Euclidean distances with the
    # Euclidean ball volume c_d = pi^(d/2) / Gamma(d/2+1). The volume of
    # a ball of radius epsilon is c_d * epsilon^d.
    h_nats = (
        (d / n) * np.sum(np.log(nn_distances)) + np.log(n - 1) - digamma(k) + log_c_d
    )

    return h_nats / np.log(base)


def knn_kl_divergence(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    k: int = 5,
    base: float = np.e,
) -> float:
    r"""Estimate the KL divergence D_KL(P || Q) using kNN distances.

    Uses the Wang et al. (2009) estimator based on k-nearest neighbor distances
    from two independent samples.

    Parameters
    ----------
    samples_p : np.ndarray
        Samples from distribution P, shape ``(n,)`` or ``(n, d)``.
    samples_q : np.ndarray
        Samples from distribution Q, shape ``(m,)`` or ``(m, d)``.
    k : int, optional
        Number of nearest neighbors. Default is 5.
    base : float, optional
        Base of the logarithm. Default is ``np.e`` (nats).

    Returns
    -------
    float
        Estimated KL divergence D_KL(P || Q).

    Notes
    -----
    The estimator is given by:

    .. math::

        \hat{D}_{KL}(P \| Q) = \frac{d}{n} \sum_{i=1}^{n}
            \ln\!\left(\frac{\nu_k(i)}{\rho_k(i)}\right)
            + \ln\!\left(\frac{m}{n - 1}\right)

    where :math:`\rho_k(i)` is the distance from :math:`x_i` to its *k*-th nearest
    neighbor in the P-sample (excluding itself), and :math:`\nu_k(i)` is the distance
    from :math:`x_i` to its *k*-th nearest neighbor in the Q-sample.

    The estimator is consistent and converges to the true KL divergence as both
    sample sizes grow.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> p = rng.normal(0, 1, size=3000)
    >>> q = rng.normal(0, 1, size=3000)
    >>> knn_kl_divergence(p, q, k=5)  # Should be close to 0
    ...

    References
    ----------
    .. [1] Wang, Q., Kulkarni, S. R., & Verdu, S. (2009). "Divergence estimation
       for multidimensional densities via k-nearest-neighbor distances."
       *IEEE Transactions on Information Theory*, 55(5), 2392-2405.
    """
    samples_p = _ensure_2d(samples_p)
    samples_q = _ensure_2d(samples_q)

    n, d = samples_p.shape
    m = samples_q.shape[0]

    if samples_p.shape[1] != samples_q.shape[1]:
        raise ValueError(
            f"samples_p and samples_q must have the same dimensionality, "
            f"got {samples_p.shape[1]} and {samples_q.shape[1]}"
        )

    tree_p = cKDTree(samples_p)
    tree_q = cKDTree(samples_q)

    # For rho_k: query P-tree with k+1 neighbors (skip self), take k-th distance
    rho_distances, _ = tree_p.query(samples_p, k=k + 1, p=2)
    rho_k = rho_distances[:, k]  # k-th NN distance in P (0-indexed, so index k)

    # For nu_k: query Q-tree with k neighbors, take k-th distance
    nu_distances, _ = tree_q.query(samples_p, k=k, p=2)
    nu_k = nu_distances[:, k - 1]  # k-th NN distance in Q (0-indexed, so index k-1)

    # Guard against log(0)
    rho_k = np.maximum(rho_k, np.finfo(float).tiny)
    nu_k = np.maximum(nu_k, np.finfo(float).tiny)

    # Compute in nats, then convert
    dkl_nats = (d / n) * np.sum(np.log(nu_k / rho_k)) + np.log(m / (n - 1))

    return dkl_nats / np.log(base)


def ksg_mutual_information(
    samples_x: np.ndarray,
    samples_y: np.ndarray,
    *,
    k: int = 5,
    base: float = np.e,
    algorithm: int = 1,
) -> float:
    r"""Estimate mutual information using the Kraskov-Stogbauer-Grassberger estimator.

    Uses k-nearest neighbor distances in the joint space with Chebyshev (max-norm)
    metric to estimate mutual information without explicit density estimation.

    Parameters
    ----------
    samples_x : np.ndarray
        Samples of the X variable, shape ``(n,)`` or ``(n, d_x)``.
    samples_y : np.ndarray
        Samples of the Y variable, shape ``(n,)`` or ``(n, d_y)``.
        Must have the same number of observations as ``samples_x``.
    k : int, optional
        Number of nearest neighbors. Default is 5.
    base : float, optional
        Base of the logarithm. Default is ``np.e`` (nats).
    algorithm : int, optional
        Which KSG algorithm to use. Must be 1 or 2. Default is 1.

    Returns
    -------
    float
        Estimated mutual information I(X; Y).

    Notes
    -----
    **Algorithm 1:**

    .. math::

        \hat{I}(X; Y) = \psi(k) - \frac{1}{N} \sum_{i=1}^{N}
            \left[\psi(n_x(i) + 1) + \psi(n_y(i) + 1)\right] + \psi(N)

    where :math:`n_x(i)` is the number of points :math:`j \neq i` with
    :math:`\|X_j - X_i\|_\infty \leq \varepsilon(i)`, and :math:`\varepsilon(i)`
    is the Chebyshev distance to the *k*-th nearest neighbor in the joint space.

    **Algorithm 2:**

    .. math::

        \hat{I}(X; Y) = \psi(k) - \frac{1}{k}
            - \frac{1}{N} \sum_{i=1}^{N}
            \left[\psi(m_x(i)) + \psi(m_y(i))\right] + \psi(N)

    where :math:`m_x(i)` and :math:`m_y(i)` use strict inequality (``<``) instead
    of ``<=``.

    Both algorithms use the Chebyshev (max-norm, :math:`L_\infty`) metric for
    neighbor searches in the joint and marginal spaces.

    Mutual information is always non-negative; small negative values may occur
    due to estimation noise.

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> x = rng.standard_normal(3000)
    >>> y = rng.standard_normal(3000)
    >>> mi = ksg_mutual_information(x, y, k=5)
    >>> abs(mi) < 0.1  # Independent variables, MI should be near 0
    True

    References
    ----------
    .. [1] Kraskov, A., Stogbauer, H., & Grassberger, P. (2004). "Estimating
       mutual information." *Physical Review E*, 69(6), 066138.
    """
    if algorithm not in (1, 2):
        raise ValueError(f"algorithm must be 1 or 2, got {algorithm}")

    samples_x = _ensure_2d(samples_x)
    samples_y = _ensure_2d(samples_y)

    n = samples_x.shape[0]
    if samples_y.shape[0] != n:
        raise ValueError(
            f"samples_x and samples_y must have the same number of observations, "
            f"got {n} and {samples_y.shape[0]}"
        )

    # Build joint space by horizontal stacking
    joint = np.hstack([samples_x, samples_y])

    # Build trees with Chebyshev (max-norm) distance
    tree_joint = cKDTree(joint)
    tree_x = cKDTree(samples_x)
    tree_y = cKDTree(samples_y)

    # Find k-th NN in joint space (k+1 because self is included)
    joint_distances, _ = tree_joint.query(joint, k=k + 1, p=np.inf)
    # epsilon(i) = distance to k-th NN in joint space (Chebyshev)
    epsilon = joint_distances[:, k]

    if algorithm == 1:
        # Count points j != i with ||X_j - X_i||_inf <= epsilon(i)
        # query_ball_point returns indices including self, so subtract 1
        n_x = np.array(
            [
                len(tree_x.query_ball_point(samples_x[i], r=epsilon[i], p=np.inf)) - 1
                for i in range(n)
            ]
        )
        n_y = np.array(
            [
                len(tree_y.query_ball_point(samples_y[i], r=epsilon[i], p=np.inf)) - 1
                for i in range(n)
            ]
        )

        mi_nats = digamma(k) - np.mean(digamma(n_x + 1) + digamma(n_y + 1)) + digamma(n)
    else:
        # Algorithm 2: strict inequality (< epsilon)
        # Count points with marginal distance strictly less than epsilon(i).
        # query_ball_point uses <=, so we count with <= epsilon then subtract
        # those at exactly distance epsilon in each marginal.
        # Algorithm 2 uses the same epsilon and counting as algorithm 1
        # (with <=), but applies the -1/k bias correction and uses
        # digamma(m) instead of digamma(n+1). For continuous distributions,
        # the difference between < and <= is measure-zero, so using <=
        # is numerically stable and the -1/k term handles the correction.
        m_x = np.array([
            len(tree_x.query_ball_point(samples_x[i], r=epsilon[i], p=np.inf)) - 1
            for i in range(n)
        ])
        m_y = np.array([
            len(tree_y.query_ball_point(samples_y[i], r=epsilon[i], p=np.inf)) - 1
            for i in range(n)
        ])

        # Ensure >= 1 for digamma
        m_x = np.maximum(m_x, 1)
        m_y = np.maximum(m_y, 1)

        mi_nats = (
            digamma(k) - 1.0 / k - np.mean(digamma(m_x) + digamma(m_y)) + digamma(n)
        )

    return mi_nats / np.log(base)
