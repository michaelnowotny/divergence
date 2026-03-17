"""Sinkhorn divergence via entropy-regularized optimal transport.

This module implements the debiased Sinkhorn divergence, which interpolates
between the Wasserstein distance (as epsilon -> 0) and the MMD (as epsilon ->
infinity), while being computationally efficient via matrix scaling.

Functions
---------
sinkhorn_divergence
    Debiased Sinkhorn divergence between two sample sets.

References
----------
.. [1] Cuturi, M. (2013). "Sinkhorn distances: Lightspeed computation of
   optimal transport." *NeurIPS*.
.. [2] Genevay, A., Peyre, G., & Cuturi, M. (2018). "Learning generative
   models with Sinkhorn divergences." *AISTATS*.
"""

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import logsumexp


def _sinkhorn_cost(
    C: np.ndarray,
    epsilon: float,
    max_iter: int,
    tol: float,
) -> float:
    """Compute the entropy-regularized OT cost via log-domain Sinkhorn iterations.

    Parameters
    ----------
    C : np.ndarray
        Cost matrix of shape ``(n, m)``.
    epsilon : float
        Regularization parameter.
    max_iter : int
        Maximum number of iterations.
    tol : float
        Convergence tolerance on the dual variable change.

    Returns
    -------
    float
        The regularized OT cost.
    """
    n, m = C.shape
    log_a = np.full(n, -np.log(n))
    log_b = np.full(m, -np.log(m))

    # Log kernel: log K_ij = -C_ij / epsilon
    log_K = -C / epsilon

    f = np.zeros(n)
    g = np.zeros(m)

    for _ in range(max_iter):
        f_prev = f.copy()
        # g_j = log(b_j) - logsumexp_i(log_K_ij + f_i)
        g = log_b - logsumexp(log_K + f[:, np.newaxis], axis=0)
        # f_i = log(a_i) - logsumexp_j(log_K_ij + g_j)
        f = log_a - logsumexp(log_K + g[np.newaxis, :], axis=1)

        if np.max(np.abs(f - f_prev)) < tol:
            break

    # Transport plan: T_ij = exp(f_i + log_K_ij + g_j)
    log_transport = f[:, np.newaxis] + log_K + g[np.newaxis, :]
    transport = np.exp(log_transport)
    return float(np.sum(transport * C))


def sinkhorn_divergence(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    *,
    epsilon: float = 0.01,
    p: int = 2,
    max_iter: int = 100,
    tol: float = 1e-6,
) -> float:
    r"""Compute the debiased Sinkhorn divergence between two sample sets.

    .. math::

        S_\varepsilon(P, Q) = OT_\varepsilon(P, Q)
            - \frac{1}{2}\bigl(OT_\varepsilon(P, P) + OT_\varepsilon(Q, Q)\bigr)

    where :math:`OT_\varepsilon` is the entropy-regularized optimal transport
    cost with regularization parameter :math:`\varepsilon`.

    Parameters
    ----------
    samples_p : np.ndarray
        Samples from distribution P. Shape ``(n,)`` for 1D or ``(n, d)`` for
        d-dimensional data.
    samples_q : np.ndarray
        Samples from distribution Q. Shape ``(m,)`` for 1D or ``(m, d)`` for
        d-dimensional data.
    epsilon : float, optional
        Entropic regularization parameter. Smaller values approximate the
        Wasserstein distance more closely but require more iterations.
        Default is 0.01.
    p : int, optional
        Exponent for the ground cost: ``c(x, y) = ||x - y||^p``.
        Default is 2.
    max_iter : int, optional
        Maximum number of Sinkhorn iterations. Default is 100.
    tol : float, optional
        Convergence tolerance. Default is 1e-6.

    Returns
    -------
    float
        The debiased Sinkhorn divergence, non-negative.

    Notes
    -----
    The debiasing correction subtracts the self-transport costs to ensure that
    :math:`S_\varepsilon(P, P) = 0`. Without debiasing, the regularized OT
    cost is always positive even for identical distributions.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> p_samples = rng.normal(0, 1, 200)
    >>> q_samples = rng.normal(2, 1, 200)
    >>> sinkhorn_divergence(p_samples, q_samples) > 0
    True
    """
    x = np.asarray(samples_p, dtype=float)
    y = np.asarray(samples_q, dtype=float)

    if x.ndim == 1:
        x = x.reshape(-1, 1)
    if y.ndim == 1:
        y = y.reshape(-1, 1)

    # Cost matrices
    if p == 2:
        C_pq = cdist(x, y, metric="sqeuclidean")
        C_pp = cdist(x, x, metric="sqeuclidean")
        C_qq = cdist(y, y, metric="sqeuclidean")
    else:
        C_pq = cdist(x, y, metric="minkowski", p=p) ** p
        C_pp = cdist(x, x, metric="minkowski", p=p) ** p
        C_qq = cdist(y, y, metric="minkowski", p=p) ** p

    ot_pq = _sinkhorn_cost(C_pq, epsilon, max_iter, tol)
    ot_pp = _sinkhorn_cost(C_pp, epsilon, max_iter, tol)
    ot_qq = _sinkhorn_cost(C_qq, epsilon, max_iter, tol)

    return ot_pq - 0.5 * (ot_pp + ot_qq)
