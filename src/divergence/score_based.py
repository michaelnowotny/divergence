"""Score-based divergence measures.

This module implements divergences that rely on score functions (gradients of
log-densities): the Fisher divergence and the kernel Stein discrepancy (KSD).

Functions
---------
fisher_divergence
    Fisher divergence between two distributions using score functions.
kernel_stein_discrepancy
    Kernel Stein discrepancy measuring sample quality against a reference.

References
----------
.. [1] Liu, Q., Lee, J., & Jordan, M. (2016). "A kernelized Stein
   discrepancy for goodness-of-fit tests." *ICML*.
.. [2] Chwialkowski, K., Strathmann, H., & Gretton, A. (2016). "A kernel
   test of goodness of fit." *ICML*.
"""

import numpy as np
from scipy.spatial.distance import cdist


def _median_bandwidth(samples: np.ndarray) -> float:
    """Compute the median heuristic bandwidth for the RBF kernel."""
    d = cdist(samples, samples, metric="euclidean")
    triu = d[np.triu_indices_from(d, k=1)]
    med = float(np.median(triu))
    return med if med > 0.0 else 1.0


def _kernel_score_estimate(
    samples: np.ndarray,
    query_points: np.ndarray,
    bandwidth: float,
) -> np.ndarray:
    r"""Estimate score function ∇log p(x) via kernel density gradient.

    Uses the RBF kernel density estimator:

    .. math::

        \hat{p}(x) = \frac{1}{n} \sum_{i=1}^{n} K_h(x - x_i)

    and its gradient:

    .. math::

        \nabla \log \hat{p}(x) = \frac{\sum_i K_h(x - x_i)(x_i - x) / h^2}
                                       {\sum_i K_h(x - x_i)}

    Parameters
    ----------
    samples : np.ndarray
        Training samples, shape ``(n, d)``.
    query_points : np.ndarray
        Points at which to estimate the score, shape ``(m, d)``.
    bandwidth : float
        Bandwidth parameter for the RBF kernel.

    Returns
    -------
    np.ndarray
        Estimated score vectors, shape ``(m, d)``.
    """
    # (m, n) pairwise squared distances
    sq_dists = cdist(query_points, samples, metric="sqeuclidean")
    gamma = 1.0 / (2.0 * bandwidth**2)

    # (m, n) kernel weights
    weights = np.exp(-gamma * sq_dists)

    # (m, 1) normalization
    weight_sums = weights.sum(axis=1, keepdims=True)
    weight_sums = np.maximum(weight_sums, np.finfo(float).tiny)

    # (m, d) weighted direction: sum_i w_i (x_i - x) / h^2
    # diff[j, i, :] = samples[i] - query_points[j]
    diff = samples[np.newaxis, :, :] - query_points[:, np.newaxis, :]  # (m, n, d)
    score = np.einsum("mn,mnd->md", weights, diff) / (bandwidth**2 * weight_sums)

    return score


def fisher_divergence(
    samples_p: np.ndarray,
    score_q,
    *,
    score_p=None,
    bandwidth: float | None = None,
) -> float:
    r"""Estimate the Fisher divergence between distributions P and Q.

    .. math::

        D_F(P \| Q) = \mathbb{E}_P\!\left[\|\nabla \log p(x) -
            \nabla \log q(x)\|^2\right]

    Parameters
    ----------
    samples_p : np.ndarray
        Samples from distribution P, shape ``(n,)`` or ``(n, d)``.
    score_q : callable
        Score function of Q: takes array of shape ``(n, d)`` and returns
        array of shape ``(n, d)`` with :math:`\nabla \log q(x)` at each point.
    score_p : callable or None, optional
        Score function of P. If None, estimated from ``samples_p`` via
        kernel density gradient with RBF kernel.
    bandwidth : float or None, optional
        Bandwidth for the kernel score estimator (used when ``score_p=None``).
        If None, the median heuristic is used.

    Returns
    -------
    float
        Estimated Fisher divergence, non-negative.
    """
    x = np.asarray(samples_p, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    # Score of Q at sample points
    sq = np.asarray(score_q(x))
    if sq.ndim == 1:
        sq = sq.reshape(-1, 1)

    # Score of P at sample points
    if score_p is not None:
        sp = np.asarray(score_p(x))
        if sp.ndim == 1:
            sp = sp.reshape(-1, 1)
    else:
        if bandwidth is None:
            bandwidth = _median_bandwidth(x)
        sp = _kernel_score_estimate(x, x, bandwidth)

    # E_P[||s_p(x) - s_q(x)||^2]
    diff = sp - sq
    return float(np.mean(np.sum(diff**2, axis=1)))


def kernel_stein_discrepancy(
    samples: np.ndarray,
    score_fn,
    *,
    kernel: str = "rbf",
    bandwidth: float | None = None,
) -> float:
    r"""Compute the kernel Stein discrepancy (KSD).

    Measures how well ``samples`` approximate the distribution P whose score
    function :math:`\nabla \log p` is provided.

    The U-statistic estimator of the squared KSD is:

    .. math::

        \widehat{\mathrm{KSD}}^2 = \frac{1}{n(n-1)} \sum_{i \neq j}
            u_p(x_i, x_j)

    where the Stein kernel is:

    .. math::

        u_p(x, y) = s_p(x)^\top s_p(y)\, k(x, y)
                   + s_p(x)^\top \nabla_y k(x, y)
                   + s_p(y)^\top \nabla_x k(x, y)
                   + \nabla_x \cdot \nabla_y k(x, y)

    Parameters
    ----------
    samples : np.ndarray
        Sample points, shape ``(n,)`` or ``(n, d)``.
    score_fn : callable
        Score function of the target distribution P. Takes array of shape
        ``(n, d)`` and returns array of shape ``(n, d)``.
    kernel : str, optional
        Kernel function. Only ``"rbf"`` is supported. Default is ``"rbf"``.
    bandwidth : float or None, optional
        Bandwidth for the RBF kernel. If None, uses the median heuristic.

    Returns
    -------
    float
        Squared KSD (U-statistic estimator). Close to 0 when samples come
        from P.
    """
    if kernel != "rbf":
        raise ValueError(f"Only 'rbf' kernel is supported, got '{kernel}'")

    x = np.asarray(samples, dtype=float)
    if x.ndim == 1:
        x = x.reshape(-1, 1)

    n, d = x.shape
    if n < 2:
        raise ValueError(f"Need at least 2 samples, got {n}")

    # Score values at all sample points
    s = np.asarray(score_fn(x))
    if s.ndim == 1:
        s = s.reshape(-1, 1)

    # Bandwidth
    if bandwidth is None:
        bandwidth = _median_bandwidth(x)

    gamma = 1.0 / (2.0 * bandwidth**2)

    # Pairwise squared distances and kernel
    sq_dists = cdist(x, x, metric="sqeuclidean")
    K = np.exp(-gamma * sq_dists)

    # Term 1: s(x_i)^T s(x_j) k(x_i, x_j)
    # s @ s.T gives (n, n) matrix where [i,j] = s_i . s_j
    term1 = (s @ s.T) * K

    # For RBF kernel k(x,y) = exp(-gamma ||x-y||^2):
    #   grad_y k(x,y) = 2*gamma*(x - y)*k(x,y)
    #   grad_x k(x,y) = -2*gamma*(x - y)*k(x,y)
    #   grad_x . grad_y k(x,y) = (2*gamma*d - 4*gamma^2*||x-y||^2)*k(x,y)

    # diff[i,j,:] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (n, n, d)

    # Term 2: s(x_i)^T grad_{x_j} k(x_i, x_j)
    #       = s(x_i)^T * 2*gamma*(x_i - x_j) * k_ij
    s_dot_diff = np.einsum("id,ijd->ij", s, diff)  # (n, n)
    term2 = 2.0 * gamma * s_dot_diff * K

    # Term 3: s(x_j)^T grad_{x_i} k(x_i, x_j)
    #       = s(x_j)^T * (-2*gamma)*(x_i - x_j) * k_ij
    s_dot_diff_j = np.einsum("jd,ijd->ij", s, diff)  # (n, n)
    term3 = -2.0 * gamma * s_dot_diff_j * K

    # Term 4: grad_{x_i} . grad_{x_j} k = (2*gamma*d - 4*gamma^2*||x_i-x_j||^2)*k
    term4 = (2.0 * gamma * d - 4.0 * gamma**2 * sq_dists) * K

    stein_kernel = term1 + term2 + term3 + term4

    # U-statistic: exclude diagonal
    np.fill_diagonal(stein_kernel, 0.0)
    ksd_sq = np.sum(stein_kernel) / (n * (n - 1))

    return float(ksd_sq)
