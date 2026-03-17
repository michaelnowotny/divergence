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
.. [3] Gorham, J. & Mackey, L. (2017). "Measuring sample quality with
   kernels." *ICML*.
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
        ``(n, d)`` and returns array of shape ``(n, d)`` with
        :math:`\nabla \log p(x)` evaluated at each sample point.
    kernel : str, optional
        Kernel function: ``"rbf"`` (Gaussian, default) or ``"imq"``
        (inverse multiquadric). The IMQ kernel
        :math:`k(x,y) = (c^2 + \|x-y\|^2)^{-1/2}` has provable
        convergence control guarantees that the RBF kernel lacks.
    bandwidth : float or None, optional
        Bandwidth parameter. For the RBF kernel this is :math:`\sigma`
        in :math:`k(x,y) = \exp(-\|x-y\|^2 / (2\sigma^2))`. For the
        IMQ kernel this is :math:`c` in
        :math:`k(x,y) = (c^2 + \|x-y\|^2)^{-1/2}`. If ``None``, the
        median heuristic is used for both kernels.

    Returns
    -------
    float
        Squared KSD (U-statistic estimator). Close to 0 when samples
        come from P.

    Raises
    ------
    ValueError
        If ``kernel`` is not ``"rbf"`` or ``"imq"``, or if fewer than
        2 samples are provided.

    Notes
    -----
    The **RBF kernel** :math:`k(x,y) = \exp(-\|x-y\|^2/(2\sigma^2))`
    is the standard choice. The **IMQ kernel**
    :math:`k(x,y) = (c^2 + \|x-y\|^2)^{-1/2}` is recommended for
    MCMC convergence diagnostics because it provides *convergence
    control*: :math:`\mathrm{KSD}(\mu_n, \pi) \to 0` implies
    :math:`\mu_n \Rightarrow \pi` (weak convergence) and tightness
    of :math:`\{\mu_n\}` [3]_.

    References
    ----------
    .. [1] Liu, Q., Lee, J., & Jordan, M. (2016). "A kernelized Stein
       discrepancy for goodness-of-fit tests." *ICML*.
    .. [2] Chwialkowski, K., Strathmann, H., & Gretton, A. (2016).
       "A kernel test of goodness of fit." *ICML*.
    .. [3] Gorham, J. & Mackey, L. (2017). "Measuring sample quality
       with kernels." *ICML*.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(42)
    >>> samples = rng.standard_normal(1000)
    >>> ksd = kernel_stein_discrepancy(samples, lambda x: -x, kernel="imq")
    >>> abs(ksd) < 0.1  # close to zero for matching distribution
    True
    """
    if kernel not in ("rbf", "imq"):
        raise ValueError(f"Unsupported kernel '{kernel}'. Use 'rbf' or 'imq'.")

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

    # Bandwidth (median heuristic)
    if bandwidth is None:
        bandwidth = _median_bandwidth(x)

    # Pairwise squared distances
    sq_dists = cdist(x, x, metric="sqeuclidean")

    # diff[i,j,:] = x[i] - x[j]
    diff = x[:, np.newaxis, :] - x[np.newaxis, :, :]  # (n, n, d)

    # s[i] . (x[i] - x[j]) and s[j] . (x[i] - x[j])
    s_dot_diff_i = np.einsum("id,ijd->ij", s, diff)  # (n, n)
    s_dot_diff_j = np.einsum("jd,ijd->ij", s, diff)  # (n, n)

    # s[i] . s[j]
    ss = s @ s.T  # (n, n)

    if kernel == "rbf":
        stein_kernel = _rbf_stein_kernel(
            ss, s_dot_diff_i, s_dot_diff_j, sq_dists, bandwidth, d
        )
    else:  # imq
        stein_kernel = _imq_stein_kernel(
            ss, s_dot_diff_i, s_dot_diff_j, sq_dists, bandwidth, d
        )

    # U-statistic: exclude diagonal
    np.fill_diagonal(stein_kernel, 0.0)
    ksd_sq = np.sum(stein_kernel) / (n * (n - 1))

    return float(ksd_sq)


def _rbf_stein_kernel(
    ss: np.ndarray,
    s_dot_diff_i: np.ndarray,
    s_dot_diff_j: np.ndarray,
    sq_dists: np.ndarray,
    bandwidth: float,
    d: int,
) -> np.ndarray:
    """Compute the Stein kernel matrix for the RBF kernel.

    Parameters
    ----------
    ss : np.ndarray
        Pairwise score dot products ``s[i] . s[j]``, shape ``(n, n)``.
    s_dot_diff_i : np.ndarray
        ``s[i] . (x[i] - x[j])``, shape ``(n, n)``.
    s_dot_diff_j : np.ndarray
        ``s[j] . (x[i] - x[j])``, shape ``(n, n)``.
    sq_dists : np.ndarray
        Pairwise squared Euclidean distances, shape ``(n, n)``.
    bandwidth : float
        Kernel bandwidth :math:`\\sigma`.
    d : int
        Dimensionality.

    Returns
    -------
    np.ndarray
        Stein kernel matrix, shape ``(n, n)``.
    """
    gamma = 1.0 / (2.0 * bandwidth**2)
    K = np.exp(-gamma * sq_dists)

    term1 = ss * K
    term2 = 2.0 * gamma * s_dot_diff_i * K
    term3 = -2.0 * gamma * s_dot_diff_j * K
    term4 = (2.0 * gamma * d - 4.0 * gamma**2 * sq_dists) * K

    return term1 + term2 + term3 + term4


def _imq_stein_kernel(
    ss: np.ndarray,
    s_dot_diff_i: np.ndarray,
    s_dot_diff_j: np.ndarray,
    sq_dists: np.ndarray,
    c: float,
    d: int,
    beta: float = -0.5,
) -> np.ndarray:
    r"""Compute the Stein kernel matrix for the IMQ kernel.

    The inverse multiquadric kernel is:

    .. math::

        k(x, y) = (c^2 + \|x - y\|^2)^{\beta}

    with :math:`\beta = -1/2` by default.

    Parameters
    ----------
    ss : np.ndarray
        Pairwise score dot products ``s[i] . s[j]``, shape ``(n, n)``.
    s_dot_diff_i : np.ndarray
        ``s[i] . (x[i] - x[j])``, shape ``(n, n)``.
    s_dot_diff_j : np.ndarray
        ``s[j] . (x[i] - x[j])``, shape ``(n, n)``.
    sq_dists : np.ndarray
        Pairwise squared Euclidean distances, shape ``(n, n)``.
    c : float
        Scale parameter (typically the median heuristic bandwidth).
    d : int
        Dimensionality.
    beta : float, optional
        Exponent of the IMQ kernel. Default is ``-0.5``.

    Returns
    -------
    np.ndarray
        Stein kernel matrix, shape ``(n, n)``.

    References
    ----------
    .. [1] Gorham, J. & Mackey, L. (2017). "Measuring sample quality
       with kernels." *ICML*.
    """
    base = c**2 + sq_dists  # (n, n)

    # k(x, y) = base^beta
    K = base**beta

    # grad_y k = -2*beta*(x - y)*base^(beta-1)
    # grad_x k =  2*beta*(x - y)*base^(beta-1)
    K_bm1 = base ** (beta - 1)

    # Term 1: s_i . s_j * k
    term1 = ss * K

    # Term 2: s_i^T grad_y k = -2*beta * s_dot_diff_i * base^(beta-1)
    term2 = -2.0 * beta * s_dot_diff_i * K_bm1

    # Term 3: s_j^T grad_x k = 2*beta * s_dot_diff_j * base^(beta-1)
    term3 = 2.0 * beta * s_dot_diff_j * K_bm1

    # Term 4: trace Hessian
    # nabla_x . nabla_y k = -2*beta * base^(beta-2) * [d*base + 2*(beta-1)*r^2]
    K_bm2 = base ** (beta - 2)
    term4 = -2.0 * beta * K_bm2 * (d * base + 2.0 * (beta - 1.0) * sq_dists)

    return term1 + term2 + term3 + term4
