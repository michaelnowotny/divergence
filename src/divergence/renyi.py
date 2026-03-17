r"""Renyi entropy and divergence estimators.

Provides Renyi entropy and Renyi divergence for both discrete and
continuous distributions, estimated from samples.

The Renyi entropy of order alpha for a discrete distribution P is

    H_alpha(P) = 1 / (1 - alpha) * log(sum_i p_i^alpha)

and the Renyi divergence of order alpha of P from Q is

    D_alpha(P || Q) = 1 / (alpha - 1) * log(sum_i p_i^alpha * q_i^(1 - alpha))

Both reduce to the Shannon entropy and KL divergence respectively as
alpha -> 1.

References
----------
.. [1] Renyi, A. (1961). "On measures of entropy and information."
       Proc. 4th Berkeley Symp. Math. Stat. Prob., 1, 547-561.
.. [2] Van Erven, T. & Harremoes, P. (2014). "Renyi divergence and
       Kullback-Leibler divergence." IEEE Trans. Inform. Theory, 60(7),
       3797-3820.
"""

import numpy as np
import statsmodels.api as sm

from divergence.base import _select_vectorized_log_fun_for_base
from divergence.continuous import (
    continuous_entropy_from_sample,
    continuous_relative_entropy_from_sample,
)
from divergence.discrete import discrete_entropy
from divergence.f_divergences import _aligned_frequencies


# ---------------------------------------------------------------------------
# Renyi entropy
# ---------------------------------------------------------------------------
def renyi_entropy(
    sample: np.ndarray,
    *,
    alpha: float,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    r"""Compute the Renyi entropy of order alpha from a sample.

    Parameters
    ----------
    sample : np.ndarray
        Sample from the distribution.
    alpha : float
        Order of the Renyi entropy. Must be non-negative.
        Special cases:

        - alpha = 0: Hartley entropy (log of support size).
        - alpha -> 1: Shannon entropy.
        - alpha = 2: collision entropy.
        - alpha = +inf: min-entropy.
    base : float, optional
        Base of the logarithm (default: e for nats, 2 for bits, 10 for
        hartleys).
    discrete : bool, optional
        If True, treat the sample as draws from a discrete distribution
        and compute frequencies directly. If False (default), estimate
        the density via kernel density estimation.

    Returns
    -------
    float
        The estimated Renyi entropy of order alpha.

    Notes
    -----
    For a discrete distribution P = (p_1, ..., p_k), the Renyi entropy
    of order alpha is defined as

    .. math::

        H_\alpha(P) = \frac{1}{1 - \alpha} \log\!\left(\sum_{i=1}^{k}
        p_i^\alpha\right)

    Key properties:

    - **Non-negative** for discrete distributions: H_alpha >= 0.
    - **Monotonically decreasing** in alpha: H_alpha1 >= H_alpha2 when
      alpha1 < alpha2.
    - **Reduces to Shannon entropy** as alpha -> 1.
    - **Hartley entropy** at alpha = 0: H_0 = log(|support|).
    - **Min-entropy** at alpha = +inf: H_inf = -log(max_i p_i).

    For continuous distributions, the density is estimated via KDE and the
    integral is computed using the trapezoidal rule on the KDE grid.

    Examples
    --------
    >>> import numpy as np
    >>> from divergence.renyi import renyi_entropy
    >>> rng = np.random.default_rng(42)
    >>> sample = rng.choice([0, 1, 2], size=10000, p=[0.2, 0.3, 0.5])
    >>> renyi_entropy(sample, alpha=2, base=np.e, discrete=True)
    0.97...

    References
    ----------
    .. [1] Renyi, A. (1961). "On measures of entropy and information."
           Proc. 4th Berkeley Symp. Math. Stat. Prob., 1, 547-561.
    """
    if alpha < 0:
        raise ValueError("alpha must be non-negative")

    log_fun = _select_vectorized_log_fun_for_base(base)

    # --- alpha -> 1: Shannon entropy ---
    if np.isclose(alpha, 1.0):
        if discrete:
            return discrete_entropy(sample=sample, base=base)
        else:
            return continuous_entropy_from_sample(sample=sample, base=base)

    # --- alpha = +inf: min-entropy ---
    if np.isinf(alpha):
        if discrete:
            _, counts = np.unique(sample, return_counts=True)
            frequencies = counts / len(sample)
            return -log_fun(np.max(frequencies))
        else:
            kde = sm.nonparametric.KDEUnivariate(sample)
            kde.fit()
            return -log_fun(np.max(kde.density))

    # --- alpha = 0: Hartley entropy ---
    if alpha == 0:
        if discrete:
            support_size = len(np.unique(sample))
            return log_fun(support_size)
        else:
            kde = sm.nonparametric.KDEUnivariate(sample)
            kde.fit()
            support = kde.support
            # Support length for continuous distributions
            support_length = support[-1] - support[0]
            return log_fun(support_length)

    # --- General alpha ---
    if discrete:
        _, counts = np.unique(sample, return_counts=True)
        frequencies = counts / len(sample)
        power_sum = np.sum(frequencies**alpha)
        return (1.0 / (1.0 - alpha)) * log_fun(power_sum)
    else:
        kde = sm.nonparametric.KDEUnivariate(sample)
        kde.fit()
        support = kde.support
        density = kde.density

        # Integrate p(x)^alpha over the support
        mask = density > 0
        integrand = np.where(mask, density**alpha, 0.0)
        integral = np.trapezoid(integrand, support)

        if integral <= 0:
            return 0.0

        return (1.0 / (1.0 - alpha)) * log_fun(integral)


# ---------------------------------------------------------------------------
# Renyi divergence
# ---------------------------------------------------------------------------
def renyi_divergence(
    sample_p: np.ndarray,
    sample_q: np.ndarray,
    *,
    alpha: float,
    base: float = np.e,
    discrete: bool = False,
) -> float:
    r"""Compute the Renyi divergence of order alpha from samples.

    Parameters
    ----------
    sample_p : np.ndarray
        Sample from distribution P.
    sample_q : np.ndarray
        Sample from distribution Q.
    alpha : float
        Order of the Renyi divergence. Must be positive.
        Special cases:

        - alpha -> 1: KL divergence D_KL(P || Q).
        - alpha = 2: sometimes called the "exponential divergence".
        - alpha = +inf: max-divergence.
    base : float, optional
        Base of the logarithm (default: e for nats, 2 for bits, 10 for
        hartleys).
    discrete : bool, optional
        If True, treat samples as draws from discrete distributions. If
        False (default), estimate densities via KDE.

    Returns
    -------
    float
        The estimated Renyi divergence of order alpha.

    Notes
    -----
    For discrete distributions P and Q, the Renyi divergence of order
    alpha is defined as

    .. math::

        D_\alpha(P \| Q) = \frac{1}{\alpha - 1} \log\!\left(
        \sum_{i=1}^{k} p_i^\alpha \, q_i^{1-\alpha}\right)

    Key properties:

    - **Non-negative**: D_alpha(P || Q) >= 0, with equality iff P = Q.
    - **Monotonically non-decreasing** in alpha: D_alpha1 <= D_alpha2
      when alpha1 < alpha2.
    - **Reduces to KL divergence** as alpha -> 1.

    For continuous distributions, the densities are estimated via KDE and
    the integral is computed using the trapezoidal rule. Log-space
    arithmetic is used for numerical stability.

    Examples
    --------
    >>> import numpy as np
    >>> from divergence.renyi import renyi_divergence
    >>> rng = np.random.default_rng(42)
    >>> p = rng.choice([0, 1, 2], size=10000, p=[0.2, 0.3, 0.5])
    >>> q = rng.choice([0, 1, 2], size=10000, p=[0.3, 0.3, 0.4])
    >>> renyi_divergence(p, q, alpha=2, base=np.e, discrete=True)
    0.03...

    References
    ----------
    .. [1] Renyi, A. (1961). "On measures of entropy and information."
           Proc. 4th Berkeley Symp. Math. Stat. Prob., 1, 547-561.
    .. [2] Van Erven, T. & Harremoes, P. (2014). "Renyi divergence and
           Kullback-Leibler divergence." IEEE Trans. Inform. Theory, 60(7),
           3797-3820.
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive for Renyi divergence")

    log_fun = _select_vectorized_log_fun_for_base(base)

    # --- alpha -> 1: KL divergence ---
    if np.isclose(alpha, 1.0):
        if discrete:
            freq_p, freq_q = _aligned_frequencies(sample_p, sample_q)
            violation = (freq_p > 0) & (freq_q == 0)
            if np.any(violation):
                return np.inf
            mask = (freq_p > 0) & (freq_q > 0)
            kl = float(np.sum(np.where(mask, freq_p * np.log(freq_p / freq_q), 0.0)))
            if base != np.e:
                kl /= np.log(base)
            return kl
        else:
            return continuous_relative_entropy_from_sample(
                sample_p=sample_p, sample_q=sample_q, base=base
            )

    # --- alpha = +inf: max-divergence ---
    if np.isinf(alpha):
        if discrete:
            freq_p, freq_q = _aligned_frequencies(sample_p, sample_q)
            mask = freq_q > 0
            ratios = np.where(mask, freq_p / np.where(mask, freq_q, 1.0), 0.0)
            # For entries where p > 0 and q == 0, divergence is infinite
            if np.any((freq_p > 0) & (freq_q == 0)):
                return np.inf
            return log_fun(np.max(ratios))
        else:
            kde_p = sm.nonparametric.KDEUnivariate(sample_p)
            kde_p.fit()
            kde_q = sm.nonparametric.KDEUnivariate(sample_q)
            kde_q.fit()

            if len(kde_p.support) >= len(kde_q.support):
                support = kde_p.support
            else:
                support = kde_q.support

            p = np.interp(support, kde_p.support, kde_p.density, left=0.0, right=0.0)
            q = np.interp(support, kde_q.support, kde_q.density, left=0.0, right=0.0)

            mask = q > 0
            ratios = np.where(mask, p / np.where(mask, q, 1.0), 0.0)
            return log_fun(np.max(ratios))

    # --- General alpha ---
    if discrete:
        freq_p, freq_q = _aligned_frequencies(sample_p, sample_q)

        # Check absolute continuity: where p > 0, q must be > 0
        violation = (freq_p > 0) & (freq_q == 0)
        if np.any(violation):
            return np.inf

        mask = (freq_p > 0) & (freq_q > 0)
        # Compute sum_i p_i^alpha * q_i^(1-alpha) in log space for stability
        log_terms = alpha * np.log(freq_p[mask]) + (1.0 - alpha) * np.log(freq_q[mask])
        log_sum = np.logaddexp.reduce(log_terms)

        return (1.0 / (alpha - 1.0)) * (log_sum / np.log(base))
    else:
        kde_p = sm.nonparametric.KDEUnivariate(sample_p)
        kde_p.fit()
        kde_q = sm.nonparametric.KDEUnivariate(sample_q)
        kde_q.fit()

        # Use the finer grid as the integration domain
        if len(kde_p.support) >= len(kde_q.support):
            support = kde_p.support
        else:
            support = kde_q.support

        p = np.interp(support, kde_p.support, kde_p.density, left=0.0, right=0.0)
        q = np.interp(support, kde_q.support, kde_q.density, left=0.0, right=0.0)

        # Integrate p(x)^alpha * q(x)^(1-alpha) over the common support
        # Use log-space for numerical stability
        mask = (p > 0) & (q > 0)
        log_integrand = np.where(
            mask,
            alpha * np.log(np.where(mask, p, 1.0))
            + (1.0 - alpha) * np.log(np.where(mask, q, 1.0)),
            -np.inf,
        )
        integrand = np.where(mask, np.exp(log_integrand), 0.0)
        integral = np.trapezoid(integrand, support)

        if integral <= 0:
            return 0.0

        return (1.0 / (alpha - 1.0)) * log_fun(integral)
