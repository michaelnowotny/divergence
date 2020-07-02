import numpy as np
import quadpy
import scipy as sp
import statsmodels.api as sm
import typing as tp

from divergence.base import _select_vectorized_log_fun_for_base


def intersection(a0: float,
                 b0: float,
                 a1: float,
                 b1: float) \
        -> tp.Optional[tp.Tuple[float, float]]:
    """
    Calculate the intersection of two intervals [a0, b0] and [a1, b1]. If the intervals do not
    overlap the function returns None. The parameters must satisfy a0 <= b0 and a1 <= b1.

    Parameters
    ----------
    a0: beginning of the first interval
    b0: end of the first interval
    a1: beginning of the second interval
    b1: end of the second interval

    Returns
    -------

    """
    assert a0 <= b0
    assert a1 <= b1

    if a0 >= b1:
        return None

    if b0 < a1:
        return None

    return max(a0, a1), min(b0, b1)


################################################################################
# Entropy
################################################################################
def entropy_from_density_with_support(pdf: tp.Callable,
                                      a: float,
                                      b: float,
                                      base: float = np.e,
                                      eps_abs: float = 1.49e-08,
                                      eps_rel: float = 1.49e-08) \
        -> float:
    """
    Compute the entropy

                H(p) = - E_p[log(p)]

    of the density given in pdf via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    pdf: a function of a scalar parameter which computes the probability density at that point
    a: lower bound of the integration region
    b: upper bound of the integration region
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The entropy of the density given by pdf
    """
    # def entropy_integrand(x: float):
    #     return pdf(x) * log_fun(pdf(x)) if pdf(x) > 0.0 else 0.0
    #
    # return -sp.integrate.quad(entropy_integrand, a=a, b=b, epsabs=eps_abs, epsrel=eps_rel)[0]

    log_fun = _select_vectorized_log_fun_for_base(base)

    def entropy_integrand_vectorized_fast(x: np.ndarray):
        p = pdf(x)
        return - np.where(p > 0.0, p * log_fun(p), 0.0)

    # def entropy_integrand_vectorized_slow(x):
    #     p = pdf(x)
    #
    #     result = np.zeros_like(x)
    #
    #     positive_index = p > 0.0
    #     p_positive = p[positive_index]
    #     result[positive_index] = - p_positive * log_fun(p_positive)
    #
    #     return result

    return quadpy.line_segment.integrate_adaptive(f=entropy_integrand_vectorized_fast,
                                                  intervals=[a, b],
                                                  eps_abs=eps_abs,
                                                  eps_rel=eps_rel,
                                                  kronrod_degree=10)[0]


def entropy_from_kde(kde: sm.nonparametric.KDEUnivariate,
                     base: float = np.e,
                     eps_abs: float = 1.49e-08,
                     eps_rel: float = 1.49e-08) -> float:
    """
    Compute the entropy

                H(p) = - E_p[log(p)]

    of the density given by the statsmodels kde object via numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    kde: statsmodels kde object representing an approximation of the density
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The entropy of the density approximated by the kde
    """
    a = min(kde.support)
    b = max(kde.support)
    return entropy_from_density_with_support(pdf=kde.evaluate,
                                             a=a,
                                             b=b,
                                             base=base,
                                             eps_abs=eps_abs,
                                             eps_rel=eps_rel)


def continuous_entropy_from_sample(sample: np.ndarray,
                                   base: float = np.e,
                                   eps_abs: float = 1.49e-08,
                                   eps_rel: float = 1.49e-08) -> float:
    """
    Compute the entropy

                H(p) = - E_p[log(p)]

    of a sample via approximation by a kernel density estimate and numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    sample: a sample of draws from the density represented as a 1-dimensional NumPy array
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The entropy of the density approximated by the sample
    """
    kde = sm.nonparametric.KDEUnivariate(sample)
    kde.fit()
    return entropy_from_kde(kde=kde,
                            base=base,
                            eps_abs=eps_abs,
                            eps_rel=eps_rel)


################################################################################
# Cross Entropy
################################################################################
def _cross_entropy_integrand(p: tp.Callable,
                             q: tp.Callable,
                             x: float,
                             log_fun: tp.Callable) -> float:
    """
    Compute the integrand p(x) * log(q(x)) at a given point x for the calculation of cross entropy.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    x: the point at which to evaluate the integrand
    base: the base of the logarithm used to control the units of measurement for the result

    Returns
    -------
    Integrand for the cross entropy calculation
    """
    # return p(x) * log_fun(q(x) + 1e-12)
    qx = q(x)
    px = p(x)
    if qx == 0.0:
        if px == 0.0:
            return 0.0
        else:
            raise ValueError(f'q(x) is zero at x={x} but p(x) is not')
    elif px == 0.0:
        return 0.0
    else:
        return px * log_fun(qx)


def _vectorized_cross_entropy_integrand(p: tp.Callable,
                                        q: tp.Callable,
                                        x: np.ndarray,
                                        log_fun: tp.Callable) -> np.ndarray:
    """
    Compute the integrand p(x) * log(q(x)) vectorized at given points x for the calculation of cross
    entropy.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    x: the point at which to evaluate the integrand
    base: the base of the logarithm used to control the units of measurement for the result

    Returns
    -------
    Integrand for the cross entropy calculation
    """
    # return p(x) * log_fun(q(x) + 1e-12)
    qx = q(x)
    px = p(x)

    q_positive_index = qx > 0.0
    p_positive_index = px > 0.0

    q_zero_but_p_positive_index = ~q_positive_index & p_positive_index
    if np.any(q_zero_but_p_positive_index):
        raise ValueError(f'q(x) is zero at x={x[q_zero_but_p_positive_index]} but p(x) is not')

    return -np.where(p_positive_index, px * log_fun(qx), 0.0)


def cross_entropy_from_densities_with_support(p: tp.Callable,
                                              q: tp.Callable,
                                              a: float,
                                              b: float,
                                              base: float = np.e,
                                              eps_abs: float = 1.49e-08,
                                              eps_rel: float = 1.49e-08) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p

                H_q(p) = - E_p [log(q)]

    via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    return -sp.integrate.quad(lambda x: _cross_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun),
                              a=a,
                              b=b,
                              epsabs=eps_abs,
                              epsrel=eps_rel)[0]

    # return (quadpy
    #         .line_segment
    #         .integrate_adaptive(
    #             f=lambda x: _vectorized_cross_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun),
    #             intervals=[a, b],
    #             eps_abs=eps_abs,
    #             eps_rel=eps_rel,
    #             kronrod_degree=15)[0])


def _does_support_overlap(p: sm.nonparametric.KDEUnivariate,
                          q: sm.nonparametric.KDEUnivariate) -> bool:
    """
    Determine whether the support of distributions of kernel density estimates p and q overlap.

    Parameters
    ----------
    p: statsmodels kde object representing an approximation of the distribution p
    q: statsmodels kde object representing an approximation of the distribution q

    Returns
    -------
    whether the support of distributions of kernel density estimates p and q overlap
    """
    return intersection(min(p.support), max(p.support), min(q.support), max(q.support)) is not None


def cross_entropy_from_kde(p: sm.nonparametric.KDEUnivariate,
                           q: sm.nonparametric.KDEUnivariate,
                           base: float = np.e,
                           eps_abs: float = 1.49e-08,
                           eps_rel: float = 1.49e-08) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p

                H_q(p) = - E_p [log(q)]

    given by the statsmodels kde objects via numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.
    """
    if not _does_support_overlap(p, q):
        raise ValueError('The support of p and q does not overlap.')

    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))

    return cross_entropy_from_densities_with_support(p=p.evaluate,
                                                     q=q.evaluate,
                                                     a=a,
                                                     b=b,
                                                     base=base,
                                                     eps_abs=eps_abs,
                                                     eps_rel=eps_rel)


def continuous_cross_entropy_from_sample(sample_p: np.ndarray,
                                         sample_q: np.ndarray,
                                         base: float = np.e,
                                         eps_abs: float = 1.49e-08,
                                         eps_rel: float = 1.49e-08) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p

                H_q(p) = - E_p [log(q)]

    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.

    """
    kde_p = sm.nonparametric.KDEUnivariate(sample_p)
    kde_p.fit()
    kde_q = sm.nonparametric.KDEUnivariate(sample_q)
    kde_q.fit()

    return cross_entropy_from_kde(kde_p, kde_q, base=base, eps_abs=eps_abs, eps_rel=eps_rel)


################################################################################
# Relative Entropy (KL Divergence)
################################################################################
def _relative_entropy_integrand(p: tp.Callable,
                                q: tp.Callable,
                                x: float,
                                log_fun: tp.Callable = np.log) -> float:
    """
    Compute the integrand p(x) * log(p(x) / q(x)) at a given point x for the calculation of relative
    entropy.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    x: the point at which to evaluate the integrand
    base: the base of the logarithm used to control the units of measurement for the result

    Returns
    -------
    Integrand for the relative entropy calculation
    """
    qx = q(x)
    px = p(x)
    if qx == 0.0:
        if px == 0.0:
            return 0.0
        else:
            raise ValueError(f'q(x) is zero at x={x} but p(x) is not')
    elif px == 0.0:
        return 0.0
    else:
        return px * log_fun(px / qx)


def _vectorized_relative_entropy_integrand(p: tp.Callable,
                                           q: tp.Callable,
                                           x: np.ndarray,
                                           log_fun: tp.Callable = np.log) -> np.ndarray:
    """
    Compute the integrand p(x) * log(p(x) / q(x)) vectorized at given points x for the calculation
    of relative entropy.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    x: the point at which to evaluate the integrand
    base: the base of the logarithm used to control the units of measurement for the result

    Returns
    -------
    Integrand for the cross entropy calculation
    """
    # return p(x) * log_fun(q(x) + 1e-12)
    qx = q(x)
    px = p(x)

    q_positive_index = qx > 0.0
    p_positive_index = px > 0.0

    q_zero_but_p_positive_index = ~q_positive_index & p_positive_index
    if np.any(q_zero_but_p_positive_index):
        raise ValueError(f'q(x) is zero at x={x[q_zero_but_p_positive_index]} but p(x) is not')

    return np.where(p_positive_index, px * log_fun(px / qx), 0.0)


def relative_entropy_from_densities_with_support(p: tp.Callable,
                                                 q: tp.Callable,
                                                 a: float,
                                                 b: float,
                                                 base: float = np.e,
                                                 eps_abs: float = 1.49e-08,
                                                 eps_rel: float = 1.49e-08
                                                 ) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p

                D_KL(p||q) = E_p [log(p/q)]

    via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    def integrand(x: float):
        return _relative_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun)

    return sp.integrate.quad(integrand,
                             a=a,
                             b=b,
                             epsabs=eps_abs,
                             epsrel=eps_rel)[0]

    # return (quadpy
    #         .line_segment
    #         .integrate_adaptive(
    #             f=lambda x: _vectorized_relative_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun),
    #             intervals=[a, b],
    #             eps_abs=eps_abs,
    #             eps_rel=eps_rel,
    #             kronrod_degree=10)[0])


def relative_entropy_from_kde(p: sm.nonparametric.KDEUnivariate,
                              q: sm.nonparametric.KDEUnivariate,
                              base: float = np.e,
                              eps_abs: float = 1.49e-08,
                              eps_rel: float = 1.49e-08) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p

                D_KL(p||q) E_p [log(p/q)]

    given by the statsmodels kde objects via numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    if not _does_support_overlap(p, q):
        raise ValueError('The support of p and q does not overlap.')

    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))
    return relative_entropy_from_densities_with_support(p=p.evaluate,
                                                        q=q.evaluate,
                                                        a=a,
                                                        b=b,
                                                        base=base,
                                                        eps_abs=eps_abs,
                                                        eps_rel=eps_rel)


def continuous_relative_entropy_from_sample(sample_p: np.ndarray,
                                            sample_q: np.ndarray,
                                            base: float = np.e,
                                            eps_abs: float = 1.49e-08,
                                            eps_rel: float = 1.49e-08) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p

                D_KL(p||q) = E_p [log(p/q)]

    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    kde_p = sm.nonparametric.KDEUnivariate(sample_p)
    kde_p.fit()
    kde_q = sm.nonparametric.KDEUnivariate(sample_q)
    kde_q.fit()

    return relative_entropy_from_kde(p=kde_p,
                                     q=kde_q,
                                     base=base,
                                     eps_abs=eps_abs,
                                     eps_rel=eps_rel)


################################################################################
# Jensen-Shannon Divergence
###############################################################################
def _relative_entropy_from_densities_with_support_for_shannon_divergence(
        p: tp.Callable,
        q: tp.Callable,
        a: float,
        b: float,
        log_fun: tp.Callable = np.log,
        eps_abs: float = 1.49e-08,
        eps_rel: float = 1.49e-08) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p

                D_KL(p||q) = E_p [log(p/q)]

    via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.

    """
    def integrand(x):
        return p(x) * log_fun(p(x) / q(x)) if p(x) > 0.0 else 0.0

    return sp.integrate.quad(integrand, a=a, b=b, epsabs=eps_abs, epsrel=eps_rel)[0]


def jensen_shannon_divergence_from_densities_with_support(p: tp.Callable,
                                                          q: tp.Callable,
                                                          a: float,
                                                          b: float,
                                                          base: float = np.e,
                                                          eps_abs: float = 1.49e-08,
                                                          eps_rel: float = 1.49e-08) \
        -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q

                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)

    via numerical integration from a to b.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.

    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    m = lambda x: 0.5 * (p(x) + q(x))
    D_PM = _relative_entropy_from_densities_with_support_for_shannon_divergence(
                p=p,
                q=m,
                a=a,
                b=b,
                log_fun=log_fun,
                eps_abs=eps_abs,
                eps_rel=eps_rel)

    D_QM = _relative_entropy_from_densities_with_support_for_shannon_divergence(
                p=q,
                q=m,
                a=a,
                b=b,
                log_fun=log_fun,
                eps_abs=eps_abs,
                eps_rel=eps_rel)

    return 0.5 * D_PM + 0.5 * D_QM


def jensen_shannon_divergence_from_kde(p: sm.nonparametric.KDEUnivariate,
                                       q: sm.nonparametric.KDEUnivariate,
                                       base: float = np.e,
                                       eps_abs: float = 1.49e-08,
                                       eps_rel: float = 1.49e-08) \
        -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q

                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)

    given by the statsmodels kde objects via numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.

    """
    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))
    return jensen_shannon_divergence_from_densities_with_support(p=p.evaluate,
                                                                 q=q.evaluate,
                                                                 a=a,
                                                                 b=b,
                                                                 base=base,
                                                                 eps_abs=eps_abs,
                                                                 eps_rel=eps_rel)


def continuous_jensen_shannon_divergence_from_sample(sample_p: np.ndarray,
                                                     sample_q: np.ndarray,
                                                     base: float = np.e,
                                                     eps_abs: float = 1.49e-08,
                                                     eps_rel: float = 1.49e-08) -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q

                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)

    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    sample_p: sample from the distribution p
    sample_q: sample from the distribution q
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.

    """
    kde_p = sm.nonparametric.KDEUnivariate(sample_p)
    kde_p.fit()
    kde_q = sm.nonparametric.KDEUnivariate(sample_q)
    kde_q.fit()

    return jensen_shannon_divergence_from_kde(kde_p,
                                              kde_q,
                                              base=base,
                                              eps_abs=eps_abs,
                                              eps_rel=eps_rel)


################################################################################
# Mutual Information
###############################################################################
def mutual_information_from_densities_with_support(pdf_x: tp.Callable,
                                                   pdf_y: tp.Callable,
                                                   pdf_xy: tp.Callable,
                                                   x_min: float,
                                                   x_max: float,
                                                   y_min: float,
                                                   y_max: float,
                                                   base: float = np.e,
                                                   eps_abs: float = 1.49e-08,
                                                   eps_rel: float = 1.49e-08
                                                   ) -> float:
    """
    Compute mutual information of the random variables x and y with joint density p_{x, y} and
    marginal densities p_x and p_y defined as the KL divergence between the product of marginal
    densities and the joint density, i.e.

            I(X; Y) = D_KL(p_{x, y}|| p_x \otimes p_y) =
            E_{p_{x, y}} \left[ \log \left( \frac{p_{x, y} (x, y)}{p_x(x) p_y(y)} \right) \right]

    via numerical integration on a rectangular domain aligned with the axes.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    pdf_x: probability density function of the random variable x
    pdf_y: probability density function of the random variable y
    pdf_xy: joint probability density function of the random variables x and y
    x_min: lower bound of the integration domain for x
    x_max: upper bound of the integration domain for x
    y_min: lower bound of the integration domain for y
    y_max: upper bound of the integration domain for y
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The mutual information of the random variables x and y
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    def mutual_information_integrand(x: float, y: float):
        pxy = pdf_xy((x, y))
        px = pdf_x(x)
        py = pdf_y(y)

        return pxy * log_fun(pxy / (px * py))

    return sp.integrate.dblquad(mutual_information_integrand,
                                a=x_min,
                                b=x_max,
                                gfun=lambda x: y_min,
                                hfun=lambda x: y_max,
                                epsabs=eps_abs,
                                epsrel=eps_rel)[0]


def mutual_information_from_kde(kde_x: sm.nonparametric.KDEUnivariate,
                                kde_y: sm.nonparametric.KDEUnivariate,
                                kde_xy: sp.stats.kde.gaussian_kde,
                                base: float = np.e,
                                eps_abs: float = 1.49e-08,
                                eps_rel: float = 1.49e-08) -> float:
    """
    Compute mutual information of the random variables x and y with joint density p_{x, y} and
    marginal densities p_x and p_y defined as the KL divergence between the product of marginal
    densities and the joint density, i.e.

            I(X; Y) = D_KL(p_{x, y}|| p_x \otimes p_y) =
            E_{p_{x, y}} \left[ \log \left( \frac{p_{x, y} (x, y)}{p_x(x) p_y(y)} \right) \right]

    given by the statsmodels kde objects for the marginal densities and a SciPy gaussian_kde object
    for the joint density via numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    kde_x: statsmodels kde object approximating the marginal density of x
    kde_y: statsmodels kde object approximating the marginal density of y
    kde_xy: SciPy gaussian_kde object approximating the joint density of x and y
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The mutual information of the random variables x and y
    """
    x_min = min(kde_x.support)
    x_max = max(kde_x.support)
    y_min = min(kde_y.support)
    y_max = max(kde_y.support)

    return mutual_information_from_densities_with_support(pdf_x=kde_x.evaluate,
                                                          pdf_y=kde_y.evaluate,
                                                          pdf_xy=kde_xy.pdf,
                                                          x_min=x_min,
                                                          x_max=x_max,
                                                          y_min=y_min,
                                                          y_max=y_max,
                                                          base=base,
                                                          eps_abs=eps_abs,
                                                          eps_rel=eps_rel)


def continuous_mutual_information_from_samples(sample_x: np.ndarray,
                                               sample_y: np.ndarray,
                                               base: float = np.e,
                                               eps_abs: float = 1.49e-08,
                                               eps_rel: float = 1.49e-08) -> float:
    """
    Compute mutual information of the random variables x and y with joint density p_{x, y} and
    marginal densities p_x and p_y defined as the KL divergence between the product of marginal
    densities and the joint density, i.e.

            I(X; Y) = D_KL(p_{x, y}|| p_x \otimes p_y) =
            E_{p_{x, y}} \left[ \log \left( \frac{p_{x, y} (x, y)}{p_x(x) p_y(y)} \right) \right]

    from samples of the two distributions via approximation by kernel density estimates and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    sample_x: x-component of the sample from the joint density p_{x, y}
    sample_y: y-component of the sample from the joint density p_{x, y}
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The mutual information of the random variables x and y
    """
    kde_x = sm.nonparametric.KDEUnivariate(sample_x)
    kde_x.fit()
    kde_y = sm.nonparametric.KDEUnivariate(sample_y)
    kde_y.fit()

    kde_xy = sp.stats.gaussian_kde([sample_x, sample_y])

    return mutual_information_from_kde(kde_x=kde_x,
                                       kde_y=kde_y,
                                       kde_xy=kde_xy,
                                       base=base,
                                       eps_abs=eps_abs,
                                       eps_rel=eps_rel)


################################################################################
# Joint Entropy
###############################################################################
def joint_entropy_from_densities_with_support(pdf_xy: tp.Callable,
                                              x_min: float,
                                              x_max: float,
                                              y_min: float,
                                              y_max: float,
                                              base: float = np.e,
                                              eps_abs: float = 1.49e-08,
                                              eps_rel: float = 1.49e-08) -> float:
    """
    Compute joint entropy of the random variables x and y with joint density p_{x, y} defined as

            H(X, Y) = - E_{p_{x, y}} \left[ \log p_{x, y} (x, y) \right]

    via numerical integration on a rectangular domain aligned with the axes.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    pdf_xy: joint probability density function of the random variables x and y
    x_min: lower bound of the integration domain for x
    x_max: upper bound of the integration domain for x
    y_min: lower bound of the integration domain for y
    y_max: upper bound of the integration domain for y
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The joint entropy of the random variables x and y
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    def joint_entropy_integrand(x: float, y: float):
        pxy = pdf_xy((x, y))

        return pxy * log_fun(pxy)

    return - sp.integrate.dblquad(joint_entropy_integrand,
                                  a=x_min,
                                  b=x_max,
                                  gfun=lambda x: y_min,
                                  hfun=lambda x: y_max,
                                  epsabs=eps_abs,
                                  epsrel=eps_rel)[0]


def joint_entropy_from_kde(kde_xy: sp.stats.kde.gaussian_kde,
                           x_min: float,
                           x_max: float,
                           y_min: float,
                           y_max: float,
                           base: float = np.e,
                           eps_abs: float = 1.49e-08,
                           eps_rel: float = 1.49e-08) -> float:
    """
    Compute joint entropy of the random variables x and y with joint density p_{x, y} defined as

            H(X, Y) = - E_{p_{x, y}} \left[ \log p_{x, y} (x, y) \right]

    via numerical integration, where the joint density is given by a SciPy gaussian_kde object.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    kde_xy: SciPy gaussian_kde object approximating the joint density of x and y
    x_min: lower bound of the integration domain for x
    x_max: upper bound of the integration domain for x
    y_min: lower bound of the integration domain for y
    y_max: upper bound of the integration domain for y
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The joint entropy of the random variables x and y
    """

    return joint_entropy_from_densities_with_support(pdf_xy=kde_xy.pdf,
                                                     x_min=x_min,
                                                     x_max=x_max,
                                                     y_min=y_min,
                                                     y_max=y_max,
                                                     base=base,
                                                     eps_abs=eps_abs,
                                                     eps_rel=eps_rel)


def continuous_joint_entropy_from_samples(sample_x: np.ndarray,
                                          sample_y: np.ndarray,
                                          base: float = np.e,
                                          eps_abs: float = 1.49e-08,
                                          eps_rel: float = 1.49e-08) -> float:
    """
    Compute joint entropy of the random variables x and y with joint density p_{x, y} defined as

            H(X, Y) = - E_{p_{x, y}} \left[ \log p_{x, y} (x, y) \right]

    from samples of the two distributions via approximation by kernel density estimates and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    sample_x: x-component of the sample from the joint density p_{x, y}
    sample_y: y-component of the sample from the joint density p_{x, y}
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The joint entropy of the random variables x and y
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    kde_x = sm.nonparametric.KDEUnivariate(sample_x)
    kde_x.fit()
    kde_y = sm.nonparametric.KDEUnivariate(sample_y)
    kde_y.fit()

    kde_xy = sp.stats.gaussian_kde([sample_x, sample_y])

    x_min = min(kde_x.support)
    x_max = max(kde_x.support)
    y_min = min(kde_y.support)
    y_max = max(kde_y.support)

    return joint_entropy_from_kde(kde_xy=kde_xy,
                                  x_min=x_min,
                                  x_max=x_max,
                                  y_min=y_min,
                                  y_max=y_max,
                                  base=base,
                                  eps_abs=eps_abs,
                                  eps_rel=eps_rel)


################################################################################
# Conditional Entropy
###############################################################################
def conditional_entropy_from_densities_with_support(pdf_x: tp.Callable,
                                                    pdf_xy: tp.Callable,
                                                    x_min: float,
                                                    x_max: float,
                                                    y_min: float,
                                                    y_max: float,
                                                    base: float = np.e,
                                                    eps_abs: float = 1.49e-08,
                                                    eps_rel: float = 1.49e-08
                                                    ) -> float:
    """
    Compute conditional entropy of the random variables x and y with joint density p_{x, y} and
    marginal density p_x defined as

            H(Y|X) = - E_{p_{x, y}} \left[ \log \frac{p_{x, y} (x, y)}{p_x(x)} \right]

    via numerical integration on a rectangular domain aligned with the axes.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    pdf_x: probability density function of the random variable x
    pdf_xy: joint probability density function of the random variables x and y
    x_min: lower bound of the integration domain for x
    x_max: upper bound of the integration domain for x
    y_min: lower bound of the integration domain for y
    y_max: upper bound of the integration domain for y
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The conditional entropy of the random variables x and y
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    def conditional_entropy_integrand(x: float, y: float):
        pxy = pdf_xy((x, y))
        px = pdf_x(x)

        return pxy * log_fun(pxy / px)

    return - sp.integrate.dblquad(conditional_entropy_integrand,
                                  a=x_min,
                                  b=x_max,
                                  gfun=lambda x: y_min,
                                  hfun=lambda x: y_max,
                                  epsabs=eps_abs,
                                  epsrel=eps_rel)[0]


def conditional_entropy_from_kde(kde_x: sm.nonparametric.KDEUnivariate,
                                 kde_xy: sp.stats.kde.gaussian_kde,
                                 y_min: float,
                                 y_max: float,
                                 base: float = np.e,
                                 eps_abs: float = 1.49e-08,
                                 eps_rel: float = 1.49e-08) -> float:
    """
    Compute conditional entropy of the random variables x and y with joint density p_{x, y} and
    marginal density p_x defined as

            H(Y|X) = - E_{p_{x, y}} \left[ \log \frac{p_{x, y} (x, y)}{p_x(x)} \right]

    via numerical integration, where the marginal density of x is given by a statsmodels kde object
    and the joint density by a SciPy gaussian_kde object.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    kde_x: statsmodels kde object approximating the marginal density of x
    kde_y: statsmodels kde object approximating the marginal density of y
    kde_xy: SciPy gaussian_kde object approximating the joint density of x and y
    y_min: lower bound of the integration domain for y
    y_max: upper bound of the integration domain for y
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The conditional entropy of the random variables x and y
    """
    x_min = min(kde_x.support)
    x_max = max(kde_x.support)

    return conditional_entropy_from_densities_with_support(pdf_x=kde_x.evaluate,
                                                           pdf_xy=kde_xy.pdf,
                                                           x_min=x_min,
                                                           x_max=x_max,
                                                           y_min=y_min,
                                                           y_max=y_max,
                                                           base=base,
                                                           eps_abs=eps_abs,
                                                           eps_rel=eps_rel)


def continuous_conditional_entropy_from_samples(sample_x: np.ndarray,
                                                sample_y: np.ndarray,
                                                base: float = np.e,
                                                eps_abs: float = 1.49e-08,
                                                eps_rel: float = 1.49e-08) -> float:
    """
    Compute conditional entropy of the random variables x and y with joint density p_{x, y} and
    marginal density p_x defined as

            H(Y|X) = - E_{p_{x, y}} \left[ \log \frac{p_{x, y} (x, y)}{p_x(x)} \right]

    from samples of the two distributions via approximation by kernel density estimates and
    numerical integration.
    The argument base can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    sample_x: x-component of the sample from the joint density p_{x, y}
    sample_y: y-component of the sample from the joint density p_{x, y}
    base: the base of the logarithm used to control the units of measurement for the result
    eps_abs: absolute error tolerance for numerical integration
    eps_rel: relative error tolerance for numerical integration

    Returns
    -------
    The conditional entropy of the random variables x and y
    """
    kde_x = sm.nonparametric.KDEUnivariate(sample_x)
    kde_x.fit()
    kde_y = sm.nonparametric.KDEUnivariate(sample_y)
    kde_y.fit()

    kde_xy = sp.stats.gaussian_kde([sample_x, sample_y])

    return conditional_entropy_from_kde(kde_x=kde_x,
                                        kde_xy=kde_xy,
                                        y_min=min(kde_y.support),
                                        y_max=max(kde_y.support),
                                        base=base,
                                        eps_abs=eps_abs,
                                        eps_rel=eps_rel)
