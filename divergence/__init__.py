import numpy as np
import scipy as sp
import statsmodels.api as sm
import typing as tp


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
def compute_entropy_from_density_with_support(pdf: tp.Callable,
                                              a: float,
                                              b: float,
                                              log_fun: tp.Callable = np.log) \
        -> float:
    """
    Compute the entropy

                H(p) = - E_p[log(p)]

    of the density given in pdf via numerical integration from a to b.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    pdf: a function of a scalar parameter which computes the probability density at that point
    a: lower bound of the integration region
    b: upper bound of the integration region
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The entropy of the density given by pdf
    """
    def integrand(x):
        return pdf(x) * log_fun(pdf(x)) if pdf(x) > 0.0 else 0.0

    return -sp.integrate.quad(integrand, a=a, b=b)[0]


def compute_entropy_from_kde(kde: sm.nonparametric.KDEUnivariate,
                             log_fun: tp.Callable = np.log) -> float:
    """
    Compute the entropy

                H(p) = - E_p[log(p)]

    of the density given by the statsmodels kde object via numerical integration.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    kde: statsmodels kde object representing an approximation of the density
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The entropy of the density approximated by the kde
    """
    a = min(kde.support)
    b = max(kde.support)
    return compute_entropy_from_density_with_support(pdf=kde.evaluate,
                                                     a=a,
                                                     b=b,
                                                     log_fun=log_fun)


def compute_entropy_from_samples(samples: np.ndarray,
                                 log_fun: tp.Callable = np.log) -> float:
    """
    Compute the entropy

                H(p) = - E_p[log(p)]

    of a sample via approximation by a kernel density estimate and numerical integration.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    samples: a 1-dimensional numpy array of samples from the density
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The entropy of the density approximated by the draws of samples
    """
    kde = sm.nonparametric.KDEUnivariate(samples)
    return compute_entropy_from_kde(kde=kde,
                                    log_fun=log_fun)


################################################################################
# Cross Entropy
################################################################################
def _cross_entropy_integrand(p: tp.Callable,
                             q: tp.Callable,
                             x: float,
                             log_fun: tp.Callable = np.log) -> float:
    """
    Compute the integrand p(x) * log(q(x)) at a given point x for the calculation of cross entropy.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    x: the point at which to evaluate the integrand
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    Integrand for the cross entropy calculation
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
        return px * log_fun(qx)


def compute_cross_entropy_from_densities_with_support(p: tp.Callable,
                                                      q: tp.Callable,
                                                      a: float,
                                                      b: float,
                                                      log_fun: tp.Callable = np.log) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p

                H(p, q) = - E_p [log(q)]

    via numerical integration from a to b.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.
    """

    return -sp.integrate.quad(lambda x: _cross_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun),
                              a=a,
                              b=b)[0]


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


def compute_cross_entropy_from_kde(p: sm.nonparametric.KDEUnivariate,
                                   q: sm.nonparametric.KDEUnivariate,
                                   log_fun: tp.Callable = np.log) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p

                H(p, q) = - E_p [log(q)]

    given by the statsmodels kde objects via numerical integration.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.
    """
    if not _does_support_overlap(p, q):
        raise ValueError('The support of p and q does not overlap.')

    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))

    return compute_cross_entropy_from_densities_with_support(p=p.evaluate,
                                                             q=q.evaluate,
                                                             a=a,
                                                             b=b,
                                                             log_fun=log_fun)


def compute_cross_entropy_from_samples(samples_p: np.ndarray,
                                       samples_q: np.ndarray,
                                       log_fun: tp.Callable = np.log) -> float:
    """
    Compute the cross entropy of the distribution q relative to the distribution p

                H(p, q) = - E_p [log(q)]

    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    samples_p: numpy array of samples from the distribution p
    samples_q: numpy array of samples from the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The cross entropy of the distribution q relative to the distribution p.
    """
    kde_p = sm.nonparametric.KDEUnivariate(samples_p)
    kde_q = sm.nonparametric.KDEUnivariate(samples_q)

    return compute_cross_entropy_from_kde(kde_p, kde_q, log_fun=log_fun)


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
    log_fun: logarithmic function to control the units of measurement for the result

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


def compute_relative_entropy_from_densities_with_support(p: tp.Callable,
                                                         q: tp.Callable,
                                                         a: float,
                                                         b: float,
                                                         log_fun: tp.Callable = np.log) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p

                D_KL(p||q) = E_p [log(p/q)]

    via numerical integration from a to b.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    return sp.integrate.quad(lambda x: _relative_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun),
                             a=a,
                             b=b)[0]


def compute_relative_entropy_from_kde(p: sm.nonparametric.KDEUnivariate,
                                      q: sm.nonparametric.KDEUnivariate,
                                      log_fun: tp.Callable = np.log) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p

                D_KL(p||q) E_p [log(p/q)]

    given by the statsmodels kde objects via numerical integration.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    if not _does_support_overlap(p, q):
        raise ValueError('The support of p and q does not overlap.')

    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))
    return compute_relative_entropy_from_densities_with_support(p=p.evaluate,
                                                                q=q.evaluate,
                                                                a=a,
                                                                b=b,
                                                                log_fun=log_fun)


def compute_relative_entropy_from_samples(samples_p: np.ndarray,
                                          samples_q: np.ndarray,
                                          log_fun: tp.Callable = np.log) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p

                D_KL(p||q) = E_p [log(p/q)]

    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    samples_p: numpy array of samples from the distribution p
    samples_q: numpy array of samples from the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    kde_p = sm.nonparametric.KDEUnivariate(samples_p)
    kde_q = sm.nonparametric.KDEUnivariate(samples_q)

    return compute_relative_entropy_from_kde(kde_p, kde_q, log_fun=log_fun)


################################################################################
# Jensen-Shannon Divergence
###############################################################################
def _compute_relative_entropy_from_densities_with_support_for_shannon_divergence(
        p: tp.Callable,
        q: tp.Callable,
        a: float,
        b: float,
        log_fun: tp.Callable = np.log) -> float:
    """
    Compute the relative entropy of the distribution q relative to the distribution p

                D_KL(p||q) = E_p [log(p/q)]

    via numerical integration from a to b.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The relative entropy of the distribution q relative to the distribution p.
    """
    def integrand(x):
        return p(x) * log_fun(p(x) / q(x)) if p(x) > 0.0 else 0.0

    return sp.integrate.quad(integrand, a=a, b=b)[0]


def compute_jensen_shannon_divergence_from_densities_with_support(p: tp.Callable,
                                                                  q: tp.Callable,
                                                                  a: float,
                                                                  b: float,
                                                                  log_fun: tp.Callable = np.log) \
        -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q

                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)

    via numerical integration from a to b.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: probability density function of the distribution p
    q: probability density function of the distribution q
    a: lower bound of the integration region
    b: upper bound of the integration region
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.
    """
    m = lambda x: 0.5 * (p(x) + q(x))
    D_PM = _compute_relative_entropy_from_densities_with_support_for_shannon_divergence(
                p=p,
                q=m,
                a=a,
                b=b,
                log_fun=log_fun)

    D_QM = _compute_relative_entropy_from_densities_with_support_for_shannon_divergence(
                p=q,
                q=m,
                a=a,
                b=b,
                log_fun=log_fun)

    return 0.5 * D_PM + 0.5 * D_QM


def compute_jensen_shannon_divergence_from_kde(p: sm.nonparametric.KDEUnivariate,
                                               q: sm.nonparametric.KDEUnivariate,
                                               log_fun: tp.Callable = np.log) \
        -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q

                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)

    given by the statsmodels kde objects via numerical integration.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    p: statsmodels kde object approximating the probability density function of the distribution p
    q: statsmodels kde object approximating the probability density function of the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.
    """
    a = min(min(p.support), min(q.support))
    b = max(max(p.support), max(q.support))
    return compute_jensen_shannon_divergence_from_densities_with_support(p=p.evaluate,
                                                                         q=q.evaluate,
                                                                         a=a,
                                                                         b=b,
                                                                         log_fun=log_fun)


def compute_jensen_shannon_divergence_from_samples(samples_p: np.ndarray,
                                                   samples_q: np.ndarray,
                                                   log_fun: tp.Callable = np.log) -> float:
    """
    Compute the Jensen-Shannon divergence between distributions p and q

                JSD(p||q) = 0.5 * (D_KL(p||m) + D_KL(q||m)), with m = 0.5 * (p + q)

    from samples of the two distributions via approximation by a kernel density estimate and
    numerical integration.
    The argument log_fun can be used to specify the units in which the entropy is measured.
    The default choice is the natural logarithm.

    Parameters
    ----------
    samples_p: numpy array of samples from the distribution p
    samples_q: numpy array of samples from the distribution q
    log_fun: logarithmic function to control the units of measurement for the result

    Returns
    -------
    The Jensen-Shannon divergence between distributions p and q.
    """
    kde_p = sm.nonparametric.KDEUnivariate(samples_p)
    kde_q = sm.nonparametric.KDEUnivariate(samples_q)

    return compute_jensen_shannon_divergence_from_kde(kde_p, kde_q, log_fun=log_fun)
