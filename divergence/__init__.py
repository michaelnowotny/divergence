import numpy as np
import scipy as sp
import statsmodels.api as sm
import typing as tp


def intersection(a0: float,
                 b0: float,
                 a1: float,
                 b1: float) \
        -> tp.Optional[tp.Tuple[float, float]]:
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
    def integrand(x):
        return pdf(x) * log_fun(pdf(x)) if pdf(x) > 0.0 else 0.0

    return -sp.integrate.quad(integrand, a=a, b=b)[0]


def compute_entropy_from_kde(kde: sm.nonparametric.KDEUnivariate,
                             log_fun: tp.Callable = np.log) -> float:
    a = min(kde.support)
    b = max(kde.support)
    return compute_entropy_from_density_with_support(pdf=kde.evaluate,
                                                     a=a,
                                                     b=b,
                                                     log_fun=log_fun)


def compute_entropy_from_samples(samples: np.ndarray,
                                 log_fun: tp.Callable = np.log) -> float:
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
    # print(f'computing cross entropy from {a} to {b}')

    return -sp.integrate.quad(lambda x: _cross_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun),
                              a=a,
                              b=b)[0]


def _does_support_overlap(p: sm.nonparametric.KDEUnivariate,
                          q: sm.nonparametric.KDEUnivariate) -> bool:
    return intersection(min(p.support), max(p.support), min(q.support), max(q.support)) is not None


def compute_cross_entropy_from_kde(p: sm.nonparametric.KDEUnivariate,
                                   q: sm.nonparametric.KDEUnivariate,
                                   log_fun: tp.Callable = np.log) -> float:
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
    # print(f'computing relative entropy from {a} to {b}')

    return sp.integrate.quad(lambda x: _relative_entropy_integrand(p=p, q=q, x=x, log_fun=log_fun),
                             a=a,
                             b=b)[0]


def compute_relative_entropy_from_kde(p: sm.nonparametric.KDEUnivariate,
                                      q: sm.nonparametric.KDEUnivariate,
                                      log_fun: tp.Callable = np.log) -> float:
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
    def integrand(x):
        return p(x) * log_fun(p(x) / q(x)) if p(x) > 0.0 else 0.0

    return sp.integrate.quad(integrand, a=a, b=b)[0]


def compute_jensen_shannon_divergence_from_densities_with_support(p: tp.Callable,
                                                                  q: tp.Callable,
                                                                  a: float,
                                                                  b: float,
                                                                  log_fun: tp.Callable = np.log) \
        -> float:
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


def compute_jensen_shannon_divergence_from_kde(kde_p: sm.nonparametric.KDEUnivariate,
                                               kde_q: sm.nonparametric.KDEUnivariate,
                                               log_fun: tp.Callable = np.log) \
        -> float:
    a = min(min(kde_p.support), min(kde_q.support))
    b = max(max(kde_p.support), max(kde_q.support))
    return compute_jensen_shannon_divergence_from_densities_with_support(p=kde_p.evaluate,
                                                                         q=kde_q.evaluate,
                                                                         a=a,
                                                                         b=b,
                                                                         log_fun=log_fun)


def compute_jensen_shannon_divergence_from_samples(samples_p: np.ndarray,
                                                   samples_q: np.ndarray,
                                                   log_fun: tp.Callable = np.log) -> float:
    kde_p = sm.nonparametric.KDEUnivariate(samples_p)
    kde_q = sm.nonparametric.KDEUnivariate(samples_q)

    return compute_jensen_shannon_divergence_from_kde(kde_p, kde_q, log_fun=log_fun)
