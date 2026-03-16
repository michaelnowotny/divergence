"""Fast estimators for continuous information-theoretic measures.

Grid estimators: integrate over the KDE's pre-computed grid using the trapezoidal
rule, with interpolation (np.interp) instead of expensive KDE re-evaluation.

Resubstitution estimators: evaluate the KDE at the sample points and average.

These are orders of magnitude faster than cubature-based integration for the
`*_from_sample()` and `*_from_kde()` functions, while producing comparable accuracy.
"""

import numpy as np
import scipy as sp
import statsmodels.api as sm

from divergence.base import _select_vectorized_log_fun_for_base


def _interp_kde_density(
    kde: sm.nonparametric.KDEUnivariate, points: np.ndarray
) -> np.ndarray:
    """Interpolate a KDE's pre-computed density onto new points.

    O(M) via np.interp instead of O(N*M) from kde.evaluate().
    Returns 0 for points outside the KDE's support range.
    """
    return np.interp(points, kde.support, kde.density, left=0.0, right=0.0)


# ---------------------------------------------------------------------------
# 1D Grid-based estimators (use KDE's pre-computed density grid)
# ---------------------------------------------------------------------------
def entropy_from_kde_grid(
    kde: sm.nonparametric.KDEUnivariate,
    base: float = np.e,
) -> float:
    r"""Compute entropy H(p) = -E_p[log p] using the KDE's grid.

    Uses the trapezoidal rule over the KDE's support grid, which is computed
    during ``kde.fit()``. This avoids expensive adaptive numerical integration.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)
    p = kde.density
    support = kde.support
    mask = p > 0
    integrand = np.where(mask, -p * log_fun(p), 0.0)
    return float(np.trapezoid(integrand, support))


def cross_entropy_from_kde_grid(
    kde_p: sm.nonparametric.KDEUnivariate,
    kde_q: sm.nonparametric.KDEUnivariate,
    base: float = np.e,
) -> float:
    r"""Compute cross entropy H_q(p) = -E_p[log q] using KDE grids.

    Interpolates q's density onto p's support grid (O(M) via np.interp),
    then integrates with the trapezoidal rule.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)
    support = kde_p.support
    p = kde_p.density
    q = _interp_kde_density(kde_q, support)

    mask = p > 0
    q_safe = np.where(q > 0, q, 1.0)
    integrand = np.where(mask, -p * log_fun(q_safe), 0.0)
    return float(np.trapezoid(integrand, support))


def relative_entropy_from_kde_grid(
    kde_p: sm.nonparametric.KDEUnivariate,
    kde_q: sm.nonparametric.KDEUnivariate,
    base: float = np.e,
) -> float:
    r"""Compute D_KL(p || q) = E_p[log(p/q)] using KDE grids."""
    log_fun = _select_vectorized_log_fun_for_base(base)
    support = kde_p.support
    p = kde_p.density
    q = _interp_kde_density(kde_q, support)

    mask = p > 0
    q_safe = np.where(q > 0, q, 1.0)
    integrand = np.where(mask, p * log_fun(p / q_safe), 0.0)
    return float(np.trapezoid(integrand, support))


def jensen_shannon_divergence_from_kde_grid(
    kde_p: sm.nonparametric.KDEUnivariate,
    kde_q: sm.nonparametric.KDEUnivariate,
    base: float = np.e,
) -> float:
    r"""Compute JSD(p || q) = 0.5 * D_KL(p||m) + 0.5 * D_KL(q||m) using KDE grids.

    Builds a common support grid and interpolates both KDEs onto it.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    # Build a common grid spanning both supports
    a = min(kde_p.support[0], kde_q.support[0])
    b = max(kde_p.support[-1], kde_q.support[-1])
    n_grid = max(len(kde_p.support), len(kde_q.support))
    support = np.linspace(a, b, n_grid)

    p = _interp_kde_density(kde_p, support)
    q = _interp_kde_density(kde_q, support)
    m = 0.5 * (p + q)

    m_safe = np.where(m > 0, m, 1.0)
    p_safe = np.where(p > 0, p, 1.0)
    q_safe = np.where(q > 0, q, 1.0)

    kl_pm = np.where(p > 0, p * log_fun(p_safe / m_safe), 0.0)
    kl_qm = np.where(q > 0, q * log_fun(q_safe / m_safe), 0.0)

    return float(
        0.5 * np.trapezoid(kl_pm, support) + 0.5 * np.trapezoid(kl_qm, support)
    )


# ---------------------------------------------------------------------------
# 2D Resubstitution estimators (evaluate densities at sample points)
# ---------------------------------------------------------------------------
def mutual_information_resubstitution(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    base: float = np.e,
) -> float:
    r"""Compute I(X;Y) via resubstitution: evaluate KDEs at sample points.

    I(X;Y) = E_{p_{xy}} [log(p_{xy}/(p_x * p_y))]
           ≈ (1/N) Σ log(p̂_{xy}(x_i,y_i) / (p̂_x(x_i) * p̂_y(y_i)))
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    # Fit marginal KDEs
    kde_x = sm.nonparametric.KDEUnivariate(sample_x)
    kde_x.fit()
    kde_y = sm.nonparametric.KDEUnivariate(sample_y)
    kde_y.fit()

    # Fit joint KDE
    kde_xy = sp.stats.gaussian_kde(np.vstack([sample_x, sample_y]))

    # Evaluate at sample points
    # Use interp for marginals (fast), gaussian_kde.evaluate for joint (BLAS)
    px = _interp_kde_density(kde_x, sample_x)
    py = _interp_kde_density(kde_y, sample_y)
    pxy = kde_xy.evaluate(np.vstack([sample_x, sample_y]))

    # Only include points where all densities are positive
    mask = (px > 0) & (py > 0) & (pxy > 0)
    if not np.any(mask):
        return 0.0

    ratios = pxy[mask] / (px[mask] * py[mask])
    return float(np.mean(log_fun(ratios)))


def mutual_information_from_kde_fast(
    kde_x: sm.nonparametric.KDEUnivariate,
    kde_y: sm.nonparametric.KDEUnivariate,
    kde_xy: sp.stats.gaussian_kde,
    base: float = np.e,
    n_grid: int = 100,
) -> float:
    r"""Compute I(X;Y) using a 2D grid + trapezoidal rule.

    Evaluates the joint and marginal KDEs on a meshgrid and integrates.
    Much faster than cubature for KDE-based densities.
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    x_grid = np.linspace(kde_x.support[0], kde_x.support[-1], n_grid)
    y_grid = np.linspace(kde_y.support[0], kde_y.support[-1], n_grid)
    xx, yy = np.meshgrid(x_grid, y_grid)
    positions = np.vstack([xx.ravel(), yy.ravel()])

    # Evaluate densities on the grid
    pxy = kde_xy.evaluate(positions).reshape(n_grid, n_grid)
    px = _interp_kde_density(kde_x, x_grid)
    py = _interp_kde_density(kde_y, y_grid)
    px_grid, py_grid = np.meshgrid(px, py)

    # Compute integrand: pxy * log(pxy / (px * py))
    denom = px_grid * py_grid
    mask = (pxy > 0) & (denom > 0)
    integrand = np.where(
        mask, pxy * log_fun(pxy / np.where(denom > 0, denom, 1.0)), 0.0
    )

    # 2D trapezoidal integration
    dx = x_grid[1] - x_grid[0]
    dy = y_grid[1] - y_grid[0]
    return float(np.trapezoid(np.trapezoid(integrand, dx=dy, axis=0), dx=dx))


def joint_entropy_resubstitution(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    base: float = np.e,
) -> float:
    r"""Compute H(X,Y) via resubstitution.

    H(X,Y) = -E_{p_{xy}} [log p_{xy}(x,y)]
           ≈ -(1/N) Σ log p̂_{xy}(x_i, y_i)
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    kde_xy = sp.stats.gaussian_kde(np.vstack([sample_x, sample_y]))
    pxy = kde_xy.evaluate(np.vstack([sample_x, sample_y]))

    mask = pxy > 0
    if not np.any(mask):
        return 0.0

    return float(-np.mean(log_fun(pxy[mask])))


def conditional_entropy_resubstitution(
    sample_x: np.ndarray,
    sample_y: np.ndarray,
    base: float = np.e,
) -> float:
    r"""Compute H(Y|X) via resubstitution.

    H(Y|X) = -E_{p_{xy}} [log(p_{xy}(x,y) / p_x(x))]
           ≈ -(1/N) Σ log(p̂_{xy}(x_i,y_i) / p̂_x(x_i))
    """
    log_fun = _select_vectorized_log_fun_for_base(base)

    kde_x = sm.nonparametric.KDEUnivariate(sample_x)
    kde_x.fit()
    kde_xy = sp.stats.gaussian_kde(np.vstack([sample_x, sample_y]))

    px = _interp_kde_density(kde_x, sample_x)
    pxy = kde_xy.evaluate(np.vstack([sample_x, sample_y]))

    mask = (px > 0) & (pxy > 0)
    if not np.any(mask):
        return 0.0

    return float(-np.mean(log_fun(pxy[mask] / px[mask])))
