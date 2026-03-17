"""ArviZ integration for Bayesian information-theoretic diagnostics.

Provides functions that accept ArviZ InferenceData (xarray DataTree) and
compute information-theoretic quantities: prior-to-posterior information gain,
inter-chain distributional divergence, predictive uncertainty decomposition,
per-observation Bayesian surprise, prior sensitivity analysis, and model
comparison via divergence measures.

ArviZ and xarray are optional dependencies. Functions raise ``ImportError``
with installation instructions if they are not available.

References
----------
.. [1] Kullback, S. & Leibler, R. A. (1951). "On Information and
       Sufficiency." Annals of Math. Stat., 22(1), 79-86.
.. [2] Lindley, D. V. (1956). "On a measure of the information provided
       by an experiment." Annals of Math. Stat., 27(4), 986-1005.
"""

from __future__ import annotations

import typing as tp

import numpy as np

if tp.TYPE_CHECKING:
    from divergence._types import ChainKSDResult, ChainTestResult, MixingDiagnostic
from scipy.special import logsumexp

from divergence.f_divergences import (
    squared_hellinger_distance,
    total_variation_distance,
)
from divergence.ipms import (
    energy_distance,
    maximum_mean_discrepancy,
    wasserstein_distance,
)
from divergence.knn import knn_entropy, knn_kl_divergence


# ---------------------------------------------------------------------------
# Lazy imports for optional dependencies
# ---------------------------------------------------------------------------
def _import_arviz():
    """Import arviz, raising a clear error if not installed."""
    try:
        import arviz as az
    except ImportError:
        raise ImportError(
            "ArviZ is required for Bayesian diagnostics. "
            "Install with: pip install divergence[bayesian]"
        ) from None
    return az


# ---------------------------------------------------------------------------
# Divergence dispatch
# ---------------------------------------------------------------------------
_1D_ONLY_METHODS = frozenset({"wasserstein", "hellinger", "tv", "js"})


def _get_divergence_fn(
    method: str,
) -> tp.Callable[[np.ndarray, np.ndarray], float]:
    """Map a divergence method name to a callable.

    Parameters
    ----------
    method : str
        One of ``"kl"``, ``"js"``, ``"hellinger"``, ``"tv"``,
        ``"wasserstein"``, ``"mmd"``, ``"energy"``.

    Returns
    -------
    callable
        Function ``(samples_p, samples_q) -> float``.

    Raises
    ------
    ValueError
        If *method* is not recognized.
    """
    if method == "kl":
        return lambda p, q: knn_kl_divergence(p, q, k=5)
    if method == "js":
        from divergence import jensen_shannon_divergence_from_samples

        return lambda p, q: jensen_shannon_divergence_from_samples(p, q)
    if method == "hellinger":
        return lambda p, q: squared_hellinger_distance(p, q)
    if method == "tv":
        return lambda p, q: total_variation_distance(p, q)
    if method == "wasserstein":
        return lambda p, q: wasserstein_distance(p, q)
    if method == "mmd":
        return lambda p, q: maximum_mean_discrepancy(p, q)
    if method == "energy":
        return lambda p, q: energy_distance(p, q)
    raise ValueError(
        f"Unknown divergence method '{method}'. "
        f"Supported: 'kl', 'js', 'hellinger', 'tv', 'wasserstein', 'mmd', 'energy'."
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _get_groups(idata: tp.Any) -> list[str]:
    """Return group names from an InferenceData or DataTree object.

    Handles both ArviZ ``InferenceData`` (which may have a ``.groups()``
    method) and raw ``xarray.DataTree`` (which has a ``.children``
    mapping).  For ``DataTree``, ``.groups`` returns path strings like
    ``('/', '/posterior', '/prior')``; this function strips the leading
    ``/`` and filters out the root.
    """
    # xarray DataTree: .children is a Mapping of child name -> DataTree
    if hasattr(idata, "children"):
        return list(idata.children)
    # ArviZ InferenceData: .groups() may be a method or a tuple property
    if hasattr(idata, "groups"):
        groups = idata.groups
        raw = list(groups()) if callable(groups) else list(groups)
        # DataTree.groups returns paths like ('/', '/posterior', '/prior')
        return [g.strip("/") for g in raw if g != "/"]
    return list(idata.keys())


def _validate_group(idata: tp.Any, group: str) -> None:
    """Check that *idata* contains *group*, raise with available groups."""
    available = _get_groups(idata)
    if group not in available:
        raise ValueError(
            f"InferenceData does not contain group '{group}'. "
            f"Available groups: {available}"
        )


def _get_var_names(
    dataset: tp.Any,
    var_names: list[str] | None,
    dataset2: tp.Any | None = None,
) -> list[str]:
    """Resolve variable names from one or two datasets.

    If *var_names* is ``None``, returns all data variables (or the
    intersection of two datasets' variables if *dataset2* is given).
    """
    if var_names is not None:
        for v in var_names:
            if v not in dataset.data_vars:
                raise KeyError(f"Variable '{v}' not found in dataset")
            if dataset2 is not None and v not in dataset2.data_vars:
                raise KeyError(f"Variable '{v}' not found in second dataset")
        return list(var_names)

    vars1 = set(dataset.data_vars)
    if dataset2 is not None:
        vars2 = set(dataset2.data_vars)
        return sorted(vars1 & vars2)
    return sorted(vars1)


def _get_values(dataset: tp.Any, var_name: str) -> np.ndarray:
    """Extract variable values in ``(chain, draw, ...)`` order.

    ArviZ backends may produce dimensions in either ``(chain, draw)`` or
    ``(draw, chain)`` order.  This helper inspects the xarray dimension
    names and transposes if necessary to guarantee ``(chain, draw, ...)``
    output.
    """
    var = dataset[var_name]
    arr = var.values
    dims = tuple(var.dims) if hasattr(var, "dims") else ()

    # Detect (draw, chain, ...) order and transpose to (chain, draw, ...)
    if len(dims) >= 2 and dims[0] == "draw" and dims[1] == "chain":
        # Move axis 1 (chain) to position 0
        arr = np.moveaxis(arr, 1, 0)

    return arr


def _flatten_samples(dataset: tp.Any, var_name: str) -> np.ndarray:
    """Extract and flatten samples, merging chain and draw dims.

    Returns shape ``(n_samples,)`` for scalar params or
    ``(n_samples, K)`` for vector params with K components.
    """
    arr = _get_values(dataset, var_name)  # (chain, draw, ...)
    n_chains, n_draws = arr.shape[:2]
    trailing = arr.shape[2:]
    if trailing:
        return arr.reshape(n_chains * n_draws, *trailing)
    return arr.reshape(n_chains * n_draws)


def _compute_divergence(
    samples_p: np.ndarray,
    samples_q: np.ndarray,
    div_fn: tp.Callable[[np.ndarray, np.ndarray], float],
    method: str,
    multivariate: bool = False,
) -> float | np.ndarray:
    """Compute divergence, handling scalar vs vector parameters.

    For 1D samples, calls *div_fn* directly.  For 2D samples
    (vector parameters), iterates over components unless
    *multivariate* is True.
    """
    if samples_p.ndim == 1:
        return float(div_fn(samples_p, samples_q))

    # Vector parameter: shape (n_samples, K)
    if multivariate:
        if method in _1D_ONLY_METHODS:
            raise ValueError(
                f"Method '{method}' only supports scalar (1D) parameters. "
                f"Use multivariate=False for component-wise computation, "
                f"or choose 'kl', 'mmd', or 'energy'."
            )
        return float(div_fn(samples_p, samples_q))

    # Component-wise
    k = samples_p.shape[1]
    results = np.empty(k)
    for i in range(k):
        results[i] = div_fn(samples_p[:, i], samples_q[:, i])
    return results


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def information_gain(
    idata: tp.Any,
    *,
    var_names: list[str] | None = None,
    method: str = "kl",
    multivariate: bool = False,
) -> dict[str, float | np.ndarray]:
    r"""Compute how much the data updated beliefs about each parameter.

    Measures D(posterior || prior) per parameter using the specified
    divergence. This is the information-theoretic analogue of asking
    "how informative was the data?"

    Parameters
    ----------
    idata : DataTree
        ArviZ InferenceData (``xarray.DataTree``). Must contain
        ``"posterior"`` and ``"prior"`` groups.
    var_names : list of str, optional
        Parameters to analyze. If ``None``, all parameters shared between
        posterior and prior are used.
    method : str, optional
        Divergence measure: ``"kl"`` (default, kNN-based), ``"js"``,
        ``"hellinger"``, ``"tv"``, ``"wasserstein"``, ``"mmd"``,
        ``"energy"``.
    multivariate : bool, optional
        If ``True``, compute multivariate divergence for vector parameters
        (requires ``"kl"``, ``"mmd"``, or ``"energy"``). If ``False``
        (default), compute per component.

    Returns
    -------
    dict[str, float | np.ndarray]
        Maps parameter names to divergence values. Scalar parameters
        produce ``float``; vector parameters produce ``np.ndarray``
        of per-component values (or ``float`` when *multivariate* is True).

    Raises
    ------
    ImportError
        If ArviZ is not installed.
    ValueError
        If required groups are missing or *method* is invalid.
    KeyError
        If a requested variable is not found in both groups.

    Notes
    -----
    The KL divergence :math:`D_{\text{KL}}(\text{posterior} \| \text{prior})`
    quantifies the information gained from the data in nats. A value
    near 0 means the data was uninformative for that parameter; a large
    value means the posterior differs substantially from the prior.

    The kNN-based KL estimator is the default because it is fast, requires
    no density estimation, and works in any dimension.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata = az.from_dict({
    ...     "posterior": {"mu": rng.normal(1, 0.5, (2, 500))},
    ...     "prior": {"mu": rng.normal(0, 1, (2, 500))},
    ... })
    >>> result = information_gain(idata)
    >>> result["mu"] > 0
    True

    References
    ----------
    .. [1] Kullback, S. & Leibler, R. A. (1951). "On Information and
           Sufficiency." Annals of Math. Stat., 22(1), 79-86.
    .. [2] Lindley, D. V. (1956). "On a measure of the information
           provided by an experiment." Annals of Math. Stat., 27(4),
           986-1005.
    """
    _import_arviz()
    _validate_group(idata, "posterior")
    _validate_group(idata, "prior")

    posterior = idata["posterior"]
    prior = idata["prior"]
    names = _get_var_names(posterior, var_names, prior)
    div_fn = _get_divergence_fn(method)

    results: dict[str, float | np.ndarray] = {}
    for name in names:
        post_samples = _flatten_samples(posterior, name)
        prior_samples = _flatten_samples(prior, name)
        results[name] = _compute_divergence(
            post_samples, prior_samples, div_fn, method, multivariate
        )
    return results


def chain_divergence(
    idata: tp.Any,
    *,
    var_names: list[str] | None = None,
    method: str = "energy",
    group: str = "posterior",
) -> dict[str, np.ndarray]:
    r"""Compute pairwise distributional divergence between MCMC chains.

    Complements R-hat with a full distributional comparison: R-hat checks
    whether chains have similar means and variances, while this function
    checks whether they sample from the same distribution (detecting
    differences in shape, skewness, and multimodality).

    Parameters
    ----------
    idata : DataTree
        ArviZ InferenceData with multiple chains in the specified *group*.
    var_names : list of str, optional
        Parameters to analyze. If ``None``, all parameters are used.
    method : str, optional
        Divergence measure. Default is ``"energy"`` (symmetric, works
        in any dimension, no density estimation). Other options:
        ``"mmd"``, ``"js"``, ``"kl"``, ``"hellinger"``, ``"tv"``,
        ``"wasserstein"``.
    group : str, optional
        Group to analyze. Default is ``"posterior"``.

    Returns
    -------
    dict[str, np.ndarray]
        Maps parameter names to pairwise divergence matrices of shape
        ``(n_chains, n_chains)``. Diagonal entries are zero (or near-zero);
        entry ``(i, j)`` is the divergence between chains *i* and *j*.

    Notes
    -----
    For symmetric methods (energy, MMD, JS, Hellinger, TV, Wasserstein),
    the matrix is symmetric. For KL divergence, entry ``(i, j)`` is
    :math:`D_{\text{KL}}(\text{chain}_i \| \text{chain}_j)`.

    Well-mixed chains should have small pairwise divergences. Large
    off-diagonal values indicate chains exploring different regions of
    parameter space.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata = az.from_dict({
    ...     "posterior": {"mu": rng.normal(0, 1, (4, 500))}
    ... })
    >>> result = chain_divergence(idata)
    >>> result["mu"].shape
    (4, 4)
    """
    _import_arviz()
    _validate_group(idata, group)
    dataset = idata[group]
    names = _get_var_names(dataset, var_names)
    div_fn = _get_divergence_fn(method)

    results: dict[str, np.ndarray] = {}
    for name in names:
        arr = _get_values(dataset, name)  # (chain, draw, ...)
        n_chains = arr.shape[0]
        matrix = np.zeros((n_chains, n_chains))

        # Extract per-chain samples
        chain_samples = []
        for c in range(n_chains):
            s = arr[c]  # (draw, ...) or (draw,)
            if s.ndim > 1:
                # Vector param: flatten draw dimension only
                s = s.reshape(s.shape[0], -1) if s.ndim > 1 else s
            chain_samples.append(s)

        for i in range(n_chains):
            for j in range(i + 1, n_chains):
                d = float(div_fn(chain_samples[i], chain_samples[j]))
                matrix[i, j] = d
                matrix[j, i] = d  # assume symmetric for display

        results[name] = matrix
    return results


def bayesian_surprise(
    idata: tp.Any,
    *,
    log_likelihood_group: str = "log_likelihood",
    var_name: str | None = None,
) -> dict[str, np.ndarray]:
    r"""Compute per-observation surprise (self-information).

    For each observation *i*, the surprise is:

    .. math::
        S(y_i) = -\log \mathbb{E}_\theta[p(y_i \mid \theta)]

    approximated via the log-sum-exp trick over MCMC draws.

    Parameters
    ----------
    idata : DataTree
        ArviZ InferenceData. Must contain the log-likelihood group.
    log_likelihood_group : str, optional
        Name of the log-likelihood group. Default is ``"log_likelihood"``.
    var_name : str, optional
        Variable name within the group. If ``None`` and the group contains
        exactly one variable, that variable is used.

    Returns
    -------
    dict[str, np.ndarray]
        Maps variable names to arrays of per-observation surprise values.
        Higher values indicate more surprising (influential or outlying)
        observations.

    Raises
    ------
    ValueError
        If the log-likelihood group is missing or *var_name* cannot be
        resolved.

    Notes
    -----
    Observations with high surprise are:

    - Influential data points driving the posterior
    - Potential outliers the model struggles to explain
    - Regions of model misspecification

    This is closely related to the pointwise WAIC/LOO computation but
    expressed in the information-theoretic framework.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata = az.from_dict({
    ...     "log_likelihood": {"y": rng.standard_normal((2, 500, 20))}
    ... })
    >>> result = bayesian_surprise(idata)
    >>> result["y"].shape
    (20,)

    References
    ----------
    .. [1] Itti, L. & Baldi, P. (2009). "Bayesian surprise attracts human
           attention." Vision Research, 49(10), 1295-1306.
    """
    _import_arviz()
    _validate_group(idata, log_likelihood_group)
    ll_group = idata[log_likelihood_group]

    # Resolve variable name
    ll_vars = list(ll_group.data_vars)
    if var_name is not None:
        if var_name not in ll_vars:
            raise ValueError(
                f"Variable '{var_name}' not found in '{log_likelihood_group}'. "
                f"Available: {ll_vars}"
            )
        names = [var_name]
    elif len(ll_vars) == 1:
        names = ll_vars
    else:
        names = ll_vars

    results: dict[str, np.ndarray] = {}
    for name in names:
        ll = _get_values(ll_group, name)  # (chain, draw, obs...)
        # Flatten chain and draw into sample dimension
        n_chains, n_draws = ll.shape[:2]
        obs_shape = ll.shape[2:]
        ll_flat = ll.reshape(n_chains * n_draws, *obs_shape)  # (n_samples, obs...)

        # S(y_i) = -log E_theta[p(y_i|theta)]
        #        = -(logsumexp(log_lik[:, i]) - log(n_samples))
        n_samples = ll_flat.shape[0]
        log_mean_lik = logsumexp(ll_flat, axis=0) - np.log(n_samples)
        surprise = -log_mean_lik
        results[name] = surprise.ravel() if obs_shape else np.array([surprise])

    return results


def model_divergence(
    idata1: tp.Any,
    idata2: tp.Any,
    *,
    var_names: list[str] | None = None,
    method: str = "energy",
    group: str = "posterior_predictive",
) -> dict[str, float]:
    r"""Compare predictive distributions from two models.

    Computes the divergence between corresponding variables in the
    specified group of two InferenceData objects.

    Parameters
    ----------
    idata1, idata2 : DataTree
        Two ArviZ InferenceData objects. Both must contain *group*.
    var_names : list of str, optional
        Variables to compare. If ``None``, all shared variables are used.
    method : str, optional
        Divergence measure. Default is ``"energy"`` (symmetric, no
        density estimation). Other options: ``"mmd"``, ``"js"``, ``"kl"``.
    group : str, optional
        Group to compare. Default is ``"posterior_predictive"``.

    Returns
    -------
    dict[str, float]
        Maps variable names to divergence values.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata1 = az.from_dict({
    ...     "posterior_predictive": {"y": rng.normal(0, 1, (2, 500))}
    ... })
    >>> idata2 = az.from_dict({
    ...     "posterior_predictive": {"y": rng.normal(1, 1, (2, 500))}
    ... })
    >>> result = model_divergence(idata1, idata2)
    >>> result["y"] > 0
    True
    """
    _import_arviz()
    _validate_group(idata1, group)
    _validate_group(idata2, group)

    ds1 = idata1[group]
    ds2 = idata2[group]
    names = _get_var_names(ds1, var_names, ds2)
    div_fn = _get_divergence_fn(method)

    results: dict[str, float] = {}
    for name in names:
        s1 = _flatten_samples(ds1, name)
        s2 = _flatten_samples(ds2, name)
        # Flatten any trailing dims for comparison
        if s1.ndim > 1:
            s1 = s1.reshape(s1.shape[0], -1)
            s2 = s2.reshape(s2.shape[0], -1)
        results[name] = float(div_fn(s1, s2))
    return results


def prior_sensitivity(
    idata: tp.Any,
    reference_prior_samples: dict[str, np.ndarray],
    *,
    var_names: list[str] | None = None,
    method: str = "kl",
) -> dict[str, dict[str, float]]:
    r"""Quantify how sensitive posteriors are to the choice of prior.

    Compares information gain under the actual prior versus a reference
    prior. High sensitivity means conclusions depend on the prior choice.

    Parameters
    ----------
    idata : DataTree
        ArviZ InferenceData with ``"posterior"`` and ``"prior"`` groups.
    reference_prior_samples : dict[str, np.ndarray]
        Maps parameter names to 1D numpy arrays of samples from a
        reference (e.g. vague/uninformative) prior.
    var_names : list of str, optional
        Parameters to analyze. If ``None``, all parameters present in
        posterior, prior, and *reference_prior_samples* are used.
    method : str, optional
        Divergence measure. Default is ``"kl"``.

    Returns
    -------
    dict[str, dict[str, float]]
        Maps parameter names to dicts with:

        - ``"actual"``: D(posterior || actual_prior)
        - ``"reference"``: D(posterior || reference_prior)
        - ``"sensitivity"``: ``abs(actual - reference)``

    Notes
    -----
    If sensitivity is high for a parameter, this suggests the data is
    not very informative for that parameter and conclusions depend on
    the prior. If sensitivity is low, the data dominates and the prior
    choice matters little.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata = az.from_dict({
    ...     "posterior": {"mu": rng.normal(1, 0.5, (2, 500))},
    ...     "prior": {"mu": rng.normal(0, 1, (2, 500))},
    ... })
    >>> ref = {"mu": rng.normal(0, 10, 5000)}
    >>> result = prior_sensitivity(idata, ref)
    >>> "sensitivity" in result["mu"]
    True
    """
    _import_arviz()
    _validate_group(idata, "posterior")
    _validate_group(idata, "prior")

    posterior = idata["posterior"]
    prior = idata["prior"]
    div_fn = _get_divergence_fn(method)

    # Resolve var_names: intersection of posterior, prior, and reference
    if var_names is None:
        post_vars = set(posterior.data_vars)
        prior_vars = set(prior.data_vars)
        ref_vars = set(reference_prior_samples.keys())
        names = sorted(post_vars & prior_vars & ref_vars)
    else:
        names = list(var_names)

    results: dict[str, dict[str, float]] = {}
    for name in names:
        post_samples = _flatten_samples(posterior, name)
        prior_samples = _flatten_samples(prior, name)
        ref_samples = reference_prior_samples[name]

        actual = float(div_fn(post_samples, prior_samples))
        reference = float(div_fn(post_samples, ref_samples))
        results[name] = {
            "actual": actual,
            "reference": reference,
            "sensitivity": abs(actual - reference),
        }
    return results


def uncertainty_decomposition(
    idata: tp.Any,
    *,
    var_names: list[str] | None = None,
    method: str = "knn",
    group: str = "posterior_predictive",
) -> dict[str, dict[str, float]]:
    r"""Decompose predictive uncertainty into aleatoric and epistemic.

    Uses the information-theoretic decomposition:

    .. math::
        \underbrace{H[p(y^* \mid y)]}_{\text{total}}
        = \underbrace{\mathbb{E}_\theta[H[p(y^* \mid \theta)]]}_{\text{aleatoric}}
        + \underbrace{I(y^*; \theta \mid y)}_{\text{epistemic}}

    Parameters
    ----------
    idata : DataTree
        ArviZ InferenceData with the posterior predictive group.
    var_names : list of str, optional
        Predicted variable names. If ``None``, all variables in the
        group are used.
    method : str, optional
        Entropy estimation method. ``"knn"`` (default) uses the
        Kozachenko-Leonenko kNN estimator. ``"kde"`` uses kernel
        density estimation.
    group : str, optional
        Predictive group name. Default is ``"posterior_predictive"``.

    Returns
    -------
    dict[str, dict[str, float]]
        Maps variable names to dicts with:

        - ``"total"``: H[p(y* | y)] — marginal predictive entropy
        - ``"aleatoric"``: E_theta[H[p(y* | theta)]] — average
          conditional entropy
        - ``"epistemic"``: total - aleatoric = I(y*; theta | y)

    Raises
    ------
    ValueError
        If the predictive group is missing or the observation dimension
        is too small (< 10) for reliable entropy estimation.

    Notes
    -----
    **Total entropy** is estimated from all posterior predictive samples
    pooled across chains and draws.

    **Aleatoric entropy** is estimated per MCMC draw: each
    ``(chain, draw)`` pair corresponds to a fixed parameter value
    theta, and the observation dimension provides the samples from
    ``p(y* | theta)``. The aleatoric estimate is the mean of these
    per-draw entropies.

    This requires the posterior predictive to have an observation
    dimension (e.g., shape ``(chain, draw, obs)``). If there is only
    one predicted value per draw, per-draw entropy cannot be estimated.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata = az.from_dict({
    ...     "posterior_predictive": {"y": rng.normal(0, 1, (2, 500, 50))}
    ... })
    >>> result = uncertainty_decomposition(idata)
    >>> result["y"]["epistemic"] >= -0.1
    True

    References
    ----------
    .. [1] Depeweg, S. et al. (2018). "Decomposition of Uncertainty in
           Bayesian Deep Learning for Efficient and Risk-sensitive
           Learning." ICML.
    """
    _import_arviz()
    _validate_group(idata, group)
    dataset = idata[group]
    names = _get_var_names(dataset, var_names)

    # Select entropy function
    if method == "knn":

        def entropy_fn(s: np.ndarray) -> float:
            return knn_entropy(s, k=5)

    elif method == "kde":
        from divergence import entropy_from_samples

        def entropy_fn(s: np.ndarray) -> float:
            return entropy_from_samples(s)

    else:
        raise ValueError(f"Unknown entropy method '{method}'. Supported: 'knn', 'kde'.")

    results: dict[str, dict[str, float]] = {}
    for name in names:
        arr = _get_values(dataset, name)  # (chain, draw, obs...)
        if arr.ndim < 3:
            raise ValueError(
                f"Variable '{name}' has shape {arr.shape} — need at least "
                f"3 dimensions (chain, draw, obs) for uncertainty "
                f"decomposition. The observation dimension is required "
                f"to estimate per-draw conditional entropy."
            )

        n_chains, n_draws = arr.shape[:2]
        obs_dim = arr.shape[2]
        if obs_dim < 10:
            raise ValueError(
                f"Variable '{name}' has only {obs_dim} observations per "
                f"draw. Need at least 10 for reliable entropy estimation."
            )

        # Total entropy: pool all samples, flatten to (n_total,)
        all_samples = arr.reshape(-1)
        total = float(entropy_fn(all_samples))

        # Aleatoric: average conditional entropy per (chain, draw)
        conditional_entropies = []
        for c in range(n_chains):
            for d in range(n_draws):
                obs_samples = arr[c, d]  # (obs,) or (obs, ...)
                if obs_samples.ndim > 1:
                    obs_samples = obs_samples.ravel()
                h = float(entropy_fn(obs_samples))
                conditional_entropies.append(h)

        aleatoric = float(np.mean(conditional_entropies))
        epistemic = total - aleatoric

        results[name] = {
            "total": total,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
        }
    return results


# ---------------------------------------------------------------------------
# MCMC Convergence Diagnostics
# ---------------------------------------------------------------------------
def chain_ksd(
    idata: tp.Any,
    score_fn: tp.Callable[[np.ndarray], np.ndarray],
    *,
    var_names: list[str] | None = None,
    group: str = "posterior",
    kernel: str = "imq",
    bandwidth: float | None = None,
    split: bool = True,
) -> dict[str, ChainKSDResult]:
    r"""Compute per-chain kernel Stein discrepancy against the target.

    Unlike :func:`chain_divergence`, which tests whether chains agree
    *with each other*, this function tests whether each chain has
    converged to the **correct target distribution**.  It is the only
    diagnostic that can detect the situation where *all* chains have
    converged to the *wrong* distribution.

    The KSD requires only the *score function*
    :math:`\nabla \log \pi(x)` of the target, which is available as a
    byproduct of gradient-based MCMC (HMC, NUTS).

    Parameters
    ----------
    idata : DataTree or InferenceData
        ArviZ inference data containing the specified *group*.
    score_fn : callable
        Score function of the target distribution.  Takes an array of
        shape ``(n, d)`` and returns an array of shape ``(n, d)`` with
        :math:`\nabla_x \log \pi(x)` at each point.  For scalar
        parameters (``d = 1``), the input and output shapes may be
        ``(n, 1)``.
    var_names : list of str, optional
        Parameters to analyze.  If ``None``, all parameters in the
        group are used.
    group : str, optional
        InferenceData group.  Default is ``"posterior"``.
    kernel : str, optional
        Kernel for the Stein operator: ``"imq"`` (inverse multiquadric,
        default) or ``"rbf"`` (Gaussian).  The IMQ kernel provides
        provable convergence control guarantees [1]_.
    bandwidth : float or None, optional
        Kernel bandwidth / scale parameter.  If ``None``, the median
        heuristic is used.
    split : bool, optional
        If ``True`` (default), additionally compute KSD for the first
        and second halves of each chain.  This is analogous to
        split-:math:`\hat{R}` and can detect non-stationarity: if the
        first-half KSD is much larger than the second-half KSD, the
        chain was still converging during the first half.

    Returns
    -------
    dict[str, ChainKSDResult]
        Maps parameter names to :class:`~divergence.ChainKSDResult`
        named tuples with fields ``ksd_per_chain``,
        ``ksd_split_first``, ``ksd_split_second``, and ``ksd_pooled``.

    Raises
    ------
    ImportError
        If ArviZ is not installed.
    ValueError
        If the specified group is missing from *idata*.

    Notes
    -----
    For the IMQ kernel :math:`k(x,y) = (c^2 + \|x-y\|^2)^{-1/2}`,
    Gorham and Mackey (2017) [1]_ proved that
    :math:`\mathrm{KSD}(\mu_n, \pi) \to 0` implies
    :math:`\mu_n \Rightarrow \pi` (weak convergence) and tightness of
    :math:`\{\mu_n\}`.  This is strictly stronger than what
    :math:`\hat{R}` can guarantee.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata = az.from_dict({
    ...     "posterior": {"mu": rng.normal(0, 1, (4, 500))}
    ... })
    >>> result = chain_ksd(idata, lambda x: -x)
    >>> result["mu"].ksd_pooled < 0.1
    True

    References
    ----------
    .. [1] Gorham, J. & Mackey, L. (2017). "Measuring sample quality
       with kernels." *ICML*.
    .. [2] Liu, Q., Lee, J., & Jordan, M. (2016). "A kernelized Stein
       discrepancy for goodness-of-fit tests." *ICML*.
    """
    from divergence._types import ChainKSDResult
    from divergence.score_based import kernel_stein_discrepancy

    _import_arviz()
    _validate_group(idata, group)
    dataset = idata[group]
    names = _get_var_names(dataset, var_names)

    results: dict[str, ChainKSDResult] = {}
    for name in names:
        arr = _get_values(dataset, name)  # (chain, draw, ...)
        n_chains, n_draws = arr.shape[:2]
        trailing = arr.shape[2:]

        ksd_per_chain = np.empty(n_chains)
        ksd_split_first = np.empty(n_chains) if split else None
        ksd_split_second = np.empty(n_chains) if split else None

        for c in range(n_chains):
            chain_samples = arr[c]  # (draw, ...)
            if trailing:
                chain_samples = chain_samples.reshape(n_draws, -1)
            ksd_per_chain[c] = kernel_stein_discrepancy(
                chain_samples, score_fn, kernel=kernel, bandwidth=bandwidth
            )

            if split:
                mid = n_draws // 2
                first = arr[c, :mid]
                second = arr[c, mid:]
                if trailing:
                    first = first.reshape(mid, -1)
                    second = second.reshape(n_draws - mid, -1)
                ksd_split_first[c] = kernel_stein_discrepancy(
                    first, score_fn, kernel=kernel, bandwidth=bandwidth
                )
                ksd_split_second[c] = kernel_stein_discrepancy(
                    second, score_fn, kernel=kernel, bandwidth=bandwidth
                )

        # Pooled: all chains combined
        pooled = _flatten_samples(dataset, name)
        ksd_pooled = kernel_stein_discrepancy(
            pooled, score_fn, kernel=kernel, bandwidth=bandwidth
        )

        results[name] = ChainKSDResult(
            ksd_per_chain=ksd_per_chain,
            ksd_split_first=ksd_split_first,
            ksd_split_second=ksd_split_second,
            ksd_pooled=float(ksd_pooled),
        )
    return results


def chain_two_sample_test(
    idata: tp.Any,
    *,
    var_names: list[str] | None = None,
    method: str = "mmd",
    group: str = "posterior",
    n_permutations: int = 500,
    seed: int | None = None,
) -> dict[str, ChainTestResult]:
    r"""Pairwise two-sample permutation tests between MCMC chains.

    Upgrades :func:`chain_divergence` from raw divergence magnitudes to
    **calibrated p-values**.  A small p-value for chains *i* and *j*
    indicates that they are sampling from detectably different
    distributions, suggesting a convergence failure.

    Parameters
    ----------
    idata : DataTree or InferenceData
        ArviZ inference data containing the specified *group*.
    var_names : list of str, optional
        Parameters to analyze.  If ``None``, all parameters are used.
    method : str, optional
        Test statistic: ``"mmd"`` (default), ``"energy"``, or
        ``"kl_knn"``.  These correspond to the methods supported by
        :func:`~divergence.two_sample_test`.
    group : str, optional
        InferenceData group.  Default is ``"posterior"``.
    n_permutations : int, optional
        Number of permutations for the null distribution.  Default is
        500.  Higher values give more precise p-values at increased
        computational cost.
    seed : int or None, optional
        Base random seed for reproducibility.  Per-pair seeds are
        derived as ``seed + i * n_chains + j`` to ensure statistical
        independence across chain pairs while remaining reproducible.

    Returns
    -------
    dict[str, ChainTestResult]
        Maps parameter names to :class:`~divergence.ChainTestResult`
        named tuples with fields ``p_value_matrix``,
        ``statistic_matrix``, ``min_p_value``, and ``any_significant``.

    Raises
    ------
    ImportError
        If ArviZ is not installed.
    ValueError
        If the specified group is missing or *method* is invalid.

    Notes
    -----
    For well-mixed chains, all pairwise p-values should be large
    (> 0.05).  If any pair has a small p-value, the chains are
    sampling from detectably different distributions, indicating a
    convergence problem.

    The MMD test statistic is recommended as the default because it is
    symmetric, works in any dimension, and is consistent against all
    alternatives when using a characteristic kernel.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata = az.from_dict({
    ...     "posterior": {"mu": rng.normal(0, 1, (4, 500))}
    ... })
    >>> result = chain_two_sample_test(idata, n_permutations=200, seed=42)
    >>> result["mu"].any_significant
    False

    References
    ----------
    .. [1] Gretton, A. et al. (2012). "A kernel two-sample test."
       *JMLR*, 13, 723-773.
    .. [2] Szekely, G. J. & Rizzo, M. L. (2004). "Testing for equal
       distributions in high dimension." *InterStat*.
    """
    from divergence._types import ChainTestResult
    from divergence.testing import two_sample_test

    _import_arviz()
    _validate_group(idata, group)
    dataset = idata[group]
    names = _get_var_names(dataset, var_names)

    results: dict[str, ChainTestResult] = {}
    for name in names:
        arr = _get_values(dataset, name)  # (chain, draw, ...)
        n_chains = arr.shape[0]

        # Extract per-chain samples
        chain_samples = []
        for c in range(n_chains):
            s = arr[c]  # (draw, ...) or (draw,)
            if s.ndim > 1:
                s = s.reshape(s.shape[0], -1)
            chain_samples.append(s)

        p_matrix = np.ones((n_chains, n_chains))
        s_matrix = np.zeros((n_chains, n_chains))

        for i in range(n_chains):
            for j in range(i + 1, n_chains):
                pair_seed = seed + i * n_chains + j if seed is not None else None
                test_result = two_sample_test(
                    chain_samples[i],
                    chain_samples[j],
                    method=method,
                    n_permutations=n_permutations,
                    seed=pair_seed,
                )
                p_matrix[i, j] = test_result.p_value
                p_matrix[j, i] = test_result.p_value
                s_matrix[i, j] = test_result.statistic
                s_matrix[j, i] = test_result.statistic

        triu_idx = np.triu_indices(n_chains, k=1)
        min_p = float(np.min(p_matrix[triu_idx])) if n_chains > 1 else 1.0

        results[name] = ChainTestResult(
            p_value_matrix=p_matrix,
            statistic_matrix=s_matrix,
            min_p_value=min_p,
            any_significant=min_p < 0.05,
        )
    return results


def mixing_diagnostic(
    idata: tp.Any,
    *,
    var_names: list[str] | None = None,
    group: str = "posterior",
    lag: int = 1,
    knn_k: int = 5,
) -> dict[str, MixingDiagnostic]:
    r"""Diagnose chain mixing using transfer entropy.

    Applies :func:`~divergence.transfer_entropy` to detect two types
    of mixing failure:

    1. **Non-stationarity within chains**: If the first half of a chain
       predicts the second half (transfer entropy > 0), the chain has
       not yet reached its stationary distribution.

    2. **Spurious dependence between chains**: If one chain's trace
       predicts another's (transfer entropy > 0), the chains are not
       truly independent — indicating shared non-stationarity or
       coupling artifacts.

    Parameters
    ----------
    idata : DataTree or InferenceData
        ArviZ inference data containing the specified *group*.
    var_names : list of str, optional
        Parameters to analyze.  If ``None``, all parameters are used.
    group : str, optional
        InferenceData group.  Default is ``"posterior"``.
    lag : int, optional
        Embedding dimension for the target series in the transfer
        entropy computation.  Default is 1.
    knn_k : int, optional
        Number of nearest neighbors for the kNN entropy estimator
        used internally by transfer entropy.  Default is 5.

    Returns
    -------
    dict[str, MixingDiagnostic]
        Maps parameter names to :class:`~divergence.MixingDiagnostic`
        named tuples with fields ``stationarity_te`` (shape
        ``(n_chains,)``) and ``cross_chain_te`` (shape
        ``(n_chains - 1,)``).

    Raises
    ------
    ImportError
        If ArviZ is not installed.
    ValueError
        If the group is missing or chains are too short for the
        requested embedding dimensions.

    Notes
    -----
    For **vector parameters** (shape ``(chain, draw, K)``), transfer
    entropy is computed for each of the *K* components separately and
    the results are averaged.  This avoids the curse of dimensionality
    in kNN entropy estimation.

    Transfer entropy is related to Granger causality: for jointly
    Gaussian processes the two are equivalent [2]_.  The key advantage
    is that TE is fully nonparametric — it detects any form of directed
    statistical dependence.

    Examples
    --------
    >>> import numpy as np
    >>> import arviz as az
    >>> rng = np.random.default_rng(42)
    >>> idata = az.from_dict({
    ...     "posterior": {"mu": rng.normal(0, 1, (4, 500))}
    ... })
    >>> result = mixing_diagnostic(idata)
    >>> result["mu"].stationarity_te.shape
    (4,)

    References
    ----------
    .. [1] Schreiber, T. (2000). "Measuring information transfer."
       *Physical Review Letters*, 85(2), 461-464.
    .. [2] Barnett, L., Barrett, A. B., & Seth, A. K. (2009). "Granger
       causality and transfer entropy are equivalent for Gaussian
       variables." *Physical Review Letters*, 103(23), 238701.
    """
    from divergence._types import MixingDiagnostic
    from divergence.causal import transfer_entropy

    _import_arviz()
    _validate_group(idata, group)
    dataset = idata[group]
    names = _get_var_names(dataset, var_names)

    results: dict[str, MixingDiagnostic] = {}
    for name in names:
        arr = _get_values(dataset, name)  # (chain, draw, ...)
        n_chains, n_draws = arr.shape[:2]
        trailing = arr.shape[2:]

        # Validate chain length
        min_half = max(1, lag) + knn_k + 2
        if n_draws // 2 < min_half:
            raise ValueError(
                f"Variable '{name}' has {n_draws} draws per chain. "
                f"Need at least {2 * min_half} for lag={lag}, knn_k={knn_k}."
            )

        def _te_1d(source: np.ndarray, target: np.ndarray) -> float:
            return transfer_entropy(source, target, k=1, lag=lag, knn_k=knn_k)

        def _te_maybe_multivariate(source: np.ndarray, target: np.ndarray) -> float:
            if source.ndim == 1:
                return _te_1d(source, target)
            n_components = source.shape[1]
            te_vals = [_te_1d(source[:, i], target[:, i]) for i in range(n_components)]
            return float(np.mean(te_vals))

        # Stationarity: TE from first half to second half of each chain
        stationarity_te = np.empty(n_chains)
        for c in range(n_chains):
            trace = arr[c]  # (draw, ...)
            if trailing:
                trace = trace.reshape(n_draws, -1)
            mid = n_draws // 2
            stationarity_te[c] = _te_maybe_multivariate(trace[:mid], trace[mid:])

        # Cross-chain: TE between consecutive chain pairs
        cross_chain_te = np.empty(max(n_chains - 1, 0))
        for c in range(n_chains - 1):
            trace_a = arr[c]
            trace_b = arr[c + 1]
            if trailing:
                trace_a = trace_a.reshape(n_draws, -1)
                trace_b = trace_b.reshape(n_draws, -1)
            cross_chain_te[c] = _te_maybe_multivariate(trace_a, trace_b)

        results[name] = MixingDiagnostic(
            stationarity_te=stationarity_te,
            cross_chain_te=cross_chain_te,
        )
    return results
