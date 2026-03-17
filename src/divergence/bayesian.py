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

    Handles both ArviZ InferenceData (which has a ``.groups()`` method)
    and raw xarray DataTree (which has a ``.children`` mapping).
    """
    # xarray DataTree: .children is a Mapping of child name -> DataTree
    if hasattr(idata, "children"):
        return list(idata.children)
    # ArviZ InferenceData: .groups() is a method returning a list of str
    if hasattr(idata, "groups"):
        groups = idata.groups
        return list(groups()) if callable(groups) else list(groups)
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
            if v not in dataset.ds.data_vars:
                raise KeyError(f"Variable '{v}' not found in dataset")
            if dataset2 is not None and v not in dataset2.ds.data_vars:
                raise KeyError(f"Variable '{v}' not found in second dataset")
        return list(var_names)

    vars1 = set(dataset.ds.data_vars)
    if dataset2 is not None:
        vars2 = set(dataset2.ds.data_vars)
        return sorted(vars1 & vars2)
    return sorted(vars1)


def _flatten_samples(dataset: tp.Any, var_name: str) -> np.ndarray:
    """Extract and flatten samples, merging chain and draw dims.

    Returns shape ``(n_samples,)`` for scalar params or
    ``(n_samples, K)`` for vector params with K components.
    """
    arr = dataset[var_name].values  # (chain, draw, ...)
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
        arr = dataset[name].values  # (chain, draw, ...)
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
    ll_vars = list(ll_group.ds.data_vars)
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
        ll = ll_group[name].values  # (chain, draw, obs...)
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
        post_vars = set(posterior.ds.data_vars)
        prior_vars = set(prior.ds.data_vars)
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
        arr = dataset[name].values  # (chain, draw, obs...)
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
