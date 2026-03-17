"""Causal and temporal information-theoretic measures.

This module provides transfer entropy estimation for detecting directed
information flow between time series.

Functions
---------
transfer_entropy
    Schreiber's transfer entropy via the 4-entropy formulation.

References
----------
.. [1] Schreiber, T. (2000). "Measuring information transfer." *Physical
   Review Letters*, 85(2), 461-464.
"""

import numpy as np

from .knn import knn_entropy


def _lag_embedding(series: np.ndarray, depth: int, first_future_idx: int) -> np.ndarray:
    """Construct a lag-embedded matrix from a 1D time series.

    For each future index t = first_future_idx, first_future_idx + 1, ..., T-1,
    build a row of past values [series[t-1], series[t-2], ..., series[t-depth]].

    Parameters
    ----------
    series : np.ndarray
        1D time series of shape ``(T,)``.
    depth : int
        Number of lagged values per row.
    first_future_idx : int
        Index of the first "future" time step.

    Returns
    -------
    np.ndarray
        Matrix of shape ``(T - first_future_idx, depth)``.
    """
    n_valid = len(series) - first_future_idx
    result = np.empty((n_valid, depth))
    for j in range(depth):
        start = first_future_idx - 1 - j
        result[:, j] = series[start : start + n_valid]
    return result


def transfer_entropy(
    source: np.ndarray,
    target: np.ndarray,
    *,
    k: int = 1,
    lag: int = 1,
    base: float = np.e,
    knn_k: int = 5,
) -> float:
    r"""Estimate transfer entropy from source to target time series.

    .. math::

        TE_{X \to Y} = I(Y_{t+1}; X_t^{(k)} \mid Y_t^{(l)})

    Computed via the 4-entropy formulation:

    .. math::

        TE = H(Y_f, Y_p) + H(Y_p, X_p) - H(Y_f, Y_p, X_p) - H(Y_p)

    where :math:`Y_f` is the future of Y, :math:`Y_p` is the past of Y
    (lag embedding of depth l), and :math:`X_p` is the past of X (lag
    embedding of depth k).

    Parameters
    ----------
    source : np.ndarray
        Source time series, shape ``(T,)``.
    target : np.ndarray
        Target time series, shape ``(T,)``.
    k : int, optional
        Embedding dimension for the source. Default is 1.
    lag : int, optional
        Embedding dimension for the target. Default is 1.
    base : float, optional
        Logarithm base. Default is ``np.e`` (nats).
    knn_k : int, optional
        Number of nearest neighbors for the kNN entropy estimator.
        Default is 5.

    Returns
    -------
    float
        Estimated transfer entropy TE_{source -> target}.

    Raises
    ------
    ValueError
        If the time series are too short for the given embedding dimensions.
    """
    source = np.asarray(source, dtype=float).ravel()
    target = np.asarray(target, dtype=float).ravel()

    T = len(source)
    if len(target) != T:
        raise ValueError(
            f"source and target must have the same length, got {T} and {len(target)}"
        )

    # We need at least max(k, lag) past values plus 1 future value
    offset = max(k, lag)
    n_valid = T - offset
    if n_valid < 2:
        raise ValueError(
            f"Time series too short (T={T}) for embedding dimensions k={k}, lag={lag}. "
            f"Need at least {offset + 2} observations."
        )

    # Build embedding vectors
    # Y_future: target[offset], target[offset+1], ..., target[T-1]
    y_future = target[offset : offset + n_valid].reshape(-1, 1)

    # Y_past: for each future t, past target values [t-1, t-2, ..., t-lag]
    y_past = _lag_embedding(target, depth=lag, first_future_idx=offset)

    # X_past: for each future t, past source values [t-1, t-2, ..., t-k]
    x_past = _lag_embedding(source, depth=k, first_future_idx=offset)

    # 4-entropy formulation of conditional MI:
    # TE = H(Y_f, Y_p) + H(Y_p, X_p) - H(Y_f, Y_p, X_p) - H(Y_p)
    yf_yp = np.hstack([y_future, y_past])
    yp_xp = np.hstack([y_past, x_past])
    yf_yp_xp = np.hstack([y_future, y_past, x_past])

    h_yf_yp = knn_entropy(yf_yp, k=knn_k, base=base)
    h_yp_xp = knn_entropy(yp_xp, k=knn_k, base=base)
    h_yf_yp_xp = knn_entropy(yf_yp_xp, k=knn_k, base=base)
    h_yp = knn_entropy(y_past, k=knn_k, base=base)

    return h_yf_yp + h_yp_xp - h_yf_yp_xp - h_yp
