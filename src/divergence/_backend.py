"""Backend detection for CPU/GPU dispatch.

Checks for GPU availability via JAX.  The GPU path is used only when
explicitly requested (``backend="gpu"``) or when auto-detection is
enabled (``backend="auto"``) and a CUDA device is found.

The environment variable ``DIVERGENCE_BACKEND`` can override detection:

- ``DIVERGENCE_BACKEND=cpu``: always use CPU, even if GPU is available.
- ``DIVERGENCE_BACKEND=gpu``: require GPU; raise if unavailable.
- (unset or ``auto``): use GPU if available, fall back to CPU.
"""

import os

_gpu_checked = False
_gpu_available = False


def _check_gpu() -> bool:
    """Check once whether JAX can see a GPU."""
    global _gpu_checked, _gpu_available
    if _gpu_checked:
        return _gpu_available
    _gpu_checked = True
    try:
        from divergence._gpu_kernels import gpu_available

        _gpu_available = gpu_available()
    except Exception:
        _gpu_available = False
    return _gpu_available


def get_backend(requested: str | None = None) -> str:
    """Resolve the compute backend.

    Parameters
    ----------
    requested : str or None
        ``"cpu"``, ``"gpu"``, ``"auto"``, or ``None``.  ``None`` reads
        from the ``DIVERGENCE_BACKEND`` environment variable, defaulting
        to ``"auto"``.

    Returns
    -------
    str
        ``"cpu"`` or ``"gpu"``.

    Raises
    ------
    RuntimeError
        If ``"gpu"`` is requested but no GPU is available.
    """
    if requested is None:
        requested = os.environ.get("DIVERGENCE_BACKEND", "auto").lower()

    if requested == "cpu":
        return "cpu"
    if requested == "gpu":
        if not _check_gpu():
            raise RuntimeError(
                "GPU backend requested but no CUDA device found by JAX. "
                "Install jax[cuda12] or set DIVERGENCE_BACKEND=cpu."
            )
        return "gpu"
    # auto
    return "gpu" if _check_gpu() else "cpu"
