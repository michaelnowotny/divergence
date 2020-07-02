import typing as tp

import numba
import numpy as np


def _select_vectorized_log_fun_for_base(base: float) -> tp.Callable:
    if base == 2:
        return np.log2
    if base == np.e:
        return np.log
    if base == 10:
        return np.log10

    raise ValueError('base not supported')


spec = [('base', numba.float64)]


@numba.jitclass(spec)
class Logarithm:
    def __init__(self, base):
        self.base = base

    def log(self, x):
        if self.base == 2:
            return np.log2(x)
        if self.base == np.e:
            return np.log(x)
        if self.base == 10:
            return np.log10(x)

        raise ValueError('base not supported')
