from __future__ import absolute_import, division, print_function

import numpy as np

from . import defaults


def identity(n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return np.identity(n, dtype=dtype)


def row(j, v, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = np.zeros((n, n), dtype=dtype)
    for i in range(min(n, len(v))):
        result[i, j] = v[i]
    return result


def col(i, v, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = np.zeros((n,n), dtype=dtype)
    for j in range(min(n, len(v))):
        result[i, j] = v[j]
    return result


def translate(v, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = identity(n, dtype=dtype)
    for i in range(min(n, len(v))):
        result[-1, i] = v[i]
    return result


def scale(v, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = identity(n, dtype=dtype)
    for i in range(min(n, len(v))):
        result[i, i] = v[i]
    return result


def rotate(i, j, a, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = identity(n=n, dtype=dtype)

    if i != j:
        a = +a if i < j else -a
        result[i, i] = np.cos(a)
        result[i, j] = +np.sin(a)
        result[j, i] = -np.sin(a)
        result[j, j] = np.cos(a)

    return result


def translate_scale(w, v, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = identity(n, dtype=dtype)
    for i in range(min(n, len(v))):
        result[i, i] = v[i]
    for i in range(min(n, len(w))):
        result[-1, i] = v[i] * w[i]
    return result


def scale_translate(v, w, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = identity(n, dtype=dtype)
    for i in range(min(n, len(v))):
        result[i, i] = v[i]
    for i in range(min(n, len(w))):
        result[-1, i] = w[i]
    return result
