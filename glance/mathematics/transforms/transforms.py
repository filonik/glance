from __future__ import absolute_import, division, print_function

import numpy as np

from .. import defaults

from . import rotations


DEFAULT_ROTATIONS = {
    1: rotations.rotate1d,
    2: rotations.rotate2d,
    3: rotations.rotate3d,
    4: rotations.rotate4d,
    5: rotations.rotate5d,
    6: rotations.rotate6d,
    7: rotations.rotate7d,
}


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


def rows(vs, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = np.zeros((n, n), dtype=dtype)
    for i, v in enumerate(vs):
        result[i,:len(v)] = v
    result[-1,-1] = 1.0
    return result


def cols(vs, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = np.zeros((n, n), dtype=dtype)
    for i, v in enumerate(vs):
        result[:len(v),i] = v
    result[-1,-1] = 1.0
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


def rotate(rotation, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = identity(n=m, dtype=dtype)
    
    try:
        result[:n,:n] = DEFAULT_ROTATIONS[n](rotation, dtype=dtype)
    except KeyError:
        pass
    
    return result


def rotate_axes(i, j, a, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
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
        vi = v[i] if i < len(v) else 1
        result[-1, i] = w[i] * vi
    return result


def scale_translate(v, w, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = identity(n, dtype=dtype)
    for i in range(min(n, len(v))):
        result[i, i] = v[i]
    for i in range(min(n, len(w))):
        result[-1, i] = w[i]
    return result

    
def normalized(m):
    try:
        return m * ((1.0/np.linalg.norm(m, axis=1)[:,np.newaxis]))
    except np.linalg.LinAlgError:
        return np.zeros_like(m)


def transposed(m):
    return np.transpose(m)


def inversed(m):
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return np.zeros_like(m)


def dualized(m):
    result = np.zeros_like(m)
    result[:-1,:-1] = normalized(transposed(inversed(m[:-1,:-1])))
    result[-1,-1] = 1.0
    return result


normalize = normalized
transpose = transposed
inverse = inversed
