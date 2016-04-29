import itertools as it

import numpy as np

from .. import defaults, transforms, vectors


def center(x, n):
    return (((2*x+1)-n)/n)


def axes(n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    for i in range(n):
        yield vectors.unit(i, n=m, dtype=dtype)


def perpendicular_axis(i, offset, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = transforms.scale_translate(vectors.unit(np.mod(i, n), n=m-1), offset, n=m)
    return result


def parallel_axis(i, j, offset, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = transforms.scale_translate(np.zeros(m-1, dtype=dtype), offset, n=m, dtype=dtype)
    value = transforms.identity(n=1, dtype=dtype)
    for s, t in np.ndindex(*value.shape):
        result[np.mod(i+s, n), np.mod(j+t, m-1)] = value[s, t]
    return result


def perpendicular_axes(direction=None, k=0, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    direction = vectors.unit(0, n=n, dtype=dtype) if direction is None else direction
    for i in range(0 - k, n + k):
        yield perpendicular_axis(i, direction*center(i, n), n=n, m=m,  dtype=dtype)


def parallel_axes(direction=None, k=0, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    direction = vectors.unit(0, n=n, dtype=dtype) if direction is None else direction
    for i in range(0 - k, n + k):
        yield parallel_axis(i, 1, direction*center(i, n), n=n, m=m, dtype=dtype)


def multiple_axes(direction=None, k=0, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    direction = vectors.units([0,1], n=m, dtype=dtype) if direction is None else direction
    result = transforms.scale(np.zeros(m-1, dtype=dtype), n=m, dtype=dtype)
    for i, j in it.combinations(range(n), 2):
        result[:2] = vectors.units([i,j], n=m, dtype=dtype)
        offset = vectors.vector([center(i, n-1), center(j, n+1)], n=2) * 1.2
        yield vectors.dot(result.T, transforms.scale_translate(vectors.full(1/(n-1), n=m-1), vectors.dot(offset, direction)))
