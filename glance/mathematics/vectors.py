import functools as ft
import itertools as it

import numpy as np

from encore import utilities

from . import defaults


def full(value, *args, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = np.full(n, value, dtype=dtype)
    lower, upper = 0, min(n, len(args))
    result[lower:upper] = args[lower:upper]
    return result


def zeros(*args, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return full(0.0, *args, n=n, dtype=dtype)


def ones(*args, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return full(1.0, *args, n=n, dtype=dtype)


def homogeneous(value, w, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = zeros(*value, n=n, dtype=dtype)
    result[n-1] = w
    return result


def vector(value, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return homogeneous(value, 0.0, n=n, dtype=dtype)


def point(value, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return homogeneous(value, 1.0, n=n, dtype=dtype)


def factory(n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    def _factory(*args):
        return zeros(*args, n=n, dtype=dtype)
    return _factory


def clamp(value, lower, upper):
    return np.minimum(np.maximum(value, lower), upper)


def clamp_range(values, bounds):
    lower, upper = bounds
    return (clamp(values[0], lower, upper), clamp(values[1], lower, upper))


def interpolate_linear(a, b):
    """ ((b-a)*alpha) + a """ 
    def _interpolate_linear(alpha):
        # See exponential
        #(b-a)*alpha + a
        return a * (1.0 - alpha) + b * alpha
    return _interpolate_linear


def interpolate_exponential(a, b):
    """ ((b/a)**alpha) * a """ 
    def _interpolate_exponential(alpha):
        #Non-commutative (Order of multiplication matters!)
        #a * (a.inv()*b)**alpha
        #(b*a.inv())**alpha * a
        return ((b/a)**alpha) * a
    return _interpolate_exponential


vec2 = factory(n=2)
vec3 = factory(n=3)
vec4 = factory(n=4)

fvec = factory(dtype=np.float32)
dvec = factory(dtype=np.float64)
ivec = factory(dtype=np.int32)

fvec2 = factory(n=2, dtype=np.float32)
fvec3 = factory(n=3, dtype=np.float32)
fvec4 = factory(n=4, dtype=np.float32)

dvec2 = factory(n=2, dtype=np.float64)
dvec3 = factory(n=3, dtype=np.float64)
dvec4 = factory(n=4, dtype=np.float64)

ivec2 = factory(n=2, dtype=np.int32)
ivec3 = factory(n=3, dtype=np.int32)
ivec4 = factory(n=4, dtype=np.int32)


def unit(i, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = zeros(n=n, dtype=dtype)
    result[i] = 1
    return result


def dot(*args):
    return utilities.foldl(np.dot, args)


def normalized(v):
    try:
        return v * (1.0/np.linalg.norm(v))
    except np.linalg.LinAlgError:
        return np.zeros_like(v)


normalize = normalized


def units(k, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    indices = range(k) if isinstance(k, int) else k
    return np.array([unit(i, n=n, dtype=dtype) for i in indices], dtype=dtype)


def combinations(k, n=defaults.DEFAULT_M):
    return it.combinations(range(n), k)


def unit_indices(k, n=defaults.DEFAULT_M):
    source = set(range(n))
    for index in combinations(k, n=n):
        yield index, source - set(index)


def diagonals(k, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    from .geometries import cubes
    return np.array([vector(normalized(d), n=n, dtype=dtype) for d in cubes.cube_diagonals(k)], dtype=dtype).reshape(-1, n)


def chunk1d(shape):
    def _chunk1d(arr):
        count = np.asarray(arr.shape)//np.asarray(shape)
        return arr.reshape(count[0], shape[0])
    return _chunk1d


def chunk2d(shape):
    def _chunk2d(arr):
        count = np.asarray(arr.shape)//np.asarray(shape)
        return arr.reshape(count[0], shape[0], count[1], shape[1]).swapaxes(1,2).reshape(count[0], count[1], shape[0], shape[1])
    return _chunk2d


default_chunk1d = chunk1d((defaults.CHUNK_SIZE,))
default_chunk2d = chunk2d((defaults.CHUNK_SIZE, defaults.CHUNK_SIZE,))
