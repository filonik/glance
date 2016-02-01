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


def vector(value, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return zeros(*value, n=n, dtype=dtype)


def homogeneous(value, w, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = vector(value, n=n, dtype=dtype)
    result[n-1] = w
    return result


def point(value, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return homogeneous(value, 1.0, n=n, dtype=dtype)


def factory(n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    def _factory(*args):
        return zeros(*args, n=n, dtype=dtype)
    return _factory


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


def units(indices, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return np.array([unit(i, n=n, dtype=dtype) for i in indices], dtype=dtype).reshape(-1, n)


def dot(*args):
    return utilities.foldl(np.dot, args)


def inverse(m):
    try:
        return np.linalg.inv(m)
    except np.linalg.LinAlgError:
        return np.zeros_like(m)


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