import numpy as np

from . import defaults, transforms, vectors


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


def perpendicular_axes(direction=None, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    direction = vectors.unit(0, n=n, dtype=dtype) if direction is None else direction
    for i in range(-1, n+1):
        yield perpendicular_axis(i, direction*center(i, n), n=n, m=m,  dtype=dtype)


def parallel_axes(direction=None, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    direction = vectors.unit(0, n=n, dtype=dtype) if direction is None else direction
    for i in range(-1, n+1):
        yield parallel_axis(i, 1, direction*center(i, n), n=n, m=m, dtype=dtype)


def spherical1(alphas, dtype=defaults.DEFAULT_DTYPE):
    s, c = np.sin(alphas), np.cos(alphas)
    return np.array([
        [+c[0], -s[0], 0],
        [+s[0], +c[0], 0],
        [0, 0, 1],
    ], dtype=dtype)


def spherical2(alphas, dtype=defaults.DEFAULT_DTYPE):
    s, c = np.sin(alphas), np.cos(alphas)
    return np.array([
        [+c[0], -s[0]*s[1], -s[0]*c[1], 0],
        [0, +c[1], -s[1], 0],
        [+s[0], +s[1]*c[0], +c[0]*c[1], 0],
        [0, 0, 0, 1],
    ], dtype=dtype)


def spherical3(alphas, dtype=defaults.DEFAULT_DTYPE):
    s, c = np.sin(alphas), np.cos(alphas)
    return np.array([
        [+c[0], -s[0]*s[1], -s[0]*s[2]*c[1], -s[0]*c[1]*c[2], 0],
        [0, +c[1], -s[1]*s[2], -s[1]*c[2], 0],
        [0, 0, +c[2], -s[2], 0],
        [+s[0], +s[1]*c[0], +s[2]*c[0]*c[1], +c[0]*c[1]*c[2], 0],
        [0, 0, 0, 0, 1],
    ], dtype=dtype)


def spherical4(alphas, dtype=defaults.DEFAULT_DTYPE):
    s, c = np.sin(alphas), np.cos(alphas)
    return np.array([
        [+c[0], -s[0]*s[1], -s[0]*s[2]*c[1], -s[0]*s[3]*c[1]*c[2], -s[0]*c[1]*c[2]*c[3], 0],
        [0, +c[1], -s[1]*s[2], -s[1]*s[3]*c[2], -s[1]*c[2]*c[3], 0],
        [0, 0, +c[2], -s[2]*s[3], -s[2]*c[3], 0],
        [0, 0, 0, +c[3], -s[3], 0],
        [+s[0], +s[1]*c[0], +s[2]*c[0]*c[1], +s[3]*c[0]*c[1]*c[2], +c[0]*c[1]*c[2]*c[3], 0],
        [0, 0, 0, 0, 0, 1],
    ], dtype=dtype)
