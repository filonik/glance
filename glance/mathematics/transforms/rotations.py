import numpy as np

from .. import defaults


def rotate1d(v, dtype=defaults.DEFAULT_DTYPE):
    s = np.sin(v)
    c = np.cos(v)
    return np.array([
        [1],
    ], dtype=dtype)


def rotate2d(v, dtype=defaults.DEFAULT_DTYPE):
    s = np.sin(v)
    c = np.cos(v)
    return np.array([
        [+c[0], -s[0]],
        [+s[0], +c[0]],
    ], dtype=dtype)


def rotate3d(v, dtype=defaults.DEFAULT_DTYPE):
    s = np.sin(v)
    c = np.cos(v)
    return np.array([
        [+c[0], 0, -s[0]],
        [-s[0]*s[1], +c[1], -c[0]*s[1]],
        [+s[0]*c[1], +s[1], +c[0]*c[1]],
    ], dtype=dtype)


def rotate4d(v, dtype=defaults.DEFAULT_DTYPE):
    s = np.sin(v)
    c = np.cos(v)
    return np.array([
        [+c[0], 0, 0, -s[0]],
        [-s[0]*s[1], +c[1], 0, -c[0]*s[1]],
        [-s[0]*c[1]*s[2], -s[1]*s[2], +c[2], -c[0]*c[1]*s[2]],
        [+s[0]*c[1]*c[2], +s[1]*c[2], +s[2], +c[0]*c[1]*c[2]],
    ], dtype=dtype)


def rotate5d(v, dtype=defaults.DEFAULT_DTYPE):
    s = np.sin(v)
    c = np.cos(v)
    return np.array([
        [+c[0], 0, 0, 0, -s[0]], 
        [-s[0]*s[1], +c[1], 0, 0, -c[0]*s[1]], 
        [-s[0]*c[1]*s[2], -s[1]*s[2], +c[2], 0, -c[0]*c[1]*s[2]], 
        [-s[0]*c[1]*c[2]*s[3], -s[1]*c[2]*s[3], -s[2]*s[3], +c[3], -c[0]*c[1]*c[2]*s[3]], 
        [+s[0]*c[1]*c[2]*c[3], +s[1]*c[2]*c[3], +s[2]*c[3], +s[3], +c[0]*c[1]*c[2]*c[3]],
    ], dtype=dtype)


def rotate6d(v, dtype=defaults.DEFAULT_DTYPE):
    s = np.sin(v)
    c = np.cos(v)
    return np.array([
        [+c[0], 0, 0, 0, 0, -s[0]], 
        [-s[0]*s[1], +c[1], 0, 0, 0, -c[0]*s[1]], 
        [-s[0]*c[1]*s[2], -s[1]*s[2], +c[2], 0, 0, -c[0]*c[1]*s[2]], 
        [-s[0]*c[1]*c[2]*s[3], -s[1]*c[2]*s[3], -s[2]*s[3], +c[3], 0, -c[0]*c[1]*c[2]*s[3]], 
        [-s[0]*c[1]*c[2]*c[3]*s[4], -s[1]*c[2]*c[3]*s[4], -s[2]*c[3]*s[4], -s[3]*s[4], +c[4], -c[0]*c[1]*c[2]*c[3]*s[4]], 
        [+s[0]*c[1]*c[2]*c[3]*c[4], +s[1]*c[2]*c[3]*c[4], +s[2]*c[3]*c[4], +s[3]*c[4], +s[4], +c[0]*c[1]*c[2]*c[3]*c[4]],
    ], dtype=dtype)


def rotate7d(v, dtype=defaults.DEFAULT_DTYPE):
    s = np.sin(v)
    c = np.cos(v)
    return np.array([
        [+c[0], 0, 0, 0, 0, 0, -s[0]],
        [-s[0]*s[1], +c[1], 0, 0, 0, 0, -c[0]*s[1]],
        [-s[0]*c[1]*s[2], -s[1]*s[2], +c[2], 0, 0, 0, -c[0]*c[1]*s[2]],
        [-s[0]*c[1]*c[2]*s[3], -s[1]*c[2]*s[3], -s[2]*s[3], +c[3], 0, 0, -c[0]*c[1]*c[2]*s[3]],
        [-s[0]*c[1]*c[2]*c[3]*s[4], -s[1]*c[2]*c[3]*s[4], -s[2]*c[3]*s[4], -s[3]*s[4], +c[4], 0, -c[0]*c[1]*c[2]*c[3]*s[4]],
        [-s[0]*c[1]*c[2]*c[3]*c[4]*s[5], -s[1]*c[2]*c[3]*c[4]*s[5], -s[2]*c[3]*c[4]*s[5], -s[3]*c[4]*s[5], -s[4]*s[5], +c[5], -c[0]*c[1]*c[2]*c[3]*c[4]*s[5]],
        [+s[0]*c[1]*c[2]*c[3]*c[4]*c[5], +s[1]*c[2]*c[3]*c[4]*c[5], +s[2]*c[3]*c[4]*c[5], +s[3]*c[4]*c[5], +s[4]*c[5], +s[5], +c[0]*c[1]*c[2]*c[3]*c[4]*c[5]],
    ], dtype=dtype)


'''
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
'''
