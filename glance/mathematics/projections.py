import numpy as np

from . import defaults, transforms, vectors


def identity(m, n=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    return np.eye(m, n, dtype=dtype)


def orthogonal(l, r, b, t, n, f, dtype=defaults.DEFAULT_DTYPE):
    return np.array([
        [+2.0/(r-l), 0, 0, 0],
        [0, +2.0/(t-b), 0, 0],
        [0, 0, +2.0/(f-n), 0],
        [-(r+l)/(r-l), -(t+b)/(t-b), -(f+n)/(f-n), +1.0],
    ], dtype=dtype)


def frustum(l, r, b, t, n, f, dtype=defaults.DEFAULT_DTYPE):
    return np.array([
        [+2.0*(n/(r-l)), 0, 0, 0],
        [0, +2.0*(n/(t-b)), 0, 0],
        [-(r+l)/(r-l), -(t+b)/(t-b), +(f+n)/(f-n), +1.0],
        [0, 0, -2.0*(n/(f-n))*f, 0]
    ], dtype=dtype)

'''
def perspective(fov, n, f, dtype=defaults.DEFAULT_DTYPE):
    cotf = 1.0/np.tan(fov/2.0)
    return np.array([
        [cotf, 0, 0, 0],
        [0, cotf, 0, 0],
        [0, 0, +(f+n)/(f-n), +1.0],
        [0, 0, -2.0*(n/(f-n))*f, 0]
    ], dtype=dtype)
'''
def perspective(fov, n, f, dtype=defaults.DEFAULT_DTYPE):
    cotf = n/np.tan(fov/2.0)
    return frustum(-cotf, +cotf, -cotf, +cotf, n, f, dtype=dtype)

def perspective_inverted(fov, n, f, dtype=defaults.DEFAULT_DTYPE):
    result = perspective(fov, n, f, dtype=dtype)
    result[2, 2] *= -1.0
    result[2, 3] *= -1.0
    return result
#'''


def general_orthogonal(lower, upper, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    sx = upper + lower
    dx = upper - lower
    return transforms.scale_translate(+2.0*(1.0/dx), -(sx/dx), n=m, dtype=dtype)


def general_orthogonal_inverted(lower, upper, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = general_orthogonal(lower, upper, n=n, m=m, dtype=dtype)
    result[n-1, n-1] *= -1.0
    return result


def general_frustum(lower, upper, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    i, j = n-1, m-1
    near, far = lower[i], upper[i]
    sx = upper + lower
    dx = upper - lower
    result = transforms.scale_translate(+2.0*(near/dx), -(sx/dx), n=m, dtype=dtype)
    result[i,i] *= -1.0 * far
    result[j,i] *= -1.0
    result[i], result[j] = np.copy(result[j]), np.copy(result[i])
    return result


def general_perspective(fov, near, far, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    cotf = near/np.tan(fov/2.0)
    lower = -vectors.full(cotf, n=n)
    upper = +vectors.full(cotf, n=n)
    lower[n-1] = near
    upper[n-1] = far
    return general_frustum(lower, upper, n=n, m=m, dtype=dtype)


def general_perspective_inverted(fov, near, far, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = general_perspective(fov, near, far, n=n, m=m, dtype=dtype)
    result[n-1, n-1] *= -1.0
    result[n-1, m-1] *= -1.0
    return result


def diagonal(k, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = np.zeros((m, m), dtype=dtype)
    result[m-1, m-1] = 1
    
    ds = np.concatenate([vectors.units(k), vectors.diagonals(k)], axis=0)
    o = min(len(ds), m-1)
    result[:o] = ds[:o]
    
    return result

'''
def diagonal(dst, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = transforms.scale(vectors.vector(np.ones(dst, dtype=dtype), n=m-1, dtype=dtype))

    if dst == 2:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([np.dot(d, transforms.rotate_axes(0, 1, a*np.pi/2, n=dst, dtype=dtype)) for a in range(2)])
        k = max(0, min(2, n-dst))
        result[dst:dst+k, :dst] = diagonals[:k]
    elif dst == 3:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([np.dot(d, transforms.rotate_axes(0, 1, a*np.pi/2, n=dst, dtype=dtype)) for a in range(4)])
        k = max(0, min(4, n-dst))
        result[dst:dst+k, :dst] = diagonals[:k]
    elif dst == 4:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([np.dot(d, transforms.rotate_axes(0, 1, a*np.pi/2, n=dst, dtype=dtype)) for a in range(8)])
        k = max(0, min(8, n-dst))
        result[dst:dst+k, :dst] = diagonals[:k]
    
    return result
'''
