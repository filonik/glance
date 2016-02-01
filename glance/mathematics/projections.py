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
#'''


def general_orthogonal(lower, upper, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    sx = upper + lower
    dx = upper - lower
    return transforms.scale_translate(+2.0*(1.0/dx), -(sx/dx), n=m, dtype=dtype)


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


def diagonal(dst, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=np.float32):
    result = transforms.scale(vectors.vector(np.ones(dst, dtype=dtype), n=m-1, dtype=dtype))

    if dst == 2:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([np.dot(d, transforms.rotate(0, 1, a*np.pi/2, n=dst, dtype=dtype)) for a in range(2)])
        k = max(0, min(2, n-dst))
        result[dst:dst+k, :dst] = diagonals[:k]
    elif dst == 3:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([np.dot(d, transforms.rotate(0, 1, a*np.pi/2, n=dst, dtype=dtype)) for a in range(4)])
        k = max(0, min(4, n-dst))
        result[dst:dst+k, :dst] = diagonals[:k]
    elif dst == 4:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([np.dot(d, transforms.rotate(0, 1, a*np.pi/2, n=dst, dtype=dtype)) for a in range(8)])
        k = max(0, min(8, n-dst))
        result[dst:dst+k, :dst] = diagonals[:k]

    return result

'''
def shear_diagonal(src, dst, n=DEFAULT_N, dtype=np.float32):
    projection = transforms.identity(n, dtype=dtype)
    if dst==2:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([dot(d, transforms.rotate2_xy(a*np.pi/2, n=dst)) for a in range(2)])
        k = max(0, min(2, src-dst))
        projection[dst:dst+k,:dst] = diagonals[:k]
    elif dst==3:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([dot(d, transforms.rotate3_xy(a*np.pi/2, n=dst)) for a in range(4)])
        k = max(0, min(4, src-dst))
        projection[dst:dst+k,:dst] = diagonals[:k]
    elif dst==4:
        d = np.ones(dst, dtype=dtype)
        d = d/np.linalg.norm(d)
        diagonals = np.asarray([dot(d, transforms.rotate4_xy(a*np.pi/2, n=dst)) for a in range(8)])
        k = max(0, min(8, src-dst))
        projection[dst:dst+k,:dst] = diagonals[:k]

    return projection[:n,:dst]
'''
'''
def shear(m, n=DEFAULT_N, dtype=np.float32):
    if n==2:
        i = transforms.identity(n, dtype=dtype)
        r = dot(transforms.rotate2_xy(np.pi/4, n=n))
        projection = np.r_[tuple([i, r])]
    elif n==3:
        i = transforms.identity(n, dtype=dtype)
        r = dot(transforms.rotate3_xy(np.pi/4, n=n), transforms.rotate3_yz(np.pi/4, n=n))
        projection = np.r_[tuple([i, r])]
    elif n==4:
        i = transforms.identity(n, dtype=dtype)
        r = dot(transforms.rotate4_xy(np.pi/4, n=n), transforms.rotate4_yz(np.pi/4, n=n), transforms.rotate4_zw(np.pi/4, n=n))
        projection = np.r_[tuple([i, r])]

    return projection[:m]
'''