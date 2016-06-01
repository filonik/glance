# Do not import defaults here! (Cyclic Dependency)


def chunk_count(x, n):
    return ((x - 1)//n) + 1


def next_multiple(n):
    def _next_multiple(x):
        return chunk_count(x, n)*n
    return _next_multiple


def encode_binary(i, n):
    def binary_digits(i, n):
        for j in range(n):
            yield (i >> j) & 1
    return list(binary_digits(i, n))


def decode_binary(b):
    def decimal_parts(b):
        for i, d in enumerate(b):
            yield (1 << i) * d
    return sum(decimal_parts(b))


def range_to_normal(r):
    ''' Map values from [lower, upper] to [-1, +1]. '''
    center = (r[1] + r[0]) / 2.0
    extent = (r[1] - r[0]) / 2.0
    def _range_to_normal(x):
        return (x - center) / extent
    return _range_to_normal


def normal_to_range(r):
    ''' Map values from [-1, +1] to [lower, upper]. '''
    center = (r[1] + r[0]) / 2.0
    extent = (r[1] - r[0]) / 2.0
    def _normal_to_range(x):
        return center + (x * extent)
    return _normal_to_range


def flip(r):
    r2n = range_to_normal(r)
    n2r = normal_to_range(r)
    def _flip(x):
        return n2r(-1*r2n(x))
    return _flip


def range_to_positive(r):
    ''' Map values from [lower, upper] to [0, 1]. '''
    lower = r[0]
    upper = r[1]
    delta = (upper - lower)
    def _range_to_positive(x):
        return (x - lower) / delta
    return _range_to_positive


def positive_to_range(r):
    ''' Map values from [0, 1] to [lower, upper]. '''
    lower = r[0]
    upper = r[1]
    delta = (upper - lower)
    def _positive_to_range(x):
        return lower + (x * delta)
    return _positive_to_range


def range_to_range(src, dst):
    ''' Map values from src to dst. '''
    r2n = range_to_positive(src)
    n2r = positive_to_range(dst)
    def _range_to_range(x):
        return n2r(r2n(x))
    return _range_to_range
