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


def range_to_normal(lower, upper):
    ''' Map values from [lower, upper] to [0, 1]. '''
    delta = (upper - lower)
    def _range_to_normal(x):
        return (x - lower) / delta
    return _range_to_normal


def normal_to_range(lower, upper):
    ''' Map values from [0, 1] to [lower, upper]. '''
    delta = (upper - lower)
    def _normal_to_range(x):
        return lower + (x * delta)
    return _normal_to_range


def range_to_range(src, dst):
    ''' Map values from [src[0], src[1]] to [dst[0], dst[1]]. '''
    r2n = range_to_normal(src[0], src[1])
    n2r = normal_to_range(dst[0], dst[1])
    def _range_to_range(x):
        return n2r(r2n(x))
    return _range_to_range
