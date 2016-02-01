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
