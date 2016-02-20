import numpy as np

from encore import iterables


def simplex_strip_to_simplex_indices(n):
    def _simplex_strip_to_simplex_indices(indices, dtype=np.uint32):
        result = []
        for i, index in enumerate(iterables.windowed(indices, n+1)):
            result.extend(index if i%2 else reversed(index))
        return np.array(result, dtype=dtype)
    return _simplex_strip_to_simplex_indices

line_strip_to_line_indices = simplex_strip_to_simplex_indices(1)
triangle_strip_to_triangle_indices = simplex_strip_to_simplex_indices(2)