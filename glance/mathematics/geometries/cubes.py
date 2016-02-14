import itertools as it
import operator

import numpy as np

from encore import iterables, predicates

from .. import combinatorics, mathematics, vectors

def as_binary_array(iterable, n):
    return [mathematics.encode_binary(i, n) for i in iterable]

def replace_all(iterable, predicate, values):
    def _replace_all(iterable, predicate, values):
        i = iter(values)
        for item in iterable:
            if predicate(item):
                yield next(i)
            else:
                yield item
    return list(_replace_all(iterable, predicate, values))

def replace_all_with_binary_numbers(predicate, iterable, n):
    num = iterables.count(iterable, predicate)
    for b in as_binary_array(range(2**num), n):
        yield replace_all(iterable, predicate, b)

# k-faces on n-cube (hypercube)
# scipy.misc.comb(n, k)*(2**(n-k))

def cube_vertex_count(n):
    return 2**n

def cube_face_count(n, k):
    if k > n:
        return 0
    else:
        return combinatorics.comb(n, k) * 2**(n-k)

def cube_vertex(i, n, dtype=np.float32):
    return 2.0 * np.array(mathematics.encode_binary(i, n), dtype=dtype) - 1.0

def cube_vertices(n, dtype=np.float32):
    return np.asarray([cube_vertex(i, n, dtype=dtype) for i in range(cube_vertex_count(n))])

def cube_tex_coord(i, n, dtype=np.float32):
    return np.array(mathematics.encode_binary(i, n), dtype=dtype)

def cube_tex_coords(n, dtype=np.float32):
    return np.asarray([cube_tex_coord(i, n, dtype=dtype) for i in range(cube_vertex_count(n))])

def cube_color(i, n, dtype=np.float32):
    from ...graphics import colors
    return colors.Color.fromordinal(i+1).rgba  #np.ones(4, dtype=dtype)

def cube_colors(n, dtype=np.float32):
    return np.asarray([cube_color(i, n, dtype=dtype) for i in range(cube_vertex_count(n))])

def cube_indices(n):
    return range(cube_vertex_count(n))

def cube_binary_indices(n, flip=None, dtype=np.uint32):
    for i in cube_indices(n):
        value = np.array(mathematics.encode_binary(i, n), dtype=np.bool_)
        if flip is not None:
            value ^= flip
        yield np.array(value, dtype=dtype)

def cube_signed_indices(n, dtype=np.float32):
    for i in cube_binary_indices(n, dtype=dtype):
        yield 2.0 * i - 1.0

def cube_face_bases(k, n, dtype=np.float32):
    A, B = 'A', 'B'
    is_a, is_b = predicates.eq_(A), predicates.eq_(B)
    for path in combinatorics.lexicographic_permutations((n-k)*[A] + (k)*[B]):
        a = vectors.units(iterables.indices(path, is_a), n=n, dtype=dtype)
        b = vectors.units(iterables.indices(path, is_b), n=n, dtype=dtype)
        yield a, b

def cube_face_positions(k, n, dtype=np.float32):
    def _cube_face_positions(k, n, dtype=np.float32):
        d = vectors.ones(n=k, dtype=dtype)
        d[0] *= -1.0
        for a, b in cube_face_bases(k, n, dtype=dtype):
            d[0] *= -1.0
            for i in cube_signed_indices(n-k, dtype=dtype):
                d[0] *= -1.0
                yield [np.dot(i, a) + np.dot(j * d, b) for j in cube_signed_indices(k, dtype=dtype)]
    def flatten(iterable):
        yield from [item for items in iterable for item in items]
    return np.array(list(flatten(_cube_face_positions(k, n, dtype=dtype))), dtype=dtype)

def cube_face_tex_coords(k, n, dtype=np.float32):
    def _cube_face_tex_coords(k, n, dtype=np.float32):
        for a, b in cube_face_bases(k, n, dtype=dtype):
            for i in cube_binary_indices(n-k, dtype=dtype):
                yield list(cube_binary_indices(k, dtype=dtype))
    def flatten(iterable):
        yield from [item for items in iterable for item in items]
    return np.array(list(flatten(_cube_face_tex_coords(k, n, dtype=dtype))), dtype=dtype)

def cube_face_indices(k, n, dtype=np.uint32):
    def _cube_face_indices(k, n, dtype=np.uint32):
        m = cube_vertex_count(k)
        for i in range(cube_face_count(n, k)):
            yield np.array(list(range(m)), dtype=dtype) + i * m
    return np.array(list(_cube_face_indices(k, n, dtype=dtype)), dtype=dtype)

def cube_faces(k, n):
    A, B = 'A', 'B'
    isA, isB = predicates.eq_(A), predicates.eq_(B)

    def _cube_faces(path, o):
        #ia, ib = where(isA, path), where(isB, path)
        p0 = path
        r0 = []
        for i, p1 in enumerate(replace_all_with_binary_numbers(isB, p0, n)):
            r1 = []
            for j, p2 in enumerate(replace_all_with_binary_numbers(isA, p1, n)):
                '''
                if ia is not None:
                    #p2[ia] *= (-1)**i * (-1)**o # Alternate Orientation
                    p2[ia] ^= (1 + i + o) % 2 # Alternate Orientation
                '''
                r1.append(p2)
            r0.append(r1)
        return r0

    if k > n:
        return
    elif k == 0:
        for i in range(2**n): yield [i]
        return
    elif k == n:
        yield [i for i in range(2**n)]
        return

    for i, path in enumerate(combinatorics.lexicographic_permutations((k)*[A] + (n-k)*[B])):
        for face in _cube_faces(path, i):
            yield map(mathematics.decode_binary, face)

'''
def simplex_indices(k, n, dtype=np.uint32):
    from . import simplices

    return np.ascontiguousarray(map(simplices.simplex_to_cube, simplices.simplex_indices(k, n, dtype=dtype)),  dtype=dtype)

def cube_indices(k, n, dtype=np.uint32):
    from . import geometries

    faces = map(geometries.simplex_strip_to_simplex_indices(k), cube_faces(k, n))

    return np.ascontiguousarray(list(iterables.flatten(faces)), dtype=dtype)
'''

# Old Version:

'''
def cube_vertices(size, dtype=np.float32):
    def _cube_vertices(size, i, n):
        head, tail = iterables.pop(size)
        result = np.zeros(n, dtype=dtype)
        if head is None:
            yield result
        else:
            vertices = _cube_vertices(tail, i+1, n)
            result[i] = head/2.0
            for vertex in vertices:
                yield vertex - result
                yield vertex + result
    return np.ascontiguousarray(list(_cube_vertices(size, 0, len(size))), dtype=dtype)

def cube_faces(n, k):
    if k > n:
        return []
    elif k == 0:
        return [[i] for i in range(2**n)]
    elif k == n:
        return [[i for i in range(2**n)]]
    else:
        offset = cube_vertex_count(n-1)
        
        old_faces_0 = cube_faces(n-1, k)
        old_faces_1 = iterables.recursive_map(lambda i: i+offset, old_faces_0)
        
        #old_faces_0 = map(compose(list, reversed), old_faces_0)
        #old_faces_1 = map(compose(list, reversed), old_faces_1)

        sub_faces_0 = cube_faces(n-1, k-1)
        sub_faces_1 = iterables.recursive_map(lambda i: i+offset, sub_faces_0)

        new_faces = [a+b for a, b in zip(sub_faces_0, sub_faces_1)]
        
        return old_faces_0 + old_faces_1 + new_faces

def cube_vertices_by_face(size, k, dtype=np.float32):
    vertices, faces = cube_vertices(size, dtype=dtype), cube_faces(len(size), k)
    return np.ascontiguousarray([[vertices[i] for i in face] for face in faces], dtype=dtype)
'''