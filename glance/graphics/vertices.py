import collections
import enum

import numpy as np

from encore import generators, utilities

from glue.utilities import reversedict

from ..mathematics import defaults, vectors


def chunk_shape(shape, chunk_size=defaults.CHUNK_SIZE):
    chunk_count = np.asarray(shape)//chunk_size
    if len(shape) == 1:
        if all(chunk_count == 1):
            return (chunk_size,)
        else:
            return (chunk_count[0], chunk_size,)
    if len(shape) == 2:
        if all(chunk_count == 1):
            return (chunk_size, chunk_size,)
        else:
            return (chunk_count[0]//chunk_size, chunk_count[1]//chunk_size, chunk_size, chunk_size,)
    return shape


def pad_shape(shape):
    if not shape:
        return (4,)
    return shape


class ComponentType(object):
    @property
    def name(self):
        return self._name
    
    @property
    def rank(self):
        return self._rank
    
    @property
    def dtype(self):
        return self._dtype
    
    @property
    def chunk(self):
        return self._chunk
    
    @property
    def pad(self):
        return self._pad
    
    def __init__(self, name, rank, dtype, chunk, pad, factory, m=None):
        self._name = name
        self._rank = rank
        self._dtype = dtype
        self._chunk = chunk
        self._pad = pad
        self._factory = factory
        self._m = m
    
    def shape(self, m=defaults.DEFAULT_M):
        # The instance 'm' takes priority!
        m = m if self._m is None else self._m
        return (m,) * self._rank
    
    def chunked_shape(self, m=defaults.DEFAULT_M):
        return chunk_shape(pad_shape(self.shape(m=m)))
    
    def padded_shape(self, m=defaults.DEFAULT_M):
        return pad_shape(self.shape(m=m))
    
    def generator(self, m=defaults.DEFAULT_M):
        dtype = self.dtype 
        shape = self.shape(m=m)
        return self._factory(shape, dtype)
    
    def record_dtype(self, m=defaults.DEFAULT_M, chunked=True):
        name = self.name
        dtype = self.dtype
        shape = self.chunked_shape(m=m) if chunked else self.shape(m=m)
        return (name, dtype, shape) if shape else (name, dtype)


def zero_generator(shape, dtype):
    return generators.constant(np.zeros(shape, dtype=dtype))


def one_generator(shape, dtype):
    return generators.constant(np.ones(shape, dtype=dtype))


def identity_generator(shape, dtype):
    assert len(shape) == 2 and shape[0] == shape[1]
    return generators.constant(np.identity(shape[0], dtype=dtype))


def identifier_generator(shape, dtype):
    return generators.autoincrement()


def zero_pad(m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    def _zero_pad(value):
        return vectors.zeros(value, n=m, dtype=dtype)
    return _zero_pad


def one_pad(m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    def _one_pad(value):
        return vectors.ones(value, n=m, dtype=dtype)
    return _one_pad


class Component(enum.IntEnum):
    Position = (1 << 0)
    TexCoord = (1 << 1)
    TexCoord0 = (1 << 2)
    TexCoord1 = (1 << 3)
    Color = (1 << 4)
    Color0 = (1 << 5)
    Color1 = (1 << 6)
    Size = (1 << 7)
    Offset = (1 << 8)
    Transform = (1 << 9)
    Identifier = (1 << 10)
    InstancePosition = (1 << 11)
    InstanceTexCoord = (1 << 12)

FORMAT_PLAIN = Component.Position | Component.TexCoord0 | Component.TexCoord1 | Component.Color0 | Component.Color1
FORMAT_SIZE = FORMAT_PLAIN | Component.Size
FORMAT_OFFSET = FORMAT_PLAIN | Component.Offset
FORMAT_SIZE_OFFSET = FORMAT_PLAIN | Component.Size | Component.Offset
FORMAT_TRANSFORM = FORMAT_PLAIN | Component.Transform
FORMAT_INSTANCE = Component.InstancePosition | Component.InstanceTexCoord

DEFAULT_FORMAT = FORMAT_PLAIN


COMPONENT_TO_COMPONENT_TYPE = {
    Component.Position: ComponentType('position', 1, defaults.DEFAULT_DTYPE, vectors.default_chunk1d, utilities.identity, zero_generator),
    Component.TexCoord: ComponentType('tex_coord', 1, np.float32, utilities.identity, utilities.identity, zero_generator, m=4),
    Component.TexCoord0: ComponentType('tex_coord0', 1, np.float32, utilities.identity, utilities.identity, zero_generator, m=4),
    Component.TexCoord1: ComponentType('tex_coord1', 1, np.float32, utilities.identity, utilities.identity, zero_generator, m=4),
    Component.Color: ComponentType('color', 1, np.float32, utilities.identity, utilities.identity, one_generator, m=4),
    Component.Color0: ComponentType('color0', 1, np.float32, utilities.identity, utilities.identity, one_generator, m=4),
    Component.Color1: ComponentType('color1', 1, np.float32, utilities.identity, utilities.identity, one_generator, m=4),
    Component.Size: ComponentType('size', 1, defaults.DEFAULT_DTYPE, vectors.default_chunk1d, utilities.identity, zero_generator),
    Component.Offset: ComponentType('offset', 1, defaults.DEFAULT_DTYPE, vectors.default_chunk1d, utilities.identity, zero_generator),
    Component.Transform: ComponentType('transform', 2, defaults.DEFAULT_DTYPE, vectors.default_chunk2d, utilities.identity, identity_generator),
    Component.Identifier: ComponentType('identifier', 0, np.uint32, utilities.identity, zero_pad(m=4, dtype=np.uint32), identifier_generator),
    Component.InstancePosition: ComponentType('instance_position', 1, defaults.DEFAULT_DTYPE, vectors.default_chunk1d, utilities.identity, zero_generator),
    Component.InstanceTexCoord: ComponentType('instance_tex_coord', 1, np.float32, utilities.identity, utilities.identity, zero_generator, m=4),
}

COMPONENT_TO_NAME = {key: value.name for key, value in COMPONENT_TO_COMPONENT_TYPE.items()}
NAME_TO_COMPONENT = reversedict(COMPONENT_TO_NAME)

COMPONENT_TO_DTYPE = {key: value.dtype for key, value in COMPONENT_TO_COMPONENT_TYPE.items()}


def components(format):
    return [component for component in Component if component & format]


def default_component_generators(m=defaults.DEFAULT_M, dtypes=COMPONENT_TO_DTYPE):
    return {key: value.generator(m=m) for key, value in COMPONENT_TO_COMPONENT_TYPE.items()}


def default_record_dtypes(chunked=True, m=defaults.DEFAULT_M, dtypes=COMPONENT_TO_DTYPE):
    return {key: value.record_dtype(m=m, chunked=chunked) for key, value in COMPONENT_TO_COMPONENT_TYPE.items()}


def record_dtypes(format=DEFAULT_FORMAT, chunked=True, m=defaults.DEFAULT_M, dtypes=COMPONENT_TO_DTYPE):
    default_dtypes = default_record_dtypes(chunked=chunked, m=m, dtypes=dtypes)
    return [default_dtypes[component] for component in components(format)]


def fromiters(iterables, format=DEFAULT_FORMAT, m=defaults.DEFAULT_M, dtypes=COMPONENT_TO_DTYPE):
    default_generators = default_component_generators(m=m, dtypes=dtypes)

    gs = collections.ChainMap(iterables, default_generators)

    cs = components(format)

    data = zip(*zip(*[gs[component] for component in cs]))

    return {component: np.ascontiguousarray(values, dtype=COMPONENT_TO_COMPONENT_TYPE[component].dtype) for component, values in zip(cs, data)}


def empty(count, format=DEFAULT_FORMAT, chunked=True, m=defaults.DEFAULT_M, dtypes=COMPONENT_TO_DTYPE):
    return np.empty(count, dtype=record_dtypes(format=format, chunked=chunked, m=m, dtypes=dtypes))


def full(iterables, format=DEFAULT_FORMAT, chunked=True, m=defaults.DEFAULT_M, dtypes=COMPONENT_TO_DTYPE):
    arrays = fromiters(iterables, format=format, m=m, dtypes=dtypes)
    counts = [len(array) for array in arrays.values()]
    result = empty(max(counts), format=format, chunked=chunked, m=m, dtypes=dtypes)
    
    for component in components(format):
        func = utilities.compose(COMPONENT_TO_COMPONENT_TYPE[component].chunk, COMPONENT_TO_COMPONENT_TYPE[component].pad)
        result[COMPONENT_TO_NAME[component]] = list(map(func, arrays[component]))
    
    return result


def concatenate(*args):
    return np.concatenate(args)
