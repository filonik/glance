import itertools as it
import logging

import numpy as np

from glue import gl
from glue.gl import GL

from ..mathematics import defaults as mdefaults, coordinates, extents, geometries as mgeometries, mathematics, projections, transforms, vectors
from ..graphics import defaults as gdefaults, buffers, primitives, renderers, shaders, textures, vertices
from ..scenes import actors, animations, materials, geometries, scenes

from .. import colors, shapes, palettes
from .. import painters

logger = logging.getLogger(__name__.split(".").pop())


def interleaved_indices(m, n):
    for i in range(m):
        for j in range(n):
            yield i + m * j


def interleaved(sequence, m, n):
    for i in range(0, len(sequence), n * m):
        for j in interleaved_indices(m, n):
            yield sequence[i + j]


class AxisActor(actors.SingleBufferComputeActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self.origin_count = 1
        self.origin_offset = 0
        self.origin_stride = 1
        
        self.axis_count = mdefaults.DEFAULT_N
        self.axis_offset = self.origin_offset + self.origin_count * self.origin_stride
        self.axis_stride = 2
        
        self.tick_count = mdefaults.DEFAULT_N
        self.tick_offset = self.axis_offset + self.axis_count * self.axis_stride
        self.tick_stride = 10
    
    def generate_output_index_data(self, vertex_data):
        #print("generate axis indices")
        
        size = 0 if vertex_data is None else len(vertex_data)
        count = self._multiple_count
        
        origin_indices = self.origin_offset * count + np.asarray(range(self.origin_count * self.origin_stride * count), dtype=np.uint32)
        axis_indices = self.axis_offset * count + np.asarray(range(self.axis_count * self.axis_stride * count), dtype=np.uint32)
        tick_indices = self.tick_offset * count + np.asarray(range(self.tick_count * self.tick_stride * count), dtype=np.uint32)
        
        indices = it.chain(*[
            interleaved(origin_indices, count, 1),
            interleaved(axis_indices, count, 2),
            interleaved(tick_indices, count, 1),
        ])
        
        result = np.ascontiguousarray(list(indices), dtype=np.uint32)
        
        return result


class TickActor(actors.SingleBufferComputeActor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._label_data = []
        #self._label_provider = painters.Atlas(shape=(8, 128), size=(2048, 2048))
        self._label_provider = painters.Atlas(shape=(4, 8), size=(2048, 512))
        self._label_atlas = textures.Texture(self._label_provider, filter=(GL.GL_NEAREST, GL.GL_NEAREST))


class MarkActor(actors.DoubleBufferComputeActor):
    @property
    def extent(self):
        return self._extent1
    
    @extent.setter
    def extent(self, value):
        self._extent0, self._extent1 = self._extent1, value
    
    @property
    def order(self):
        return self._order1
    
    @order.setter
    def order(self, value):
        self._order0, self._order1 = self._order1, value
    
    @property
    def basis(self):
        return self._basis1
    
    @basis.setter
    def basis(self, value):
        self._basis0, self._basis1 = self._basis1, value 
    
    @property
    def dual_basis(self):
        return self._dual_basis1
    
    @dual_basis.setter
    def dual_basis(self, value):
        self._dual_basis0, self._dual_basis1 = self._dual_basis1, value
    
    @property
    def rank(self):
        return len(self._basis1)
    
    @property
    def bases(self):
        return np.concatenate((self._basis1, self._dual_basis1), axis=0)
    
    @property
    def instance_count(self):
        return self._instance_count
    
    @instance_count.setter
    def instance_count(self, value):
        self._instance_count = value
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._alpha = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        
        self._extent0 = self._extent1 = extents.Extent()
        self._order0 = self._order1 = 0
        self._basis0 = self._basis1 = np.empty((0, mdefaults.DEFAULT_M), dtype=mdefaults.DEFAULT_DTYPE)
        self._dual_basis0 = self._dual_basis1 = transforms.identity(n=mdefaults.DEFAULT_M)
        
        self._size = 0.0
        self._motion = 0.0
        
        self._label_data = []
        #self._label_provider = painters.Atlas(shape=(8, 128), size=(2048, 2048))
        self._label_provider = painters.Atlas(shape=(8, 32), size=(2048, 2048))
        self._label_atlas = textures.Texture(self._label_provider, filter=(GL.GL_NEAREST, GL.GL_NEAREST))
        
        self._join_index_data = buffers.BufferData()
        self._join_index_buffer = buffers.IndexBuffer(self._join_index_data)
        
        self._geometry.index_buffers["joins"] = self._join_index_buffer
        self._geometry.index_buffers["instance_joins"] = self._join_index_buffer
    
    def current_rank(self, time):
        rank0, rank1 = len(self._basis0), len(self._basis1)
        if self._alpha.has_finished(time):
            return rank1
        else:
            return max(rank0, rank1)
    
    def current_bases(self, time):
        rank0, rank1 = len(self._basis0), len(self._basis1)
        #print("B0:", self._basis0, self._dual_basis0)
        #print("B1:", self._basis1, self._dual_basis1)
        if self.current_rank(time) == rank1:
            return np.concatenate((self._basis1, self._dual_basis1), axis=0)
        if self.current_rank(time) == rank0:
            return np.concatenate((self._basis0, self._dual_basis0), axis=0)
    
    def current_extent(self, time):
        extent = extents.interpolate_linear(self._extent0, self._extent1)
        return extent(self._alpha(time))
    
    def generate_output_index_data(self, vertex_data):
        #print("generate mark indices")
        
        size = 0 if vertex_data is None else len(vertex_data)
        count = self._multiple_count
        
        marks_join0_indices = [i*count + np.array(range(count), dtype=np.uint32) for i in range(size)]
        marks_join0_indices = [primitives.line_strip_to_line_indices(line_strip) for line_strip in marks_join0_indices]
        
        self._join_index_data.data = np.ascontiguousarray(marks_join0_indices, dtype=np.uint32).ravel()
        
        return super().generate_output_index_data(vertex_data)
