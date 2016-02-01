import numpy as np

from .. import colors

from ..mathematics import extents
from ..graphics import buffers

from . import materials, geometries, scenes


class Actor(scenes.Node):    
    def __init__(self, material=None, geometry=None, transform=None, parent=None):
        super().__init__(transform=transform, parent=parent)
        
        self._material = material
        self._geometry = geometry


class SingleBufferComputeActor(Actor):
    def __init__(self, transform=None, parent=None):
        super().__init__(transform=transform, parent=parent)
        
        self._input_vertex_data = buffers.BufferData()
        self._output_vertex_data = buffers.BufferData()
        
        self._input_index_data = buffers.BufferData()
        self._output_index_data = buffers.BufferData()
        
        self._input_vertex_buffer = buffers.ShaderStorageBuffer(self._input_vertex_data, divisor=1)
        self._output_vertex_buffer = buffers.ShaderStorageBuffer(self._output_vertex_data, divisor=1)
        
        self._input_index_buffer = buffers.IndexBuffer(self._input_index_data)
        self._output_index_buffer = buffers.IndexBuffer(self._output_index_data)
        
        self._input_geometry = geometries.CustomGeometry({
            "position": self._input_vertex_buffer,
            "tex_coord0": self._input_vertex_buffer,
            "tex_coord1": self._input_vertex_buffer,
            "color0": self._input_vertex_buffer,
            "color1": self._input_vertex_buffer,
            "size": self._input_vertex_buffer,
            "offset": self._input_vertex_buffer,
        }, indices={
            "default": self._input_index_buffer,
        })
        
        self._output_geometry = geometries.CustomGeometry({
            "position": self._output_vertex_buffer,
            "tex_coord0": self._input_vertex_buffer,
            "tex_coord1": self._input_vertex_buffer,
            "color0": self._input_vertex_buffer,
            "color1": self._input_vertex_buffer,
            "size": self._output_vertex_buffer,
            "offset": self._output_vertex_buffer,
        }, indices={
            "default": self._output_index_buffer,
        })
        
        self._material = materials.MultiMaterial([
            materials.PhongMaterial(colors={
                "ambient": colors.svg.white.rgb,
                "diffuse": colors.svg.white.rgb,
                "specular": colors.svg.black.rgb,
                "emissive": colors.svg.black.rgb,
            }, opacity=1.0, shininess=32.0),  materials.PhongMaterial(colors={
                "ambient": colors.svg.white.rgb,
                "diffuse": colors.svg.white.rgb,
                "specular": colors.svg.black.rgb,
                "emissive": colors.svg.black.rgb,
            }, opacity=1.0, shininess=32.0)
        ])
        self._geometry = self._input_geometry


class DoubleBufferComputeActor(Actor):
    @property
    def input(self):
        return self._input0_vertex_data.data

    @input.setter
    def input(self, value):
        self._input0_vertex_data.data = value
        self._input1_vertex_data.data = value

        count = len(self._transforms)

        self._output_vertex_data.size = self._input0_vertex_data.size * count
        self._output_vertex_data.type = self._input0_vertex_data.type
    
    @property
    def input0_vertex_data(self):
        return self._input0_vertex_data.data

    @input0_vertex_data.setter
    def input0_vertex_data(self, value):
        self._input0_vertex_data.data = value

        count = len(self._transforms)

        self._output_vertex_data.size = self._input0_vertex_data.size * count
        self._output_vertex_data.type = self._input0_vertex_data.type

    @property
    def input1_vertex_data(self):
        return self._input1_vertex_data.data

    @input1_vertex_data.setter
    def input1_vertex_data(self, value):
        self._input1_vertex_data.data = value

        count = len(self._transforms)

        self._output_vertex_data.size = self._input1_vertex_data.size * count
        self._output_vertex_data.type = self._input1_vertex_data.type
    
    def __init__(self, transform=None, parent=None):
        super().__init__(transform=transform, parent=parent)
        
        self._input0_vertex_data = buffers.BufferData()
        self._input1_vertex_data = buffers.BufferData()
        self._output_vertex_data = buffers.BufferData()
        
        self._input0_index_data = buffers.BufferData()
        self._input1_index_data = buffers.BufferData()
        self._output_index_data = buffers.BufferData()
        
        self._input0_vertex_buffer = buffers.ShaderStorageBuffer(self._input0_vertex_data)
        self._input1_vertex_buffer = buffers.ShaderStorageBuffer(self._input1_vertex_data)
        self._output_vertex_buffer = buffers.ShaderStorageBuffer(self._output_vertex_data)
        
        self._input0_index_buffer = buffers.IndexBuffer(self._input0_index_data)
        self._input1_index_buffer = buffers.IndexBuffer(self._input1_index_data)
        self._output_index_buffer = buffers.IndexBuffer(self._output_index_data)
        
        self._geometry = geometries.CustomGeometry({
            "position": self._output_vertex_buffer,
            "tex_coord0": self._output_vertex_buffer,
            "tex_coord1": self._output_vertex_buffer,
            "color0": self._output_vertex_buffer,
            "color1": self._output_vertex_buffer,
            "size": self._output_vertex_buffer,
            "offset": self._output_vertex_buffer,
            "identifier": self._output_vertex_buffer,
        }, indices={
            "default": self._output_index_buffer,
        })
        
        self._material = materials.MultiMaterial([
            materials.PhongMaterial(colors={
                "ambient": colors.svg.white.rgb,
                "diffuse": colors.svg.white.rgb,
                "specular": colors.svg.black.rgb,
                "emissive": colors.svg.black.rgb,
            }, opacity=1.0, shininess=32.0),  materials.PhongMaterial(colors={
                "ambient": colors.svg.white.rgb,
                "diffuse": colors.svg.white.rgb,
                "specular": colors.svg.black.rgb,
                "emissive": colors.svg.black.rgb,
            }, opacity=1.0, shininess=32.0)
        ])
        
        self._transforms = []
    
    def update_index_data(self, vertex_data):
        size = 0 if vertex_data is None else len(vertex_data) 
        
        value = np.ascontiguousarray(range(size), dtype=np.uint32)
        self._input0_index_data.data, self._input1_index_data.data = self._input1_index_data.data, value
        
        count = len(self._transforms)
        
        value = np.ascontiguousarray(range(size*count), dtype=np.uint32)
        self._output_index_data.data = value
    
    def update_vertex_data(self, vertex_data):
        self.input0_vertex_data, self.input1_vertex_data = self.input1_vertex_data, vertex_data
        
        self.update_index_data(vertex_data)
