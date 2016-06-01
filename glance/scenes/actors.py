import numpy as np

from .. import colors

from ..mathematics import extents
from ..graphics import buffers

from . import materials, geometries, scenes


class Actor(scenes.Node):
    @property
    def instance_count(self):
        return self._instance_count
    
    @instance_count.setter
    def instance_count(self, value):
        self._instance_count = value
    
    @property
    def multiple_count(self):
        return self._multiple_count
    
    @multiple_count.setter
    def multiple_count(self, value):
        self._multiple_count = value
    
    @property
    def visible(self):
        return self._enabled and self._visible
    
    def __init__(self, material=None, geometry=None, transform=None, parent=None):
        super().__init__(transform=transform, parent=parent)
        
        self._instance_count = 1
        self._multiple_count = 1
        
        self._material = material
        self._geometry = geometry
        
        self._enabled = True
        self._visible = True


class SingleBufferComputeActor(Actor):
    @Actor.multiple_count.setter
    def multiple_count(self, value):
        if self._multiple_count != value:
            self._multiple_count = value
            
            self._output_vertex_data.size = self._input_vertex_data.size * self._multiple_count
            self._output_vertex_data.type = self._input_vertex_data.type
            
            self.update_output_index_data(self.generate_output_index_data(self.input_vertex_data))
    
    @property
    def input_vertex_data(self):
        return self._input_vertex_data.data

    @input_vertex_data.setter
    def input_vertex_data(self, value):
        self._input_vertex_data.data = value
        
        self._output_vertex_data.size = self._input_vertex_data.size * self._multiple_count
        self._output_vertex_data.type = self._input_vertex_data.type
    
    def __init__(self, transform=None, parent=None):
        super().__init__(transform=transform, parent=parent)
        
        self._input_vertex_data = buffers.BufferData()
        self._output_vertex_data = buffers.BufferData()
        
        self._input_index_data = buffers.BufferData()
        self._output_index_data = buffers.BufferData()
        
        self._input_vertex_buffer = buffers.ShaderStorageBuffer(self._input_vertex_data)
        self._output_vertex_buffer = buffers.ShaderStorageBuffer(self._output_vertex_data)
        
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
            "identifier": self._input_vertex_buffer,
        }, indices={
            "default": self._input_index_buffer,
        })
        
        self._output_geometry = geometries.CustomGeometry({
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
        
        self._geometry = self._output_geometry
        
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
    
    def set_instanced(self, value):
        value = 1 if value else 0
        self._input_vertex_buffer._divisor = value
        self._output_vertex_buffer._divisor = value
    
    def generate_input_index_data(self, vertex_data):
        size = 0 if vertex_data is None else len(vertex_data) 
        
        return np.ascontiguousarray(range(size), dtype=np.uint32)
    
    def generate_output_index_data(self, vertex_data):
        size = 0 if vertex_data is None else len(vertex_data) 
        count = self._multiple_count
        
        return np.ascontiguousarray(range(size*count), dtype=np.uint32)
    
    def update_input_index_data(self, index_data):
        self._input_index_data.data = index_data
    
    def update_output_index_data(self, index_data):
        self._output_index_data.data = index_data
    
    def update_vertex_data(self, vertex_data):
        self.input_vertex_data = vertex_data
        
        input_index_data = self.generate_input_index_data(vertex_data)
        self.update_input_index_data(input_index_data)
        
        output_index_data = self.generate_output_index_data(vertex_data)
        self.update_output_index_data(output_index_data)


class DoubleBufferComputeActor(Actor):
    @Actor.multiple_count.setter
    def multiple_count(self, value):
        if self._multiple_count != value:
            self._multiple_count = value
            
            self._output_vertex_data.size = self._input1_vertex_data.size * self._multiple_count
            self._output_vertex_data.type = self._input1_vertex_data.type
            
            self.update_output_index_data(self.generate_output_index_data(self.input_vertex_data))
    
    @property
    def input_vertex_data(self):
        return self._input1_vertex_data.data

    @input_vertex_data.setter
    def input_vertex_data(self, value):
        self._input0_vertex_data.data, self._input1_vertex_data.data = self._input1_vertex_data.data, value
        
        self._output_vertex_data.size = self._input1_vertex_data.size * self._multiple_count
        self._output_vertex_data.type = self._input1_vertex_data.type
    
    @property
    def input0_vertex_data(self):
        return self._input0_vertex_data.data

    @input0_vertex_data.setter
    def input0_vertex_data(self, value):
        self._input0_vertex_data.data = value
        
        self._output_vertex_data.size = self._input0_vertex_data.size * self._multiple_count
        self._output_vertex_data.type = self._input0_vertex_data.type

    @property
    def input1_vertex_data(self):
        return self._input1_vertex_data.data

    @input1_vertex_data.setter
    def input1_vertex_data(self, value):
        self._input1_vertex_data.data = value
        
        self._output_vertex_data.size = self._input1_vertex_data.size * self._multiple_count
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
    
    def set_instanced(self, value):
        value = 1 if value else 0
        self._input0_vertex_buffer._divisor = value
        self._input1_vertex_buffer._divisor = value
        self._output_vertex_buffer._divisor = value
    
    def generate_input_index_data(self, vertex_data):
        size = 0 if vertex_data is None else len(vertex_data) 
        
        return np.ascontiguousarray(range(size), dtype=np.uint32)
    
    def generate_output_index_data(self, vertex_data):
        size = 0 if vertex_data is None else len(vertex_data) 
        count = self._multiple_count
        
        return np.ascontiguousarray(range(size*count), dtype=np.uint32)
    
    def update_input_index_data(self, index_data):
        self._input0_index_data.data, self._input1_index_data.data = self._input1_index_data.data, index_data
    
    def update_output_index_data(self, index_data):
        self._output_index_data.data = index_data
    
    def update_vertex_data(self, vertex_data):
        self.input0_vertex_data, self.input1_vertex_data = self.input1_vertex_data, vertex_data
        
        input_index_data = self.generate_input_index_data(vertex_data)
        self.update_input_index_data(input_index_data)
        
        output_index_data = self.generate_output_index_data(vertex_data)
        self.update_output_index_data(output_index_data)
    