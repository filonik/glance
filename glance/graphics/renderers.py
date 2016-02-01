import contextlib as cl
import enum

import numpy as np

from glue import gl
from glue.gl import GL

from encore import decorators

from ..mathematics import vectors

from . import resources, shaders


class Shade(enum.IntEnum):
    Flat = 1
    Phong = 2

    
class Space(enum.IntEnum):
    Model = 1
    World = 2
    View = 3


class Wrap(enum.IntEnum):
    Clamp = 1
    Repeat = 2
    ClampRepeat = 3


class Renderer(object):
    @property
    def context(self):
        return self._context

    def __init__(self):
        super(Renderer, self).__init__()

        self._context = gl.backend.Context(handle=-1)

        self._program = None
        self._material = None

        self._texture_unit = 0
        self._light_unit = 0

    def available_texture_unit(self):
        self._texture_unit += 1
        return self._texture_unit - 1

    def available_light_unit(self):
        self._light_unit += 1
        return self._light_unit - 1
    
    @decorators.indexedproperty
    def uniforms(self, key):
        return self._program._value.uniforms[key]

    @uniforms.setter
    def uniforms(self, key, value):
        if isinstance(value, resources.Resource):
            self._program._value.uniforms[key] = value._value
        else:
            self._program._value.uniforms[key] = value

    @decorators.indexedproperty
    def inputs(self, key):
        return self._program._value.inputs[key]

    @inputs.setter
    def inputs(self, key, value):
        if isinstance(value, resources.Resource):
            self._program._value.inputs[key] = value._value
        else:
            self._program._value.inputs[key] = value

    @decorators.indexedproperty
    def outputs(self, key):
        return self._program._value.outputs[key]

    @decorators.indexedproperty
    def uniform_blocks(self, key):
        return self._program._value.uniform_blocks[key]

    @uniform_blocks.setter
    def uniform_blocks(self, key, value):
        if isinstance(value, resources.Resource):
            self._program._value.uniform_blocks[key] = value._value

    @decorators.indexedproperty
    def shader_storage_blocks(self, key):
        return self._program._value.shader_storage_blocks[key]

    @shader_storage_blocks.setter
    def shader_storage_blocks(self, key, value):
        if isinstance(value, resources.Resource):
            self._program._value.shader_storage_blocks[key] = value._value

    def set_uniform_array_1d(self, key, value):
        for i, in np.ndindex(value.shape[:1]):
            self.uniforms["%s[%s]" % (key, i)] = value[i]

    def set_uniform_array_2d(self, key, value):
        for i, j in np.ndindex(value.shape[:2]):
            self.uniforms["%s[%s][%s]" % (key, i, j)] = value[i][j]

    def set_model_view_projection(self, model, view, projection):
        model_view = np.dot(model, view)
        model_view_projection = np.dot(model_view, projection)
        view_projection = np.dot(view, projection)

        self.uniforms['model'] = model
        self.uniforms['model_view'] = model_view
        self.uniforms['model_view_projection'] = model_view_projection
        self.uniforms['view'] = view
        self.uniforms['view_projection'] = view_projection
        self.uniforms['projection'] = projection

        self.uniforms['inverse_model'] = vectors.inverse(model)
        self.uniforms['inverse_view'] = vectors.inverse(view)
        
        #print(self.uniforms['inverse_view'])
        
        #self.uniforms['dual_model'] = normalized_basis(transpose_inverse(model[:3,:3]))
        #self.uniforms['dual_model_view'] = normalized_basis(transpose_inverse(model_view[:3,:3]))
    
    def set_model_nd(self, model_nd):
        model_nd_chunked = vectors.default_chunk2d(model_nd)
        self.set_uniform_array_2d("model_nd", model_nd_chunked)
        
        inverse_model_nd = vectors.inverse(model_nd)
        inverse_model_nd_chunked = vectors.default_chunk2d(inverse_model_nd)
        self.set_uniform_array_2d("inverse_model_nd", inverse_model_nd_chunked)
    
        #dual_model_nd = np.transpose(unit_row(inverse_model_nd, len(inverse_model_nd)-1))
        #dual_model_nd_chunked = vectors.default_chunk2d(dual_model_nd)

    def set_view_nd(self, view_nd):
        view_nd_chunked = vectors.default_chunk2d(view_nd)
        self.set_uniform_array_2d("view_nd", view_nd_chunked)
        
        inverse_view_nd = vectors.inverse(view_nd)
        inverse_view_nd_chunked = vectors.default_chunk2d(inverse_view_nd)
        self.set_uniform_array_2d("inverse_view_nd", inverse_view_nd_chunked)
        
        #dual_view_nd = np.transpose(unit_row(inverse_view_nd, len(inverse_view_nd)-1))
        #dual_view_nd_chunked = vectors.default_chunk2d(dual_view_nd)
    
    def set_projection_nd(self, projection_nd):
        projection_nd_chunked = vectors.default_chunk2d(projection_nd)
        self.set_uniform_array_2d("projection_nd", projection_nd_chunked)
        
    def set_model_view_nd(self, model_nd, view_nd):
        self.set_model_nd(model_nd)
        self.set_view_nd(view_nd)
        
        model_view_nd = np.dot(model_nd, view_nd)
        model_view_nd_chunked = vectors.default_chunk2d(model_view_nd)
        self.set_uniform_array_2d("model_view_nd", model_view_nd_chunked)
        
        inverse_model_view_nd = vectors.inverse(model_view_nd)
        inverse_model_view_nd_chunked = vectors.default_chunk2d(inverse_model_view_nd)
        self.set_uniform_array_2d("inverse_model_view_nd", inverse_model_view_nd_chunked)
    
    def set_model_view_projection_nd(self, model_nd, view_nd, projection_nd):
        self.set_model_nd(model_nd)
        self.set_view_nd(view_nd)
        self.set_projection_nd(projection_nd)
        
        model_view_nd = np.dot(model_nd, view_nd)
        model_view_nd_chunked = vectors.default_chunk2d(model_view_nd)
        self.set_uniform_array_2d("model_view_nd", model_view_nd_chunked)
        
        model_view_projection_nd = np.dot(model_view_nd, projection_nd)
        model_view_projection_nd_chunked = vectors.default_chunk2d(model_view_projection_nd)
        self.set_uniform_array_2d("model_view_projection_nd", model_view_projection_nd_chunked)
        
        view_projection_nd = np.dot(view_nd, projection_nd)
        view_projection_nd_chunked = vectors.default_chunk2d(view_projection_nd)
        self.set_uniform_array_2d("view_projection_nd", view_projection_nd_chunked)
    
    def activate(self, resource):
        if isinstance(resource, shaders.Program):
            self._program = resource
            self._texture_unit = 0
            self._light_unit = 0

        resource.activate(renderer=self)

    def deactivate(self, resource):
        resource.deactivate(renderer=self)

    @cl.contextmanager
    def activated(self, *args, **kwargs):
        for arg in args:
            self.activate(arg)

        yield

        for arg in reversed(args):
            self.deactivate(arg)
