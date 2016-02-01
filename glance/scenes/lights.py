import os
import enum

import numpy as np

from glue import gl
from glue.gl import GL
from glue.gl import Unspecified, specified, getspecified

from encore import decorators

from ..graphics import defaults, resources, buffers

from .. import colors


class Light(resources.Resource):
    def __init__(self, ambient=defaults.DEFAULT_AMBIENT_COLOR, diffuse=defaults.DEFAULT_DIFFUSE_COLOR, specular=defaults.DEFAULT_SPECULAR_COLOR, *args, **kwargs):
        super(Light, self).__init__(*args, **kwargs)
        
        self.enabled = True
        
        self.ambient = ambient
        self.diffuse = diffuse
        self.specular = specular
        
        self.attenuation = np.array([0.5, 0.001, 0.0001], dtype=np.float32)
        self.position = np.zeros(4, dtype=np.float32)
        
        self.target = 0
    
    def activate(self, renderer=Unspecified):
        self.unit = self.target #renderer.available_light_unit()

        ambient = self.ambient if self.enabled else colors.svg.black
        diffuse = self.diffuse if self.enabled else colors.svg.black
        specular = self.specular if self.enabled else colors.svg.black

        renderer.uniforms['lights[%s].ambient' % (self.unit,)] = np.asarray(ambient[:3], dtype=np.float32)
        renderer.uniforms['lights[%s].diffuse' % (self.unit,)] = np.asarray(diffuse[:3], dtype=np.float32)
        renderer.uniforms['lights[%s].specular' % (self.unit,)] = np.asarray(specular[:3], dtype=np.float32)

        renderer.uniforms['lights[%s].attenuation' % (self.unit,)] = np.asarray(self.attenuation, dtype=np.float32)
        renderer.uniforms['lights[%s].position' % (self.unit,)] = np.asarray(self.position, dtype=np.float32)
    
    def deactivate(self, renderer):
        pass


class MultiLight(resources.Resource):
    def __init__(self, lights=None):
        super().__init__()

        self._items = [] if lights is None else lights
    
    def __getitem__(self, key):
        return self._items[key]
        
    def __setitem__(self, key, value):
        self._items[key] = value
        
    def __delitem__(self, key):
        del self._items[key]
    
    def prepare(self, renderer):
        return True

    def activate(self, renderer):
        if self.prepare(renderer=renderer):
            for target, light in enumerate(self._items):
                light.target = target
                light.activate(renderer)
    
    def deactivate(self, renderer):
        for light in self._items:
            light.target = 0
            light.deactivate(renderer)
