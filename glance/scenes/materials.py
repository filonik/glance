import os

import collections
import enum

from glue import gl
from glue.gl import GL
from glue.gl import Unspecified, specified, getspecified

from encore import decorators

from ..graphics import defaults, resources, buffers, textures

from ..colors import svg


class Component(enum.IntEnum):
    Ambient = (1 << 0)
    Diffuse = (1 << 1)
    Specular = (1 << 2)
    Emissive = (1 << 3)


DEFAULT_FORMAT = Component.Ambient | Component.Diffuse | Component.Specular | Component.Emissive


class CustomMaterial(resources.Resource):
    def __init__(self, data=None, *args, **kwargs):
        super().__init__()
        
        self._data = {} if data is None else data
        self._data.update(kwargs)
        
        self.target = 0
    
    def __getitem__(self, key):
        return self._data[key]
    
    def __setitem__(self, key, value):
        self._data[key] = value
    
    def __delitem__(self, key):
        del self._data[key]
    
    def prepare(self, renderer):
        def _prepare(data):
            if isinstance(data, collections.Mapping):
                for key, value in data.items():
                    _prepare(value)
            elif isinstance(data, textures.Texture):
                data.prepare(renderer)
            else:
                pass
        
        _prepare(self._data)
        
        return True
    
    def activate(self, renderer):
        if self.prepare(renderer):
            def _activate(path, data):
                if isinstance(data, collections.Mapping):
                    for key, value in data.items():
                        sub_path = ".".join([path, key])
                        _activate(sub_path, value)
                elif isinstance(data, textures.Texture):
                    data.activate(renderer)
                    renderer.uniforms[path] = data.unit
                else:
                    renderer.uniforms[path] = data
            
            path = 'materials[%s]' % (self.target)
            
            _activate(path, self._data)
    
    def deactivate(self, renderer):
        def _deactivate(path, data):
            if isinstance(data, collections.Mapping):
                for key, value in data.items():
                    sub_path = ".".join([path, key])
                    _deactivate(sub_path, value)
            elif isinstance(data, textures.Texture):
                data.deactivate(renderer)
        
        path = 'materials[%s]' % (self.target)
        
        _deactivate(path, self._data)


class PhongMaterial(CustomMaterial):
    @property
    def colors(self):
        return self._data.get("colors", {})

    @property
    def textures(self):
        return self._data.get("textures", {})
    
    @property
    def opacity(self):
        return self._data.get("opacity", 1.0)
    
    @property
    def shininess(self):
        return self._data.get("shininess", 32.0)
    
    def __init__(self, *args, **kwargs):
        colors = kwargs.setdefault("colors", {})
        colors.setdefault("ambient", svg.white.rgb)
        colors.setdefault("diffuse", svg.white.rgb)
        colors.setdefault("specular", svg.black.rgb)
        colors.setdefault("emissive", svg.black.rgb)
        
        textures = kwargs.setdefault("textures", {}) 
        textures.setdefault("ambient", defaults.DEFAULT_TEXTURE_WHITE)
        textures.setdefault("diffuse", defaults.DEFAULT_TEXTURE_WHITE)
        textures.setdefault("specular", defaults.DEFAULT_TEXTURE_BLACK)
        textures.setdefault("emissive", defaults.DEFAULT_TEXTURE_BLACK)
        
        kwargs.setdefault("opacity", 1.0) 
        kwargs.setdefault("shininess", 32.0) 
        
        super().__init__(*args, **kwargs)


class MultiMaterial(resources.Resource):
    def __init__(self, materials=None):
        super().__init__()

        self._items = [] if materials is None else materials
    
    def __getitem__(self, key):
        return self._items[key]
        
    def __setitem__(self, key, value):
        self._items[key] = value
        
    def __delitem__(self, key):
        del self._items[key]
    
    def prepare(self, renderer):
        result = True

        for material in self._items:
            result = result and material.prepare(renderer)
        
        return result

    def activate(self, renderer):
        if self.prepare(renderer=renderer):
            for target, material in enumerate(self._items):
                material.target = target
                material.activate(renderer)
    
    def deactivate(self, renderer):
        for material in self._items:
            material.target = 0
            material.deactivate(renderer)
