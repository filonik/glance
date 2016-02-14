import os
import sys

import numpy as np

from glue import gl
from glue.gl import GL

if sys.version_info >= (3, 0):
    import cairocffi as cairo
else:
    import cairo

from ..mathematics import transforms, vectors
from ..graphics import textures

class Canvas(textures.TextureData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target", GL.GL_TEXTURE_2D) 
        kwargs.setdefault("format", GL.GL_BGRA)
        
        super().__init__(*args, **kwargs)
        
        self._surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(self.size[0]), int(self.size[1]))
        
    def paint(self, painter):
        self._context = cairo.Context(self._surface)
        self._context.set_operator(cairo.OPERATOR_CLEAR)
        self._context.paint()

        painter(self._context, self.size)

    def provide(self):
        image = np.frombuffer(self._surface.get_data(), np.uint8)
        image.shape = (self.size[1], self.size[0], 4)
        
        self.images = [image]
        
        return self


class Atlas(textures.TextureData):
    @property
    def shape(self):
        return self._shape
        
    @shape.setter
    def shape(self, value):
        self._shape = value
        
    @property
    def tex_coord_transform(self):
        return transforms.scale(vectors.vector(1.0/self._shape, n=3), n=4)
    
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target", GL.GL_TEXTURE_2D) 
        kwargs.setdefault("format", GL.GL_BGRA)
        
        super().__init__(*args, **kwargs)
        
        self._shape = np.array(kwargs.get("shape", (1, 1)), dtype=np.uint32)
        self._surface = cairo.ImageSurface(cairo.FORMAT_ARGB32, int(self.size[0]), int(self.size[1]))
    
    def paint(self, painters):
        from . import painters as pntrs
        
        self._context = cairo.Context(self._surface)
        self._context.set_operator(cairo.OPERATOR_CLEAR)
        self._context.paint()
        
        painter = pntrs.atlas(self._shape, painters)
        
        painter(self._context, self.size)

    def provide(self):
        image = np.frombuffer(self._surface.get_data(), np.uint8)
        image.shape = (self.size[1], self.size[0], 4)
        
        self.images = [image]
        
        return self
