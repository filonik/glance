import os
import sys

import numpy as np

from glue import gl
from glue.gl import GL

if sys.version_info >= (3, 0):
    import cairocffi as cairo
else:
    import cairo

from ..graphics import textures


class Canvas(textures.TextureData):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("target", GL.GL_TEXTURE_2D) 
        kwargs.setdefault("format", GL.GL_BGRA)
        
        super(Canvas, self).__init__(*args, **kwargs)
        
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
