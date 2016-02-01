import collections

import numpy as np

from glue import gl
from glue.gl import GL

from encore import accessors, objects

from ..mathematics import defaults, vectors

from .transforms import *
from .projections import *


class Node(object):
    @property
    def relative_transform(self, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        return self._transform.to_matrix(n=n, m=m, dtype=dtype)

    @property
    def absolute_transform(self, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        result = self.relative_transform(n=n, m=m, dtype=dtype)
        if self._parent:
            result = np.dot(self._parent.absolute_transform(n=n, dtype=dtype), result)
        return result
    
    def __init__(self, transform=None, parent=None):
        super().__init__()
        
        self._parent = parent
        
        self._transform = Transform() if transform is None else transform

    def update(self, delta):
        self._transform.update(delta)


class Camera(Node):
    @property
    def projection(self, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        return self._projection.to_matrix(n=n, m=m, dtype=dtype)
    
    def __init__(self, transform=None, projection=None, parent=None):
        super().__init__(transform=transform, parent=parent)
        
        self._projection = Projection() if projection is None else projection


class View(object):
    @property
    def parent(self):
        return self._parent
        
    @parent.setter
    def parent(self, value):
        self._parent = value
    
    @property
    def clock(self):
        return self._parent.clock if self._clock is None else self._clock
        
    @clock.setter
    def clock(self, value):
        self._clock = value
    
    @property
    def size(self):
        return self._parent.size if self._size is None else self._size
        
    @size.setter
    def size(self, value):
        self._size = value
    
    @property
    def time(self):
        return self.clock.elapsed()
    
    @property
    def aspect(self):
        return self.size/np.min(self.size)
    
    @property
    def camera(self):
        return self._parent.camera if self._camera is None else self._camera
        
    @camera.setter
    def camera(self, value):
        self._camera = value
    
    @property
    def root(self):
        return self._parent.root if self._root is None else self._root
        
    @camera.setter
    def root(self, value):
        self._root = value
    
    def __init__(self, clock=None, size=None, parent=None):
        super().__init__()
        
        self._parent = parent
        
        self._clock = clock
        self._size = size
        
        self._camera = None
        self._root = None
    
    def on_key_press(self, state):
        pass
    
    def on_key_release(self, state):
        pass
    
    def create(self, renderer):
        pass
        
    def update(self, renderer):
        pass
    
    def render(self, renderer):
        pass
        
    def delete(self, renderer):
        pass

