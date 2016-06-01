import collections
import enum

import numpy as np

from ..mathematics import defaults, coordinates, projections, transforms, vectors


class Component(enum.IntEnum):
    Scale = (1 << 0)
    Rotate = (1 << 1)
    Translate = (1 << 2)
    Orientate = (1 << 3)


class Transform(object):
    @property
    def translation(self):
        return self._translation1
        
    @translation.setter
    def translation(self, value):
        n = min(len(self._translation1), len(value))
        self._translation0, self._translation1 = np.copy(self._translation1), value[:n]
    
    @property
    def rotation(self):
        return self._rotation1
        
    @rotation.setter
    def rotation(self, value):
        n = min(len(self._rotation1), len(value))
        self._rotation0, self._rotation1[:n] = np.copy(self._rotation1), value[:n]
    
    @property
    def scaling(self):
        return self._scaling1
    
    @scaling.setter
    def scaling(self, value):
        n = min(len(self._scaling1), len(value))
        self._scaling0, self._scaling1[:n] = np.copy(self._scaling1), value[:n]
    
    def __init__(self, translation=vectors.zeros(n=0), rotation=vectors.zeros(n=0), scaling=vectors.ones(n=0), reversed=False, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        super().__init__()
        
        self._translation0 = vectors.zeros(*translation, n=n, dtype=dtype)
        self._translation1 = vectors.zeros(*translation, n=n, dtype=dtype)
        self._rotation0 = vectors.zeros(*rotation, n=n, dtype=dtype)
        self._rotation1 = vectors.zeros(*rotation, n=n, dtype=dtype)
        self._scaling0 = vectors.ones(*scaling, n=n, dtype=dtype)
        self._scaling1 = vectors.ones(*scaling, n=n, dtype=dtype)
        
        self._delta_translation = vectors.zeros(n=n, dtype=dtype)
        self._delta_rotation = vectors.zeros(n=n, dtype=dtype)
        self._delta_scaling = vectors.ones(n=n, dtype=dtype)
        
        self._reversed = reversed
    
    def reset(self):
        self._translation0 = np.zeros_like(self._translation0)
        self._translation1 = np.zeros_like(self._translation1)
        self._rotation0 = np.zeros_like(self._rotation0)
        self._rotation1 = np.zeros_like(self._rotation1)
        self._scaling0 = np.ones_like(self._scaling0)
        self._scaling1 = np.ones_like(self._scaling1)
    
    def translation_at(self, alpha):
        return vectors.interpolate_linear(self._translation0, self._translation1)(alpha)
    
    def rotation_at(self, alpha):
        return vectors.interpolate_linear(self._rotation0, self._rotation1)(alpha)
    
    def scaling_at(self, alpha):
        return vectors.interpolate_linear(self._scaling0, self._scaling1)(alpha)
    
    def translate(self, value):
        self._translation1 += value

    def rotate(self, value):
        self._rotation1 += value

    def scale(self, value):
        self._scaling1 *= value
    
    def update(self, delta):
        self.translate(self._delta_translation * delta)
        self.rotate(self._delta_rotation * delta)
        #self.scale(self._delta_scaling ** delta)
    
    def to_matrix(self, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        scale = transforms.scale(self.scaling, n=m, dtype=dtype)
        rotate = transforms.rotate(self.rotation, n=n, m=m, dtype=dtype)
        rotate = transforms.inversed(rotate)
        translate = transforms.translate(self.translation, n=m, dtype=dtype)
        
        if self._reversed:
            return vectors.dot(translate, rotate, scale) #TRS - Order
        else:
            return vectors.dot(scale, rotate, translate) #SRT - Order
    
    def to_matrix_at(self, alpha, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        scale = transforms.scale(self.scaling_at(alpha), n=m, dtype=dtype)
        rotate = transforms.rotate(self.rotation_at(alpha), n=n, m=m, dtype=dtype)
        rotate = transforms.inversed(rotate)
        translate = transforms.translate(self.translation_at(alpha), n=m, dtype=dtype)
        
        if self._reversed:
            return vectors.dot(translate, rotate, scale) #TRS - Order
        else:
            return vectors.dot(scale, rotate, translate) #SRT - Order
    
    def __str__(self):
        return str(self.to_matrix())
