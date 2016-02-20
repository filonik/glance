import collections
import enum

import numpy as np

from ..mathematics import defaults, coordinates, projections, transforms, vectors


class Component(enum.IntEnum):
    Scale = (1 << 0)
    Rotate = (1 << 1)
    Translate = (1 << 2)
    Orientate = (1 << 3)


def spherical(rotation, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
    result = transforms.identity(n=m, dtype=dtype)
    if n==2:
        result[:3,:3] = coordinates.spherical1(rotation, dtype=dtype)
    elif n==3:
        result[:4,:4] = coordinates.spherical2(rotation, dtype=dtype)
    elif n==4:
        result[:5,:5] = coordinates.spherical3(rotation, dtype=dtype)
    elif n==5:
        result[:6,:6] = coordinates.spherical4(rotation, dtype=dtype)
    return result


class Transform(object):
    @property
    def translation(self):
        return self._translation
        
    @translation.setter
    def translation(self, value):
        n = min(len(self._translation), len(value))
        self._translation[:n] = value[:n]
    
    @property
    def rotation(self):
        return self._rotation
        
    @rotation.setter
    def rotation(self, value):
        n = min(len(self._rotation), len(value))
        self._rotation[:n] = value[:n]
    
    @property
    def scaling(self):
        return self._scaling
        
    @scaling.setter
    def scaling(self, value):
        n = min(len(self._scaling), len(value))
        self._scaling[:n] = value[:n]
    
    def __init__(self, translation=vectors.zeros(n=0), rotation=vectors.zeros(n=0), scaling=vectors.ones(n=0), reversed=False, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        super().__init__()
        
        self._reversed = reversed
        
        self._translation = vectors.zeros(*translation, n=m-1, dtype=dtype)
        self._rotation = vectors.zeros(*rotation, n=m-1, dtype=dtype)
        self._scaling = vectors.ones(*scaling, n=m-1, dtype=dtype)
        
        self._delta_translation = vectors.zeros(n=m-1, dtype=dtype)
        self._delta_rotation = vectors.zeros(n=m-1, dtype=dtype)
        self._delta_scaling = vectors.ones(n=m-1, dtype=dtype)
    
    def translate(self, value):
        self._translation += value

    def rotate(self, value):
        self._rotation += value

    def scale(self, value):
        self._scaling *= value
    
    def update(self, delta):
        self.translate(self._delta_translation * delta)
        self.rotate(self._delta_rotation * delta)
        #self.scale(self._delta_scaling ** delta)
    
    def to_matrix(self, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        scale = transforms.scale(self._scaling, n=m, dtype=dtype)
        rotate = spherical(self._rotation, n=n, m=m, dtype=dtype)
        translate = transforms.translate(self._translation, n=m, dtype=dtype)
        
        #np.dot(rotation, transforms.translate_scale(self._translation, self._scaling, n=m, dtype=dtype))
        #np.dot(rotation, transforms.scale_translate(self._scaling, self._translation, n=m, dtype=dtype))
        
        if self._reversed:
            return vectors.dot(translate, rotate, scale) #TRS - Order
        else:
            return vectors.dot(scale, rotate, translate) #SRT - Order

    def __str__(self):
        return str(self.to_matrix())
