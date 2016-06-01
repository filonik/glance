import numpy as np

from encore import decorators, generators, iterables, objects

from . import colorconversions


class Color(np.ndarray):
    r, g, b, a = decorators.itemproperties(range(4))
    
    @classmethod
    def from_ordinal(cls, i):
        r = ((i >> 0) & 1)
        g = ((i >> 1) & 1)
        b = ((i >> 2) & 1)
        return cls([r, g, b, 1.0])

    @classmethod
    def from_rgb(cls, r, g, b):
        return cls([r, g, b, 1.0])
    
    @classmethod
    def from_rgba(cls, r, g, b, a):
        return cls([r, g, b, a])
    
    @classmethod
    def from_hex(cls, value):
        return cls(colorconversions.load_hex(value))
    
    @classmethod
    def from_hsv(cls, h, s, v):
        r, g, b = colorconversions.hsv_to_rgb(h, s, v)
        return cls.from_rgb(r, g, b)
    
    @classmethod
    def from_hsva(cls, h, s, v, a):
        r, g, b = colorconversions.hsv_to_rgb(h, s, v)
        return cls.from_rgba(r, g, b, a)
    
    def __new__(cls, other=None):
        result = np.ones(4, dtype=np.float32)
        
        if other is not None:
            iterables.populate(result, other)
        
        return result.view(cls)
    
    def __eq__(self, other):
        return np.all(super(Color, self).__eq__(other))
    
    @property
    def rg(self):
        return self[:2]

    @property
    def rgb(self):
        return self[:3]

    @property
    def rgba(self):
        return self[:4]
    
    @property
    def hsv(self):
        h, s, v = colorconversions.rgb_to_hsv(self.r, self.g, self.b)
        return np.array([h, s, v], dtype=np.float32)
    
    @property
    def hsva(self):
        h, s, v = colorconversions.rgb_to_hsv(self.r, self.g, self.b)
        return np.array([h, s, v, self.a], dtype=np.float32)
    
    def hex(self):
        return colorconversions.save_hex(self)

    def saturate(self, factor):
        h, s, v = self.hsv
        s *= (1.0 + factor)
        return Color.from_hsv(h, s, v)

    def desaturate(self, factor):
        return self.saturate(-factor)

    def lighten(self, factor):
        h, s, v = self.hsv
        v *= (1.0 + factor)
        return Color.from_hsv(h, s, v)
        
    def darken(self, factor):
        return self.lighten(-factor)


def ordinals(repeat=1):
    return iterables.repeated((Color.from_ordinal(i) for i in generators.autoincrement()), repeat)
