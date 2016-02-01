import collections
import enum

import numpy as np

from ..mathematics import defaults, coordinates, projections, transforms, vectors


class Projection(object):
    def __init__(self, fov=np.pi/2.0, near=0.1, far=1000.0, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        super().__init__()
        
        self.perspective = False
        
        self.fov = fov
        self.near = near
        self.far = far
        
        self.lower = -vectors.ones(n=n, dtype=dtype)
        self.upper = +vectors.ones(n=n, dtype=dtype)
        
        self.lower[n-1] = near
        self.upper[n-1] = far
    
    def to_matrix(self, n=defaults.DEFAULT_N, m=defaults.DEFAULT_M, dtype=defaults.DEFAULT_DTYPE):
        if self.perspective:
            return projections.general_perspective(self.fov, self.near, self.far, n=n, m=m, dtype=dtype)
        else:
            return projections.general_orthogonal(self.lower, self.upper, n=n, m=m, dtype=dtype)
