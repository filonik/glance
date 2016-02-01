import numpy as np

from . import defaults, vectors


class Extent(object):
    def __init__(self, lower=None, upper=None, n=defaults.DEFAULT_N):
        super().__init__()
        
        self.lower = vectors.full(np.nan, n=n) if lower is None else vectors.full(np.nan, *lower, n=n)
        self.upper = vectors.full(np.nan, n=n) if upper is None else vectors.full(np.nan, *upper, n=n)
    
    def __and__(self, other):
        return Extent(np.maximum(self.lower, other.lower), np.minimum(self.upper, other.upper))
    
    def __or__(self, other):
        return Extent(np.minimum(self.lower, other.lower), np.maximum(self.upper, other.upper))
    
    def nan_to_inf_full(self):
        result = Extent(self.lower, self.upper)
        result.lower[np.where(np.isnan(result.lower))] = -np.inf
        result.upper[np.where(np.isnan(result.upper))] = +np.inf
        return result
    
    def nan_to_inf_null(self):
        result = Extent(self.lower, self.upper)
        result.lower[np.where(np.isnan(result.lower))] = +np.inf
        result.upper[np.where(np.isnan(result.upper))] = -np.inf
        return result
    
    def inf_to_nan(self):
        result = Extent(self.lower, self.upper)
        result.lower[np.where(np.isinf(result.lower))] = np.nan
        result.upper[np.where(np.isinf(result.upper))] = np.nan
        return result
    
    def __str__(self):
        return "[{},{}]".format(self.lower, self.upper)


Full = Extent(vectors.full(-np.inf, n=defaults.DEFAULT_N), vectors.full(+np.inf, n=defaults.DEFAULT_N))
Null = Extent(vectors.full(+np.inf, n=defaults.DEFAULT_N), vectors.full(-np.inf, n=defaults.DEFAULT_N))


def replace_nans(lhs, rhs):
    result = vectors.vector(lhs, n=len(lhs))
    nans = np.where(np.isnan(result))
    result[nans] = rhs[nans]
    return result


def interpolate_linear(lhs, rhs):
    lower = vectors.interpolate_linear(replace_nans(lhs.lower, rhs.lower), rhs.lower)
    upper = vectors.interpolate_linear(replace_nans(lhs.upper, rhs.upper), rhs.upper)
    #lower = vectors.interpolate_linear(np.nan_to_num(lhs.lower), np.nan_to_num(rhs.lower))
    #upper = vectors.interpolate_linear(np.nan_to_num(lhs.upper), np.nan_to_num(rhs.upper))
    def _interpolate_linear(alpha):
        return Extent(lower(alpha), upper(alpha))
    return _interpolate_linear


def _transform_extent(src_extent, dst_extent):
    src_lower = src_extent.lower
    src_upper = src_extent.upper
    
    dst_lower = dst_extent.lower
    dst_upper = dst_extent.upper
    
    src_delta = src_upper - src_lower
    dst_delta = dst_upper - dst_lower
    
    src_center = (src_lower + src_upper)/2.0
    dst_center = (dst_lower + dst_upper)/2.0
    
    translation = dst_center - src_center
    scaling = (dst_delta / src_delta)
    
    return translation, scaling


def transform_extent(src_extent, dst_extent):
    translation, scaling = _transform_extent(src_extent, dst_extent)
    
    translation[np.where(~np.isfinite(translation))] = 0.0
    scaling[np.where(~np.isfinite(scaling))] = 1.0
    
    return translation, scaling


def transform_extent_min(src_extent, dst_extent):
    translation, scaling = _transform_extent(src_extent, dst_extent)
    
    scaling = vectors.full(np.nanmin(scaling), n=len(scaling))
    
    translation[np.where(~np.isfinite(translation))] = 0.0
    scaling[np.where(~np.isfinite(scaling))] = 1.0
    
    return translation, scaling


def transform_extent_max(src_extent, dst_extent):
    translation, scaling = _transform_extent(src_extent, dst_extent)
    
    scaling = vectors.full(np.nanmax(scaling), n=len(scaling))
    
    translation[np.where(~np.isfinite(translation))] = 0.0
    scaling[np.where(~np.isfinite(scaling))] = 1.0
    
    return translation, scaling
