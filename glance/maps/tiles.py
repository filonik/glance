import numpy as np

from .. import mathematics, painters


def tile_limits(zoom):
    l, u = 0, 2**zoom
    return (l,l), (u,u)


def tile_range(lower=None, upper=None, zoom=0):
    l, u = 0, 2**zoom
    lower = (l,l) if lower is None else lower
    upper = (u,u) if upper is None else upper
    for y in range(max(l, lower[0]), min(u, upper[0])):
        for x in range(max(l, lower[1]), min(u, upper[1])):
            yield (x, y)


norm2rad = mathematics.positive_to_range((-np.pi, +np.pi))
rad2norm = mathematics.range_to_positive((-np.pi, +np.pi))

deg2rad = mathematics.range_to_range((-180.0, +180.0), (-np.pi, +np.pi))
rad2deg = mathematics.range_to_range((-np.pi, +np.pi), (-180.0, +180.0))


def norm2num(zoom):
    return mathematics.positive_to_range((0, 2**zoom))


def num2norm(zoom):
    return mathematics.range_to_positive((0, 2**zoom))


def _project(y):
    return 2.0*np.arctan(np.exp(y)) - np.pi/2.0


def _unproject(y):
    return np.log(np.tan((y + np.pi/2.0)/2.0))


def project(lon, lat):
    x = rad2norm(lon)
    y = (np.log(np.tan(lat) + 1 / np.cos(lat)))/np.pi
    return x, y


def unproject(x, y):
    lon = norm2rad(x)
    lat = np.arctan(np.sinh(y*np.pi))
    return lon, lat


def rad2num(lon, lat, zoom):
    x, y = project(lon, lat)
    y = (1 - y)/2.0
    n2n = norm2num(zoom)
    x, y = n2n(x), n2n(y)
    return (int(np.floor(x)), int(np.floor(y)))


def num2rad(x, y, zoom):
    n2n = num2norm(zoom)
    x, y = n2n(x), n2n(y)
    y = (1 - 2.0*y)
    lon, lat = unproject(x, y)
    return (lon, lat)


def deg2num(lon, lat, zoom):
    lon, lat = deg2rad(lon), deg2rad(lat)
    return rad2num(lon, lat, zoom)


def num2deg(x, y, zoom):
    lon, lat = num2rad(x, y, zoom)
    return rad2deg(lon), rad2deg(lat)


class TileData(painters.Canvas):
    def __init__(self, path, shape, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._cache = painters.ImageCache(factory=painters.load_image)
        self._path = path
        self._shape = shape
    
    def paint(self, x, y, z):
        paths = [self._path.format(x=x+i, y=y+j, z=z) for i, j in np.ndindex(*self._shape)]
        painter = painters.atlas(self._shape, [painters.image(path, cache=self._cache) for path in paths], flip=None)
        
        super().paint(painter)
