import numpy as np

from .. import painters


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
