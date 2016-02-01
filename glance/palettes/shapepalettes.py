import collections

from .. import colors

from . import palettes


class ShapePalette(palettes.Palette):
    def __getitem__(self, key):
        return "x"


class DiscretePalette(ShapePalette, collections.MutableSequence):
    def __init__(self, items=None):
        super().__init__()
        
        self._items = [] if items is None else items
    
    def __getitem__(self, key):
        key = int(key)
        return self._items[key]

    def __setitem__(self, key, value):
        key = int(key)
        self._items[key] = value
        
    def __delitem__(self, key):
        key = int(key)
        del self._items[key]
    
    def insert(self, key, value):
        return self._items.insert(key, value)
    
    def __iter__(self):
        return iter(self._items)
    
    def __len__(self):
        return len(self._items)
