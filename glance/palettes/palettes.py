import collections


def min_with_none(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return min(x, y)


def max_with_none(x, y):
    if x is None:
        return y
    if y is None:
        return x
    return max(x, y)


class Palette(object):
    def __init__(self, title=None):
        super().__init__()
        
        self.title = title
    
    def __getitem__(self, key):
        return None

    def __call__(self, alpha):
        return None


class PaletteSet(object):
    def __init__(self, title=None):
        super().__init__()
        
        self.title = title
    
    def __getitem__(self, key):
        return None
    
    def lower(self):
        return None
    
    def upper(self):
        return None
    
    def min(self, lower=0):
        key = max_with_none(self.lower(), lower)
        return self[key]

    def max(self, upper=10):
        key = min_with_none(self.upper(), upper)
        return self[key]


class DiscretePaletteSet(PaletteSet):
    def __init__(self, title=None):
        super().__init__(title=title)
        
        self._items = {}
        
    def __getitem__(self, key):
        key = int(key)
        return self._items[key]

    def __setitem__(self, key, value):
        key = int(key)
        self._items[key] = value
        
    def __delitem__(self, key):
        key = int(key)
        del self._items[key]

    def lower(self):
        return min(self._items.keys())
    
    def upper(self):
        return max(self._items.keys())


def ContinuousPaletteSetTemplate(cls, title=None):
    class ContinuousPaletteSet(PaletteSet):
        def __init__(self, title=title):
            super().__init__(title=title)
            
        def __getitem__(self, key):
            return cls(key)
    
    return ContinuousPaletteSet

