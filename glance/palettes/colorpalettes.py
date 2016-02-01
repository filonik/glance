import collections

from .. import colors

from . import palettes


class ColorPalette(palettes.Palette):
    def __getitem__(self, key):
        return colors.Color([0.0, 0.0, 0.0, 1.0])

    def __call__(self, alpha):
        return colors.Color([0.0, 0.0, 0.0, 1.0])


class ContinuousPalette(ColorPalette):
    """ Implements __getitem__ in terms of __call__. """
    
    def __init__(self, count=10):
        super().__init__()
        
        self._count = count
    
    def __getitem__(self, key):
        lower, upper = 0, self._count - 1
        
        if key < lower or upper < key:
            raise KeyError(key)
        
        alpha = (key - lower)/(upper - lower)
        
        return self(alpha)
        
    def __call__(self, alpha):
        return super().__call__(alpha)

    def __iter__(self):
        for key in range(len(self)):
            yield self[key]
    
    def __len__(self):
        return self._count


class DiscretePalette(ColorPalette, collections.MutableSequence):
    """ Implements __call__ in terms of __getitem__. """
    
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
    
    def __call__(self, alpha):
        l = len(self) - 1
        x = l * alpha
        i = int(x)
        a = x - i
        i = max(0, min(i+0, l))
        j = max(0, min(i+1, l))
        return (1.0 - a) * self[i] + a * self[j]
    
    def __iter__(self):
        return iter(self._items)
    
    def __len__(self):
        return len(self._items)


class GradientPalette(ContinuousPalette):
    def __init__(self, count=10, items=None):
        super().__init__(count=count)
        
        self._items = [] if items is None else items
    
    def __call__(self, key):
        lower, upper = self._items[0], self._items[0]
        
        for stop in self._items:
            if stop[0] <= key:
                lower = stop
            if stop[0] > key:
                upper = stop
                break
        
        alpha = (key-lower[0])/(upper[0]-lower[0])
        return lower[1] * (1.0 - alpha) + upper[1] * (alpha)


# Color Maps
# http://www.paraview.org/ParaView/index.php/Default_Color_Map

class BlackBodyPalette(GradientPalette):
    def __init__(self, count=10):
        super().__init__(count=count, items=[
            (0.00, colors.Color([0, 0, 0])),
            (0.40, colors.Color([1, 0, 0])),
            (0.75, colors.Color([1, 1, 0])),
            (1.00, colors.Color([1, 1, 1])),
        ])


class CoolToWarmPalette(GradientPalette):
    def __init__(self, count=10):
        super().__init__(count=count, items=[
            (0.00, colors.Color([0, 0, 1])),
            (0.50, colors.Color([1, 1, 1])),
            (1.00, colors.Color([1, 0, 0])),
        ])


class RainbowPalette(ContinuousPalette):
    def __call__(self, key):
        return colors.Color.from_hsv((1.0-key)*(2/3), 1.0, 1.0)


BlackBodyPaletteSet = palettes.ContinuousPaletteSetTemplate(BlackBodyPalette, title="Black Body")
CoolToWarmPaletteSet = palettes.ContinuousPaletteSetTemplate(CoolToWarmPalette, title="Cool to Warm")
RainbowPaletteSet = palettes.ContinuousPaletteSetTemplate(RainbowPalette, title="Rainbow")


#meta = Meta(PaletteLibrary, items=Meta(PaletteSet, items=Meta(DiscretePalette)))