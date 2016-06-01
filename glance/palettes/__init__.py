from .palettes import *

import codecs

from encore import accessors, objects, mappings, resources


color_palette_libraries = objects.Object(items=collections.OrderedDict())
shape_palette_libraries = objects.Object(items=collections.OrderedDict())


def color_palettes(path):
    path = None if path is None else path.split(".")
    return accessors.getitempath(color_palette_libraries, path) if path else None


def shape_palettes(path):
    path = None if path is None else path.split(".")
    return accessors.getitempath(shape_palette_libraries, path) if path else None


def _iter_palettes(palette_libraries):
    for palette_library_key, palette_library in palette_libraries.items():
        for palette_key, palette in palette_library.items():
            yield (".".join([palette_library_key, palette_key]), palette)


def iter_color_palettes():
    return _iter_palettes(color_palette_libraries)


def iter_shape_palettes():
    return _iter_palettes(shape_palette_libraries)
    

def _default_color_palettes():
    from .. import colors
    from . import colorpalettes as cp
    
    return objects.Object(attrs={"title": "Other"}, items=collections.OrderedDict([
        ("black_body", cp.BlackBodyPaletteSet()),
        ("cool_to_warm", cp.CoolToWarmPaletteSet()),
        ("rainbow", cp.RainbowPaletteSet()),
    ]))


def _load_color_palettes(path):    
    from .. import colors
    from . import colorpalettes as cp

    palette_library_schema = objects.Codec(objects.Object, items=objects.Codec(
        objects.Object, decode=objects.simple_decode(palettes.DiscretePaletteSet), items=objects.Codec(
            objects.Object, decode=objects.simple_decode(cp.DiscretePalette), items=objects.Codec(decode=colors.Color.from_hex) #objects.simple_decode(cp.DiscretePalette) #
        )
    ))
    
    reader = codecs.getreader("utf-8")
    with resources.open_resource(__name__, path) as file:
        return objects.load_file(reader(file), codec=palette_library_schema, object_pairs_hook=collections.OrderedDict)


def _load_shape_palettes(path):
    from .. import shapes
    from . import shapepalettes as sp
    
    palette_library_schema = objects.Codec(objects.Object, items=objects.Codec(
        objects.Object, decode=objects.simple_decode(palettes.DiscretePaletteSet), items=objects.Codec(
            objects.Object, decode=objects.simple_decode(sp.DiscretePalette)
        )
    ))
    
    reader = codecs.getreader("unicode_escape") # codecs.getreader("utf-8")
    with resources.open_resource(__name__, path) as file:
        return objects.load_file(reader(file), codec=palette_library_schema, object_pairs_hook=collections.OrderedDict)


def _try_import_color_palettes(name, path):
    try:
        color_palette_libraries[name] = _load_color_palettes(path)
    except FileNotFoundError:
        pass


def _try_import_shape_palettes(name, path):
    try:
        shape_palette_libraries[name] = _load_shape_palettes(path)
    except FileNotFoundError:
        pass


_try_import_color_palettes("brewer", "brewer.json",)
_try_import_color_palettes("various", "various.json",)
_try_import_shape_palettes("geometric", "geometric.json")
_try_import_shape_palettes("symbolic", "symbolic.json")

color_palette_libraries["other"] = _default_color_palettes()
