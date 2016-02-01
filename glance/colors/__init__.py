from .colors import *


def _setmoduleattr(key, name, value):
    import sys
    module = sys.modules[key]
    setattr(module, name, value)


def _load_colors(path):
    import codecs
    
    from encore import objects, mappings, resources
    
    reader = codecs.getreader("utf-8")
    with resources.open_resource(__name__, path) as file:
        data = objects.load_file(reader(file), codec=objects.Codec(objects.Object, items=objects.Codec(decode=Color.from_hex, encode=Color.hex)))
    
    return mappings.AttrMap(mappings.LowerCaseMap(data))


def _try_import_colors(name, path):
    try:
        _setmoduleattr(__name__, name, _load_colors(path))
    except FileNotFoundError:
        pass


_try_import_colors("crayola", "crayola.json",)
_try_import_colors("svg", "svg.json")
_try_import_colors("x11", "x11.json")
