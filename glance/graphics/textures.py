import os

from glue import gl
from glue.gl import GL

from encore.resources import application_path
from encore.accessors import getitem

from .resources import Resource, Status

import logging
logger = logging.getLogger(__name__)


class TextureData(object):
    @property
    def image(self):
        return getitem(self.images, 0)

    @property
    def offset(self):
        return getitem(self.offsets, 0)

    @property
    def size(self):
        return getitem(self.sizes, 0)

    @property
    def type(self):
        return getitem(self.types, 0)

    @property
    def format(self):
        return getitem(self.formats, 0)

    def __init__(self, *args, **kwargs):
        self.target = kwargs.get('target', GL.GL_TEXTURE_2D)

        image = kwargs.get('image', None)
        offset = kwargs.get('offset', None)
        size = kwargs.get('size', None)
        type = kwargs.get('type', gl.Unspecified)
        format = kwargs.get('format', gl.Unspecified)

        count = 6 if self.target == GL.GL_TEXTURE_CUBE_MAP else 1

        self.images = kwargs.get('images', [image]*count)
        self.offsets = kwargs.get('offsets', [offset]*count)
        self.sizes = kwargs.get('sizes', [size]*count)
        self.types = kwargs.get('types', [type]*count)
        self.formats = kwargs.get('formats', [format]*count)

    def provide(self):
        return self


class TextureFile(object):
    _loaders = []

    @classmethod
    def register(cls, loader):
        cls._loaders.insert(0, loader)

    def __init__(self,  paths, *args, **kwargs):
        self.paths = paths
        self.args = args
        self.kwargs = kwargs

    def provide(self):
        for loader in self._loaders:
            if loader.accepts(self.paths):
                return loader.load(self.paths, *self.args, **self.kwargs)
        return None


class TextureFileLoader(object):

    def accepts(self, paths):
        return True

    def getimage(self, path):
        import PIL.Image
        image = PIL.Image.open(path).convert("RGBA")
        return image, image.size, gl.GL.GL_UNSIGNED_BYTE, gl.GL.GL_RGBA

    def load(self, paths, *args, **kwargs):
        images = []
        sizes = []
        types = []
        formats = []

        for path in paths:
            image, size, type, format = self.getimage(path)

            images.append(image)
            sizes.append(size)
            types.append(type)
            formats.append(format)

        return TextureData(images=images, sizes=sizes, types=types, formats=formats, *args, **kwargs)


TextureFile.register(TextureFileLoader())

DEFAULT_TEXTURE_CUBE_MAP_TARGETS = [
    GL.GL_TEXTURE_CUBE_MAP_POSITIVE_X,
    GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_X,
    GL.GL_TEXTURE_CUBE_MAP_POSITIVE_Y,
    GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_Y,
    GL.GL_TEXTURE_CUBE_MAP_POSITIVE_Z,
    GL.GL_TEXTURE_CUBE_MAP_NEGATIVE_Z,
]

DEFAULT_TEXTURE_CUBE_MAP_SUFFIXES = ['ft', 'bk', 'up', 'dn', 'rt', 'lf']

DEFAULT_TEXTURE_FORMAT_TO_INTERNAL_FORMAT = {
    GL.GL_BGR: GL.GL_RGB,
    GL.GL_BGRA: GL.GL_RGBA,
}

def format_to_internal_format(format):
    return DEFAULT_TEXTURE_FORMAT_TO_INTERNAL_FORMAT.get(format, format)

def environment_paths(name):
    return [name.format(side=side) for side in DEFAULT_TEXTURE_CUBE_MAP_SUFFIXES]


class Texture(Resource):
    _path_prefix = os.path.join('data', 'images')

    @classmethod
    def relative_path(cls, *args):
        return application_path(cls._path_prefix, *args)

    @classmethod
    def from_data(cls, *args, **kwargs):
        return cls(TextureData(*args, **kwargs))

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        return cls(TextureFile([cls.relative_path(path)], *args, **kwargs))

    @classmethod
    def from_files(cls, paths, *args, **kwargs):
        return cls(TextureFile([cls.relative_path(path) for path in paths], *args, **kwargs))

    @property
    def target(self):
        return self._target

    @property
    def offset(self):
        return self._offset

    @property
    def size(self):
        return self._size

    @property
    def type(self):
        return self._type

    @property
    def unit(self):
        return self._unit

    def __init__(self, provider, filter=gl.DEFAULT_TEXTURE_FILTER, wrap=gl.DEFAULT_TEXTURE_WRAP):
        super(Texture, self).__init__()

        self._provider = provider

        self._target = None
        self._offset = None
        self._size = None
        self._type = None
        self._format = None
        
        self._filter = filter
        self._wrap = wrap

        self._unit = 0

    def create(self, renderer):
        try:
            data = self._provider.provide()

            value = gl.Texture.fromtype(data.target, filter=self._filter, wrap=self._wrap, context=renderer.context)
            value.bind(value)
            
            if data.target == GL.GL_TEXTURE_CUBE_MAP:
                for i, target in enumerate(DEFAULT_TEXTURE_CUBE_MAP_TARGETS):
                    internalFormat = format_to_internal_format(data.formats[i])
                    value.set_image(data.images[i], size=data.sizes[i], type=data.types[i], internalFormat=internalFormat, format=data.formats[i], target=target)
            else:
                internalFormat = format_to_internal_format(data.format)
                value.set_image(data.image, size=data.size, type=data.type, internalFormat=internalFormat, format=data.format)

            value.bind(None)

        except Exception as e:
            logger.warn("Create Texture: %s", e)
        else:
            self._target = data.target
            self._offset = data.offset
            self._size = data.size
            self._type = data.type
            self._format = data.format

            self._value = value

        return super(Texture, self).update()

    def update(self, renderer):
        try:
            data = self._provider.provide()

            value = self._value
            value.bind(value)

            if data.target == GL.GL_TEXTURE_CUBE_MAP:
                for i, target in enumerate(DEFAULT_TEXTURE_CUBE_MAP_TARGETS):
                    value.set_sub_image(target, data.images[i], size=data.sizes[i], type=data.types[i], format=data.formats[i], target=target)
            else:
                value.set_sub_image(data.image, size=data.size, type=data.type, format=data.format)

            value.bind(None)

        except Exception as e:
            logger.warn("Update Texture: %s", e)
        else:
            self._target = data.target
            self._offset = data.offset
            self._size = data.size
            self._type = data.type
            self._format = data.format

        return super(Texture, self).update()

    def activate(self, renderer):
        if self.prepare(renderer=renderer):
            self._unit = renderer.available_texture_unit()
            gl.GL.glActiveTexture(gl.GL.GL_TEXTURE0 + self._unit)

            self._value.bind(self._value)

    def deactivate(self, renderer):
        gl.GL.glActiveTexture(gl.GL.GL_TEXTURE0 + self._unit)

        self._value.bind(None)
