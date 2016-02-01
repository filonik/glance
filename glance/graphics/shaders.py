import os

from glue import gl
from glue.gl import GL

from encore.resources import application_path

from .resources import Observable, Resource, Status

import logging
logger = logging.getLogger(__name__)


_GENERIC_SUFFIX = '.glsl'

_SUFFIX_MAP = {
    ".vs": GL.GL_VERTEX_SHADER, ".vert": GL.GL_VERTEX_SHADER,
    ".tcs": GL.GL_TESS_CONTROL_SHADER, ".tc": GL.GL_TESS_CONTROL_SHADER, ".tesc": GL.GL_TESS_CONTROL_SHADER,
    ".tes": GL.GL_TESS_EVALUATION_SHADER, ".te": GL.GL_TESS_EVALUATION_SHADER, ".tese": GL.GL_TESS_EVALUATION_SHADER,
    ".gs": GL.GL_GEOMETRY_SHADER, ".geom": GL.GL_GEOMETRY_SHADER,
    ".fs": GL.GL_FRAGMENT_SHADER, ".frag": GL.GL_FRAGMENT_SHADER,
    ".cs": GL.GL_COMPUTE_SHADER, ".comp": GL.GL_COMPUTE_SHADER,
}


class ShaderData(Observable):
    @property
    def type(self):
        return self._type

    @type.setter
    def type(self, value):
        if self._type != value:
            self._type = value
            self.notify()

    @property
    def source(self):
        return self._source

    @source.setter
    def source(self, value):
        if self._source != value:
            self._source = value
            self.notify()

    def __init__(self, type, source):
        super(ShaderData, self).__init__()

        self._type = type
        self._source = source

    def provide(self):
        return self


class ShaderFile(Observable):
    _loaders = []

    @classmethod
    def register(cls, loader):
        cls._loaders.insert(0, loader)

    def __init__(self, path, *args, **kwargs):
        super(ShaderFile, self).__init__()

        self.path = path
        self.args = args
        self.kwargs = kwargs

    def provide(self):
        for loader in self._loaders:
            if loader.accepts(self.path):
                return loader.load(self.path, *self.args, **self.kwargs)
        return None

    def __str__(self):
        return self.path


class ShaderFileLoader(object):
    _extension_to_type = _SUFFIX_MAP

    def gettype(self, path, *args, **kwargs):
        _, extension = os.path.splitext(path)
        return self._extension_to_type[extension]

    def getsource(self, path, *args, **kwargs):
        with open(path, "r") as f:
            return f.read()

    def accepts(self, path):
        _, extension = os.path.splitext(path)
        return extension in self._extension_to_type

    def load(self, path, *args, **kwargs):
        return ShaderData(self.gettype(path, *args, **kwargs), self.getsource(path, *args, **kwargs))


ShaderFile.register(ShaderFileLoader())


class Shader(Resource):
    _path_prefix = os.path.join('data', 'shaders')

    @classmethod
    def relative_path(cls, *args):
        return application_path(cls._path_prefix, *args)

    @classmethod
    def from_data(cls, *args, **kwargs):
        return cls(ShaderData(*args, **kwargs))

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        return cls(ShaderFile(cls.relative_path(path), *args, **kwargs))

    def __init__(self, provider):
        super(Shader, self).__init__()

        self._provider = provider
        self._provider.subscribe(lambda observable: self.request_update())

    def update(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Update Shader: %s", e)
        else:
            try:
                value = gl.Shader.fromtype(data.type, context=renderer.context)
                value.set_source(data.source)
                value.compile()
            except Exception as e:
                logger.error("Compile %s: %s", self, e)

            self._value = value

        return super(Shader, self).update()

    def __str__(self):
        return "%s{%s}" % (self.__class__.__name__, str(self._provider),)


class ProgramData(Observable):
    def __init__(self, shaders):
        super(ProgramData, self).__init__()

        self.shaders = shaders

    def provide(self):
        return self

    def __str__(self):
        return ",\n".join(map(str, self.shaders))

class Program(Resource):
    _path_prefix = os.path.join('data')

    @classmethod
    def from_files(cls, paths, *args, **kwargs):
        shaders = [Shader.from_file(path, *args, **kwargs) for path in paths ]
        return cls(ProgramData(shaders))

    def __init__(self, provider):
        super(Program, self).__init__()

        self._provider = provider
        self._provider.subscribe(lambda observable: self.request_update())

        self._shaders = None

    def update(self, renderer):
        try:
            data = self._provider.provide()

            self._shaders = data.shaders

            for shader in self._shaders:
                shader.prepare(renderer=renderer)

        except Exception as e:
            logger.warn("Update Program: %s", e)
        else:
            try:
                value = gl.Program(context=renderer.context)

                for shader in data.shaders:
                    value.attach(shader._value)

                value.link()

                for shader in self._shaders:
                    value.detach(shader._value)
            except Exception as e:
                logger.error("Link %s: %s", self, e)

            self._value = value

        return super(Program, self).update()

    def activate(self, renderer):
        if self.prepare(renderer=renderer):
            self._value.bind(self._value)

    def deactivate(self, renderer):
        self._value.bind(None)

    def __str__(self):
        return "%s{\n%s\n}" % (self.__class__.__name__, str(self._provider),)
