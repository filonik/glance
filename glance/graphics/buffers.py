import os

import contextlib as cl

import ctypes

import numpy as np

from glue import gl
from glue.gl import GL

from .resources import Observable, Resource, Status

import logging
logger = logging.getLogger(__name__)


class BufferData(Observable):
    @staticmethod
    def offsetof(data):
        return 0

    @staticmethod
    def sizeof(data):
        return 0 if data is None else data.nbytes

    @staticmethod
    def typeof(data):
        return None if data is None else gl.types.gltype(data)

    @property
    def offset(self):
        return BufferData.offsetof(self._data) if self._offset is None else self._offset

    @offset.setter
    def offset(self, value):
        self._offset = value
        self.notify()

    @property
    def size(self):
        return BufferData.sizeof(self._data) if self._size is None else self._size

    @size.setter
    def size(self, value):
        self._size = value
        self.notify()

    @property
    def type(self):
        return BufferData.typeof(self._data) if self._type is None else self._type

    @type.setter
    def type(self, value):
        self._type = value
        self.notify()

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if self._data is not value:
            self._data = value
            self.notify()

    def __init__(self, data=None, *args, **kwargs):
        super(BufferData, self).__init__()

        self._data = data
        self._offset = kwargs.get('offset', None)
        self._size = kwargs.get('size', None)
        self._type = kwargs.get('type', None)

    def provide(self):
        return self

    def __str__(self):
        return "%s{%s}" % (type(self).__name__, self.data)

class Buffer(Resource):
    @classmethod
    def fromdata(cls, *args, **kwargs):
        return cls(BufferData(*args, **kwargs))

    @property
    def capacity(self):
        return self._capacity

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
    def count(self):
        if self._type is None:
            return 0
        return self._size//self._type.dtype.nbytes

    def __init__(self, provider, usage=gl.DEFAULT_BUFFER_USAGE, divisor=None):
        super(Buffer, self).__init__()

        self._provider = provider
        self._provider.subscribe(lambda observable: self.request_update())

        self._capacity = 0
        self._offset = 0
        self._size = 0
        self._type = None

        self._usage = usage
        self._divisor = divisor

    def create(self, renderer):
        #logger.info("Create %s", type(self).__name__)
        return super(Buffer, self).create(renderer)

    def update(self, renderer):
        #logger.info("Update %s", type(self).__name__)
        return super(Buffer, self).update(renderer)

    def delete(self, renderer):
        self._capacity = 0
        self._offset = 0
        self._size = 0
        self._type = None

        return super(Buffer, self).delete(renderer)

    def activate(self, renderer):
        if self.prepare(renderer=renderer):
            self._value.bind(self._value)

    def deactivate(self, renderer):
        self._value.bind(None)

    @cl.contextmanager
    def mapped(self, access=GL.GL_READ_ONLY):
        result = self._value.map(access)

        yield result

        self._value.unmap()

    def debug_dump(self, renderer):
        renderer.activate(self)
        
        type = ctypes.c_float
        #type = ctypes.c_int
        with self.mapped(GL.GL_READ_ONLY) as ptr:
            arr = np.ctypeslib.as_array(ctypes.cast(ptr, ctypes.POINTER(type)), (self.size//ctypes.sizeof(type),))
            
            logger.debug(arr)


class VertexBuffer(Buffer):
    def create(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Create VertexBuffer: %s", e)
        else:
            value = gl.VertexBufferObject(context=renderer.context)
            value._divisor = self._divisor

            self._value = value

        return super(VertexBuffer, self).create(renderer)

    def update(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Update VertexBuffer: %s", e)
        else:
            value = self._value
            value._divisor = self._divisor

            capacity = data.offset + data.size

            value.bind(value)

            if self._capacity < capacity:
                value.set_data(None, size=capacity, usage=self._usage)

            value.set_sub_data(data.data, size=data.size, offset=data.offset)

            value.bind(None)

            self._capacity = capacity
            self._offset = data.offset
            self._size = data.size
            self._type = data.type

            self._value = value

        return super(VertexBuffer, self).update(renderer)


class IndexBuffer(Buffer):
    def create(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Create IndexBuffer: %s", e)
        else:
            value = gl.IndexBufferObject(context=renderer.context)

            self._value = value

        return super(IndexBuffer, self).create(renderer)

    def update(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Update IndexBuffer: %s", e)
        else:
            value = self._value

            capacity = data.offset + data.size

            value.bind(value)

            if self._capacity < capacity:
                value.set_data(None, size=capacity, usage=self._usage)

            value.set_sub_data(data.data, size=data.size, offset=data.offset)

            value.bind(None)

            self._capacity = capacity
            self._offset = data.offset
            self._size = data.size
            self._type = data.type

            self._value = value

        return super(IndexBuffer, self).update(renderer)


class ShaderStorageBuffer(Buffer):
    def create(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Create ShaderStorageBuffer: %s", e)
        else:
            value = gl.ShaderStorageBuffer(context=renderer.context)
            value._divisor = self._divisor
            
            self._value = value

        return super(ShaderStorageBuffer, self).create(renderer)

    def update(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Update ShaderStorageBuffer: %s", e)
        else:
            value = self._value
            value._divisor = self._divisor
            
            capacity = data.offset + data.size

            value.bind(value)

            if self._capacity < capacity:
                logger.info("Resize ShaderStorageBuffer.")
                value.set_data(None, size=capacity, usage=self._usage)
                self._capacity = capacity

            value.set_sub_data(data.data, size=data.size, offset=data.offset)
            # This is a hack! Fix in glue.
            value._size = data.size
            value._type = data.type

            value.bind(None)

            self._offset = data.offset
            self._size = data.size
            self._type = data.type

            self._value = value

        return super(ShaderStorageBuffer, self).update(renderer)




class UniformBuffer(Buffer):
    def create(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Create UniformBuffer: %s", e)
        else:
            value = gl.UniformBuffer(context=renderer.context)

            self._value = value

        return super(UniformBuffer, self).create(renderer)

    def update(self, renderer):
        try:
            data = self._provider.provide()
        except Exception as e:
            logger.warn("Update UniformBuffer: %s", e)
        else:
            value = self._value

            capacity = data.offset + data.size

            value.bind(value)

            if self._capacity < capacity:
                logger.info("Resize UniformBuffer.")
                value.set_data(None, size=capacity, usage=self._usage)
                self._capacity = capacity

            value.set_sub_data(data.data, size=data.size, offset=data.offset)
            # This is a hack! Fix in glue.
            value._size = data.size
            value._type = data.type

            value.bind(None)

            self._offset = data.offset
            self._size = data.size
            self._type = data.type

            self._value = value

        return super(UniformBuffer, self).update(renderer)


class VertexArray(Resource):
    def create(self, renderer):
        self._value = gl.VertexArrayObject(context=renderer.context)
        return super(VertexArray, self).create(renderer)

    def activate(self, renderer):
        if self.prepare(renderer=renderer):
            self._value.bind(self._value)

    def deactivate(self, renderer):
        self._value.bind(None)
