import os

import logging
import ctypes

import numpy as np

from glue import gl
from glue.gl import GL
from glue.gl import Unspecified, specified, getspecified

from encore import decorators, iterables

from ..graphics import resources, buffers

logger = logging.getLogger(__name__)


class SimpleGeometry(resources.Resource):
    def __init__(self):
        super(SimpleGeometry, self).__init__()

        self._vao = buffers.VertexArray()

        self._vbo = None
        self._ibo = None

    def prepare(self, renderer):
        result = True

        if self._vao is not None:
            result = result and self._vao.prepare(renderer)

        if self._vbo is not None:
            result = result and self._vbo.prepare(renderer)

        if self._ibo is not None:
            result = result and self._vbo.prepare(renderer)

        return result and super(SimpleGeometry, self).prepare(renderer)

    def draw_arrays(self, mode=Unspecified, first=Unspecified, count=Unspecified, patch_vertices=Unspecified, renderer=Unspecified):
        mode = getspecified(mode, self._mode)
        first = getspecified(first, getspecified(self._first, 0))
        count = getspecified(count, getspecified(self._count, 0))
        patch_vertices = getspecified(patch_vertices, self._patch_vertices)

        if mode == GL.GL_PATCHES:
            GL.glPatchParameteri(GL.GL_PATCH_VERTICES, patch_vertices)

        GL.glDrawArrays(mode, first, count)

    def draw_indices(self, mode=Unspecified, first=Unspecified, count=Unspecified, patch_vertices=Unspecified, renderer=Unspecified):
        renderer.activate(self._ibo)

        _type = self._ibo.type
        _count = self._ibo.count

        mode = getspecified(mode, self._mode)
        first = getspecified(first, getspecified(self._first, 0))
        count = getspecified(count, getspecified(self._count, _count))
        patch_vertices = getspecified(patch_vertices, self._patch_vertices)

        if mode == GL.GL_PATCHES:
            GL.glPatchParameteri(GL.GL_PATCH_VERTICES, patch_vertices)

        GL.glDrawElements(mode, count, _type.dtype.type, ctypes.c_void_p(first * _type.dtype.nbytes))

    def draw_indices_instanced(self, instances, mode=Unspecified, first=Unspecified, count=Unspecified, patch_vertices=Unspecified, renderer=Unspecified):
        renderer.activate(self._ibo)

        _type = self._ibo.type
        _count = self._ibo.count

        mode = getspecified(mode, self._mode)
        first = getspecified(first, getspecified(self._first, 0))
        count = getspecified(count, getspecified(self._count, _count))
        patch_vertices = getspecified(patch_vertices, self._patch_vertices)

        if mode == GL.GL_PATCHES:
            GL.glPatchParameteri(GL.GL_PATCH_VERTICES, patch_vertices)

        GL.glDrawElementsInstanced(mode, count, _type.dtype.type, ctypes.c_void_p(first * _type.dtype.nbytes), instances)


class CustomGeometry(resources.Resource):
    def __init__(self, vertices=None, indices=None, *args, **kwargs):
        super(CustomGeometry, self).__init__(*args, **kwargs)

        self._vao = buffers.VertexArray()

        self._vbos = {} if vertices is None else vertices
        self._ibos = {} if indices is None else indices

        self._mode = Unspecified
        self._patch_vertices = Unspecified

        self._first = Unspecified
        self._count = Unspecified
    
    @decorators.indexedproperty
    def vertex_buffers(self, key):
        return self._vbos[key]
    
    @vertex_buffers.setter
    def vertex_buffers(self, key, value):
        self._vbos[key] = value
    
    @decorators.indexedproperty
    def index_buffers(self, key):
        return self._ibos[key]
    
    @index_buffers.setter
    def index_buffers(self, key, value):
        self._ibos[key] = value
    
    def prepare(self, renderer):
        result = True

        if self._vao is not None:
            result = result and self._vao.prepare(renderer)

        for name, value in self._vbos.items():
            result = result and value.prepare(renderer)

        for name, value in self._ibos.items():
            result = result and value.prepare(renderer)

        return result

    def activate(self, renderer):
        if self.prepare(renderer):
            renderer.activate(self._vao)

            for name, value in self._vbos.items():
                renderer.inputs[name] = value

    def deactivate(self, renderer):
        renderer.deactivate(self._vao)

    def draw_arrays(self, mode=Unspecified, first=Unspecified, count=Unspecified, patch_vertices=Unspecified, renderer=Unspecified):
        mode = getspecified(mode, self._mode)
        first = getspecified(first, getspecified(self._first, 0))
        count = getspecified(count, getspecified(self._count, 0))
        patch_vertices = getspecified(patch_vertices, self._patch_vertices)

        if mode == GL.GL_PATCHES:
            GL.glPatchParameteri(GL.GL_PATCH_VERTICES, patch_vertices)

        GL.glDrawArrays(mode, first, count)

    def draw_indices(self, name, mode=Unspecified, first=Unspecified, count=Unspecified, patch_vertices=Unspecified, renderer=Unspecified):
        ibo = self._ibos[name]

        renderer.activate(ibo)

        _type = ibo.type
        _count = ibo.count

        mode = getspecified(mode, self._mode)
        first = getspecified(first, getspecified(self._first, 0))
        count = getspecified(count, getspecified(self._count, _count))
        patch_vertices = getspecified(patch_vertices, self._patch_vertices)

        if mode == GL.GL_PATCHES:
            GL.glPatchParameteri(GL.GL_PATCH_VERTICES, patch_vertices)

        GL.glDrawElements(mode, count, _type.dtype.type, ctypes.c_void_p(first * _type.dtype.nbytes))

    def draw_indices_instanced(self, name, instances, mode=Unspecified, first=Unspecified, count=Unspecified, patch_vertices=Unspecified, renderer=Unspecified):
        ibo = self._ibos[name]

        renderer.activate(ibo)

        _type = ibo.type
        _count = ibo.count
        
        mode = getspecified(mode, self._mode)
        first = getspecified(first, getspecified(self._first, 0))
        count = getspecified(count, getspecified(self._count, _count))
        patch_vertices = getspecified(patch_vertices, self._patch_vertices)

        if mode == GL.GL_PATCHES:
            GL.glPatchParameteri(GL.GL_PATCH_VERTICES, patch_vertices)

        GL.glDrawElementsInstanced(mode, count, _type.dtype.type, ctypes.c_void_p(first * _type.dtype.nbytes), instances)
