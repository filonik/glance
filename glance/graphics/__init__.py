from .resources import Resource
from .buffers import VertexArray, VertexBuffer, IndexBuffer
from .shaders import Shader, Program
from .textures import Texture

from glue import gl
from glue.gl import GL

import logging
logger = logging.getLogger(__name__)


class RenderbufferData(object):
    def __init__(self, storage, size):
        self.storage = storage
        self.size = size

    def data(self):
        return self


class Renderbuffer(Resource):
    @classmethod
    def fromdata(cls, *args, **kwargs):
        return cls(RenderbufferData(*args, **kwargs))

    def __init__(self, provider):
        super(Renderbuffer, self).__init__()

        self._provider = provider

    def update(self, renderer):
        try:
            data = self._provider.data()

            value = gl.Renderbuffer(context=renderer.context)

            gl.Renderbuffer.bind(value)
            gl.Renderbuffer.set_storage(data.storage, data.size)
            gl.Renderbuffer.bind(None)

        except Exception as e:
            logger.warn("Create Renderbuffer: %s", e)
        else:
            self._value = value

        return super(Renderbuffer, self).update()

    def activate(self, renderer):
        if self.prepare(renderer=renderer):
            self._value.bind(self._value)

    def deactivate(self, renderer):
        self._value.bind(None)


class FramebufferData(object):
    def __init__(self, attachments):
        self.attachments = attachments

    def data(self):
        return self


class Framebuffer(Resource):
    @classmethod
    def fromdata(cls, *args, **kwargs):
        return cls(FramebufferData(*args, **kwargs))

    def __init__(self, provider):
        super(Framebuffer, self).__init__()

        self._provider = provider

    def update(self, renderer):
        from encore import predicates
        try:
            data = self._provider.data()

            self._attachments = data.attachments

            color_attachments = predicates.between(GL.GL_COLOR_ATTACHMENT0, GL.GL_COLOR_ATTACHMENT15 + 1)
            self._draw_buffers = list(filter(color_attachments, self._attachments.keys()))

            for attachment_point, attachment in self._attachments.items():
                attachment.prepare(renderer=renderer)

            value = gl.Framebuffer(context=renderer.context)

            gl.Framebuffer.bind(value)

            for attachment_point, attachment in self._attachments.items():
                value.attach(attachment_point, attachment._value)

            if gl.Framebuffer.status != gl.GL.GL_FRAMEBUFFER_COMPLETE:
                raise Exception("Framebuffer Error.")

            gl.Framebuffer.bind(None)

        except Exception as e:
            logger.warn("Update Framebuffer: %s", e)
        else:
            self._value = value

        return super(Framebuffer, self).update()

    def activate(self, renderer):
        if self.prepare(renderer=renderer):
            self._value.bind(self._value)

            GL.glDrawBuffers(self._draw_buffers)

    def deactivate(self, renderer):
        self._value.bind(None)