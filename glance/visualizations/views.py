import collections
import copy
import enum
import itertools as it
import logging

import numpy as np
import pandas as pd

from glue import gl
from glue.gl import GL

from encore import accessors, coercions, containers, generators, iterables, mappings, objects, observables, predicates, utilities

import medley as md
from medley import expressions

from .. import config

from ..mathematics import defaults as mdefaults, coordinates, extents, mathematics, projections, transforms, vectors
from ..mathematics.geometries import cubes
from ..graphics import defaults as gdefaults, buffers, primitives, renderers, shaders, textures, vertices
from ..scenes import animations, inputs, lights, materials, geometries, scenes

from .. import colors, shapes, palettes
from .. import painters

from . import actors, models

logger = logging.getLogger(__name__.split(".").pop())

_identifier_size = 32

_identifier_type = utilities.Bits(4, _identifier_size - 4)
_identifier_data = utilities.Bits(_identifier_size - 4, 0)

_identifier_mark_layer = utilities.Bits(4, _identifier_data.size - 4)
_identifier_mark_index = utilities.Bits(_identifier_data.size - 4, 0)

ORIGIN_TYPE = 1 << 0
AXIS_TYPE = 1 << 1
MARK_TYPE = 1 << 2
GUIDE_TYPE = 1 << 3


def identifier(type, data=0):
    return _identifier_type.encode(type) | _identifier_data.encode(data)

def origin_identifier():
    return identifier(ORIGIN_TYPE)

def axis_identifier(index):
    return identifier(AXIS_TYPE, index)

def mark_identifier(layer, index):
    return identifier(MARK_TYPE, _identifier_mark_layer.encode(layer) | _identifier_mark_index.encode(index))


GLANCE_LIGHT_COUNT = 3

DEFAULT_VERTEX_FORMAT = vertices.DEFAULT_FORMAT | vertices.Component.Transform
DEFAULT_VERTEX_ND_FORMAT = vertices.FORMAT_SIZE_OFFSET | vertices.Component.Identifier

DEFAULT_TESS_LEVEL = 256//int(2**mdefaults.DEFAULT_N)
#print("DEFAULT_TESS_LEVEL", DEFAULT_TESS_LEVEL)

DEFAULT_PROGRAM_DEFINES = mappings.DefineMap({
    "GLANCE_COLOR": 2,
    "GLANCE_LIGHT_COUNT": GLANCE_LIGHT_COUNT,
    "GLANCE_MATERIAL_COUNT": 2,
    "GLANCE_MATERIAL_SIDES": 1,
    "GLANCE_MATERIAL_FORMAT": materials.DEFAULT_FORMAT,
    "GLANCE_MATERIAL_COLOR_AMBIENT": 1,
    "GLANCE_MATERIAL_COLOR_DIFFUSE": 1,
    "GLANCE_TESS_LEVEL_INNER": DEFAULT_TESS_LEVEL,
    "GLANCE_TESS_LEVEL_OUTER": DEFAULT_TESS_LEVEL,
    "GLANCE_VERTEX_FORMAT": DEFAULT_VERTEX_FORMAT,
    "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
    "M_N": mdefaults.DEFAULT_N,
})


def unpack_args(func):
    def _unpack_args(key):
        return func(*coercions.coerce_tuple(key))
    return _unpack_args


def simplex0_nd_program(material_format=materials.DEFAULT_FORMAT, vertex_format=DEFAULT_VERTEX_ND_FORMAT):
    return gdefaults.thick_simplex_nd_program(material_format, vertex_format, 0, space=renderers.Space.View)


def simplex1_nd_program(material_format=materials.DEFAULT_FORMAT, vertex_format=DEFAULT_VERTEX_ND_FORMAT):
    return gdefaults.thick_simplex_nd_program(material_format, vertex_format, 1, space=renderers.Space.View)


def shaded_surface_program(material_format, coordinate_count, shade):
    return shaders.Program.from_files([
        "default_warp_nd.vs",
        "surface_nd.gs",
        "shade_phong.fs",
        #"default.fs",
    ], defines={
        "GLANCE_COLOR": 1,
        "GLANCE_COORDINATE_COUNT": coordinate_count,
        "GLANCE_LIGHT_COUNT": GLANCE_LIGHT_COUNT,
        "GLANCE_MATERIAL_FORMAT": material_format,
        "GLANCE_MATERIAL_COLOR_AMBIENT": 1,
        "GLANCE_MATERIAL_COLOR_DIFFUSE": 1,
        "GLANCE_MATERIAL_SIDES": 2,
        "GLANCE_SHADE": int(shade),
        "GLANCE_VERTEX_FORMAT": DEFAULT_VERTEX_FORMAT,
        "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
        "M_N": mdefaults.DEFAULT_N,
    })


def axis_program(material_format, coordinate_count):
    return shaders.Program.from_files([
        "default_nd.vs",
        "tessellate_simplex1_nd.tcs",
        "tessellate_simplex1_nd.tes",
        "thick_simplex1_warp_nd.gs",
        "default.fs",
    ], defines={
        "GLANCE_COLOR": 1,
        "GLANCE_COORDINATE_COUNT": coordinate_count,
        "GLANCE_LIGHT_COUNT": GLANCE_LIGHT_COUNT,
        "GLANCE_MATERIAL_FORMAT": material_format,
        "GLANCE_MATERIAL_COLOR_AMBIENT": 1,
        "GLANCE_MATERIAL_COLOR_DIFFUSE": 1,
        "GLANCE_MATERIAL_SIDES": 1,
        "GLANCE_TESS_LEVEL_INNER": 32,
        "GLANCE_TESS_LEVEL_OUTER": 32,
        "GLANCE_VERTEX_FORMAT": DEFAULT_VERTEX_FORMAT,
        "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
        "M_N": mdefaults.DEFAULT_N,
    })


def tick_program(material_format, coordinate_count):
    return shaders.Program.from_files([
        "default_warp_nd.vs",
        "axes_nd.gs",
        "default.fs"
    ], defines={
        "GLANCE_COLOR": 1,
        "GLANCE_COORDINATE_COUNT": coordinate_count,
        "GLANCE_LIGHT_COUNT": GLANCE_LIGHT_COUNT,
        "GLANCE_MATERIAL_COUNT": 2,
        "GLANCE_MATERIAL_SIDES": 1,
        "GLANCE_MATERIAL_FORMAT": materials.DEFAULT_FORMAT,
        "GLANCE_MATERIAL_COLOR_AMBIENT": 1,
        "GLANCE_MATERIAL_COLOR_DIFFUSE": 1,
        "GLANCE_SHADE": 0,
        "GLANCE_VERTEX_FORMAT": DEFAULT_VERTEX_FORMAT,
        "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
        "M_N": mdefaults.DEFAULT_N,
    })


def mark_program(material_format, coordinate_count, shade, i):
    return shaders.Program.from_files([
        "default_nd.vs",
        "mark{}_nd.gs".format(i),
        "shade_phong.fs",
    ], defines=DEFAULT_PROGRAM_DEFINES + {
        "GLANCE_ALPHA_TEST": 1,
        "GLANCE_COORDINATE_COUNT": coordinate_count,
        "GLANCE_MARK_DIMENSION": 2,
        "GLANCE_MATERIAL_TEXTURE_AMBIENT": 1,
        "GLANCE_MATERIAL_TEXTURE_DIFFUSE": 1,
        "GLANCE_SHADE": int(shade),
        "GLANCE_SPACE": int(renderers.Space.View),
    })


def mark_instance_program(material_format, coordinate_count, shade, i):
    return shaders.Program.from_files([
        "default_instanced_nd.vs",
        "tessellate_simplex2_instanced_nd.tcs",
        "tessellate_simplex2_instanced.tes",
        #"tessellate_simplex2_instanced_nd.tes",
        #"extrude_instanced_nd.gs",
        "shade_phong.fs",
    ], defines=DEFAULT_PROGRAM_DEFINES + {
        "GLANCE_COORDINATE_COUNT": coordinate_count,
        "GLANCE_MARK_DIMENSION": 2,
        "GLANCE_MATERIAL_COUNT": 2,
        "GLANCE_MATERIAL_SIDES": 2,
        "GLANCE_MATERIAL_TEXTURE_AMBIENT": 1,
        "GLANCE_MATERIAL_TEXTURE_DIFFUSE": 1,
        "GLANCE_MERCATOR": 1,
        "GLANCE_SHADE": int(shade),
        "GLANCE_SPACE": int(renderers.Space.View),
    })


def mark_interval_program(material_format, coordinate_count, shade, i):
    return shaders.Program.from_files([
        "default_nd.vs",
        "tessellate_simplex0_to_cube{}_nd.tcs".format(i),
        "tessellate_simplex0_to_cube{}_nd.tes".format(i),
        "extrude_simplex{}_nd.gs".format(i),
        "shade_phong.fs",
    ], defines=DEFAULT_PROGRAM_DEFINES + {
        "GLANCE_COORDINATE_COUNT": coordinate_count,
        "GLANCE_MARK_DIMENSION": 2,
        "GLANCE_MATERIAL_COUNT": 2,
        "GLANCE_MATERIAL_SIDES": 2,
        "GLANCE_MATERIAL_TEXTURE_AMBIENT": 1,
        "GLANCE_MATERIAL_TEXTURE_DIFFUSE": 1,
        "GLANCE_SHADE": int(shade),
        "GLANCE_SPACE": int(renderers.Space.View),
    })


def join_program(material_format, coordinate_count, shade, i):
    return shaders.Program.from_files([
        "default_warp_nd.vs",
        "default_simplex1_nd.gs",
        "default.fs",
    ], defines=DEFAULT_PROGRAM_DEFINES + {
        "GLANCE_COORDINATE_COUNT": coordinate_count,
        "GLANCE_MATERIAL_COLOR_AMBIENT": 1,
        "GLANCE_MATERIAL_COLOR_DIFFUSE": 1,
        "GLANCE_SHADE": int(shade),
        "GLANCE_SPACE": int(renderers.Space.View),
    })


def label_program(material_format, coordinate_count, shade=0):
    return shaders.Program.from_files([
        "default_nd.vs",
        "cube2_nd.gs",
        "plain_texture.fs",
        #"debug.fs",
    ], defines=DEFAULT_PROGRAM_DEFINES + {
        "GLANCE_ALPHA_TEST": 1,
        "GLANCE_COORDINATE_COUNT": coordinate_count,
        "GLANCE_MARK_DIMENSION": 2,
        "GLANCE_MATERIAL_TEXTURE_AMBIENT": 1,
        "GLANCE_MATERIAL_TEXTURE_DIFFUSE": 1,
        "GLANCE_SHADE": int(shade),
        "GLANCE_SPACE": int(renderers.Space.View),
    })


DEFAULT_SIMPLEX0_PROGRAMS = mappings.CustomMap(factory=unpack_args(simplex0_nd_program))
DEFAULT_SIMPLEX1_PROGRAMS = mappings.CustomMap(factory=unpack_args(simplex1_nd_program))

DEFAULT_SIMPLEX0_PROGRAM = DEFAULT_SIMPLEX0_PROGRAMS[()]
DEFAULT_SIMPLEX1_PROGRAM = DEFAULT_SIMPLEX1_PROGRAMS[()]

DEFAULT_ORIGIN_PROGRAMS = mappings.CustomMap(factory=unpack_args(shaded_surface_program))
DEFAULT_AXIS_PROGRAMS = mappings.CustomMap(factory=unpack_args(axis_program))
DEFAULT_TICK_PROGRAMS = mappings.CustomMap(factory=unpack_args(tick_program))

DEFAULT_MARK_PROGRAMS = mappings.CustomMap(factory=unpack_args(mark_program))
DEFAULT_MARK_INSTANCE_PROGRAMS = mappings.CustomMap(factory=unpack_args(mark_instance_program))
DEFAULT_MARK_INTERVAL_PROGRAMS = mappings.CustomMap(factory=unpack_args(mark_interval_program))

DEFAULT_JOIN_PROGRAMS = mappings.CustomMap(factory=unpack_args(join_program))

DEFAULT_LABEL_PROGRAMS = mappings.CustomMap(factory=unpack_args(label_program))


DEFAULT_MATERIAL_COLORS_NORMAL_RGB = {
    "ambient": colors.svg.white.rgb,
    "diffuse": colors.svg.white.rgb,
    "specular": colors.svg.black.rgb,
    "emissive": colors.svg.black.rgb,
}

DEFAULT_MATERIAL_COLORS_HOVER_RGB = {
    "ambient": colors.Color.from_hex("#66cc66").rgb,
    "diffuse": colors.Color.from_hex("#66cc66").rgb,
    "specular": colors.svg.black.rgb,
    "emissive": colors.svg.black.rgb,
}

DEFAULT_MATERIAL_COLORS_NORMAL_HSV = {
    "ambient": colors.svg.white.hsv,
    "diffuse": colors.svg.white.hsv,
    "specular": colors.svg.black.hsv,
    "emissive": colors.svg.black.hsv,
}

DEFAULT_MATERIAL_COLORS_HOVER_HSV = {
    "ambient": colors.Color.from_hex("#66cc66").hsv,
    "diffuse": colors.Color.from_hex("#66cc66").hsv,
    "specular": colors.svg.black.hsv,
    "emissive": colors.svg.black.hsv,
}


def axes_nd_program(key):
    vertex_format, wrap, transform_count = key
    return shaders.Program.from_files(["axes_clip_n_to_n_nd.cs"], defines={
        "GLANCE_COLOR": 2,
        "GLANCE_TRANSFORM_COUNT": transform_count,
        "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
        "GLANCE_WORK_GROUP_SIZE": gdefaults.DEFAULT_WORK_GROUP_SIZE,
        "GLANCE_WRAP": int(wrap),
        "M_N": mdefaults.DEFAULT_N,
    })


def axes_interpolate_nd_program(key):
    vertex_format, wrap, transform_count = key
    return shaders.Program.from_files(["axes_interpolate_clip_n_to_n_nd.cs"], defines={
        "GLANCE_COLOR": 2,
        "GLANCE_TRANSFORM_COUNT": transform_count,
        "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
        "GLANCE_WORK_GROUP_SIZE": gdefaults.DEFAULT_WORK_GROUP_SIZE,
        "GLANCE_WRAP": int(wrap),
        "M_N": mdefaults.DEFAULT_N,
    })


DEFAULT_AXES_PROGRAMS = mappings.CustomMap(factory=axes_nd_program)
DEFAULT_AXES_INTERPOLATE_PROGRAMS = mappings.CustomMap(factory=axes_interpolate_nd_program)


class BackgroundView(scenes.View):
    @property
    def texture(self):
        return self._texture
        
    @texture.setter
    def texture(self, value):
        self._texture = value
    
    def __init__(self, texture=None, parent=None):
        super().__init__(parent=parent)
        
        self._texture = texture
    
    def create(self, renderer):
        self._material = materials.CustomMaterial(textures={"background": self._texture})
        self._material.prepare(renderer)

        self._geometry = geometries.CustomGeometry()
        self._geometry.prepare(renderer)
    
    def render(self, renderer):
        if self._texture is None:
            return
        
        self._texture.prepare(renderer)
        
        model = self.parent._root._transform.to_matrix(n=3, m=4)
        view = transforms.inversed(self.parent._camera._transform.to_matrix(n=3, m=4))
        projection = projections.perspective_inverted(np.pi/2.0, 0.1, 1000.0)
        projection = np.dot(transforms.scale(1.0/self.aspect, n=4), projection)
        #projection = self.parent._camera._projection.to_matrix(n=3, m=4)
        #projection = np.dot(transforms.scale(1.0/self.aspect, n=4), projection)
        
        GL.glDepthMask(GL.GL_FALSE)
        
        program = gdefaults.DEFAULT_BACKGROUND_PROGRAMS[self._texture.target]
        with renderer.activated(program, self._texture, self._geometry):
            renderer.uniforms["background"] = self._texture.unit

            renderer.uniforms["model_view"] = np.dot(view, model)
            renderer.uniforms["projection"] = projection

            GL.glDrawArrays(GL.GL_POINTS, 0, 1)

        GL.glDepthMask(GL.GL_TRUE)


def origin_generator():
    yield vectors.point(vectors.zeros())


def axes_generator(aspect):
    for i, v in enumerate(coordinates.axes()):
        yield vectors.point(-100000.0 * aspect[i] * v)
        yield vectors.point(+100000.0 * aspect[i] * v)


def ticks_generator(aspect, count):
    for i, v in enumerate(coordinates.axes()):
        for x in np.linspace(-aspect[i], +aspect[i], count):
            yield vectors.point(x * v)


class AxesView(scenes.View):
    @property
    def axes(self):
        return self._axes_input_vertex_data.data

    @axes.setter
    def axes(self, value):
        self._axes_input_vertex_data.data = value

        self._axes_output_vertex_data.size = self._axes_input_vertex_data.size * self.coordinates_count
        self._axes_output_vertex_data.type = self._axes_input_vertex_data.type
    
    def process(self, input):
        self._hovering_axis_indices = set(map(_identifier_data.decode,  self.parent._hovering_axes))
    
    def clear(self):
        pass
    
    def on_data_changed(self, axis, key, value):
        time = self.time
        
        animation = self.parent._alphas[axis.index]
        if axis.visible and animation.reversed:
            animation.reverse(time)
        if (not axis.visible) and (not animation.reversed):
            animation.reverse(time)
        
        animation = self.parent._betas[axis.index]
        if axis.angular and animation.reversed:
            animation.reverse(time)
        if (not axis.angular) and (not animation.reversed):
            animation.reverse(time)
    
    def create_axes(self, renderer):
        actor = actors.AxisActor()
        
        def origin_identifiers():
            for index in iterables.repeated(generators.autoincrement(), actor.origin_stride):
                yield origin_identifier()
        
        def axis_identifiers():
            for index in iterables.repeated(generators.autoincrement(), actor.axis_stride):
                yield axis_identifier(index)
        
        def tick_identifiers():
            for index in iterables.repeated(generators.autoincrement(), actor.tick_stride):
                yield axis_identifier(index)
        
        origin_data = vertices.full({
            vertices.Component.Position: origin_generator(),
            vertices.Component.Identifier: origin_identifiers(),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        
        axis_data = vertices.full({
            vertices.Component.Position: axes_generator(self.parent.aspect),
            #vertices.Component.Color0: iterables.repeated((colors.Color.from_ordinal(i) for i in generators.autoincrement()), actor.axis_stride),
            vertices.Component.Identifier: axis_identifiers(),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        
        tick_data = vertices.full({
            vertices.Component.Position: ticks_generator(self.parent.aspect, actor.tick_stride),
            vertices.Component.Identifier: tick_identifiers(),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        
        vertex_data = vertices.concatenate(origin_data, axis_data, tick_data)
        #print(vertex_data)
        
        actor.update_vertex_data(vertex_data)
        
        self._axes_actor = actor
        
    def create(self, renderer):
        self._material_normal = materials.MultiMaterial([
            materials.PhongMaterial(colors={
                "ambient": colors.svg.lightgray.rgb,
                "diffuse": colors.svg.lightgray.rgb,
                "specular": colors.svg.black.rgb,
                "emissive": colors.svg.black.rgb,
            }, opacity=1.0, shininess=32.0),
            materials.PhongMaterial(colors={
                "ambient": colors.svg.lightgray.rgb,
                "diffuse": colors.svg.lightgray.rgb,
                "specular": colors.svg.black.rgb,
                "emissive": colors.svg.black.rgb,
            }, opacity=1.0, shininess=32.0)
        ])
        self._material_normal.prepare(renderer)
        
        self._material_hover = materials.MultiMaterial([
            materials.PhongMaterial(colors=DEFAULT_MATERIAL_COLORS_HOVER_RGB, opacity=1.0, shininess=32.0),
            materials.PhongMaterial(colors=DEFAULT_MATERIAL_COLORS_HOVER_RGB, opacity=1.0, shininess=32.0),
        ])
        self._material_hover.prepare(renderer)
        
        self._origin_geometry = geometries.CustomGeometry()
        self._origin_geometry.prepare(renderer)
        
        self.create_axes(renderer)
    
    def update(self, renderer):
        time = self.time
        
        actor = self._axes_actor
        actor.multiple_count = len(self.parent.transforms)
        
        model_nd = self.parent._root_nd._transform.to_matrix()
        view_nd = transforms.inversed(self.parent._camera_nd._transform.to_matrix_at(self.parent._alpha_transform(time)))
        
        #aspect_nd = vectors.default_chunk1d(self.parent.current_aspect(time))
        aspect_nd = vectors.default_chunk1d(self.parent.current_aspect(time) * self.parent._alphas_nd)
        
        wrap = 0 if self.parent.model.show_overflow else renderers.Wrap.ClampRepeat
        compute_program = DEFAULT_AXES_PROGRAMS[DEFAULT_VERTEX_ND_FORMAT, wrap, len(self.parent.transforms)]
        with renderer.activated(compute_program, self.parent._transforms0_buffer, self.parent._transforms1_buffer):
            renderer.set_model_view_nd(model_nd, view_nd)
            
            renderer.uniforms["buffer_offset"] = GL.GLuint(0)
            renderer.uniforms["buffer_stride"] = GL.GLuint(1)
            
            renderer.uniform_blocks["TransformSrcData"] = self.parent._transforms0_buffer
            renderer.uniform_blocks["TransformDstData"] = self.parent._transforms1_buffer
            
            renderer.set_uniform_array_1d("aspect_nd", aspect_nd)
            
            renderer.uniforms["alpha"] = GL.GLfloat(self.parent._alpha(time))
            renderer.uniforms["beta"] = GL.GLfloat(self.parent._beta(time))
            
            with renderer.activated(actor._input_vertex_buffer, actor._output_vertex_buffer):
                renderer.shader_storage_blocks["VertexSrcData"] = actor._input_vertex_buffer
                renderer.shader_storage_blocks["VertexDstData"] = actor._output_vertex_buffer
                
                # Origin
                renderer.set_uniform_array_1d("wrap_mask", vectors.default_chunk1d(vectors.zeros()))
                
                renderer.uniforms["buffer_offset"] = GL.GLuint(actor.origin_offset)
                
                GL.glDispatchCompute(mathematics.chunk_count(actor.origin_stride, gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                #GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                
                # Axes
                for i in range(actor.axis_count):
                    renderer.set_uniform_array_1d("wrap_mask", vectors.default_chunk1d(vectors.zeros()))
                    
                    renderer.uniforms["buffer_offset"] = GL.GLuint(actor.axis_offset + i*actor.axis_stride)
                    
                    GL.glDispatchCompute(mathematics.chunk_count(actor.axis_stride, gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                    #GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                
                # Ticks
                for i in range(actor.tick_count):
                    renderer.set_uniform_array_1d("wrap_mask", vectors.default_chunk1d(vectors.unit(i)))
                    
                    renderer.uniforms["buffer_offset"] = GL.GLuint(actor.tick_offset + i*actor.tick_stride)
                    
                    GL.glDispatchCompute(mathematics.chunk_count(actor.tick_stride, gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                    #GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                    
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
    
    def render(self, renderer):
        time = self.time

        #view_nd = transforms.inversed(self.parent._camera_nd._transform.to_matrix_at(self.parent._alpha_transform(time)))
        projection_nd = projections.diagonal(3)
        #projection_nd = transforms.identity()
        
        model = self.parent._root._transform.to_matrix(n=3, m=4)
        view = transforms.inversed(self.parent._camera._transform.to_matrix(n=3, m=4))
        projection = self.parent._camera._projection.to_matrix(n=3, m=4)
        projection = np.dot(transforms.scale(1.0/self.aspect, n=4), projection)
        
        alphas_nd = vectors.default_chunk1d(self.parent._alphas_nd)
        betas_nd = vectors.default_chunk1d(self.parent._betas_nd)
        
        shade = renderers.Shade.Phong if self.parent.model.show_shading else 0
        
        axis_count = len(self.parent.model.axes)
        coordinate_count = axis_count - 1
        
        actor = self._axes_actor
        
        origin_program = DEFAULT_ORIGIN_PROGRAMS[materials.DEFAULT_FORMAT, coordinate_count, shade]
        with renderer.activated(origin_program, self.parent._lights, self._origin_geometry):
            renderer.set_projection_nd(projection_nd)
            renderer.set_model_view_projection(model, view, projection)
            
            renderer.set_uniform_array_1d("alphas_nd", alphas_nd)
            renderer.set_uniform_array_1d("betas_nd", betas_nd)
            
            material = self._material_hover if self.parent.dragging and len(self.parent._hovering_origins) else self._material_normal
            geometry = actor._geometry
            with renderer.activated(material, geometry):
                renderer.uniforms['identifier'] = GL.GLuint(origin_identifier())
                renderer.uniforms['size'] = GL.GLfloat(self.parent.origin_size)
                
                geometry.draw_indices("default", mode=GL.GL_POINTS, first=actor.origin_offset * actor.multiple_count, count=actor.origin_stride * actor.multiple_count, renderer=renderer)
        
        
        axis_program = DEFAULT_AXIS_PROGRAMS[materials.DEFAULT_FORMAT, coordinate_count]
        with renderer.activated(axis_program, self.parent._lights, self._axes_actor._geometry):
            renderer.set_projection_nd(projection_nd)
            renderer.set_model_view_projection(model, view, projection)
            
            renderer.set_uniform_array_1d("alphas_nd", alphas_nd)
            renderer.set_uniform_array_1d("betas_nd", betas_nd)
            
            root = self.parent.model.axes
            if root is not None:
                if self.parent.dragging and len(self._hovering_axis_indices):
                    for i, axis in enumerate(root.values()):
                        material = self._material_hover if i in self._hovering_axis_indices else self._material_normal
                        geometry = actor._geometry
                        with renderer.activated(material, geometry):
                            renderer.uniforms['size'] = GL.GLfloat(self.parent.axis_size)
                            
                            #geometry.draw_indices("default", mode=GL.GL_LINES, first=actor.axis_offset + i*actor.axis_stride, count=actor.axis_stride, renderer=renderer)
                            geometry.draw_indices("default", mode=GL.GL_PATCHES, first=(actor.axis_offset + i * actor.axis_stride) * actor.multiple_count, count=actor.axis_stride * actor.multiple_count, patch_vertices=2, renderer=renderer)
        
                else:
                    material = self._material_normal
                    geometry = actor._geometry
                    with renderer.activated(material, geometry):
                        renderer.uniforms['size'] = GL.GLfloat(self.parent.axis_size)
                        
                        #geometry.draw_indices("default", mode=GL.GL_LINES, first=actor.axis_offset, count=actor.axis_count * actor.axis_stride, renderer=renderer)
                        geometry.draw_indices("default", mode=GL.GL_PATCHES, first=actor.axis_offset * actor.multiple_count, count=axis_count * actor.axis_stride * actor.multiple_count, patch_vertices=2, renderer=renderer)

        '''
        tick_program = DEFAULT_TICK_PROGRAMS[materials.DEFAULT_FORMAT, coordinate_count]
        with renderer.activated(tick_program, self.parent._lights):
            renderer.set_projection_nd(projection_nd)
            renderer.set_model_view_projection(model, view, projection)
            
            renderer.set_uniform_array_1d("alphas_nd", alphas_nd)
            renderer.set_uniform_array_1d("betas_nd", betas_nd)
            
            root = self.parent.model.axes
            if root is not None:
                if self.parent.dragging and len(self._hovering_axis_indices):
                    for i, axis in enumerate(root.values()):
                        material = self._material_hover if i in self._hovering_axis_indices else self._material_normal
                        geometry = actor._geometry
                        with renderer.activated(material, geometry):
                            renderer.uniforms['size'] = GL.GLfloat(self.parent.tick_size)
                            
                            geometry.draw_indices("default", mode=GL.GL_POINTS, first=(actor.tick_offset + i * actor.tick_stride) * actor.multiple_count, count=actor.tick_stride * actor.multiple_count, renderer=renderer)
                else:
                    material = self._material_normal
                    geometry = actor._geometry
                    with renderer.activated(material, geometry):
                        renderer.uniforms['size'] = GL.GLfloat(self.parent.tick_size)
                        
                        geometry.draw_indices("default", mode=GL.GL_POINTS, first=actor.tick_offset * actor.multiple_count, count=axis_count * actor.tick_stride * actor.multiple_count, renderer=renderer)
        '''
    
    def delete(self, renderer):
        pass


def as_number(expression):
    type = expressions.typeof(expression)
    
    if type in [md.Integer, md.Decimal, md.Interval]:
        return expression
    
    return md.ordinal(expression)


def as_position(expression):
    type = expressions.typeof(expression)
    
    if md.is_position_type(type):
        result = expression
    elif md.is_location_type(type):
        result = md.Position[3](expression.longitude, expression.latitude, expression.altitude)
    elif type in [md.Integer, md.Decimal, md.Interval]:
        result = md.Position[1](expression)
    else:
        result = md.layout_grid[2](expression)
    
    return result


def as_offset(expression):
    type = expressions.typeof(expression)
    
    return expression


def as_color(expression):
    type = expressions.typeof(expression)
    
    if md.is_color_type(type):
        return expression
    
    return md.static_ordinal(expression)


def as_shape(expression):
    type = expressions.typeof(expression)
    
    if md.is_shape_type(type):
        return expression
    
    return md.static_ordinal(expression)


def position_from_property(position):
    expression = position.expression if position.bound else None
    if expression is not None:
        return as_position(expression)


def position_from_property_items(position):
    result = md.Object()
    for key in range(mdefaults.DEFAULT_N):
        expression = position[key].expression if position[key].bound else None
        if expression is not None:
            result[key] = expression
    return result.alias("position_items_raw") if len(md.dataof(result)) else None


def position_from_property_items_as_numbers(position):
    result = md.Object()
    for key in range(mdefaults.DEFAULT_N):
        expression = position[key].expression if position[key].bound else None
        if expression is not None:
            result[key] = as_number(expression)
    return result.alias("position_items") if len(md.dataof(result)) else None


def offset_from_property(offset):
    expression = offset.expression if offset.bound else None
    if expression is not None:
        return as_offset(expression)


def offset_from_property_items_as_numbers(offset):
    result = md.Object()
    for key in range(mdefaults.DEFAULT_N):
        expression = offset[key].expression if offset[key].bound else None
        if expression is not None:
            result[key] = as_number(expression)
    return result.alias("offset_items") if len(md.dataof(result)) else None


def color_from_property(color):
    expression = color.expression if color.bound else None
    if expression is not None:
        return as_color(expression)


def color_from_property_items(color):
    result = md.Object()
    for key in ["h", "s", "v", "a"]:
        expression = color[key].expression if color[key].bound else None
        if expression is not None:
            result[key] = expression
    return result.alias("color_items_raw") if len(md.dataof(result)) else None


def color_from_property_items_as_numbers(color):
    result = md.Object()
    for key in ["h", "s", "v", "a"]:
        expression = color[key].expression if color[key].bound else None
        if expression is not None:
            result[key] = as_number(expression)
    return result.alias("color_items") if len(md.dataof(result)) else None


def shape_from_property(shape):
    expression = shape.expression if shape.bound else None
    if expression is not None:
        return as_shape(expression)

        
def delta_size_from_property(size):
    expression = size.expression if size.bound else None
    if expression is not None:
        return as_number(expression)


def delta_offset_from_property(offset):
    expression = offset.expression if offset.bound else None
    if expression is not None:
        return as_number(expression)


class MarksView(scenes.View):
    @property
    def extent(self):
        result = extents.Null
        #print("current_extent", result)
        for actor in self._mark_actors.values():
            result |= actor.extent.nan_to_inf_null()
        #print("->", result.inf_to_nan())
        return result.inf_to_nan()
    
    def current_extent(self, time):
        result = extents.Null
        #print("current_extent", result)
        for actor in self._mark_actors.values():
            result |= actor.current_extent(time).nan_to_inf_null()
        #print("->", result.inf_to_nan())
        return result.inf_to_nan()
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._has_marks = True
        self._has_joins = True
        self._has_labels = True
    
    def clear(self):
        self._mark_actors = {}
        self._tick_actors = {}
    
    def get_or_create_mark_actor(self, key):
        result = self._mark_actors.get(key)
        if result is None:
            result = actors.MarkActor()
            self._mark_actors[key] = result
        return result
    
    def get_or_create_tick_actor(self, key):
        result = self._tick_actors.get(key)
        if result is None:
            result = actors.TickActor()
            self._tick_actors[key] = result
        return result
    
    def on_data_changed(self, mark, key, value):
        if 'palette' in key:
            self.on_binding_changed(mark)
        
    def on_binding_changed(self, mark):
        #print("on_binding_changed", mark.owner.title)
        owner_key = mark.owner.title.lower()
        dataset = mark.owner.schema._parent
        
        # Query
        #position0 = position_from_property(mark["position"])
        position1 = position_from_property_items_as_numbers(mark["position"])
        position1_raw = position_from_property_items(mark["position"])
        
        #offset0 = offset_from_property(mark["offset"])
        offset1 = offset_from_property_items_as_numbers(mark["offset"])
        #offset1_raw = offset_from_property_items(mark["offset"])
        
        color0 = color_from_property(mark["color"])
        color1 = color_from_property_items_as_numbers(mark["color"])
        #color1_raw = color_from_property_items(mark["color"])
        
        shape = shape_from_property(mark["shape"])
        
        delta_size = delta_size_from_property(mark["delta_size"])
        delta_offset = delta_offset_from_property(mark["delta_offset"])
        
        label = mark["label"].expression if mark["label"].bound else None
        
        sort_expressions = [property.expression for property in mark.sortby]
        sort_orderings = list(iterables.flattened([[property.sorting == models.SortState.SortAscending] * md.leaf_count(property.expression) for property in mark.sortby]))
        
        validated = filter(None, [position1, color0, color1, shape, delta_size, delta_offset, label])
        validated = list(iterables.flattened((map(md.nameof, md.leaves(expression)) for expression in validated), depth=1))
        
        select = filter(None, [position1, color0, color1, shape, delta_size, delta_offset, label] + [position1_raw])
        sortby = filter(None, sort_expressions)
        groupby = filter(None, [position1])
        
        #print(list(iterables.flattened([md.leaves(expression) for expression in select], depth=1)))
        
        # Auto-Select Default Palette
        if color0 is not None:
            mark.default_color_palette = "various.google" if md.is_categorical(md.typeof(color0)) else "brewer.ylgnbu"
        
        #np.random.seed(1337)
        
        #print("Query Start.")
        
        query = md.Query(owner_key, select)
        result = query.manual(dataset, sortby=sortby, groupby=groupby, sort_ordering=sort_orderings)
        
        result["__valid"] = ~result[validated].isnull().any(axis=1)
        
        mark.total = len(result)
        mark.count = result["__valid"].sum()
        
        #print("Data Points: {}/{}".format(mark.count, mark.total))
        #print(result.info())
        
        #print("Query Finish.")
        
        # Actor
        actor = self.get_or_create_mark_actor(owner_key)
        actor._material = materials.MultiMaterial([
            materials.PhongMaterial(colors={
                "ambient": colors.svg.white.hsv,
                "diffuse": colors.svg.white.hsv,
                "specular": colors.svg.black.hsv,
                "emissive": colors.svg.black.hsv,
            }, textures={
                "ambient": self._shape_atlas,
                "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
            materials.PhongMaterial(colors={
                "ambient": colors.svg.white.hsv,
                "diffuse": colors.svg.white.hsv,
                "specular": colors.svg.black.hsv,
                "emissive": colors.svg.black.hsv,
            }, textures={
                "ambient": self._shape_atlas,
                "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
        ])
        actor._enabled = (mark.count > 0)
        '''
            materials.PhongMaterial(colors={
                "ambient": colors.Color.from_hex("#66cc66").hsv,
                "diffuse": colors.Color.from_hex("#66cc66").hsv,
                "specular": colors.svg.black.hsv,
                "emissive": colors.svg.black.hsv,
            }, textures={
                "ambient": self._shape_atlas,
                "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
        '''
        
        actor._geometry._vbos["instance_position"] = self._instance_geometry._vbos["instance_position"]
        actor._geometry._vbos["instance_tex_coord"] = self._instance_geometry._vbos["instance_tex_coord"]
        actor._geometry._ibos["instance_default"] = self._instance_geometry._ibos["instance_default"]
        
        actor.instance_count = mark.total
        
        if not actor._enabled:
            logger.warn("No Marks.")
            return
        
        # Populate
        def discrete_property(property):
            if property is None:
                return utilities.identity
            else:
                type = md.typeof(property)
                def _discrete_property(value):
                    if pd.isnull(value):
                        return value
                    return type.domain.index(value)
                return _discrete_property
        
        def continuous_property(property):
            if property is None:
                return utilities.identity
            else:
                name = md.nameof(property)
                normal = mathematics.range_to_positive((result[name].min(), result[name].max()))
                def _continuous_property(value):
                    if pd.isnull(value):
                        return value
                    return normal(value)
                return _continuous_property
        
        def getter(property, default, normalize=None):
            if property is None:
                def _getter(row):
                    return default
                return _getter
            else:
                normalize = utilities.identity if normalize is None else normalize
                def _getter(row):
                    return normalize(row[md.nameof(property)])
                return _getter
        
        def getposition(row):
            def _getposition(row):
                def __getposition(row, expression):
                    if md.is_interval(expression):
                        lower = accessors.getitem(row, md.nameof(expression.lower), np.nan)
                        upper = accessors.getitem(row, md.nameof(expression.upper), np.nan)
                        return (upper + lower)/2.0
                    else:
                        value = accessors.getitem(row, md.nameof(expression), np.nan)
                        return value
                
                if position1 is not None:    
                    return vectors.zeros(*[__getposition(row, position1[i]) for i in range(mdefaults.DEFAULT_N)])
                else:
                    return vectors.zeros()
            
            return vectors.homogeneous(_getposition(row), 1.0)
        
        getsizedelta = getter(delta_size, 1.0, continuous_property(delta_size)) #utilities.compose(mathematics.positive_to_range((0.05, 1.0)), continuous_property(delta_size)))
        
        def getsize(row):
            def _getsize(row):
                def __getsize(row, expression):
                    if md.is_interval(expression):
                        lower = accessors.getitem(row, md.nameof(expression.lower), np.nan)
                        upper = accessors.getitem(row, md.nameof(expression.upper), np.nan)
                        return (upper - lower)/2.0
                    else:
                        return 0.0
                
                if position1 is not None:    
                    return vectors.zeros(*[__getsize(row, position1[i]) for i in range(mdefaults.DEFAULT_N)])
                else:
                    return vectors.zeros()
            
            return vectors.homogeneous(_getsize(row), getsizedelta(row))
        
        getoffsetdelta = getter(delta_offset, 1.0, continuous_property(delta_offset)) #utilities.compose(mathematics.positive_to_range((0.00, 1.0)), continuous_property(delta_offset)))
        
        def getoffset(row):
            def _getoffset(row):
                def __getoffset(row, expression):
                    return accessors.getitem(row, md.nameof(expression), np.nan)
                
                if offset1 is not None:    
                    return vectors.zeros(*[__getoffset(row, offset1[i]) for i in range(mdefaults.DEFAULT_N)])
                else:
                    return vectors.zeros()
            
            return vectors.homogeneous(_getoffset(row), getoffsetdelta(row))
        
        if not (color0 is None or md.is_color(color0)):
            if md.is_categorical(md.typeof(color0)):
                getcolor = getter(color0, 0, discrete_property(color0))
            else:
                getcolor = getter(color0, 1.0, continuous_property(color0))
        
        if not (shape is None or md.is_shape(shape)):
            if md.is_categorical(md.typeof(shape)):
                getshape = getter(shape, 0, discrete_property(shape))
            else:
                getshape = getter(shape, 1.0, continuous_property(shape))
        
        def as_position(row):
            #if not row["__valid"]:
            #    return vectors.full(np.nan)
            return getposition(row)
        
        def as_size(row):
            #if not row["__valid"]:
            #    return vectors.full(np.nan)
            return getsize(row)
        
        def as_offset(row):
            #if not row["__valid"]:
            #    return vectors.full(np.nan)
            return getoffset(row)
        
        def as_color0(row):
            if not row["__valid"]:
                if mark.color_palettes is not None:
                    return colors.Color([1.0,1.0,1.0,0.0])
                else:
                    return colors.Color([0.0,0.0,1.0,0.0])
            
            if color0 is None:
                return colors.svg.white.hsva
            
            if md.is_color(color0):
                return colors.Color([row[md.nameof(color0[key])] for key in ['r', 'g', 'b', 'a']]).hsva 
            
            if mark.color_palettes is not None:
                value = getcolor(row)
                palette = mark.color_palettes.max(20)
                if md.is_categorical(md.typeof(color0)):
                    return palette[value % len(palette)].hsva
                else:
                    return palette(value).hsva
            
            return colors.svg.white.hsva
        
        color1_h = color1["h"] if color1 is not None and "h" in color1 else None
        color1_s = color1["s"] if color1 is not None and "s" in color1 else None
        color1_v = color1["v"] if color1 is not None and "v" in color1 else None
        color1_a = color1["a"] if color1 is not None and "a" in color1 else None
        
        getcolor_h = getter(color1_h, 1.0, continuous_property(color1_h))
        getcolor_s = getter(color1_s, 1.0, utilities.compose(mathematics.positive_to_range((0.5, 1.0)), continuous_property(color1_s)))
        getcolor_v = getter(color1_v, 1.0, utilities.compose(mathematics.positive_to_range((0.5, 1.0)), continuous_property(color1_v)))
        getcolor_a = getter(color1_a, 1.0, utilities.compose(mathematics.positive_to_range((0.5, 1.0)), continuous_property(color1_a)))
        
        def as_color1(row):
            h = getcolor_h(row)
            s = getcolor_s(row)
            v = getcolor_v(row)
            a = getcolor_a(row)
            return [h, s, v, a]
        
        def as_tex_coord(row):
            if not row["__valid"]:
                return vectors.zeros(n=4)
            
            if shape is not None:
                index = getshape(row)
                index = np.unravel_index(int(index), self._shape_atlas._provider._shape)
                return vectors.zeros(*index, n=4)
            
            return vectors.zeros(n=4)
        
        
        layer = list(self.parent.model.marks.keys()).index(owner_key)
        
        def mark_identifiers(rank):
            for index in iterables.repeated(generators.autoincrement(), rank):
                yield mark_identifier(layer, index)
        
        def as_identifier(index):
            return mark_identifier(layer, index)
        
        
        print("Data Start.")
        # TODO: This is a bottleneck...
        vertex_data = vertices.empty(len(result), format=DEFAULT_VERTEX_ND_FORMAT)
        
        position_data = vertex_data[vertices.Component.Position.type.name]
        tex_coord0_data = vertex_data[vertices.Component.TexCoord0.type.name]
        tex_coord1_data = vertex_data[vertices.Component.TexCoord1.type.name]
        color0_data = vertex_data[vertices.Component.Color0.type.name]
        color1_data = vertex_data[vertices.Component.Color1.type.name]
        size_data = vertex_data[vertices.Component.Size.type.name]
        offset_data = vertex_data[vertices.Component.Offset.type.name]
        identifier_data = vertex_data[vertices.Component.Identifier.type.name]
        
        for index, row in result.iterrows():
            tex_coord = as_tex_coord(row)
            position_data[index] = vertices.Component.Position.type(as_position(row))
            tex_coord0_data[index] = vertices.Component.TexCoord0.type(tex_coord)
            tex_coord1_data[index] = vertices.Component.TexCoord1.type(tex_coord)
            color0_data[index] = vertices.Component.Color0.type(as_color0(row))
            color1_data[index] = vertices.Component.Color1.type(as_color1(row))
            size_data[index] = vertices.Component.Size.type(as_size(row))
            offset_data[index] = vertices.Component.Offset.type(as_offset(row))
            identifier_data[index] = vertices.Component.Identifier.type(as_identifier(index))
        
        '''
        position_data = np.ascontiguousarray([as_position(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        tex_coord_data = np.ascontiguousarray([as_tex_coord(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        color0_data = np.ascontiguousarray([as_color0(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        color1_data = np.ascontiguousarray([as_color1(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        size_data = np.ascontiguousarray([as_size(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        offset_data = np.ascontiguousarray([as_offset(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        
        vertex_data = vertices.full({
            vertices.Component.Position: position_data,
            vertices.Component.TexCoord0: tex_coord_data,
            vertices.Component.TexCoord1: tex_coord_data,
            vertices.Component.Color0: color0_data,
            vertices.Component.Color1: color1_data,
            vertices.Component.Size: size_data,
            vertices.Component.Offset: offset_data,
            vertices.Component.Identifier: mark_identifiers(1),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        '''
        '''
        data = [{
            vertices.Component.Position: as_position(row),
            vertices.Component.TexCoord: as_tex_coord(row),
            vertices.Component.Color0: as_color0(row),
            vertices.Component.Color1: as_color1(row),
            vertices.Component.Size: as_size(row),
            vertices.Component.Offset: as_offset(row),
        } for index, row in result.iterrows()]
        
        position_data = np.ascontiguousarray(list(map(accessors.itemgetter(vertices.Component.Position), data)), dtype=mdefaults.DEFAULT_DTYPE)
        size_data = np.ascontiguousarray(list(map(accessors.itemgetter(vertices.Component.Size), data)), dtype=mdefaults.DEFAULT_DTYPE)
        offset_data = np.ascontiguousarray(list(map(accessors.itemgetter(vertices.Component.Offset), data)), dtype=mdefaults.DEFAULT_DTYPE)
        tex_coord_data = map(accessors.itemgetter(vertices.Component.TexCoord), data)
        color0_data = map(accessors.itemgetter(vertices.Component.Color0), data)
        color1_data = map(accessors.itemgetter(vertices.Component.Color1), data)
        '''
        print("Data Finish.")
        
        # Compute Extent
        position_data_nans = position_data.reshape(-1, mdefaults.DEFAULT_M)
        size_data_nans = size_data.reshape(-1, mdefaults.DEFAULT_M)
        offset_data_nans = offset_data.reshape(-1, mdefaults.DEFAULT_M)
        
        #TODO: Error with missing information
        #print([str(md.typeof(position1[i]).domain) if i in position1 else None for i in range(mdefaults.DEFAULT_N)])
        #print([str(md.typeof(position1[i]).domain) if i in position1 else None for i in range(mdefaults.DEFAULT_N)])
        #lower = vectors.vector([md.typeof(position1[i]).domain.static_lower if i in position1 else np.nan for i in range(mdefaults.DEFAULT_N)], n=mdefaults.DEFAULT_N)
        #upper = vectors.vector([md.typeof(position1[i]).domain.static_upper if i in position1 else np.nan for i in range(mdefaults.DEFAULT_N)], n=mdefaults.DEFAULT_N)
        
        position_mask = vectors.vector([np.nan if position1 is None or i not in position1 else 1.0 for i in range(mdefaults.DEFAULT_N)], n=mdefaults.DEFAULT_N)
        lower = position_mask * vectors.vector(np.nanmin(position_data_nans - np.abs(size_data_nans), axis=0), n=mdefaults.DEFAULT_N)
        upper = position_mask * vectors.vector(np.nanmax(position_data_nans + np.abs(size_data_nans), axis=0), n=mdefaults.DEFAULT_N)
        
        extent = extents.Extent(lower, upper)
        
        position_data[:] = np.nan_to_num(position_data)
        size_data[:] = np.nan_to_num(size_data)
        offset_data[:] = np.nan_to_num(offset_data)
        
        #print(extent)
        
        
        if label is None:
            actor._label_data = None
        else:
            getlabel = getter(label, None)
            
            def as_label(index, row):
                value = getlabel(row)
                '''
                if pd.isnull(value):
                    return "{} #{}".format(mark.owner.title, index)
                '''
                return str(value)
            
            try:
                actor._label_data = [as_label(index, row) for index, row in result.iterrows()]
            except KeyError:
                actor._label_data = None
            
            '''
            style = {"fill": colors.svg.white, "stroke": colors.svg.black, "stroke_width": 5.0, "font_size": 1.0, "font_family": "Verdana"} # "font_weight": painters.Weight.Bold
            actor._label_provider.paint([painters.text(utilities.truncate(item, 10, ".."), **style) for item in actor._label_data])
            #actor._label_provider.paint([painters.chain(painters.fill(fill=colors.svg.red), painters.text(item, **style)) for item in actor._label_data])
            actor._label_atlas.request_update()
            '''
        
        def mark_basis(position):
            def _mark_basis(position):
                for i in range(mdefaults.DEFAULT_M):
                    item = accessors.getitem(position, i, md.as_expression(np.nan))
                    if md.is_interval(item):
                        yield vectors.unit(i, n=mdefaults.DEFAULT_M)
            result = np.array(list(_mark_basis(position)), dtype=mdefaults.DEFAULT_DTYPE)
            return result.reshape((len(result), mdefaults.DEFAULT_M))
        
        def mark_dual_basis(position):
            def _mark_dual_basis(position):
                for i in range(mdefaults.DEFAULT_M):
                    item = accessors.getitem(position, i, md.as_expression(np.nan))
                    if not md.is_interval(item):
                        yield vectors.unit(i, n=mdefaults.DEFAULT_M)
            result = np.array(list(_mark_dual_basis(position)), dtype=mdefaults.DEFAULT_DTYPE)
            return result.reshape((len(result), mdefaults.DEFAULT_M))
        
        basis, dual_basis = mark_basis(position1), mark_dual_basis(position1)
        
        actor.set_instanced(len(basis))
        
        #print(len(basis), len(dual_basis))
        #print(basis, dual_basis)
        
        #print(vectors.dot(vectors.vector(range(mdefaults.DEFAULT_N)), actor.bases))
        
        if actor._input0_vertex_data.data is None:
            actor.update_vertex_data(vertex_data)
            actor.update_vertex_data(vertex_data)
        else:
            actor.update_vertex_data(vertex_data)
        
        actor.extent = extent
        actor.order = 0
        actor.basis, actor.dual_basis = basis, dual_basis
        actor._alpha.play(self.time)
        
        # Tick Labels
        
        def iter_domains(position, position_raw):
            if position is None or position_raw is None:
                return
            
            for key in range(mdefaults.DEFAULT_N):
                if key in position and key in position_raw:
                    domain, domain_raw = md.typeof(position[key]).domain.values, md.typeof(position_raw[key]).domain.values
                    if domain_raw and md.length_or_default(domain_raw):
                        yield key, domain, domain_raw
        
        def generate_tick_positions_labels(position, position_raw):
            fmt = lambda value: "{0:.2f}".format(value) if isinstance(value, float) else str(value)
            for key, domain, domain_raw in iter_domains(position, position_raw):
                for value, value_raw in zip(domain, domain_raw):
                    yield vectors.point(vectors.unit(key) * value), fmt(value_raw)
        
        tick_positions_labels = list(generate_tick_positions_labels(position1, position1_raw))
        
        actor = self.get_or_create_tick_actor(owner_key)
        actor._material = materials.MultiMaterial([
            materials.PhongMaterial(colors={
                "ambient": colors.svg.white.hsv,
                "diffuse": colors.svg.white.hsv,
                "specular": colors.svg.black.hsv,
                "emissive": colors.svg.black.hsv,
            }, textures={
                "ambient": actor._label_atlas,
                "diffuse": actor._label_atlas,
            }, opacity=1.0, shininess=32.0),
            materials.PhongMaterial(colors={
                "ambient": colors.svg.white.hsv,
                "diffuse": colors.svg.white.hsv,
                "specular": colors.svg.black.hsv,
                "emissive": colors.svg.black.hsv,
            }, textures={
                "ambient": actor._label_atlas,
                "diffuse": actor._label_atlas,
            }, opacity=1.0, shininess=32.0),
        ])
        actor._enabled = bool(tick_positions_labels)
        
        if not actor._enabled:
            logger.warn("No Labels.")
            return
        
        tick_positions, tick_labels = zip(*tick_positions_labels)
        
        vertex_data = vertices.full({
            vertices.Component.Position: tick_positions,
            vertices.Component.Size: generators.constant(vectors.ones(0,0,0)),
            vertices.Component.Offset: generators.constant(-0.25 * vectors.ones()),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        
        actor._label_data = tick_labels
        
        style = {"fill": colors.svg.white, "stroke": colors.svg.black, "stroke_width": 5.0, "font_size": 1.0, "font_family": "Verdana", "align": painters.Alignment.Left} # "font_weight": painters.Weight.Bold
        actor._label_provider.paint([painters.text(utilities.truncate(item, 10, ".."), **style) for item in actor._label_data])
        #actor._label_provider.paint([painters.chain(painters.fill(fill=colors.svg.red), painters.text(item, **style)) for item in actor._label_data])
        actor._label_atlas.request_update()
        
        actor.update_vertex_data(vertex_data)
    
    def get_label(self, key, index):
        actor = self._mark_actors.get(key)
        return accessors.getitem(actor._label_data, index, None)
    
    def process(self, input):
        self._hovering_mark_layers = set(map(_identifier_mark_layer.decode, self.parent._hovering_marks))
        self._hovering_mark_indices = set(map(_identifier_mark_index.decode,  self.parent._hovering_marks))
        
        if len(self._hovering_mark_layers) and len(self._hovering_mark_indices):
            mark_layer = next(iter(self._hovering_mark_layers))
            mark_index = next(iter(self._hovering_mark_indices))
            key = list(self.parent.model.marks.keys())[mark_layer] #containers.key(self.parent.model.marks, mark_layer) #???
            label = self.get_label(key, mark_index)
            self.parent.status = "" if label is None else label
    
    def create(self, renderer):
        self._shape_atlas = textures.Texture(self.parent.shape_provider, filter=(GL.GL_NEAREST, GL.GL_NEAREST))
        self._shape_atlas.prepare(renderer)
        
        self._texture_atlas = textures.Texture(self.parent.texture_provider, filter=(GL.GL_NEAREST, GL.GL_NEAREST))
        self._texture_atlas.prepare(renderer)
        
        self._material_normal = materials.MultiMaterial([
            materials.PhongMaterial(colors=copy.copy(DEFAULT_MATERIAL_COLORS_NORMAL_HSV), textures={
                "ambient": self._shape_atlas, "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
            materials.PhongMaterial(colors=copy.copy(DEFAULT_MATERIAL_COLORS_NORMAL_HSV), textures={
                "ambient": self._shape_atlas, "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
        ])
        self._material_normal.prepare(renderer)
        
        self._material_hover = materials.MultiMaterial([
            materials.PhongMaterial(colors=copy.copy(DEFAULT_MATERIAL_COLORS_HOVER_HSV), textures={
                "ambient": self._shape_atlas, "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
            materials.PhongMaterial(colors=copy.copy(DEFAULT_MATERIAL_COLORS_HOVER_HSV), textures={
                "ambient": self._shape_atlas, "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
        ])
        self._material_hover.prepare(renderer)
        
        cube_face2_positions = cubes.cube_face_positions(2, mdefaults.DEFAULT_N)
        cube_face2_tex_coords = cubes.cube_face_tex_coords(2, mdefaults.DEFAULT_N)
        cube_face2_indices = cubes.cube_face_indices(2, mdefaults.DEFAULT_N)
        
        vertex_data = vertices.full({
            vertices.Component.InstancePosition: [vectors.zeros(*position, n=mdefaults.DEFAULT_M) for position in cube_face2_positions],
            vertices.Component.InstanceTexCoord: [vectors.zeros(*tex_coord, n=4) for tex_coord in cube_face2_tex_coords],
        }, format=vertices.FORMAT_INSTANCE)
        
        #index_data = np.array(list(cube_face2_indices), dtype=np.uint32).ravel()
        index_data = np.array(list(map(primitives.triangle_strip_to_triangle_indices, cube_face2_indices)), dtype=np.uint32).ravel()
        
        #print(vertex_data, index_data, sep='\n')
        
        self._instance_vertex_buffer = buffers.ShaderStorageBuffer(buffers.BufferData(vertex_data))
        self._instance_index_buffer = buffers.IndexBuffer(buffers.BufferData(index_data))
        
        self._instance_geometry = geometries.CustomGeometry({
            "instance_position": self._instance_vertex_buffer,
            "instance_tex_coord": self._instance_vertex_buffer,
        }, indices={
            "instance_default": self._instance_index_buffer,
        })
        self._instance_geometry.prepare(renderer)
        
        self._mark_actors = {}
        self._tick_actors = {}
    
    def update(self, renderer):
        time = self.time
        
        self._has_joins = self.parent.arrangement in [models.Arrangement.PerpendicularDisjoint, models.Arrangement.Parallel] #len(self.parent.transforms) > 1
        
        for key, mark in self.parent.model.marks.items():
            actor = self.get_or_create_mark_actor(key)
            actor.multiple_count = len(self.parent.transforms)
            
            if actor.rank == 0:
                texture0 = self._shape_atlas
                texture1 = self._texture_atlas #actor._label_atlas
                tex_coord_transform = self.parent.shape_provider.tex_coord_transform
                tex_coord_modifiers = vectors.zeros(*np.unravel_index(int(mark.shape_index), self.parent.shape_provider._shape), n=4)
            else:
                texture0 = gdefaults.DEFAULT_TEXTURE_WHITE if mark.texture is None else mark.texture
                texture1 = self._texture_atlas #actor._label_atlas
                tex_coord_transform = transforms.identity(n=4)
                tex_coord_modifiers = vectors.zeros(n=4)
            
            primaryColor = mark.color_primary_hsva
            actor._material[0].colors["ambient"] = primaryColor[:3]
            actor._material[0].colors["diffuse"] = primaryColor[:3]
            actor._material[0]["opacity"] = primaryColor[3]
            
            actor._material[0].textures["ambient"] = texture0
            actor._material[0].textures["diffuse"] = texture1
            
            secondaryColor = mark.color_secondary_hsva
            actor._material[1].colors["ambient"] = secondaryColor[:3]
            actor._material[1].colors["diffuse"] = secondaryColor[:3]
            actor._material[1]["opacity"] = secondaryColor[3]
            
            actor._material[1].textures["ambient"] = texture0
            actor._material[1].textures["diffuse"] = texture1
            
            actor._color_modifiers = mark.color_modifiers_hsva
            actor._tex_coord_transform = tex_coord_transform
            actor._tex_coord_modifiers = tex_coord_modifiers
            actor._size = mark.delta_size
            actor._motion = mark.delta_offset
            actor._visible = mark.visible
            
            actor = self.get_or_create_tick_actor(key)
            actor.multiple_count = len(self.parent.transforms)
            actor._visible = mark.visible
        
        
        model_nd = self.parent._root_nd._transform.to_matrix()
        view_nd = transforms.inversed(self.parent._camera_nd._transform.to_matrix_at(self.parent._alpha_transform(time)))
        
        #aspect_nd = vectors.default_chunk1d(self.parent.current_aspect(time))
        aspect_nd = vectors.default_chunk1d(self.parent.current_aspect(time) * self.parent._alphas_nd)
        
        wrap = 0 if self.parent.model.show_overflow else renderers.Wrap.Clamp
        compute_program = DEFAULT_AXES_INTERPOLATE_PROGRAMS[DEFAULT_VERTEX_ND_FORMAT, wrap, len(self.parent.transforms)]
        with renderer.activated(compute_program, self.parent._transforms0_buffer, self.parent._transforms1_buffer):
            renderer.set_model_view_nd(model_nd, view_nd)
            
            renderer.uniforms["buffer_offset"] = GL.GLuint(0)
            renderer.uniforms["buffer_stride"] = GL.GLuint(1)
            
            renderer.uniform_blocks["TransformSrcData"] = self.parent._transforms0_buffer
            renderer.uniform_blocks["TransformDstData"] = self.parent._transforms1_buffer
            
            renderer.set_uniform_array_1d("aspect_nd", aspect_nd)
            
            renderer.uniforms["beta"] = GL.GLfloat(self.parent._beta(time))
            
            for i, key in enumerate(self.parent.model.marks.keys()):
                actor = self._mark_actors.get(key)
                
                if not (actor and actor.visible):
                    continue
                
                renderer.uniforms["alpha"] = GL.GLfloat(actor._alpha(time))
                
                with renderer.activated(actor._input0_vertex_buffer, actor._input1_vertex_buffer, actor._output_vertex_buffer):
                    renderer.shader_storage_blocks["VertexSrcData0"] = actor._input0_vertex_buffer
                    renderer.shader_storage_blocks["VertexSrcData1"] = actor._input1_vertex_buffer
                    renderer.shader_storage_blocks["VertexDstData"] = actor._output_vertex_buffer
                    
                    GL.glDispatchCompute(mathematics.chunk_count(accessors.lenitems(actor._input0_vertex_data.data), gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                    #GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        
        
        wrap = 0 if self.parent.model.show_overflow else renderers.Wrap.Clamp
        compute_program = DEFAULT_AXES_PROGRAMS[DEFAULT_VERTEX_ND_FORMAT, wrap, len(self.parent.transforms)]
        with renderer.activated(compute_program, self.parent._transforms0_buffer, self.parent._transforms1_buffer):
            renderer.set_model_view_nd(model_nd, view_nd)
            
            renderer.uniforms["buffer_offset"] = GL.GLuint(0)
            renderer.uniforms["buffer_stride"] = GL.GLuint(1)
            
            renderer.uniform_blocks["TransformSrcData"] = self.parent._transforms0_buffer
            renderer.uniform_blocks["TransformDstData"] = self.parent._transforms1_buffer
            
            renderer.set_uniform_array_1d("aspect_nd", aspect_nd)
            
            renderer.uniforms["beta"] = GL.GLfloat(self.parent._beta(time))
            
            for i, key in enumerate(self.parent.model.marks.keys()):
                actor = self._tick_actors.get(key)
                
                if not (actor and actor.visible):
                    continue
                
                #renderer.uniforms["alpha"] = GL.GLfloat(actor._alpha(time))
                
                with renderer.activated(actor._input_vertex_buffer, actor._output_vertex_buffer):
                    renderer.shader_storage_blocks["VertexSrcData"] = actor._input_vertex_buffer
                    renderer.shader_storage_blocks["VertexDstData"] = actor._output_vertex_buffer
                    
                    GL.glDispatchCompute(mathematics.chunk_count(accessors.lenitems(actor._input_vertex_data.data), gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                    #GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
            
            GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        
        '''
        for actor in self._mark_actors.values():
            print(actor._input0_vertex_data.data)
            print(actor._input1_vertex_data.data)
        '''
        '''
        for actor in self._tick_actors.values():
            print(actor._output_vertex_buffer._value._handle)
            #actor._input0_vertex_buffer.debug_dump(renderer)
            #actor._input1_vertex_buffer.debug_dump(renderer)
            #actor._output_vertex_buffer.debug_dump(renderer)
        '''
        '''
        for actor in self._tick_actors.values():
            try:
                print(actor._output_vertex_buffer._value._handle)
                actor._input_vertex_buffer.debug_dump(renderer)
                actor._output_vertex_buffer.debug_dump(renderer)
            except:
                pass
        '''
    
    def render(self, renderer):
        time = self.time
        
        #view_nd = transforms.inversed(self.parent._camera_nd._transform.to_matrix_at(self.parent._alpha_transform(time)))
        projection_nd = projections.diagonal(3)
        #projection_nd = transforms.identity()
        
        model = self.parent._root._transform.to_matrix(n=3, m=4)
        view = transforms.inversed(self.parent._camera._transform.to_matrix(n=3, m=4))
        projection = self.parent._camera._projection.to_matrix(n=3, m=4)
        projection = np.dot(transforms.scale(1.0/self.aspect, n=4), projection)
        
        alphas_nd = vectors.default_chunk1d(self.parent._alphas_nd)
        betas_nd = vectors.default_chunk1d(self.parent._betas_nd)
        
        shade = renderers.Shade.Phong if self.parent.model.show_shading else 0
        
        axis_count = len(self.parent.model.axes)
        coordinate_count = axis_count - 1
        
        selection = vectors.zeros(*self.parent.model.selection, n=4, dtype=np.uint32)
        
        if self.parent.model.show_marks and self._has_marks:
            for i, key in enumerate(self.parent.model.marks.keys()):
                actor = self._mark_actors.get(key)
                
                if not (actor and actor.visible):
                    continue
                
                rank = actor.rank #actor.current_rank(time)
                order = actor.order
                bases = actor.bases #actor.current_bases(time)
                
                if rank == 0:
                    marks_program = DEFAULT_MARK_PROGRAMS[materials.DEFAULT_FORMAT, coordinate_count, shade, order]
                else:
                    #marks_program = DEFAULT_MARK_INTERVAL_PROGRAMS[materials.DEFAULT_FORMAT, shade, rank]
                    #actor.set_instanced(False)
                    marks_program = DEFAULT_MARK_INSTANCE_PROGRAMS[materials.DEFAULT_FORMAT, coordinate_count, shade, rank]
                
                with renderer.activated(marks_program, self.parent._lights):
                    renderer.set_projection_nd(projection_nd)
                    renderer.set_model_view_projection(model, view, projection)

                    renderer.set_uniform_array_1d("alphas_nd", alphas_nd)
                    renderer.set_uniform_array_1d("betas_nd", betas_nd)
                    
                    renderer.set_uniform_array_2d("basis_nd", vectors.default_chunk2d(bases))
                    
                    renderer.uniforms['time'] = GL.GLfloat(time)
                    
                    material = actor._material
                    geometry = actor._geometry
                    
                    renderer.uniforms['selection_color'] = self.parent.accent.hsva
                    
                    if self.parent.dragging and (i in self._hovering_mark_layers):
                        # TODO: This is pretty bad...
                        actor._material[0].colors["ambient"] = self._material_hover[0].colors["ambient"]
                        actor._material[0].colors["diffuse"] = self._material_hover[0].colors["diffuse"]
                        
                        actor._material[1].colors["ambient"] = self._material_hover[1].colors["ambient"] 
                        actor._material[1].colors["diffuse"] = self._material_hover[1].colors["diffuse"]
                        
                        actor._color_modifiers = vectors.ones(n=4)
                    
                    renderer.uniforms['tex_coord_transform'] = actor._tex_coord_transform 
                    
                    with renderer.activated(material, geometry):
                        #renderer.uniforms['identifier'] = GL.GLuint(mark_identifier(i, 0))
                        renderer.uniforms['selection'] = selection
                        
                        renderer.uniforms['rank'] = GL.GLint(rank)
                        
                        renderer.uniforms['tex_coord_modifiers'] = actor._tex_coord_modifiers
                        renderer.uniforms['color_modifiers'] = actor._color_modifiers
                        renderer.uniforms['size'] = GL.GLfloat(actor._size * self.parent.mark_size)
                        renderer.uniforms['motion'] = GL.GLfloat(actor._motion)
                        
                        if rank == 0:
                            mode = [GL.GL_POINTS,GL.GL_LINES,GL.GL_TRIANGLES][order]
                            geometry.draw_indices("default", mode=mode, renderer=renderer)
                        else:
                            #geometry.draw_indices("default", mode=GL.GL_PATCHES, patch_vertices=1, renderer=renderer)
                            #geometry.draw_indices_instanced("instance_default", actor.instance_count, mode=GL.GL_TRIANGLES, renderer=renderer)
                            geometry.draw_indices_instanced("instance_default", actor.instance_count * actor.multiple_count, mode=GL.GL_PATCHES, patch_vertices=3, renderer=renderer)
        
        
        if self.parent.model.show_joins and self._has_joins:
            for i, key in enumerate(self.parent.model.marks.keys()):
                actor = self._mark_actors.get(key)
                
                if not (actor and actor.visible):
                    continue
                
                rank = actor.rank
                
                joins_program = DEFAULT_JOIN_PROGRAMS[materials.DEFAULT_FORMAT, coordinate_count, shade, rank]
                with renderer.activated(joins_program, self.parent._lights):
                    renderer.set_projection_nd(projection_nd)
                    renderer.set_model_view_projection(model, view, projection)

                    renderer.set_uniform_array_1d("alphas_nd", alphas_nd)
                    renderer.set_uniform_array_1d("betas_nd", betas_nd)
                    
                    renderer.set_uniform_array_2d("basis_nd", vectors.default_chunk2d(bases))
                    
                    renderer.uniforms['time'] = GL.GLfloat(time)
                    
                    material = actor._material
                    geometry = actor._geometry
                    
                    renderer.uniforms['selection_color'] = self.parent.accent.hsva
                    
                    with renderer.activated(material, geometry):
                        #renderer.uniforms['identifier'] = GL.GLuint(mark_identifier(i, 0))
                        renderer.uniforms['selection'] = selection
                        
                        renderer.uniforms['rank'] = GL.GLint(rank)
                        
                        renderer.uniforms['tex_coord_modifiers'] = actor._tex_coord_modifiers
                        renderer.uniforms['color_modifiers'] = actor._color_modifiers
                        renderer.uniforms['size'] = GL.GLfloat(actor._size * self.parent.mark_size)
                        renderer.uniforms['motion'] = GL.GLfloat(actor._motion)
                        
                        if rank == 0:
                            geometry.draw_indices("joins", mode=GL.GL_LINES, renderer=renderer)
                        else:
                            geometry.draw_indices_instanced("instance_joins", actor.instance_count * actor.multiple_count, mode=GL.GL_LINES, renderer=renderer)
        
        
        if self.parent.model.show_labels and self._has_labels:
            gl.clear(GL.GL_DEPTH_BUFFER_BIT)
            
            for i, key in enumerate(self.parent.model.marks.keys()):
                actor = self._tick_actors.get(key)
                
                if not (actor and actor.visible):
                    continue
                
                labels_program = DEFAULT_LABEL_PROGRAMS[materials.DEFAULT_FORMAT, coordinate_count]
                with renderer.activated(labels_program, self.parent._lights):
                    renderer.set_projection_nd(projection_nd)
                    renderer.set_model_view_projection(model, view, projection)

                    renderer.set_uniform_array_1d("alphas_nd", alphas_nd)
                    renderer.set_uniform_array_1d("betas_nd", betas_nd)
                    
                    renderer.uniforms['time'] = GL.GLfloat(time)
                    
                    material = actor._material
                    geometry = actor._geometry
                    
                    renderer.uniforms['tex_atlas_shape'] = actor._label_provider.shape
                    renderer.uniforms['tex_coord_transform'] = actor._label_provider.tex_coord_transform
                    
                    with renderer.activated(material, geometry):
                        renderer.uniforms['size'] = GL.GLfloat(1.0) #GL.GLfloat(self.parent.label_size)
                        renderer.uniforms['offset'] = GL.GLfloat(1.0) #GL.GLfloat(actor._size * self.parent.mark_size)
                        
                        geometry.draw_indices("default", mode=GL.GL_POINTS, renderer=renderer) # first=actor.origin_offset * actor.multiple_count, count=actor.origin_stride * actor.multiple_count, renderer=renderer)
    
    def delete(self, renderer):
        pass


class VisualizationView(scenes.View):
    @property
    def hovering(self):
        return self._hovering
        
    @hovering.setter
    def hovering(self, value):
        if self._hovering is not value:
            self._hovering = value
            self.data_changed(self, ("hovering",), value)
    
    @property
    def dragging(self):
        return self._dragging
    
    @dragging.setter
    def dragging(self, value):
        if self._dragging is not value:
            self._dragging = value
            self.data_changed(self, ("dragging",), value)
    
    @property
    def accent(self):
        return self._accent
    
    @accent.setter
    def accent(self, value):
        if self._accent is not value:
            self._accent = value
            self.data_changed(self, ("accent",), value)
    
    @property
    def status(self):
        return self._status
    
    @status.setter
    def status(self, value):
        if self._status is not value:
            self._status = value
            self.data_changed(self, ("status",), value)
    
    @property
    def preserve_aspect(self):
        return self._preserve_aspect
    
    @preserve_aspect.setter
    def preserve_aspect(self, value):
        if self._preserve_aspect is not value:
            self._preserve_aspect = value
            self.data_changed(self, ("preserve_aspect",), value)
    
    @property
    def orbit(self):
        return self._orbit
        
    @orbit.setter
    def orbit(self, value):
        if self._orbit is not value:
            self._orbit = value
            self.data_changed(self, ("orbit",), value)
    
    @property
    def track(self):
        return self._track
        
    @track.setter
    def track(self, value):
        if self._track is not value:
            self._track = value
            self.data_changed(self, ("track",), value)
    
    @property
    def perspective(self):
        return self._camera._projection.perspective
        
    @perspective.setter
    def perspective(self, value):
        self._camera._projection.perspective = value
    
    @property
    def aspect(self):
        return self._aspect1
    
    @aspect.setter
    def aspect(self, value):
        self._aspect0, self._aspect1 = self._aspect1, value
    
    @property
    def arrangement(self):
        return self._arrangement1

    @arrangement.setter
    def arrangement(self, value):
        self._arrangement0, self._arrangement1 = self._arrangement1, value
        
        self.update_transforms()
    
    @property
    def transforms(self):
        return self._transforms1
    
    @property
    def model(self):
        return self._model
    
    @model.setter
    def model(self, value):
        self._model = value
        self._model.data_changed.subscribe(self.on_data_changed)
        self._model.axis_data_changed.subscribe(self._axes.on_data_changed)
        self._model.mark_data_changed.subscribe(self._marks.on_data_changed)
        self._model.binding_changed.subscribe(self.on_binding_changed)
    
    @property
    def color_provider(self):
        return self._color_provider
    
    @color_provider.setter
    def color_provider(self, value):
        self._color_provider = value
    
    @property
    def shape_provider(self):
        return self._shape_provider
    
    @shape_provider.setter
    def shape_provider(self, value):
        self._shape_provider = value
    
    @property
    def texture_provider(self):
        return self._texture_provider
    
    @texture_provider.setter
    def texture_provider(self, value):
        self._texture_provider = value
    
    def __init__(self, model=None, clock=None, size=None, parent=None):
        super().__init__(clock=clock, size=size, parent=parent)
        
        self.data_changed = observables.Observable()
        
        self._environment = BackgroundView(None, parent=self)
        self._axes = AxesView(parent=self)
        self._marks = MarksView(parent=self)
        
        self._accent = colors.svg.red
        self._status = "Drop data onto canvas to compose visualization, touch canvas to interactively explore."
        
        self.origin_size = 1.0/32.0
        self.axis_size = 1.0/128.0
        self.mark_size = 1.0/16.0
        self.tick_size = 1.0/32.0
        self.label_size = 1.0
        
        self._hovering = False
        self._dragging = False
        self._dropping = False
        
        self._model = model
        
        self._preserve_aspect = True
        self._track = True
        self._orbit = False
        
        self._hovering_origins = set()
        self._hovering_axes = set()
        self._hovering_marks = set()
        
        self._view_extent = extents.Extent(-vectors.ones(), +vectors.ones())
        self._model_extent = extents.Extent(-vectors.ones(), +vectors.ones())
                
        self._color_canvas = None
        self._shape_canvas = None
        self._texture_canvas = None
        
        self._environment_textures = collections.OrderedDict([("none", None)])
        self._tile_textures = collections.OrderedDict([("none", None)])
        
        self._orientations = [vectors.zeros(*([np.pi/2]*i), n=mdefaults.DEFAULT_N) for i in range(mdefaults.DEFAULT_N)]
        #self._orientations = [vectors.zeros(n=mdefaults.DEFAULT_N)] + [vectors.unit(i)*np.pi/2 for i in range(mdefaults.DEFAULT_N-1)]
        self._current_orientation = 0
        
        self._current_environment_index = 0
        self._current_tile_index = 0
        
        self._lights = lights.MultiLight([lights.Light() for _ in range(GLANCE_LIGHT_COUNT)])
        
        self._lights[0] = lights.Light(colors.svg.white.darken(0.5), colors.svg.white.darken(0.5), colors.svg.white.darken(0.75))
        self._lights[0].position = np.asarray([+0.25, +0.5, +1.0, 0.0], dtype=np.float32)
        
        '''
        for light, color in zip(self.lights, [colors.SVG.LightGreen, colors.SVG.LightSkyBlue, colors.SVG.LightSalmon]):
            light.ambient = colors.SVG.Black
            light.diffuse = color.darken(0.25)
            light.specular = color.darken(0.50)
            light.attenuation =  np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        '''
        
        self.clear()
    
    def current_aspect(self, time):
        value = vectors.interpolate_linear(self._aspect0, self._aspect1)
        return value(self._gamma(time))
    
    def current_transforms(self, time):
        value = vectors.interpolate_linear(self._transforms0, self._transforms1)
        return value(self._beta(time))
    
    def add_environment_provider(self, key, provider):
        texture = textures.Texture(provider, wrap=(GL.GL_CLAMP_TO_EDGE,)*3)
        self._environment_textures[key] = texture
    
    def add_tile_provider(self, key, provider):
        texture = textures.Texture(provider)
        self._tile_textures[key] = texture
    
    def select_environment(self, index):
        self._current_environment_index = index
        keys = list(self._environment_textures.keys())
        if len(keys) > 0:
            self._environment.texture = self._environment_textures[keys[self._current_environment_index % len(keys)]]
    
    def process(self, input):
        hovering_origins = set()
        hovering_axes = set()
        hovering_marks = set()
        
        for touch in input.touches.values():
            type = _identifier_type.decode(touch.target)
            if type == ORIGIN_TYPE:
                hovering_origins.add(touch.target)
                continue
            if type == AXIS_TYPE:
                hovering_axes.add(touch.target)
                continue
            if type == MARK_TYPE:
                hovering_marks.add(touch.target)
                continue
        
        self.model.selection -= self._hovering_marks
        
        self._hovering_origins = hovering_origins
        self._hovering_axes = hovering_axes
        self._hovering_marks = hovering_marks
        
        self.model.selection |= self._hovering_marks
        
        delta = vectors.zeros(n=mdefaults.DEFAULT_N)
        if input.keys.get(inputs.Key.Left) or input.keys.get(inputs.Key.Right):
            for key in range(inputs.Key._1, inputs.Key._1 + mdefaults.DEFAULT_N):
                if key in input.keys:
                    if input.keys.get(inputs.Key.Left):
                        delta[key - inputs.Key._1] -= 1.0 * self._camera_nd._transform.scaling[key - inputs.Key._1]
                    if input.keys.get(inputs.Key.Right):
                        delta[key - inputs.Key._1] += 1.0 * self._camera_nd._transform.scaling[key - inputs.Key._1]
        
        self._camera_nd._transform._delta_translation = delta
        
        if self._orbit and not self._hovering:
            self._root._transform._delta_rotation = vectors.zeros(0.25, 0, 0, n=3)
        else:
            self._root._transform._delta_rotation = vectors.zeros(0.00, 0, 0, n=3)
        
        self._axes.process(input)
        self._marks.process(input)
        
        #self.hovering = bool(self._hovering_origins) or bool(self._hovering_axes) or bool(self._hovering_marks)
        #print(self.hovering)
    
    def on_data_changed(self, obj, key, value):
        time = self.time
        
        if "arrangement" in key:
            self.arrangement = value
            self._beta.play(time)
        
    def on_binding_changed(self, obj):
        self._marks.on_binding_changed(obj)
        
        self.update_transforms()
    
    def on_key_press(self, state):
        pass
    
    def on_key_release(self, state):
        '''
        is_valid_dimension = predicates.between(inputs.Key._1, inputs.Key._1 + mdefaults.DEFAULT_N)
        if is_valid_dimension(state.identifier):
            i = state.identifier - inputs.Key._1
            self._betas[i].reverse(self.time)
        '''
        if state.identifier == inputs.Key.X:
            self._gamma.reverse(self.time)
        if state.identifier == inputs.Key.A:
            self._current_orientation = (self._current_orientation - 1) % len(self._orientations)
            self._camera_nd._transform.rotation = self._orientations[self._current_orientation]
            self._alpha_transform.play(self.time)
        if state.identifier == inputs.Key.D:
            self._current_orientation = (self._current_orientation + 1) % len(self._orientations)
            self._camera_nd._transform.rotation = self._orientations[self._current_orientation]
            self._alpha_transform.play(self.time)
        if state.identifier == inputs.Key.N:
            self.select_environment(self._current_environment_index + 1)
        if state.identifier == inputs.Key.M:
            self.select_environment(self._current_environment_index - 1)
    
    def on_drop(self, state):
        drop_type = _identifier_type.decode(state.target) # & self._type_identifier_mask
        drop_data = _identifier_data.decode(state.target) # & self._data_identifier_mask
        
        expression = state.data
        owner = expressions.ownerof(expression)
        
        if owner is None:
            logger.warn('Ignoring ownerless expression "{}".'.format(expression))
        else:
            key = owner.title.lower()
            
            '''
            for face in owner.faces[1]:
                key0 = face[0].title.lower()
                key1 = face[1].title.lower()
                mark0 = accessors.getitem(self._model.marks, key0, None)
                mark1 = accessors.getitem(self._model.marks, key1, None)
                if not (mark0 is None or mark1 is None):
                    mark = self._model.get_or_create_mark(key, order=1, owner=owner, options={"texture": self._tile_textures})
            
            for face in owner.faces[0]:
                key0 = face[0].title.lower()
                mark0 = accessors.getitem(self._model.marks, key0, None)
                if not (mark0 is None):
                    mark = self._model.get_or_create_mark(key, order=0, owner=owner, options={"texture": self._tile_textures})
            '''
            
            mark = self._model.get_or_create_mark(key, order=0, owner=owner, options={"texture": self._tile_textures})
            
            if drop_type == ORIGIN_TYPE:
                self._model.auto_bind_origin(expression, mark)
            elif drop_type == AXIS_TYPE:
                axis = self._model.get_axis_by_index(drop_data)
                self._model.auto_bind_axis(expression, mark, axis)
            elif drop_type == MARK_TYPE:
                self._model.auto_bind_mark(expression, mark)
            else:
                self._model.auto_bind(expression, mark)
    
    def clear(self):
        self._alpha = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        self._beta = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        self._gamma = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        
        self._alphas = [animations.Animation(0.0, 1.0, ease=animations.ease_sine_in_out, reversed=False) for _ in range(mdefaults.DEFAULT_N)]
        self._betas = [animations.Animation(0.0, 1.0, ease=animations.ease_sine_in_out, reversed=True) for _ in range(mdefaults.DEFAULT_N)]
        
        self._alpha_transform = animations.Animation(0.0, 1.0, ease=animations.ease_cubic_out)
        
        self._camera = scenes.Camera(transform=scenes.Transform(reversed=True, n=3, m=4), projection=scenes.Projection(fov=np.pi/1.5, n=3, m=4))
        self._camera._transform.translate(vectors.zeros(0, 0, 1.5, n=3))
        self._camera._transform.scale(vectors.full(np.pi, n=3))
        
        self._camera_nd = scenes.Camera() #transform=scenes.Transform(reversed=False)
        #self._camera_nd._transform._delta_rotation = vectors.zeros(0.0, 1.0, 0.0, n=mdefaults.DEFAULT_N)
        
        self._root = scenes.Node(transform=scenes.Transform(n=3, m=4))
        self._root_nd = scenes.Node()
        
        self._aspect0 = vectors.full(np.pi/2.0)
        self._aspect1 = vectors.full(np.pi/2.0)
        self._aspect1[0] *= 2.0
        
        self._arrangement0 = models.Arrangement.Perpendicular
        self._arrangement1 = models.Arrangement.Perpendicular
        
        self._view_extent = extents.Extent(-vectors.ones(), +vectors.ones())
        self._model_extent = extents.Extent(-vectors.ones(), +vectors.ones())
        
        self._axes.clear()
        self._marks.clear()
    
    def scale(self, value):
        self._camera._transform.scale(vectors.full(value, n=3))
        #self._camera_nd._transform.scale(value)
    
    def translate(self, value):
        self._camera_nd._transform.translate(value)
    
    def update_camera_transform(self, translation, scaling):
        '''
        a = transforms.translate(-src_center)
        b = transforms.scale(2.0/src_delta)
        c = transforms.scale(dst_delta/2.0)
        d = transforms.translate(+dst_center)
        print(transforms.inversed(vectors.dot(a,b,c,d)))
        print(-translation, 1.0/scaling)
        '''
        
        self._root_nd._transform.translation = translation * scaling
        self._root_nd._transform.scaling = scaling
        
        #self._camera_nd._transform.translation = -translation
        #self._camera_nd._transform.scaling = 1.0 / scaling
        
        #print(translation, scaling)
    
    def update_transforms(self):
        default_transforms = {
            models.Arrangement.Perpendicular: lambda n=mdefaults.DEFAULT_N: [transforms.identity()],
            models.Arrangement.PerpendicularDisjoint: lambda n=mdefaults.DEFAULT_N: list(coordinates.perpendicular_axes(vectors.zeros(n=mdefaults.DEFAULT_N), n=n)),
            models.Arrangement.Parallel: lambda n=mdefaults.DEFAULT_N: list(coordinates.parallel_axes(vectors.unit(0, n=mdefaults.DEFAULT_N) * np.pi, n=n)),
            models.Arrangement.Multiple: lambda n=mdefaults.DEFAULT_N: list(coordinates.multiple_axes(vectors.units([0,1], n=mdefaults.DEFAULT_N) * np.pi, n=n)),
        }
        
        def pad(items, count):
            i = len(items)
            if i == 0:
                return [transforms.identity()] * count
            elif i > count:
                return items[:count]
            elif i < count:
                return items + [items[-1]] * (count - i)
            return items
        
        def get_transforms(arrangement):
            if config.enable_axis_arrangements: 
                n = len(self.model.axes)
                m = max(1, max(n, (n*(n-1))//2))
                return pad(default_transforms[arrangement](n), m)
            return [transforms.identity()]
        
        self._transforms0 = get_transforms(self._arrangement0)
        self._transforms1 = get_transforms(self._arrangement1)
        
        _transforms0_data = np.ascontiguousarray(list(map(vectors.default_chunk2d, self._transforms0)), dtype=mdefaults.DEFAULT_DTYPE)
        _transforms1_data = np.ascontiguousarray(list(map(vectors.default_chunk2d, self._transforms1)), dtype=mdefaults.DEFAULT_DTYPE)
        
        self._transforms0_data.data = _transforms0_data
        self._transforms1_data.data = _transforms1_data
        
        #print(self._transforms0)
        #print(self._transforms1)
    
    def zoom_extent(self):
        marks_extent = self._marks.extent
        
        self._model_extent = extents.Extent(vectors.full(np.nan, *(marks_extent.lower)), vectors.full(np.nan, *(marks_extent.upper)))
        self._view_extent = extents.Extent(-vectors.vector(self.aspect), +vectors.vector(self.aspect))
        
        transform = extents.transform_extent_min if self._preserve_aspect else extents.transform_extent
        self.update_camera_transform(*transform(self._model_extent,  self._view_extent))
    
    def zoom_extent_preserve_aspect_min(self):
        marks_extent = self._marks.extent
        
        self._model_extent = extents.Extent(vectors.full(np.nan, *(marks_extent.lower)), vectors.full(np.nan, *(marks_extent.upper)))
        self._view_extent = extents.Extent(-vectors.vector(self.aspect), +vectors.vector(self.aspect))
        
        self.update_camera_transform(*extents.transform_extent_min(self._model_extent,  self._view_extent))
    
    def zoom_extent_preserve_aspect_max(self):
        marks_extent = self._marks.extent
        
        self._model_extent = extents.Extent(vectors.full(np.nan, *(marks_extent.lower)), vectors.full(np.nan, *(marks_extent.upper)))
        self._view_extent = extents.Extent(-vectors.vector(self.aspect), +vectors.vector(self.aspect))
    
        self.update_camera_transform(*extents.transform_extent_max(self._model_extent,  self._view_extent))
    
    def create(self, renderer):
        self._transforms0_data = buffers.BufferData()
        self._transforms0_buffer = buffers.UniformBuffer(self._transforms0_data)
        
        self._transforms1_data = buffers.BufferData()
        self._transforms1_buffer = buffers.UniformBuffer(self._transforms1_data)
        
        self.update_transforms()
        
        #s = vectors.full(0.25, n=mdefaults.DEFAULT_N)
        #self.transforms = [transforms.translate_scale([-np.pi,-np.pi], s), transforms.translate_scale([0.0], s), transforms.translate_scale([+np.pi,+np.pi], s)]
        #self.transforms = [coordinates.perpendicular_axis(0, vectors.zeros(n=mdefaults.DEFAULT_N)), coordinates.perpendicular_axis(1, vectors.zeros(n=mdefaults.DEFAULT_N)), coordinates.perpendicular_axis(2, vectors.zeros(n=mdefaults.DEFAULT_N))]
        #self.transforms = [coordinates.perpendicular_axis(1, -vectors.unit(0, n=mdefaults.DEFAULT_N)), coordinates.perpendicular_axis(1, vectors.zeros(n=mdefaults.DEFAULT_N)), coordinates.perpendicular_axis(1, +vectors.unit(0, n=mdefaults.DEFAULT_N))]
        #self.transforms = list(coordinates.perpendicular_axes(vectors.zeros(n=mdefaults.DEFAULT_N)))
        
        self.select_environment(self._current_environment_index)
        
        self._environment.create(renderer)
        self._axes.create(renderer)
        self._marks.create(renderer)
    
    def update(self, renderer):
        time = self.time
        delta = 1.0/30.0
        
        self._alphas_nd = vectors.full(1.0, *[animation(time) for animation in self._alphas])
        self._betas_nd = vectors.full(0.0, *[animation(time) for animation in self._betas])
        
        self._camera.update(delta)
        self._camera_nd.update(delta)
        
        self._root.update(delta)
        self._root_nd.update(delta)
        
        #model_nd = transforms.scale([animation(time) for animation in self._alphas])
        #view_nd = transforms.identity()
        #projection_nd = transforms.identity()
        
        if self._track:
            aspect = self.current_aspect(time)
            marks_extent = self._marks.current_extent(time)
            
            self._model_extent = extents.Extent(vectors.full(np.nan, *(marks_extent.lower)), vectors.full(np.nan, *(marks_extent.upper)))
            self._view_extent = extents.Extent(-vectors.vector(aspect), +vectors.vector(aspect))
            
            transform = extents.transform_extent_min if self._preserve_aspect else extents.transform_extent
            self.update_camera_transform(*transform(self._model_extent,  self._view_extent))
        
        self._environment.update(renderer)
        self._axes.update(renderer)
        self._marks.update(renderer)
        
        #gl.cleanup(context=renderer.context)
    
    def render(self, renderer):
        #GL.glFrontFace(GL.GL_CW)
        #GL.glDepthFunc(GL.GL_GREATER)
        #GL.glDepthRange(1.0, 0.0)
        
        #GL.glDepthFunc(GL.GL_GREATER)
        GL.glEnable(GL.GL_DEPTH_TEST)
        
        gl.clear_color([0.0, 0.0, 0.0, 0.25])
        #gl.clear_depth(0.0)
        gl.clear()
        
        if self._model.show_background:
            self._environment.render(renderer)
        
        if self._model.show_axes or (self._dragging or self._dropping):
            self._axes.render(renderer)
        
        self._marks.render(renderer)
    
    def delete(self, renderer):
        self._marks.update(renderer)
        self._axes.update(renderer)
        self._background.delete(renderer)
