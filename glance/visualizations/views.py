import collections
import enum
import logging

import numpy as np

from glue import gl
from glue.gl import GL

from encore import accessors, generators, iterables, mappings, objects, utilities

import medley as md
from medley import expressions

from ..mathematics import defaults as mdefaults, coordinates, extents, mathematics, projections, transforms, vectors
from ..mathematics.geometries import cubes
from ..graphics import defaults as gdefaults, buffers, renderers, shaders, textures, vertices
from ..scenes import actors, animations, inputs, lights, materials, geometries, scenes

from .. import colors, shapes, palettes
from .. import painters

logger = logging.getLogger(__name__.split(".").pop())


GLANCE_LIGHT_COUNT = 3

DEFAULT_VERTEX_FORMAT = vertices.DEFAULT_FORMAT | vertices.Component.Transform
DEFAULT_VERTEX_ND_FORMAT = vertices.FORMAT_SIZE_OFFSET | vertices.Component.Identifier

DEFAULT_PROGRAM_DEFINES = mappings.DefineMap({
    "GLANCE_ALPHA_TEST": 1,
    "GLANCE_COLOR": 2,
    "GLANCE_LIGHT_COUNT": GLANCE_LIGHT_COUNT,
    "GLANCE_MATERIAL_COUNT": 2,
    "GLANCE_MATERIAL_SIDES": 1,
    "GLANCE_MATERIAL_FORMAT": materials.DEFAULT_FORMAT,
    "GLANCE_MATERIAL_COLOR_AMBIENT": 1,
    "GLANCE_MATERIAL_COLOR_DIFFUSE": 1,
    "GLANCE_TESS_LEVEL_INNER": 8,
    "GLANCE_TESS_LEVEL_OUTER": 8,
    "GLANCE_VERTEX_FORMAT": DEFAULT_VERTEX_FORMAT,
    "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
    "M_N": mdefaults.DEFAULT_N,
})

def unpack_args(func):
    def _unpack_args(key):
        return func(*key)
    return _unpack_args


def simplex0_nd_program(material_format=materials.DEFAULT_FORMAT, vertex_format=DEFAULT_VERTEX_ND_FORMAT):
    return gdefaults.thick_simplex_nd_program(material_format, vertex_format, 0, space=renderers.Space.View)


def simplex1_nd_program(material_format=materials.DEFAULT_FORMAT, vertex_format=DEFAULT_VERTEX_ND_FORMAT):
    return gdefaults.thick_simplex_nd_program(material_format, vertex_format, 1, space=renderers.Space.View)


def shaded_surface_program(material_format, shade):
    return shaders.Program.from_files(["default_nd.vs", "surface_nd.gs", "shade_phong.fs"], defines={
        "GLANCE_COLOR": 1,
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


def axis_program(material_format):
    return shaders.Program.from_files(["default_nd.vs", "thick_simplex1_nd.gs", "default.fs"], defines={
        "GLANCE_COLOR": 1,
        "GLANCE_LIGHT_COUNT": GLANCE_LIGHT_COUNT,
        "GLANCE_MATERIAL_FORMAT": material_format,
        "GLANCE_MATERIAL_COLOR_AMBIENT": 1,
        "GLANCE_MATERIAL_COLOR_DIFFUSE": 1,
        "GLANCE_MATERIAL_SIDES": 1,
        "GLANCE_VERTEX_FORMAT": DEFAULT_VERTEX_FORMAT,
        "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
        "M_N": mdefaults.DEFAULT_N,
    })


def tick_program(material_format):
    return shaders.Program.from_files(["default_nd.vs", "axes_nd.gs", "default.fs"], defines={
        "GLANCE_COLOR": 1,
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


def mark_program(material_format, shade, i):
    return shaders.Program.from_files([
        "default_nd.vs",
        "mark{}_nd.gs".format(i),
        "shade_phong.fs",
    ], defines=DEFAULT_PROGRAM_DEFINES + {
        "GLANCE_MARK_DIMENSION": 2,
        "GLANCE_MATERIAL_TEXTURE_AMBIENT": 1,
        "GLANCE_MATERIAL_TEXTURE_DIFFUSE": 1,
        "GLANCE_SHADE": int(shade),
        "GLANCE_SPACE": int(renderers.Space.View),
    })


def mark_instance_program(material_format, shade, i):
    return shaders.Program.from_files([
        "default_instanced_nd.vs",
        "tessellate_simplex2_instanced_nd.tcs",
        "tessellate_simplex2_instanced.tes",
        #"tessellate_simplex2_instanced_nd.tes",
        #"extrude_instanced_nd.gs",
        "shade_phong.fs",
    ], defines=DEFAULT_PROGRAM_DEFINES + {
        "GLANCE_MARK_DIMENSION": 2,
        "GLANCE_MATERIAL_COUNT": 2,
        "GLANCE_MATERIAL_SIDES": 2,
        "GLANCE_SHADE": int(shade),
        "GLANCE_SPACE": int(renderers.Space.View),
    })

def mark_interval_program(material_format, shade, i):
    return shaders.Program.from_files([
        "default_nd.vs",
        "tessellate_simplex0_to_cube{}_nd.tcs".format(i),
        "tessellate_simplex0_to_cube{}_nd.tes".format(i),
        "extrude_simplex{}_nd.gs".format(i),
        "shade_phong.fs"
    ], defines=DEFAULT_PROGRAM_DEFINES + {
        "GLANCE_MARK_DIMENSION": 2,
        "GLANCE_MATERIAL_COUNT": 2,
        "GLANCE_MATERIAL_SIDES": 2,
        "GLANCE_SHADE": int(shade),
        "GLANCE_SPACE": int(renderers.Space.View),
    })


DEFAULT_MARK_PROGRAMS = mappings.CustomMap(factory=unpack_args(mark_program))
DEFAULT_MARK_INSTANCE_PROGRAMS = mappings.CustomMap(factory=unpack_args(mark_instance_program))
DEFAULT_MARK_INTERVAL_PROGRAMS = mappings.CustomMap(factory=unpack_args(mark_interval_program))


DEFAULT_SHADED_SURFACE_PROGRAMS = mappings.CustomMap(factory=unpack_args(shaded_surface_program))

DEFAULT_SURFACE_PROGRAMS = {
    False: DEFAULT_SHADED_SURFACE_PROGRAMS[materials.DEFAULT_FORMAT, 0],
    True: DEFAULT_SHADED_SURFACE_PROGRAMS[materials.DEFAULT_FORMAT, renderers.Shade.Phong],
}

DEFAULT_SIMPLEX0_PROGRAMS = mappings.CustomMap(factory=unpack_args(simplex0_nd_program))
DEFAULT_SIMPLEX1_PROGRAMS = mappings.CustomMap(factory=unpack_args(simplex1_nd_program))

DEFAULT_SIMPLEX0_PROGRAM = DEFAULT_SIMPLEX0_PROGRAMS[()]
DEFAULT_SIMPLEX1_PROGRAM = DEFAULT_SIMPLEX1_PROGRAMS[()]

DEFAULT_AXIS_PROGRAM = axis_program(materials.DEFAULT_FORMAT)
DEFAULT_TICK_PROGRAM = tick_program(materials.DEFAULT_FORMAT)


DEFAULT_MATERIAL_NORMAL_RGB = materials.PhongMaterial(colors={
    "ambient": colors.svg.white.rgb,
    "diffuse": colors.svg.white.rgb,
    "specular": colors.svg.black.rgb,
    "emissive": colors.svg.black.rgb,
}, opacity=1.0, shininess=32.0)

DEFAULT_MATERIAL_HOVER_RGB = materials.PhongMaterial(colors={
    "ambient": colors.Color.from_hex("#66cc66").rgb,
    "diffuse": colors.Color.from_hex("#66cc66").rgb,
    "specular": colors.svg.black.rgb,
    "emissive": colors.svg.black.rgb,
}, opacity=1.0, shininess=32.0)

DEFAULT_MATERIAL_NORMAL_HSV = materials.PhongMaterial(colors={
    "ambient": colors.svg.white.hsv,
    "diffuse": colors.svg.white.hsv,
    "specular": colors.svg.black.hsv,
    "emissive": colors.svg.black.hsv,
}, opacity=1.0, shininess=32.0)

DEFAULT_MATERIAL_HOVER_HSV = materials.PhongMaterial(colors={
    "ambient": colors.Color.from_hex("#66cc66").hsv,
    "diffuse": colors.Color.from_hex("#66cc66").hsv,
    "specular": colors.svg.black.hsv,
    "emissive": colors.svg.black.hsv,
}, opacity=1.0, shininess=32.0)


def axes_nd_program(key):
    vertex_format, wrap, transform_count = key
    return shaders.Program.from_files(["axes_nd.cs"], defines={
        "GLANCE_COLOR": 2,
        "GLANCE_TRANSFORM_COUNT": transform_count,
        "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
        "GLANCE_WORK_GROUP_SIZE": gdefaults.DEFAULT_WORK_GROUP_SIZE,
        "GLANCE_WRAP": int(wrap),
        "M_N": mdefaults.DEFAULT_N,
    })


def axes_interpolate_nd_program(key):
    vertex_format, wrap, transform_count = key
    return shaders.Program.from_files(["axes_interpolate_nd.cs"], defines={
        "GLANCE_COLOR": 2,
        "GLANCE_TRANSFORM_COUNT": transform_count,
        "GLANCE_VERTEX_ND_FORMAT": DEFAULT_VERTEX_ND_FORMAT,
        "GLANCE_WORK_GROUP_SIZE": gdefaults.DEFAULT_WORK_GROUP_SIZE,
        "GLANCE_WRAP": int(wrap),
        "M_N": mdefaults.DEFAULT_N,
    })


DEFAULT_AXES_PROGRAMS = mappings.CustomMap(factory=axes_nd_program)
DEFAULT_AXES_INTERPOLATE_PROGRAMS = mappings.CustomMap(factory=axes_interpolate_nd_program)


def replace(predicate, value):
    def _replace(x):
        return value if predicate(x) else x
    return _replace



class BackgroundView(scenes.View):
    def __init__(self, path, size=None, parent=None):
        super().__init__(size=size, parent=parent)
        
        self._path = path

    def create(self, renderer):
        self._texture = textures.Texture.from_files(textures.environment_paths(self._path), target=GL.GL_TEXTURE_CUBE_MAP, wrap=(GL.GL_CLAMP_TO_EDGE,)*3)
        self._texture.prepare(renderer)
        
        self._material = materials.CustomMaterial(textures={"background": self._texture})
        self._material.prepare(renderer)

        self._geometry = geometries.CustomGeometry()
        self._geometry.prepare(renderer)

    def render(self, renderer):
        model = self.parent._root._transform.to_matrix(n=3, m=4)
        view = vectors.inverse(self.parent._camera._transform.to_matrix(n=3, m=4))
        projection = projections.perspective(np.pi/2.0, 0.1, 1000.0) #self.parent._camera._projection.to_matrix(n=3, m=4)
        projection = np.dot(transforms.scale(1.0/self.aspect, n=4), projection)
        
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

        
class AxisActor(actors.DoubleBufferComputeActor):
    pass


class AxesView(scenes.View):
    @property
    def axes(self):
        return self._axes_input_vertex_data.data

    @axes.setter
    def axes(self, value):
        self._axes_input_vertex_data.data = value

        self._axes_output_vertex_data.size = self._axes_input_vertex_data.size * self.coordinates_count
        self._axes_output_vertex_data.type = self._axes_input_vertex_data.type
    
    def clear(self):
        pass
    
    def create_axes(self, renderer):
        self._axes_actor = actors.SingleBufferComputeActor()
        
        self.origin_count = 1
        self.origin_offset = 0
        self.origin_stride = 1
        
        self.axis_count = mdefaults.DEFAULT_N
        self.axis_offset = self.origin_offset + self.origin_count * self.origin_stride
        self.axis_stride = 2
        
        self.tick_count = mdefaults.DEFAULT_N
        self.tick_offset = self.axis_offset + self.axis_count * self.axis_stride
        self.tick_stride = 10
        
        def axis_identifiers():
            for identifier in iterables.repeated(generators.autoincrement(), self.axis_stride):
                yield self.parent._axes_identifier | identifier
        
        def tick_identifiers():
            for identifier in iterables.repeated(generators.autoincrement(), self.tick_stride):
                yield self.parent._axes_identifier | identifier
        
        origin_data = vertices.full({
            vertices.Component.Position: origin_generator(),
            vertices.Component.Identifier: generators.autoincrement(),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        
        axis_data = vertices.full({
            vertices.Component.Position: axes_generator(self.parent._view_aspect),
            vertices.Component.Identifier: axis_identifiers(),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        
        tick_data = vertices.full({
            vertices.Component.Position: ticks_generator(self.parent._view_aspect, self.tick_stride),
            vertices.Component.Identifier: tick_identifiers(),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        
        vertex_data = vertices.concatenate(origin_data, axis_data, tick_data)
        #print(vertex_data)
        
        self._axes_actor = AxisActor()
        self._axes_actor._transforms = self.parent._transforms
        
        actor = self._axes_actor
        
        if actor._input0_vertex_data.data is None:
            actor.update_vertex_data(vertex_data)
            actor.update_vertex_data(vertex_data)
        else:
            actor.update_vertex_data(vertex_data)
    
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
        
        self._material_hover = materials.MultiMaterial([DEFAULT_MATERIAL_HOVER_RGB, DEFAULT_MATERIAL_HOVER_RGB])
        self._material_hover.prepare(renderer)
        
        self._origin_geometry = geometries.CustomGeometry()
        self._origin_geometry.prepare(renderer)
        
        self.create_axes(renderer)
    
    def update(self, renderer):
        time = self.time
        
        model_nd = self.parent._root_nd._transform.to_matrix()
        view_nd = vectors.inverse(self.parent._camera_nd._transform.to_matrix())
        
        wrap = 0 if self.parent.model.show_overflow else renderers.Wrap.ClampRepeat
        compute_program = DEFAULT_AXES_INTERPOLATE_PROGRAMS[DEFAULT_VERTEX_ND_FORMAT, wrap, len(self.parent._transforms)]
        with renderer.activated(compute_program, self.parent._transforms_buffer):
            renderer.set_model_view_nd(model_nd, view_nd)
            
            renderer.uniforms["buffer_offset"] = GL.GLuint(0)
            renderer.uniforms["buffer_stride"] = GL.GLuint(1)
            
            renderer.uniform_blocks["TransformsData"] = self.parent._transforms_buffer
            
            renderer.set_uniform_array_1d("aspect_nd", vectors.default_chunk1d(self.parent._view_aspect))
            
            renderer.uniforms["alpha"] = GL.GLfloat(self.parent._alpha(time))
            
            actor = self._axes_actor
            with renderer.activated(actor._input0_vertex_buffer, actor._input1_vertex_buffer, actor._output_vertex_buffer):
                renderer.shader_storage_blocks["VertexSrcData0"] = actor._input0_vertex_buffer
                renderer.shader_storage_blocks["VertexSrcData1"] = actor._input1_vertex_buffer
                renderer.shader_storage_blocks["VertexDstData"] = actor._output_vertex_buffer
                
                # Origin
                renderer.set_uniform_array_1d("wrap_mask", vectors.default_chunk1d(vectors.zeros()))
                
                renderer.uniforms["buffer_offset"] = GL.GLuint(self.origin_offset)
                
                GL.glDispatchCompute(mathematics.chunk_count(self.origin_stride, gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                
                # Axes
                for i in range(self.axis_count):
                    renderer.set_uniform_array_1d("wrap_mask", vectors.default_chunk1d(vectors.zeros()))
                    
                    renderer.uniforms["buffer_offset"] = GL.GLuint(self.axis_offset + i*self.axis_stride)
                    
                    GL.glDispatchCompute(mathematics.chunk_count(self.axis_stride, gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
                
                # Ticks
                for i in range(self.tick_count):
                    renderer.set_uniform_array_1d("wrap_mask", vectors.default_chunk1d(vectors.unit(i)))
                    
                    renderer.uniforms["buffer_offset"] = GL.GLuint(self.tick_offset + i*self.tick_stride)
                    
                    GL.glDispatchCompute(mathematics.chunk_count(self.tick_stride, gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
    
    def render(self, renderer):
        projection_nd = projections.diagonal(3)
        
        model = self.parent._root._transform.to_matrix(n=3, m=4)
        view = vectors.inverse(self.parent._camera._transform.to_matrix(n=3, m=4))
        projection = self.parent._camera._projection.to_matrix(n=3, m=4)
        projection = np.dot(transforms.scale(1.0/self.aspect, n=4), projection)
        
        origin_program = DEFAULT_SURFACE_PROGRAMS[self.parent.model.show_shading]
        with renderer.activated(origin_program, self.parent._lights, self._origin_geometry):
            renderer.set_projection_nd(projection_nd)
            renderer.set_model_view_projection(model, view, projection)
            
            actor = self._axes_actor
            
            material = self._material_hover if len(self.parent._hovering_origins) else self._material_normal
            geometry = actor._geometry
            with renderer.activated(material, geometry):
                renderer.uniforms['identifier'] = GL.GLuint(self.parent._origin_identifier)
                renderer.uniforms['size'] = GL.GLfloat(self.parent.origin_size)
                
                geometry.draw_indices("default", mode=GL.GL_POINTS, first=self.origin_offset, count=self.origin_stride, renderer=renderer)
        
        
        axis_program = DEFAULT_AXIS_PROGRAM
        with renderer.activated(axis_program, self.parent._lights, self._axes_actor._geometry):
            renderer.set_projection_nd(projection_nd)
            renderer.set_model_view_projection(model, view, projection)
            
            actor = self._axes_actor
            #'''
            root = self.parent.model.axes.get('root')
            if root is not None:
                for i, axis in enumerate(root.values()):
                    material = self._material_hover if i in self.parent._hovering_axes else self._material_normal
                    geometry = actor._geometry
                    with renderer.activated(material, geometry):
                        renderer.uniforms['identifier'] = GL.GLuint(self.parent._axes_identifier | i)
                        renderer.uniforms['size'] = GL.GLfloat(self.parent.axis_size)
                        
                        geometry.draw_indices("default", mode=GL.GL_LINES, first=self.axis_offset + i*self.axis_stride, count=self.axis_stride, renderer=renderer)
            '''
            i = 0
            material = self._material_hover if i in self.parent._hovering_axes else self._material_normal
            geometry = actor._geometry
            with renderer.activated(material, geometry):
                #renderer.uniforms['identifier'] = GL.GLuint(self.parent._axes_identifier | i)
                renderer.uniforms['size'] = GL.GLfloat(self.parent.axis_size)
                
                geometry.draw_indices("default", mode=GL.GL_LINES, first=self.axis_offset, count=self.axis_count * self.axis_stride, renderer=renderer)
            #'''
        
        tick_program = DEFAULT_TICK_PROGRAM
        with renderer.activated(tick_program, self.parent._lights):
            renderer.set_projection_nd(projection_nd)
            renderer.set_model_view_projection(model, view, projection)
            
            actor = self._axes_actor
            #'''
            root = self.parent.model.axes.get('root')
            if root is not None:
                for i, axis in enumerate(root.values()):
                    material = self._material_hover if i in self.parent._hovering_axes else self._material_normal
                    geometry = actor._geometry
                    with renderer.activated(material, geometry):
                        renderer.uniforms['identifier'] = GL.GLuint(self.parent._axes_identifier | i)
                        renderer.uniforms['size'] = GL.GLfloat(self.parent.tick_size)
                        
                        geometry.draw_indices("default", mode=GL.GL_POINTS, first=self.tick_offset + i*self.tick_stride, count=self.tick_stride, renderer=renderer)
            '''
            i = 0
            material = self._material_hover if i in self.parent._hovering_axes else self._material_normal
            geometry = actor._geometry
            with renderer.activated(material, geometry):
                renderer.uniforms['identifier'] = GL.GLuint(self.parent._axes_identifier | i)
                renderer.uniforms['size'] = GL.GLfloat(self.parent.tick_size)
                
                geometry.draw_indices("default", mode=GL.GL_POINTS, first=self.tick_offset, count=self.tick_count * self.tick_stride, renderer=renderer)
            #'''
            
    def delete(self, renderer):
        pass


class MarkActor(actors.DoubleBufferComputeActor):
    @property
    def extent(self):
        return self._extent1
    
    @extent.setter
    def extent(self, value):
        self._extent0, self._extent1 = self._extent1, value
    
    @property
    def basis(self):
        return self._basis1
    
    @basis.setter
    def basis(self, value):
        self._basis0, self._basis1 = self._basis1, value 
    
    @property
    def dual_basis(self):
        return self._dual_basis1
    
    @dual_basis.setter
    def dual_basis(self, value):
        self._dual_basis0, self._dual_basis1 = self._dual_basis1, value
    
    @property
    def rank(self):
        return len(self._basis1)
    
    @property
    def bases(self):
        return np.concatenate((self._basis1, self._dual_basis1), axis=0)
    
    @property
    def instance_count(self):
        return self._instance_count
    
    @instance_count.setter
    def instance_count(self, value):
        self._instance_count = value
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        self._alpha = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        
        self._extent0 = extents.Extent()
        self._extent1 = extents.Extent()
        
        self._basis0 = np.empty((0, mdefaults.DEFAULT_M), dtype=mdefaults.DEFAULT_DTYPE)
        self._basis1 = np.empty((0, mdefaults.DEFAULT_M), dtype=mdefaults.DEFAULT_DTYPE)
        
        self._dual_basis0 = transforms.identity(n=mdefaults.DEFAULT_M)
        self._dual_basis1 = transforms.identity(n=mdefaults.DEFAULT_M)
        
        self._size = 0.0
        self._motion = 0.0
        self._visible = True
        
        self._instance_count = 0
    
    def set_instanced(self, value):
        if value:
            self._input0_vertex_buffer._divisor = 1
            self._input1_vertex_buffer._divisor = 1
            self._output_vertex_buffer._divisor = 1
        else:
            self._input0_vertex_buffer._divisor = 0
            self._input1_vertex_buffer._divisor = 0
            self._output_vertex_buffer._divisor = 0
    
    def current_rank(self, time):
        rank0, rank1 = len(self._basis0), len(self._basis1)
        if self._alpha.has_finished(time):
            return rank1
        else:
            return max(rank0, rank1)
    
    def current_bases(self, time):
        rank0, rank1 = len(self._basis0), len(self._basis1)
        #print("B0:", self._basis0, self._dual_basis0)
        #print("B1:", self._basis1, self._dual_basis1)
        if self.current_rank(time) == rank1:
            return np.concatenate((self._basis1, self._dual_basis1), axis=0)
        if self.current_rank(time) == rank0:
            return np.concatenate((self._basis0, self._dual_basis0), axis=0)
    
    def current_extent(self, time):
        extent = extents.interpolate_linear(self._extent0, self._extent1)
        return extent(self._alpha(time))


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
    
    def clear(self):
        self._mark_actors = {}
    
    def get_or_create_actor(self, key):
        result = self._mark_actors.get(key)
        if result is None:
            result = MarkActor()
            self._mark_actors[key] = result
        return result
    
    def on_data_changed(self, mark, key, value):
        if 'palette' in key:
            self.on_binding_changed(mark)
        
    def on_binding_changed(self, mark):
        #print("on_binding_changed", mark.owner.title)
        
        key = mark.owner.title.lower()
        dataset = mark.owner.schema._parent
        
        
        # Query
        position = mark["position"].expression
        color = mark["color"].expression if mark["color"].bound else None
        color_h = mark["color"]["h"].expression if mark["color"]["h"].bound else None
        color_s = mark["color"]["s"].expression if mark["color"]["s"].bound else None
        color_v = mark["color"]["v"].expression if mark["color"]["v"].bound else None
        color_a = mark["color"]["a"].expression if mark["color"]["a"].bound else None
        shape = mark["shape"].expression if mark["shape"].bound else None
        size = mark["size"].expression if mark["size"].bound else None
        motion = mark["motion"].expression if mark["motion"].bound else None
        
        select = filter(None, [position, color, color_s, color_v, color_a, shape, size, motion])
        
        #np.random.seed(1337)
        
        query = md.Query(key, select)
        result = query(dataset)
        #print(result.head())
        
        
        # Actor
        actor = self.get_or_create_actor(key)
        actor._material =_material = materials.MultiMaterial([
            materials.PhongMaterial(colors={
                "ambient": colors.svg.white.hsv,
                "diffuse": colors.svg.white.hsv,
                "specular": colors.svg.black.hsv,
                "emissive": colors.svg.black.hsv,
            }, textures={
                "ambient": self._texture_atlas,
                "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
            materials.PhongMaterial(colors={
                "ambient": colors.svg.white.hsv,
                "diffuse": colors.svg.white.hsv,
                "specular": colors.svg.black.hsv,
                "emissive": colors.svg.black.hsv,
            }, textures={
                "ambient": self._texture_atlas,
                "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
        ])
        '''
            materials.PhongMaterial(colors={
                "ambient": colors.Color.from_hex("#66cc66").hsv,
                "diffuse": colors.Color.from_hex("#66cc66").hsv,
                "specular": colors.svg.black.hsv,
                "emissive": colors.svg.black.hsv,
            }, textures={
                "ambient": self._texture_atlas,
                "diffuse": self._texture_atlas,
            }, opacity=1.0, shininess=32.0),
        '''
        
        actor._transforms = self.parent._transforms
        actor._geometry._vbos["instance_position"] = self._instance_geometry._vbos["instance_position"]
        actor._geometry._vbos["instance_tex_coord"] = self._instance_geometry._vbos["instance_tex_coord"]
        actor._geometry._ibos["instance_default"] = self._instance_geometry._ibos["instance_default"]
        
        actor.instance_count = len(result)
        
        
        # Populate
        def discrete_property(property):
            if property is None:
                return utilities.identity
            else:
                key = md.nameof(property)
                values = list(iterables.distinct(result[key]))
                def _discrete_property(value):
                    return values.index(value)
                return _discrete_property
        
        def continuous_property(property):
            if property is None:
                return utilities.identity
            else:
                key = md.nameof(property)
                lower, upper = result[key].min(), result[key].max()
                def _continuous_property(value):
                    return (value - lower)/(upper - lower)
                return _continuous_property
        
        def getter(property, default, continuity=utilities.identity):
            if property is None:
                def _getter(row):
                    return default
                return _getter
            else:
                normalize = continuity(property)
                def _getter(row):
                    return normalize(row[md.nameof(property)])
                return _getter
        
        def getposition(row):
            def _getposition(row):
                def __getposition(row, expression):
                    if md.is_interval(expression):
                        return (row[md.nameof(expression.upper)] + row[md.nameof(expression.lower)])/2.0
                    else:
                        return row[md.nameof(expression)]
                
                if position is not None:    
                    return vectors.zeros(*[__getposition(row, position[i]) for i in range(mdefaults.DEFAULT_N)])
                else:
                    return vectors.zeros()
            
            return vectors.homogeneous(_getposition(row), 1.0)
        
        getsizemodifier = getter(size, 1.0, continuous_property)
        
        def getsize(row):
            def _getsize(row):
                def __getsize(row, expression):
                    if md.is_interval(expression):
                        return (row[md.nameof(expression.upper)] - row[md.nameof(expression.lower)])/2.0
                    else:
                        return 0.0
                
                if position is not None:    
                    return vectors.zeros(*[__getsize(row, position[i]) for i in range(mdefaults.DEFAULT_N)])
                else:
                    return vectors.zeros()
            
            return vectors.homogeneous(_getsize(row), getsizemodifier(row))
        
        getoffsetmodifier = getter(motion, 1.0, continuous_property)
        
        def getoffset(row):
            def _getoffset(row):
                return vectors.zeros()
            
            return vectors.homogeneous(_getoffset(row), getoffsetmodifier(row))
        
        if not (color is None or md.is_color(color)):
            if md.is_categorical(md.typeof(color)):
                getcolor = getter(color, 0, discrete_property)
            else:
                getcolor = getter(color, 1.0, continuous_property)
        
        getcolor_h = getter(color_h, 1.0, continuous_property)
        getcolor_s = getter(color_s, 1.0, continuous_property)
        getcolor_v = getter(color_v, 1.0, continuous_property)
        getcolor_a = getter(color_a, 1.0, continuous_property)
        
        if not (shape is None): # or md.is_shape(shape):
            if md.is_categorical(md.typeof(shape)):
                getshape = getter(shape, 0, discrete_property)
            else:
                getshape = getter(shape, 1.0, continuous_property)
        
        def as_position(row):
            return getposition(row)
        
        def as_size(row):
            return getsize(row)
        
        def as_offset(row):
            return getoffset(row)
        
        def as_color0(row):
            if color is not None:
                if md.is_color(color):
                    return colors.Color([row[md.nameof(color[key])] for key in ['r', 'g', 'b', 'a']]).hsva 
                
                if mark.color_palettes is not None:
                    value = getcolor(row)
                    if md.is_categorical(md.typeof(color)):
                        return mark.color_palettes.max()[value].hsva
                    else:
                        return mark.color_palettes.max()(value).hsva
            
            return colors.svg.white.hsva
        
        def as_color1(row):
            h = getcolor_h(row)
            s = getcolor_s(row)
            v = getcolor_v(row)
            a = getcolor_a(row)
            return [h, s, v, a]
        
        def as_tex_coord(row):
            if shape is not None:
                index = getshape(row)
                index = np.unravel_index(index, self._texture_atlas._shape)
                return vectors.zeros(*index, n=4)
            
            return vectors.zeros(n=4)
        
        position_data = np.ascontiguousarray([as_position(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        tex_coord_data = np.ascontiguousarray([as_tex_coord(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        color0_data = np.ascontiguousarray([as_color0(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        color1_data = np.ascontiguousarray([as_color1(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        size_data = np.ascontiguousarray([as_size(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        offset_data = np.ascontiguousarray([as_offset(row) for index, row in result.iterrows()], dtype=mdefaults.DEFAULT_DTYPE)
        
        #print(position_data, size_data)
        #print(size_data)
        
        vertex_data = vertices.full({
            vertices.Component.Position: np.nan_to_num(position_data),
            vertices.Component.TexCoord0: tex_coord_data,
            vertices.Component.TexCoord1: tex_coord_data,
            vertices.Component.Color0: color0_data,
            vertices.Component.Color1: color1_data,
            vertices.Component.Size: size_data,
            vertices.Component.Offset: offset_data,
            vertices.Component.Identifier: generators.autoincrement(),
        }, format=DEFAULT_VERTEX_ND_FORMAT)
        
        """
        #'''
        n = 1000
        vertex_data = vertices.full({
            vertices.Component.Position: [vectors.point(2.0 * np.random.ranf(mdefaults.DEFAULT_N) - 1.0) for _ in range(n)],
            vertices.Component.Color: [vectors.point(2.0 * np.random.ranf(mdefaults.DEFAULT_N) - 1.0) for _ in range(n)],
            vertices.Component.Identifier: generators.autoincrement(),
        }, format=self._vertex_format)
        '''
        vertex_data = vertices.full({
            vertices.Component.Position: self.positions(result),
            vertices.Component.Color: self.colors(result),
        }, format=self._vertex_format)
        #'''
        """
        lower = vectors.vector(np.amin(position_data - np.abs(size_data), axis=0), n=mdefaults.DEFAULT_N)
        upper = vectors.vector(np.amax(position_data + np.abs(size_data), axis=0), n=mdefaults.DEFAULT_N)
        
        extent = extents.Extent(lower, upper)
        
        def mark_basis(position):
            def _mark_basis(position):
                for i in range(mdefaults.DEFAULT_M):
                    item = accessors.getitem(position, i, None)
                    if md.is_interval(item):
                        yield vectors.unit(i, n=mdefaults.DEFAULT_M)
            result = np.array(list(_mark_basis(position)), dtype=mdefaults.DEFAULT_DTYPE)
            return result.reshape((len(result), mdefaults.DEFAULT_M))
        
        def mark_dual_basis(position):
            def _mark_dual_basis(position):
                for i in range(mdefaults.DEFAULT_M):
                    item = accessors.getitem(position, i, None)
                    if not md.is_interval(item):
                        yield vectors.unit(i, n=mdefaults.DEFAULT_M)
            result = np.array(list(_mark_dual_basis(position)), dtype=mdefaults.DEFAULT_DTYPE)
            return result.reshape((len(result), mdefaults.DEFAULT_M))
        
        basis, dual_basis = mark_basis(position), mark_dual_basis(position)
        
        actor.set_instanced(len(basis))
        
        #print(len(basis), len(dual_basis))
        #print(basis, dual_basis)
        
        #print(vectors.dot(vectors.vector(range(mdefaults.DEFAULT_N)), actor.bases))
        
        if actor._input0_vertex_data.data is None:
            actor.update_vertex_data(vertex_data)
            actor.update_vertex_data(vertex_data)
        else:
            actor.update_vertex_data(vertex_data)
        
        actor.basis, actor.dual_basis = basis, dual_basis
        actor.extent = extent
        actor._alpha.play(self.time)
        
        #print(vertex_data, extent)
        
    def process(self, input):
        pass 
    
    def create(self, renderer):
        self._material_normal = materials.MultiMaterial([DEFAULT_MATERIAL_NORMAL_HSV, DEFAULT_MATERIAL_NORMAL_HSV])
        self._material_normal.prepare(renderer)
        
        self._material_hover = materials.MultiMaterial([DEFAULT_MATERIAL_HOVER_HSV, DEFAULT_MATERIAL_HOVER_HSV])
        self._material_hover.prepare(renderer)
        
        self._texture_atlas = textures.Texture(self.parent.shape_canvas, filter=(GL.GL_NEAREST, GL.GL_NEAREST))
        self._texture_atlas._shape = np.array([8, 8], dtype=np.uint32)
        self._texture_atlas.prepare(renderer)
        
        cube_face2_positions = cubes.cube_face_positions(2, mdefaults.DEFAULT_N)
        cube_face2_tex_coords = cubes.cube_face_tex_coords(2, mdefaults.DEFAULT_N)
        cube_face2_indices = cubes.cube_face_indices(2, mdefaults.DEFAULT_N)
        
        vertex_data = vertices.full({
            vertices.Component.InstancePosition: [vectors.zeros(*position, n=mdefaults.DEFAULT_M) for position in cube_face2_positions],
            vertices.Component.InstanceTexCoord: [vectors.zeros(*tex_coord, n=4) for tex_coord in cube_face2_tex_coords],
        }, format=vertices.FORMAT_INSTANCE)
        
        #index_data = np.array(list(cube_face2_indices), dtype=np.uint32).ravel()
        index_data = np.array(list(map(geometries.triangle_strip_to_triangle_indices, cube_face2_indices)), dtype=np.uint32).ravel()
        
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
    
    def update(self, renderer):
        time = self.time
        
        for key, mark in self.parent.model.marks.items():
            actor = self.get_or_create_actor(key)
                        
            primaryColor = mark.color_primary_hsva
            actor._material[0].colors["ambient"] = primaryColor[:3]
            actor._material[0].colors["diffuse"] = primaryColor[:3]
            actor._material[0]["opacity"] = primaryColor[3]
            
            secondaryColor = mark.color_secondary_hsva
            actor._material[1].colors["ambient"] = secondaryColor[:3]
            actor._material[1].colors["diffuse"] = secondaryColor[:3]
            actor._material[1]["opacity"] = secondaryColor[3]
            
            actor._color_modifiers = mark.color_modifiers_hsva
            actor._tex_coord_modifiers = vectors.zeros(*np.unravel_index(int(mark.shape_index), self._texture_atlas._shape), n=4)
            actor._size = mark.size
            actor._motion = mark.motion
            actor._visible = mark.visible
            
            #print(actor._color_modifiers)
            #print(primaryColor, secondaryColor)
        
        
        #model_nd = transforms.scale(vectors.vector([animation(time) for animation in self.parent._alphas], n=mdefaults.DEFAULT_M-1))
        model_nd = self.parent._root_nd._transform.to_matrix()
        view_nd = vectors.inverse(self.parent._camera_nd._transform.to_matrix())
        
        wrap = 0 if self.parent.model.show_overflow else renderers.Wrap.Clamp
        compute_program = DEFAULT_AXES_INTERPOLATE_PROGRAMS[DEFAULT_VERTEX_ND_FORMAT, wrap, len(self.parent._transforms)]
        with renderer.activated(compute_program, self.parent._transforms_buffer):
            renderer.set_model_view_nd(model_nd, view_nd)
            
            renderer.uniforms["buffer_offset"] = GL.GLuint(0)
            renderer.uniforms["buffer_stride"] = GL.GLuint(1)
            
            renderer.uniform_blocks["TransformsData"] = self.parent._transforms_buffer
            
            renderer.set_uniform_array_1d("aspect_nd", vectors.default_chunk1d(self.parent._view_aspect))
            
            for actor in self._mark_actors.values():
                renderer.uniforms["alpha"] = GL.GLfloat(actor._alpha(time))
                
                with renderer.activated(actor._input0_vertex_buffer, actor._input1_vertex_buffer, actor._output_vertex_buffer):
                    renderer.shader_storage_blocks["VertexSrcData0"] = actor._input0_vertex_buffer
                    renderer.shader_storage_blocks["VertexSrcData1"] = actor._input1_vertex_buffer
                    renderer.shader_storage_blocks["VertexDstData"] = actor._output_vertex_buffer
                    
                    GL.glDispatchCompute(mathematics.chunk_count(accessors.lenitems(actor._input0_vertex_data.data), gdefaults.DEFAULT_WORK_GROUP_SIZE), 1, 1)
                    GL.glMemoryBarrier(GL.GL_SHADER_STORAGE_BARRIER_BIT)
        
        '''
        for actor in self._mark_actors.values():
            print(actor._input0_vertex_data.data)
            print(actor._input1_vertex_data.data)
        '''
        '''
        self.parent._transforms_buffer.debug_dump(renderer)
        for actor in self._mark_actors.values():
            print(actor._output_vertex_buffer._value._handle)
            #actor._input0_vertex_buffer.debug_dump(renderer)
            #actor._input1_vertex_buffer.debug_dump(renderer)
            #actor._output_vertex_buffer.debug_dump(renderer)
        '''
    
    def render(self, renderer):
        time = self.time
        
        projection_nd = projections.diagonal(3)
        
        model = self.parent._root._transform.to_matrix(n=3, m=4)
        view = vectors.inverse(self.parent._camera._transform.to_matrix(n=3, m=4))
        projection = self.parent._camera._projection.to_matrix(n=3, m=4)
        projection = np.dot(transforms.scale(1.0/self.aspect, n=4), projection)
        
        for actor in self._mark_actors.values():
            if not actor._visible:
                continue
            
            shade = renderers.Shade.Phong if self.parent.model.show_shading else 0
            rank = actor.rank #actor.current_rank(time)
            bases = actor.bases #actor.current_bases(time)
            
            if rank == 0:
                marks_program = DEFAULT_MARK_PROGRAMS[materials.DEFAULT_FORMAT, shade, rank]
            else:
                #marks_program = DEFAULT_MARK_INTERVAL_PROGRAMS[materials.DEFAULT_FORMAT, shade, rank]
                #actor.set_instanced(False)
                marks_program = DEFAULT_MARK_INSTANCE_PROGRAMS[materials.DEFAULT_FORMAT, shade, rank]
            
            with renderer.activated(marks_program, self.parent._lights):
                renderer.set_projection_nd(projection_nd)
                renderer.set_model_view_projection(model, view, projection)
                
                renderer.uniforms['tex_coord_transform'] = transforms.scale(vectors.vector(1.0/self._texture_atlas._shape, n=3), n=4)
                
                renderer.set_uniform_array_2d("basis_nd", vectors.default_chunk2d(bases))
                
                renderer.uniforms['time'] = GL.GLfloat(time)
                
                material = self._material_hover if len(self.parent._hovering_marks) else actor._material
                geometry = actor._geometry
                
                renderer.uniforms['material_index'] = GL.GLint(len(self.parent._hovering_marks))
                
                with renderer.activated(material, geometry):
                    renderer.uniforms['identifier'] = GL.GLuint(self.parent._marks_identifier)
                    renderer.uniforms['rank'] = GL.GLint(rank)
                    renderer.uniforms['tex_coord_modifiers'] = actor._tex_coord_modifiers
                    renderer.uniforms['color_modifiers'] = actor._color_modifiers
                    renderer.uniforms['size'] = GL.GLfloat(actor._size * self.parent.mark_size)
                    renderer.uniforms['motion'] = GL.GLfloat(actor._motion)
                    
                    if rank == 0:
                        geometry.draw_indices("default", mode=GL.GL_POINTS, renderer=renderer)
                    else:
                        #geometry.draw_indices("default", mode=GL.GL_PATCHES, patch_vertices=1, renderer=renderer)
                        #geometry.draw_indices_instanced("instance_default", actor.instance_count, mode=GL.GL_TRIANGLES, renderer=renderer)
                        geometry.draw_indices_instanced("instance_default", actor.instance_count, mode=GL.GL_PATCHES, patch_vertices=3, renderer=renderer)
                
    def delete(self, renderer):
        pass


class VisualizationView(scenes.View):
    _type_identifier_size = 4
    _data_identifier_size = 32 - _type_identifier_size
    
    _type_identifier_offset = _data_identifier_size
    _data_identifier_offset = 0
    
    _type_identifier_mask = (2**_type_identifier_size - 1) << _type_identifier_offset
    _data_identifier_mask = (2**_data_identifier_size - 1) << _data_identifier_offset
    
    _origin_identifier = (1 << (0 + _type_identifier_offset))
    _axes_identifier = (1 << (1 + _type_identifier_offset))
    _marks_identifier = (1 << (2 + _type_identifier_offset))
    
    @property
    def preserve_aspect(self):
        return self._preserve_aspect
        
    @preserve_aspect.setter
    def preserve_aspect(self, value):
        self._preserve_aspect = value
    
    @property
    def orbit(self):
        return self._orbit
        
    @orbit.setter
    def orbit(self, value):
        self._orbit = value
    
    @property
    def track(self):
        return self._track
        
    @track.setter
    def track(self, value):
        self._track = value
    
    @property
    def hover(self):
        return self._hover
        
    @hover.setter
    def hover(self, value):
        self._hover = value
    
    @property
    def perspective(self):
        return self._camera._projection.perspective
        
    @perspective.setter
    def perspective(self, value):
        self._camera._projection.perspective = value
    
    @property
    def transforms(self):
        return self._transforms

    @transforms.setter
    def transforms(self, value):
        self._transforms = value
        
        self._transforms_data.data = np.ascontiguousarray(list(map(vectors.default_chunk2d, self._transforms)), dtype=mdefaults.DEFAULT_DTYPE)
    
    @property
    def model(self):
        return self._model
        
    @model.setter
    def model(self, value):
        self._model = value
        self._model.data_changed.subscribe(self._marks.on_data_changed)
        self._model.binding_changed.subscribe(self._marks.on_binding_changed)
        self._model.visible_changed.subscribe(self.toggle)
    
    @property
    def color_canvas(self):
        return self._color_canvas
    
    @color_canvas.setter
    def color_canvas(self, value):
        self._color_canvas = value

    @property
    def shape_canvas(self):
        return self._shape_canvas
    
    @shape_canvas.setter
    def shape_canvas(self, value):
        self._shape_canvas = value
    
    def __init__(self, model=None, clock=None, size=None, parent=None):
        super().__init__(clock=clock, size=size, parent=parent)
        
        self._alpha = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        self._beta = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        
        self._alphas = [animations.Animation(0.0, 1.0, ease=animations.ease_sine_in_out, reversed=True) for _ in range(mdefaults.DEFAULT_N)]
        self._betas = [animations.Animation(0.0, 1.0, ease=animations.ease_sine_in_out, reversed=True) for _ in range(mdefaults.DEFAULT_N)]
        
        self._camera = scenes.Camera(transform=scenes.Transform(n=3, m=4), projection=scenes.Projection(fov=np.pi/1.5, n=3, m=4))
        self._camera._transform.translate(vectors.zeros(0, 0, -4.0, n=3))
        self._camera._transform.scale(vectors.full(2.0, n=3))
        self._camera_nd = scenes.Camera()
        
        self._root = scenes.Node(transform=scenes.Transform(n=3, m=4))
        self._root_nd = scenes.Node()
        
        self._lights = lights.MultiLight([lights.Light() for _ in range(GLANCE_LIGHT_COUNT)])
        
        self._lights[0] = lights.Light(colors.svg.white.darken(0.25), colors.svg.white.darken(0.00), colors.svg.white.darken(0.50))
        self._lights[0].position = np.asarray([-0.25, -0.5, -1.0, 0.0], dtype=np.float32)
        
        '''
        for light, color in zip(self.lights, [colors.SVG.LightGreen, colors.SVG.LightSkyBlue, colors.SVG.LightSalmon]):
            light.ambient = colors.SVG.Black
            light.diffuse = color.darken(0.25)
            light.specular = color.darken(0.50)
            light.attenuation =  np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
        '''
        
        self._background = BackgroundView("environments/dust_{side}.jpg", parent=self)
        self._axes = AxesView(parent=self)
        self._marks = MarksView(parent=self)
        
        self.origin_size = 1.0/32.0
        self.axis_size = 1.0/128.0
        self.mark_size = 1.0/16.0
        self.tick_size = 1.0/32.0
        
        self._hover = False
        self._model = model
        
        self._preserve_aspect = True
        self._track = True
        self._orbit = False
        
        self._hovering_origins = set()
        self._hovering_axes = set()
        self._hovering_marks = set()
        
        self._view_aspect = vectors.ones()
        self._view_aspect[0] = 2.0
        
        self._view_extent = extents.Extent(-vectors.ones(), +vectors.ones())
        self._model_extent = extents.Extent(-vectors.ones(), +vectors.ones())
                
        self._color_canvas = None
        self._shape_canvas = None
    
    def process(self, input):
        hovering_origins = set()
        hovering_axes = set()
        hovering_marks = set()
        
        for touch in input.touches.values():
            type = touch.target & self._type_identifier_mask
            data = touch.target & self._data_identifier_mask
            if type == self._origin_identifier:
                hovering_origins.add(data)
                continue
            if type == self._axes_identifier:
                hovering_axes.add(data)
                continue
            if type == self._marks_identifier:
                hovering_marks.add(data)
                continue
        
        self._hovering_origins = hovering_origins
        self._hovering_axes = hovering_axes
        self._hovering_marks = hovering_marks
        
        delta = vectors.zeros(n=mdefaults.DEFAULT_M-1)
        if input.keys.get(inputs.Key.Left) or input.keys.get(inputs.Key.Right):
            for key in range(inputs.Key._1, inputs.Key._1 + mdefaults.DEFAULT_N):
                if key in input.keys:
                    if input.keys.get(inputs.Key.Left):
                        delta[key - inputs.Key._1] -= 1.0
                    if input.keys.get(inputs.Key.Right):
                        delta[key - inputs.Key._1] += 1.0
        
        self._camera_nd._transform._delta_translation = delta
        
        if self._orbit and not self._hover:
            self._root._transform._delta_rotation = vectors.zeros(0.25, 0, 0, n=3)
        else:
            self._root._transform._delta_rotation = vectors.zeros(0.00, 0, 0, n=3)
        
        #print(self._hovering_axes)
        
    def on_key_press(self, state):
        pass
    
    def on_key_release(self, state):
        pass
    
    def on_drop(self, state):
        type = state.target & self._type_identifier_mask
        data = state.target & self._data_identifier_mask
        
        expression = state.data
        owner = expressions.ownerof(expression)
        
        if owner is None:
            logger.warn('Ignoring ownerless expression "{}".'.format(expression))
        else:
            key = owner.title.lower()
            axis = self._model.get_or_create_axis(key)
            mark = self._model.get_or_create_mark(key, order=0, owner=owner)
            
            if type == self._origin_identifier:
                self._model.auto_bind_origin(expression, mark, axis)
            elif type == self._axes_identifier:
                axis = self._model.get_index(axis, data)
                self._model.auto_bind_axis(expression, mark, axis)
            elif type == self._marks_identifier:
                self._model.bind_color(mark, expression)
            else:
                self._model.auto_bind(expression, mark, axis)
    
    def toggle(self, index):
        if 0 <= index < len(self._alphas):
            self._alphas[index].reverse(self.time)
    
    def clear(self):
        self._alpha = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        self._beta = animations.Animation(0.0, 2.0, ease=animations.ease_cubic_out)
        
        self._alphas = [animations.Animation(0.0, 1.0, ease=animations.ease_sine_in_out, reversed=True) for _ in range(mdefaults.DEFAULT_N)]
        self._betas = [animations.Animation(0.0, 1.0, ease=animations.ease_sine_in_out, reversed=True) for _ in range(mdefaults.DEFAULT_N)]
        
        self._camera = scenes.Camera(transform=scenes.Transform(n=3, m=4), projection=scenes.Projection(fov=np.pi/1.5, n=3, m=4))
        self._camera._transform.translate(vectors.zeros(0, 0, -4.0, n=3))
        self._camera._transform.scale(vectors.full(2.0, n=3))
        self._camera_nd = scenes.Camera()
        
        self._root = scenes.Node(transform=scenes.Transform(n=3, m=4))
        self._root_nd = scenes.Node()
        
        self._view_aspect = vectors.ones()
        self._view_aspect[0] = 2.0
        
        self._view_extent = extents.Extent(-vectors.ones(), +vectors.ones())
        self._model_extent = extents.Extent(-vectors.ones(), +vectors.ones())
        
        self._axes.clear()
        self._marks.clear()
    
    def scale(self, value):
        self._camera_nd._transform.scale(value)
    
    def translate(self, value):
        self._camera_nd._transform.translate(value)
    
    def update_camera_transform(self, translation, scaling):
        '''
        a = transforms.translate(-src_center)
        b = transforms.scale(2.0/src_delta)
        c = transforms.scale(dst_delta/2.0)
        d = transforms.translate(+dst_center)
        print(vectors.inverse(vectors.dot(a,b,c,d)))
        print(-translation, 1.0/scaling)
        '''
        self._camera_nd._transform.translation = -translation
        self._camera_nd._transform.scaling = 1.0/scaling
        
        #print(translation, scaling)
    
    def zoom_extent(self):
        marks_extent = self._marks.extent
        
        self._model_extent = extents.Extent(vectors.full(np.nan, *(marks_extent.lower)), vectors.full(np.nan, *(marks_extent.upper)))
        self._view_extent = extents.Extent(-vectors.vector(self._view_aspect), +vectors.vector(self._view_aspect))
        
        transform = extents.transform_extent_min if self._preserve_aspect else extents.transform_extent
        self.update_camera_transform(*transform(self._model_extent,  self._view_extent))
    
    def zoom_extent_preserve_aspect_min(self):
        marks_extent = self._marks.extent
        
        self._model_extent = extents.Extent(vectors.full(np.nan, *(marks_extent.lower)), vectors.full(np.nan, *(marks_extent.upper)))
        self._view_extent = extents.Extent(-vectors.vector(self._view_aspect), +vectors.vector(self._view_aspect))
        
        self.update_camera_transform(*extents.transform_extent_min(self._model_extent,  self._view_extent))
    
    def zoom_extent_preserve_aspect_max(self):
        marks_extent = self._marks.extent
        
        self._model_extent = extents.Extent(vectors.full(np.nan, *(marks_extent.lower)), vectors.full(np.nan, *(marks_extent.upper)))
        self._view_extent = extents.Extent(-vectors.vector(self._view_aspect), +vectors.vector(self._view_aspect))
    
        self.update_camera_transform(*extents.transform_extent_max(self._model_extent,  self._view_extent))
    
    def create(self, renderer):
        self._transforms_data = buffers.BufferData()
        self._transforms_buffer = buffers.UniformBuffer(self._transforms_data)
        
        self.transforms = [transforms.identity()]
        
        self._background.create(renderer)
        self._axes.create(renderer)
        self._marks.create(renderer)
    
    def update(self, renderer):
        time = self.time
        delta = 1.0/30.0
        
        self._camera.update(delta)
        self._camera_nd.update(delta)
        
        self._root.update(delta)
        self._root_nd.update(delta)
        
        #model_nd = transforms.scale([animation(time) for animation in self._alphas])
        #view_nd = transforms.identity()
        #projection_nd = transforms.identity()
        
        if self._track:
            marks_extent = self._marks.current_extent(time)
            
            self._model_extent = extents.Extent(vectors.full(np.nan, *(marks_extent.lower)), vectors.full(np.nan, *(marks_extent.upper)))
            self._view_extent = extents.Extent(-vectors.vector(self._view_aspect), +vectors.vector(self._view_aspect))
            
            transform = extents.transform_extent_min if self._preserve_aspect else extents.transform_extent
            self.update_camera_transform(*transform(self._model_extent,  self._view_extent))
        
        self._background.update(renderer)
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
            self._background.size = self.size
            self._background.render(renderer)
        
        if self._model.show_axes:
            self._axes.render(renderer)
        
        if self._model.show_marks:
            self._marks.render(renderer)
    
    def delete(self, renderer):
        self._marks.update(renderer)
        self._axes.update(renderer)
        self._background.delete(renderer)
