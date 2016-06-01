import numpy as np

from glue import gl
from glue.gl import GL

from encore import mappings

from . import renderers, shaders, textures, vertices

from .. import colors, mathematics


DEFAULT_AMBIENT_COLOR = colors.svg.white
DEFAULT_DIFFUSE_COLOR = colors.svg.white
DEFAULT_SPECULAR_COLOR = colors.svg.black
DEFAULT_EMISSIVE_COLOR = colors.svg.black

DEFAULT_TESS_LEVEL_INNER = 8
DEFAULT_TESS_LEVEL_OUTER = 8

DEFAULT_WORK_GROUP_SIZE = 64

DEFAULT_RENDERER = renderers.Renderer()

DEFAULT_TEXTURE_EMPTY = textures.Texture.from_data(target=GL.GL_TEXTURE_2D, size=(1, 1), image=np.array([[(0, 0, 0, 0)]], dtype=np.uint8), format=GL.GL_RGBA)
DEFAULT_TEXTURE_BLACK = textures.Texture.from_data(target=GL.GL_TEXTURE_2D, size=(1, 1), image=np.array([[(0, 0, 0, 255)]], dtype=np.uint8), format=GL.GL_RGBA)
DEFAULT_TEXTURE_WHITE = textures.Texture.from_data(target=GL.GL_TEXTURE_2D, size=(1, 1), image=np.array([[(255, 255, 255, 255)]], dtype=np.uint8), format=GL.GL_RGBA)

DEFAULT_SIMPLEX_PRIMITIVES = {0: GL.GL_POINTS, 1:GL.GL_LINES, 2: GL.GL_TRIANGLES}

DEFAULT_PROGRAM_DEFINES = mappings.DefineMap({
    "GLANCE_MATERIAL_COLOR_AMBIENT": 1,
    "GLANCE_MATERIAL_COLOR_DIFFUSE": 1,
    "GLANCE_MATERIAL_COUNT": 2,
    "GLANCE_MATERIAL_SIDES": 1,
    "GLANCE_TESS_LEVEL_INNER": DEFAULT_TESS_LEVEL_INNER,
    "GLANCE_TESS_LEVEL_OUTER": DEFAULT_TESS_LEVEL_OUTER,
    "GLANCE_VERTEX_FORMAT": vertices.DEFAULT_FORMAT,
    "GLANCE_VERTEX_ND_FORMAT": vertices.DEFAULT_FORMAT,
    "M_N": mathematics.defaults.DEFAULT_N,
})

DEFAULT_BACKGROUND_PROGRAMS = {
    GL.GL_TEXTURE_CUBE_MAP: shaders.Program.from_files(["noop.vs", "background.gs", "background.fs"], defines={"GLANCE_BACKGROUND_CUBE": 1}),
    GL.GL_TEXTURE_2D: shaders.Program.from_files(["noop.vs", "background.gs", "background.fs"], defines={"GLANCE_BACKGROUND_CUBE": 0}),
}


def simplex_program(material_format, vertex_format=vertices.DEFAULT_FORMAT, i=0):
    return shaders.Program.from_files([
        "default.vs",
        "default_simplex%d.gs" % (i,),
        "default.fs",
    ], defines={
        "GLANCE_MATERIAL_FORMAT": material_format,
        "GLANCE_VERTEX_FORMAT": vertex_format,
        "M_N": mathematics.defaults.DEFAULT_N,
    })


def simplex_nd_program(material_format, vertex_format=vertices.DEFAULT_FORMAT, i=0):
    return shaders.Program.from_files([
        "default_nd.vs",
        "default_simplex%d_nd.gs" % (i,),
        "default.fs",
    ], defines={
        "GLANCE_MATERIAL_FORMAT": material_format,
        "GLANCE_VERTEX_FORMAT": vertex_format,
        "M_N": mathematics.defaults.DEFAULT_N,
    })


def thick_simplex_program(material_format, vertex_format=vertices.DEFAULT_FORMAT, i=0, space=renderers.Space.World):
    return shaders.Program.from_files([
        "default.vs",
        "thick_simplex%d.gs" % (i,),
        "default.fs",
    ], defines={
        "GLANCE_SPACE": int(space),
        "GLANCE_MATERIAL_FORMAT": material_format,
        "GLANCE_VERTEX_FORMAT": vertex_format,
        "M_N": mathematics.defaults.DEFAULT_N,
    })


def thick_simplex_nd_program(material_format, vertex_format=vertices.DEFAULT_FORMAT, i=0, space=renderers.Space.World):
    return shaders.Program.from_files([
        "default_nd.vs",
        "thick_simplex%d_nd.gs" % (i,),
        "default.fs",
    ], defines={
        "GLANCE_SPACE": int(space),
        "GLANCE_MATERIAL_FORMAT": material_format,
        "GLANCE_VERTEX_FORMAT": vertex_format,
        "M_N": mathematics.defaults.DEFAULT_N,
    })


DEFAULT_SIMPLEX_PROGRAMS = mappings.CustomMap(factory=lambda key: simplex_nd_program(*key))
DEFAULT_THICK_SIMPLEX_PROGRAMS = mappings.CustomMap(factory=lambda key: thick_simplex_nd_program(*key))