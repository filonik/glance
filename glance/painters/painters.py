import os
import sys

import enum

import functools as ft
import itertools as it

import numpy as np

if sys.version_info >= (3, 0):
    import cairocffi as cairo
else:
    import cairo

from encore import accessors, coercions, iterables


def load_image(path):
    return cairo.ImageSurface.create_from_png(path)


class ImageCache(object):
    def __init__(self, factory=load_image):
        super().__init__()
        
        self._factory = factory
        self._data = {}
    
    def __getitem__(self, key):
        try:
            return accessors.getitem(self._data, key)
        except KeyError:
            result = self._factory(key)
            self._data[key] = result
            return result
    
    def __delitem__(self, key):
        del self._data[key]
    
    def clear():
        self._data.clear()

DefaultImageCache = ImageCache()


class Alignment(enum.IntEnum):
    Left = 0
    Right = 1
    Center = 2

class Weight(enum.IntEnum):
    Normal = cairo.FONT_WEIGHT_NORMAL
    Bold = cairo.FONT_WEIGHT_BOLD


def _coerce_tuple(value, n):
    return tuple(iterables.take(it.cycle(coercions.coerce_tuple(value)), n))


def _fill_path(cr, fill=None, **kwargs):
    if fill is not None:
        cr.set_source_rgb(fill[0], fill[1], fill[2])
        cr.fill_preserve()


def _stroke_path(cr, stroke=None, **kwargs):
    if stroke is not None:
        cr.set_source_rgb(stroke[0], stroke[1], stroke[2])
        cr.stroke_preserve()
 

def _normal(f, **kwargs):
    margin = _coerce_tuple(kwargs.get('margin', 0), 4)
    total_margin = np.array([margin[0]+margin[2], margin[1]+margin[3]])
    def paint(cr, size):
        aspect = size/np.min(size)
        content_size = size - total_margin
        w, h = content_size/aspect
        x0, y0, x1, y1 = margin
        cr.set_operator(cairo.OPERATOR_OVER)
        cr.save()
        cr.translate(x0, y0)
        cr.scale(0.5*w, 0.5*h)
        cr.translate(aspect[0], aspect[1])
        f(cr, size)
        cr.restore()
    return paint


def _path(f, **kwargs):
    stroke_width = kwargs.get('stroke_width', 1.0)
    stroke_dasharray = kwargs.get('stroke_dasharray', None)
    def paint(cr, size):
        if stroke_width is not None:
            cr.set_line_width(stroke_width)
        if stroke_dasharray is not None:
            cr.set_dash(stroke_dasharray)
        f(cr, size)
        _stroke_path(cr, **kwargs)
        _fill_path(cr, **kwargs)
        cr.new_path()
    return paint


def _normal_path(f, **kwargs):
    return _path(_normal(f, **kwargs), **kwargs)


def fill(**kwargs):
    foreground_color = kwargs.get('stroke')
    background_color = kwargs.get('fill')
    def paint(cr, size):
        if background_color is not None:
            cr.set_operator(cairo.OPERATOR_OVER)
            cr.set_source_rgba(
                accessors.getitem(background_color, 0, 0.0),
                accessors.getitem(background_color, 1, 0.0),
                accessors.getitem(background_color, 2, 0.0),
                accessors.getitem(background_color, 3, 1.0)
            )
            cr.paint()
    return paint


def rect(size=(2.0, 2.0), **kwargs):
    w, h = size
    def paint(cr, size):
        cr.rectangle(-w/2.0, -h/2.0, w, h)
    return _normal_path(paint, **kwargs)


def linear_gradient(stops, p0=-1.0, p1=+1.0, **kwargs):
    source = cairo.LinearGradient(
        accessors.getitem(p0, 0, p0), accessors.getitem(p0, 1, 0.0),
        accessors.getitem(p1, 0, p1), accessors.getitem(p1, 1, 0.0)
    )
    for i, color in stops:
        source.add_color_stop_rgba(i,
            accessors.getitem(color, 0, 0.0),
            accessors.getitem(color, 1, 0.0),
            accessors.getitem(color, 2, 0.0),
            accessors.getitem(color, 3, 1.0)
        )
    def paint(cr, size):
        cr.set_source(source)
        cr.paint()
    return _normal(paint, **kwargs)


def radial_gradient(stops, p0=0.0, r0=0.0, p1=0.0, r1=1.0, **kwargs):
    source = cairo.RadialGradient(
        accessors.getitem(p0, 0, p0), accessors.getitem(p0, 1, 0.0), r0,
        accessors.getitem(p1, 0, p1), accessors.getitem(p1, 1, 0.0), r1
    )
    for i, color in stops:
        source.add_color_stop_rgba(i,
            accessors.getitem(color, 0, 0.0),
            accessors.getitem(color, 1, 0.0),
            accessors.getitem(color, 2, 0.0),
            accessors.getitem(color, 3, 1.0)
        )
    def paint(cr, size):
        cr.set_source(source)
        cr.paint()
    return _normal(paint, **kwargs)


def text(content, **kwargs):
    font_family = kwargs.get('font_family', "DejaVu Sans")
    font_size = kwargs.get('font_size', 0.1)
    font_weight = kwargs.get('font_weight', Weight.Normal)
    font_weight = {
        Weight.Normal: cairo.FONT_WEIGHT_NORMAL,
        Weight.Bold: cairo.FONT_WEIGHT_BOLD,
    }[font_weight]
    align = kwargs.get('align', Alignment.Center)
    
    def paint(cr, size):
        aspect = size/np.min(size)
        
        cr.select_font_face(font_family, cairo.FONT_SLANT_NORMAL, font_weight)
        cr.set_font_size(font_size)
        
        x_bearing, y_bearing, width, height, x_advance, y_advance = cr.text_extents(content)

        #cr.rectangle(-1.0, -1.0, 2.0, 2.0)
        #cr.rectangle(-width/2.0, -height/2.0, width, height)
        if align == Alignment.Center:
            cr.move_to(-x_bearing - width/2.0, -y_bearing - height/2.0)
        elif align == Alignment.Left:
            cr.move_to(-x_bearing, -y_bearing - height/2.0)
        elif align == Alignment.Right:
            cr.move_to(-x_bearing - width, -y_bearing - height/2.0)
        
        cr.text_path(content)
    return _normal_path(paint, **kwargs)


def grid(n, **kwargs):
    foreground_color = kwargs.get('stroke')
    background_color = kwargs.get('fill')
    
    def paint(cr, size):
        cr.set_operator(cairo.OPERATOR_OVER)
        
        cr.set_source_rgba(
            accessors.getitem(background_color, 0, 0.0),
            accessors.getitem(background_color, 1, 0.0),
            accessors.getitem(background_color, 2, 0.0),
            accessors.getitem(background_color, 3, 1.0)
        )
        
        cr.paint()
        
        if n == 0: return
        
        cr.set_source_rgba(
            accessors.getitem(foreground_color, 0, 0.0),
            accessors.getitem(foreground_color, 1, 0.0),
            accessors.getitem(foreground_color, 2, 0.0),
            accessors.getitem(foreground_color, 3, 1.0)
        )
        
        cr.set_line_width(1.0/(n+1))
        
        for i in np.linspace(-1.0, +1.0, (n+1)):
            cr.move_to(i, -1.0)
            cr.line_to(i, +1.0)
            cr.move_to(-1.0, i)
            cr.line_to(+1.0, i)
    
    return _normal(_path(paint, **kwargs), **kwargs)


def grain(n, **kwargs):
    foreground_color = kwargs.get('foreground_color')
    background_color = kwargs.get('background_color')
    
    def paint(cr, size):
        def slant_grain(n):
            for i in np.linspace(-2.0, +2.0, 2.0*(n+1)):
                cr.move_to(-2.0, -2.0+i)
                cr.line_to(+2.0, +2.0+i)

        def grid_grain(n):
            for i in np.linspace(-1.0, +1.0, (n+1)):
                cr.move_to(i, -1.0)
                cr.line_to(i, +1.0)
                cr.move_to(-1.0, i)
                cr.line_to(+1.0, i)
        
        if n == 0: return

        cr.set_operator(cairo.OPERATOR_OVER)
        cr.set_source_rgba(
            accessors.getitem(foreground_color, 0, 0.0),
            accessors.getitem(foreground_color, 1, 0.0),
            accessors.getitem(foreground_color, 2, 0.0),
            accessors.getitem(foreground_color, 3, 1.0)
        )

        cr.set_line_width(1.0/(n+1))

        grid_grain(n)
        #slant_grain(n)

    return _normal(_path(paint, **kwargs), **kwargs)


def chain(*painters):
    def paint(cr, size):
        for painter in painters:
            painter(cr, size)
    return paint


def atlas(shape, painters, flip=1, **kwargs):
    def _flip(i, index, shape):
        index = np.array(index, dtype=np.int32)
        if i < len(index):
            index[i] = shape[i] - index[i] - 1
        return index
    
    def paint(cr, size):
        item_size = np.array(size, dtype=np.int32)/shape
        for index, painter in zip(np.ndindex(tuple(shape)), painters): #np.ndenumerate(painters):
            if painter is None:
                continue
            
            index = index if flip is None else _flip(flip, index, tuple(shape))
            
            item_offset = item_size * index
            
            cr.save()
            cr.translate(*item_offset)

            cr.rectangle(0, 0, item_size[0], item_size[1])
            cr.clip()

            painter(cr, item_size)

            cr.restore()
    
    return paint


def image(path, cache=DefaultImageCache, **kwargs):
    def paint(cr, size):
        image = cache[path]
        #print(path, image.get_width(), image.get_height())
        cr.set_source_surface(image, 0, 0)
        cr.paint()
    
    return paint


# TODO:
"""
class Weight(enum.IntEnum):
    Light = Pango.Weight.LIGHT
    Normal = Pango.Weight.NORMAL
    Bold = Pango.Weight.BOLD
    Heavy = Pango.Weight.HEAVY

class Alignment(enum.IntEnum):
    Left = Pango.Alignment.LEFT
    Right = Pango.Alignment.RIGHT
    Center = Pango.Alignment.CENTER

def image(path, flip=False, **kwargs):
    import os

    margin = coercions.coerce_tuple(kwargs.get('margin', 0.0), n=4)
    #surface = cairo.ImageSurface.create_from_png(path)
    #pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)

    #print(path, cairo.ImageSurface.get_width(surface), cairo.ImageSurface.get_height(surface))
    #print(path, pixbuf.get_width(), pixbuf.get_height())

    def paint(cr, size):
        if os.path.isfile(path):
            pixbuf = GdkPixbuf.Pixbuf.new_from_file(path)

            if flip:
                cr.translate(0.0, pixbuf.get_height())
                cr.scale(+1.0, -1.0)

            cr.set_operator(cairo.OPERATOR_OVER)
            #cr.set_source_surface(surface, margin[0], margin[1])
            Gdk.cairo_set_source_pixbuf(cr, pixbuf, margin[0], margin[1])

            cr.paint()
        else:
            print('File "%s" not found.' % (path,))
    return paint
  
def gradient(**style):
    foreground_color = style.get('foreground_color', (0.0, 0.0, 0.0))
    background_color = style.get('background_color', (1.0, 1.0, 1.0))
    def paint(cr, size):
        cr.set_operator(cairo.OPERATOR_OVER)
        cr.set_source_rgb(background_color[0], background_color[1], background_color[2])
        cr.paint()

        cr.set_source_rgb(foreground_color[0], foreground_color[1], foreground_color[2])
        cr.move_to(0, 0)
        cr.line_to(size[0], size[1])
        cr.stroke()
    return paint

def paragraph(text, **kwargs):
    fill_color = kwargs.get('fill', (0.0, 0.0, 0.0))
    font_family = kwargs.get('font_family', "DejaVu Sans")
    font_size = kwargs.get('font_size', 8)
    font_weight = kwargs.get('font_weight', Weight.Normal)
    text_align = kwargs.get('text_align', Alignment.Left)
    text_justify = kwargs.get('text_justify', False)
    margin = _coerce_tuple(kwargs.get('margin', 0), 4)

    #text = text.encode('utf-8')
    background_painter = fill(**kwargs)

    def paint(cr, size):
        background_painter(cr, size)

        font = Pango.font_description_from_string(str(font_family) + " " + str(font_size))
        font.set_weight(font_weight)

        layout = PangoCairo.create_layout(cr)
        layout.set_alignment(text_align)
        layout.set_justify(text_justify)
        layout.set_font_description(font)
        layout.set_wrap(Pango.WrapMode.WORD)
        '''
        w, h = layout.get_pixel_size()
        position = ((size[0] - w)/2.0, (size[1] - h)/2.0)
        cr.move_to(*position)
        '''
        cr.move_to(margin[0], margin[1])

        layout.set_width(int(size[0] * Pango.SCALE))
        layout.set_text(text, -1)

        cr.set_operator(cairo.OPERATOR_OVER)
        cr.set_source_rgb(fill_color[0], fill_color[1], fill_color[2])
        
        PangoCairo.update_layout(cr, layout)
        PangoCairo.show_layout(cr, layout)
    return paint
"""