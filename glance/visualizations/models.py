import collections
import logging

from encore import accessors, functions, objects, utilities

import numpy as np
import pandas as pd

import medley as md
from medley import expressions

from ..mathematics import defaults

logger = logging.getLogger(__name__.split(".").pop())


def recursive_itervalues(obj):
    yield obj
    for sub_obj in obj.values():
        yield from recursive_itervalues(sub_obj)


class Observable(object):
    def __init__(self):
        super(Observable, self).__init__()

        self._subscribers = []

    def subscribe(self, value):
        self._subscribers.append(value)

    def unsubscribe(self, value):
        self._subscribers.remove(value)

    def __call__(self, *args, **kwargs):
        for subscriber in self._subscribers:
            subscriber(*args, **kwargs)


class Axis(objects.Object):
    title = objects.attrproperty("title", "Untitled")
    type = objects.attrproperty("type", None)
    group = objects.attrproperty("group", False)
    visible = objects.attrproperty("visible", True)
    
    @title.getter
    def title(self):
        return accessors.getitem(objects.attrsof(self), "title", str(self.index))
    
    def __init__(self, *args, **kwargs):
        super().__init__(items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.index = -1
        
        objects.attrsof(self).update(kwargs)

Axis._codec = objects.Codec(Axis)
Axis._codec.items = Axis._codec

    
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


def as_color(expression):
    type = expressions.typeof(expression)
    
    if md.is_color_type(type):
        return expression
    
    return md.ordinal(expression)


class Property(objects.Object):
    title = objects.attrproperty("title", "Untitled")
    type = objects.attrproperty("type", None)
    expression = objects.attrproperty("expression", None)
    bound = objects.attrproperty("bound", False)
    visible = objects.attrproperty("visible", True)
    
    @property
    def data(self):
        return accessors.getitem(objects.attrsof(self), "data", self.default)
        
    @data.setter
    def data(self, value):
        accessors.setitem(objects.attrsof(self), "data", value)
        self.data_changed(self, (), value)
    
    @data.deleter
    def data(self):
        accessors.delitem(objects.attrsof(self), "data")
        self.data_changed(self, (), None)
    
    def __init__(self, *args, **kwargs):
        super().__init__(items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.default = kwargs.get("default", None)
            self.coerce = kwargs.get("coerce", utilities.identity)
            self.data_changed = Observable()
        
        objects.attrsof(self).update(kwargs)
    
    def __setitem__(self, key, value):
        def _notify(obj, key_, value_):
            self.data_changed(self, (key,) + key_, value_)
        
        old_value, new_value = accessors.getitem(objects.itemsof(self), key, None), value
        if old_value is not None:
            old_value.data_changed.unsubscribe(_notify)
        accessors.setitem(objects.itemsof(self), key, value)
        if new_value is not None:
            new_value.data_changed.subscribe(_notify)
    
    def bind(self, expression):
        self.expression = self.coerce(expression)
        self.bound = True
    
    def unbind(self):
        self.expression = None
        self.bound = False
    
    def __str__(self):
        return "{}: {}".format(self.title, self.type)

Property._codec = objects.Codec(Property)
Property._codec.items = Property._codec


class Mark(objects.Object):
    title = objects.attrproperty("title", None)
    rank = objects.attrproperty("rank", None)
    order = objects.attrproperty("order", None)
    owner = objects.attrproperty("owner", None)
    visible = objects.attrproperty("visible", True)
    
    @classmethod
    def default_properties(cls, obj):
        position_expression = md.Position[defaults.DEFAULT_N]()
        
        position = Property(title="Position", type="Position", expression=position_expression, default=None, visible=False)
        color = Property(title="Color", type="Color", coerce=as_color)
        color["primary"] = Property(title="Primary", type="Color", default="#0000ff", visible=False) 
        color["secondary"] = Property(title="Secondary", type="Color", default="#00ff00", visible=False) 
        color["palette"] = Property(title="Palette", type="String", default="brewer.ylgnbu", visible=False)
        color["h"] = Property(title="Hue", type="Decimal", coerce=md.ordinal, default=1.0, visible=False) 
        color["s"] = Property(title="Saturation", type="Decimal", coerce=md.ordinal, default=1.0) 
        color["v"] = Property(title="Value", type="Decimal", coerce=md.ordinal, default=1.0)
        color["a"] = Property(title="Alpha", type="Decimal", coerce=md.ordinal, default=1.0)
        shape = Property(title="Shape", type="Shape")
        shape["index"] = Property(title="Index", type="Integer", default=0, visible=False) 
        shape["palette"] = Property(title="Palette", type="String", default="geometric.geometric_black", visible=False)   
        texture = Property(title="Texture", type="Texture", default=None)
        size = Property(title="Size", type="Decimal", coerce=md.ordinal, default=1.0)
        label = Property(title="Label", type="String", default="", visible=False)
        motion = Property(title="Motion", type="Decimal", coerce=md.ordinal, default=0.0)
        
        obj.setdefault("position", position)
        obj.setdefault("color", color)
        obj.setdefault("shape", shape)
        obj.setdefault("texture", texture) 
        obj.setdefault("size", size)
        obj.setdefault("label", label)
        obj.setdefault("motion", motion)
    
    @property
    def position(self):
        return accessors.getattr(accessors.getitem(self, "position", None), "data", None)
    
    @property
    def color(self):
        return accessors.getattr(accessors.getitem(self, "color", None), "data", None)
    
    @property
    def shape(self):
        return accessors.getattr(accessors.getitem(self, "shape", None), "data", None)
    
    @property
    def size(self):
        return accessors.getattr(accessors.getitem(self, "size", None), "data", None)
    
    @property
    def motion(self):
        return accessors.getattr(accessors.getitem(self, "motion", None), "data", None)
    
    @property
    def color_h(self):
        return accessors.getattr(accessors.getitem(accessors.getitem(self, "color", None), "h", None), "data", None)
    
    @property
    def color_s(self):
        return accessors.getattr(accessors.getitem(accessors.getitem(self, "color", None), "s", None), "data", None)
    
    @property
    def color_v(self):
        return accessors.getattr(accessors.getitem(accessors.getitem(self, "color", None), "v", None), "data", None)
    
    @property
    def color_a(self):
        return accessors.getattr(accessors.getitem(accessors.getitem(self, "color", None), "a", None), "data", None)
    
    @property
    def color_primary(self):
        from .. import colors
        value = accessors.getattr(accessors.getitem(accessors.getitem(self, "color", None), "primary", None), "data", None)
        if isinstance(value, str):
            return colors.Color.from_hex(value)
        return value
        
    @property
    def color_secondary(self):
        from .. import colors
        value = accessors.getattr(accessors.getitem(accessors.getitem(self, "color", None), "secondary", None), "data", None)
        if isinstance(value, str):
            return colors.Color.from_hex(value)
        return value
    
    @property
    def color_primary_hsva(self):
        color = accessors.getitem(self, "color", None)
        return self.color_primary.hsva if color is not None and not color.bound else np.array([0,0,1,1], dtype=np.float32) #np.ones(4, dtype=np.float32)
        
    @property
    def color_secondary_hsva(self):
        color = accessors.getitem(self, "color", None)
        return self.color_secondary.hsva if color is not None and not color.bound else np.array([0,0,1,1], dtype=np.float32) #np.ones(4, dtype=np.float32)
    
    @property
    def color_modifiers_hsva(self):
        color = accessors.getitem(self, "color", None)
        color_h = accessors.getitem(color, "h", None)
        color_s = accessors.getitem(color, "s", None)
        color_v = accessors.getitem(color, "v", None)
        color_a = accessors.getitem(color, "a", None)
        h = accessors.getattr(color_h, "data", None) if color_h is not None else 1.0
        s = accessors.getattr(color_s, "data", None) if color_s is not None else 1.0
        v = accessors.getattr(color_v, "data", None) if color_v is not None else 1.0
        a = accessors.getattr(color_a, "data", None) if color_a is not None else 1.0
        return np.array([h, s, v, a], dtype=np.float32)
    
    @property
    def color_palettes(self):
        from .. import palettes
        value = accessors.getattr(accessors.getitem(accessors.getitem(self, "color", None), "palette", None), "data", None)
        return palettes.color_palettes(value)
    
    @property
    def shape_index(self):
        value = accessors.getattr(accessors.getitem(accessors.getitem(self, "shape", None), "index", None), "data", None)
        return value
    
    @property
    def shape_palettes(self):
        from .. import palettes
        value = accessors.getattr(accessors.getitem(accessors.getitem(self, "shape", None), "palette", None), "data", None)
        return palettes.shape_palettes(value)
    
    @property
    def title(self):
        return self.owner.title
    
    def __init__(self, order=None, owner=None):
        super().__init__(items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.data_changed = Observable()
        
        self.rank = order
        self.order = order
        self.owner = owner
        
        self.default_properties(self)
    
    def __setitem__(self, key, value):
        def _notify(obj, key_, value_):
            self.data_changed(self, (key,) + key_, value_)
        
        old_value, new_value = accessors.getitem(objects.itemsof(self), key, None), value
        if old_value is not None:
            old_value.data_changed.unsubscribe(_notify)
        accessors.setitem(objects.itemsof(self), key, value)
        if new_value is not None:
            new_value.data_changed.subscribe(_notify)

Mark._codec = objects.Codec(Mark, items=Property._codec)


class Guide(objects.Object):
    pass

Guide._codec = objects.Codec(Guide)


class Visualization(objects.Object):
    title = objects.attrproperty("title", "Untitled")
    
    @property
    def axes(self):
        return objects.attrsof(self).setdefault("axes", {})
    
    @property
    def marks(self):
        return objects.attrsof(self).setdefault("marks", {})
    
    @property
    def guides(self):
        return objects.attrsof(self).setdefault("guides", {})

    @classmethod
    def default_properties(cls, obj):
        show_background = Property(title="Background", type="Boolean", data=False)
        show_shading = Property(title="Shading", type="Boolean", data=False)
        show_overflow = Property(title="Overflow", type="Boolean", data=False)
        show_axes = Property(title="Axes", type="Boolean", data=True)
        show_marks = Property(title="Marks", type="Boolean", data=True)
        show_guides = Property(title="Guides", type="Boolean", data=True)

        obj.setdefault("show_background", show_background)
        obj.setdefault("show_shading", show_shading)
        obj.setdefault("show_overflow", show_overflow)
        obj.setdefault("show_axes", show_axes)
        obj.setdefault("show_marks", show_marks)
        obj.setdefault("show_guides", show_guides)
        
        obj.axes.setdefault("root", Axis(group=True))
    
    @property
    def show_background(self):
        return accessors.getattr(accessors.getitem(self, "show_background", None), "data", False)
    
    @property
    def show_shading(self):
        return accessors.getattr(accessors.getitem(self, "show_shading", None), "data", False)
    
    @property
    def show_overflow(self):
        return accessors.getattr(accessors.getitem(self, "show_overflow", None), "data", False)
    
    @property
    def show_axes(self):
        return accessors.getattr(accessors.getitem(self, "show_axes", None), "data", True)
    
    @property
    def show_marks(self):
        return accessors.getattr(accessors.getitem(self, "show_marks", None), "data", True)
    
    @property
    def show_guides(self):
        return accessors.getattr(accessors.getitem(self, "show_guides", None), "data", True)
    
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data=data, items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.axis_identifiers = functions.counter(0)
            self.mark_identifiers = functions.counter(0)
            
            self.data_changed = Observable()
            self.binding_changed = Observable()
            self.visible_changed = Observable()
            
        objects.attrsof(self).update(kwargs)
        
        self.default_properties(self)
    
    def toggle(self, index):
        self.visible_changed(index)
    
    def get_or_create_axis(self, key, *args, **kwargs):
        return self.axes['root']
    
    def get_or_create_mark(self, key, *args, **kwargs):
        result = accessors.getitem(self.marks, key, None)
        
        if result is None:
            result = Mark(*args, **kwargs)
            result.data_changed.subscribe(self.data_changed)
            accessors.setitem(self.marks, key, result)
        
        return result
        
    def get_index(self, axis, index):
        return next(iter(x for x in recursive_itervalues(axis) if x.index == index), None)
    
    def create_axis(self, parent, type):
        # TODO: This is a mess...
        root = self.axes['root']
        
        def next_index():
            return len([x for x in recursive_itervalues(root) if not x.group])
        
        def update_indices():
            leaves = [x for x in recursive_itervalues(root) if not x.group]
            for i, leaf in enumerate(leaves):
                leaf.index = i
        
        index = next_index()
        
        if index < defaults.DEFAULT_N:
            key = str(self.axis_identifiers())
            parent[key] = Axis(type=type.name)
            update_indices()
            return index
        else:
            logger.warn('Exhausted dimensions "{}/{}"'.format(index, defaults.DEFAULT_N))
            return None
    
    def bind(self, mark, key, expression):
        property = accessors.getitempath(mark, key)
        if property is not None:
            property.bind(expression)
            self.binding_changed(mark)
        else:
            logger.warn('Unknown property "{}".'.format(key))
    
    def unbind(self, mark, key):
        property = accessors.getitempath(mark, key)
        if property is not None:
            property.unbind()
            self.binding_changed(mark)
        else:
            logger.warn('Unknown property "{}".'.format(key))
    
    def auto_bind(self, expression, mark, axis):
        type = expressions.typeof(expression)
        
        expression = as_position(expression)
        size = expressions.typeof(expression).size
        
        result = mark["position"].expression
        for i in range(size):
            item = expression[i]
            index = self.create_axis(axis, md.typeof(item))
            
            if index is not None:
                result[index] = item
                self.toggle(index)
        
        self.bind_position(mark, result)
    
    def auto_bind_origin(self, expression, mark, axis):
        type = expressions.typeof(expression)
        
        index = self.create_axis(axis, type)
        
        if index is not None:
            self.bind_position_item(mark, index, expression)
        
        self.binding_changed(mark)
        
    def auto_bind_axis(self, expression, mark, axis):
        self.bind_position_item(mark, axis.index, expression)
        self.binding_changed(mark)
    
    def bind_position(self, mark, expression):
        mark["position"].bind(expression)
        self.binding_changed(mark)
    
    def bind_position_item(self, mark, key, expression):
        result = mark['position'].expression 
        result[key] = as_number(expression)
    
    def bind_color(self, mark, expression):
        mark["color"].bind(expression)
        self.binding_changed(mark)
    
    def bind_shape(self, mark, expression):
        mark["shape"].bind(expression)
        self.binding_changed(mark)
    
    def bind_size(self, mark, expression):
        mark["size"].bind(expression)
        self.binding_changed(mark)
    
    def bind_motion(self, mark, expression):
        mark["motion"].bind(expression)
        self.binding_changed(mark)


Visualization._codec = objects.Codec(Visualization, attrs={
    "axes": objects.Codec(items=Axis._codec),
    "marks": objects.Codec(items=Mark._codec),
    "guides": objects.Codec(items=Guide._codec),
}, items=Property._codec)
