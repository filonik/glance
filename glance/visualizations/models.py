import collections
import enum
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


class Group(objects.Object):
    title = objects.attrproperty("title", "Untitled")
    visible = objects.attrproperty("visible", True)
    
    def __init__(self, items, *args, **kwargs):
        super().__init__(items=items)
        
        with objects.direct_access(self):
            self.data_changed = Observable()
        
        objects.attrsof(self).update(kwargs)

    def increment_angular(self):
        items = objects.itemsof(self)
        angular_count = sum(item.angular for item in items.values()) 
        angular_count = (angular_count + 1) % len(items)
        for i, item in enumerate(items.values()):
            item.angular = i < angular_count


class Axis(objects.Object):
    index = objects.attrproperty("index", -1)
    title = objects.attrproperty("title", "Untitled")
    type = objects.attrproperty("type", None)
    angular = objects.attrproperty("angular", False)
    visible = objects.attrproperty("visible", True)
    
    @title.getter
    def title(self):
        return accessors.getitem(objects.attrsof(self), "title", str(self.index))
      
    @angular.setter
    def angular(self, value):
        accessors.setitem(objects.attrsof(self), "angular", value)
        self.data_changed(self, ("angular",), value)
        
    @visible.setter
    def visible(self, value):
        accessors.setitem(objects.attrsof(self), "visible", value)
        self.data_changed(self, ("visible",), value)
    
    def __init__(self, *args, **kwargs):
        super().__init__(items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.data_changed = Observable()
        
        objects.attrsof(self).update(kwargs)

    
Axis._codec = objects.Codec(Axis)

    
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
    
    @property
    def options(self):
        return self._options
    
    @options.setter
    def options(self, value):
        self._options = value
        self.options_changed(self, (), value)
    
    @options.deleter
    def options(self):
        del self._options
        self.options_changed(self, (), None)
    
    def __init__(self, *args, **kwargs):
        super().__init__(items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.data_changed = Observable()
            self.options_changed = Observable()
            
            self.options = kwargs.get("options", None)
            self.default = kwargs.get("default", None)
            self.coerce = kwargs.get("coerce", utilities.identity)
        
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
    def default_properties(cls, obj, options=None):
        position_expression = md.Position[defaults.DEFAULT_N]()
        
        position = Property(title="Position", type="Position", expression=position_expression, default=None, visible=False)
        
        color = Property(title="Color", type="Color", options=accessors.getitem(options, "color", None), coerce=as_color)
        color["primary"] = Property(title="Primary", type="Color", default="#ffffff", visible=False) 
        color["secondary"] = Property(title="Secondary", type="Color", default="#ffffff", visible=False) 
        color["palette"] = Property(title="Palette", type="String", default="brewer.ylgnbu", visible=False)
        color["h"] = Property(title="Hue", type="Decimal", coerce=md.ordinal, default=1.0, visible=False) 
        color["s"] = Property(title="Saturation", type="Decimal", coerce=md.ordinal, default=1.0) 
        color["v"] = Property(title="Value", type="Decimal", coerce=md.ordinal, default=1.0)
        color["a"] = Property(title="Alpha", type="Decimal", coerce=md.ordinal, default=1.0)
        
        shape = Property(title="Shape", type="Shape", options=accessors.getitem(options, "shape", None))
        shape["index"] = Property(title="Index", type="Integer", default=0, visible=False)
        shape["palette"] = Property(title="Palette", type="String", default="geometric.geometric_black", visible=False)
        
        texture = Property(title="Texture", type="Texture", default=None, options=accessors.getitem(options, "texture", None))
        texture["index"] = Property(title="Index", type="Integer", default=0, visible=False)
        texture["palette"] = Property(title="Palette", type="String", default="basic.grid", visible=False)
        
        size = Property(title="Size", type="Decimal", coerce=md.ordinal, default=0.5)
        offset = Property(title="Offset", type="Decimal", coerce=md.ordinal, default=0.0, visible=False)
        motion = Property(title="Motion", type="Decimal", coerce=md.ordinal, default=0.0)
        
        label = Property(title="Label", type="String", default="")
        
        obj.setdefault("position", position)
        obj.setdefault("color", color)
        obj.setdefault("shape", shape)
        obj.setdefault("texture", texture) 
        obj.setdefault("size", size)
        obj.setdefault("offset", offset)
        obj.setdefault("motion", motion)
        obj.setdefault("label", label)
    
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
    def label(self):
        return accessors.getattr(accessors.getitem(self, "label", None), "data", None)
    
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
    def texture(self):
        return accessors.getattr(accessors.getitem(self, "texture", None), "data", None)
    
    @property
    def title(self):
        return self.owner.title
    
    def __init__(self, order=None, owner=None, options=None):
        super().__init__(items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.data_changed = Observable()
        
        self.rank = order
        self.order = order
        self.owner = owner
        
        self.default_properties(self, options=options)
    
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


class Arrangement(enum.IntEnum):
    Perpendicular = 0
    Parallel = 1
    
    @classmethod
    def cycle(cls, value, step=1):
        return cls((value + step) % len(cls))

class PerpendicularAxisView(collections.Mapping):
    def __init__(self, axes):
        super().__init__()
        
        self._axes = axes
    
    def __getitem__(self, key):
        return Group(self._axes)
    
    def __iter__(self):
        return iter(["root"])
    
    def __len__(self):
        return 1


class ParallelAxisView(collections.Mapping):
    def __init__(self, axes):
        super().__init__()
        
        self._axes = axes
    
    def __getitem__(self, key):
        return Group({key: self._axes[key]})
    
    def __iter__(self):
        return iter(self._axes)
    
    def __len__(self):
        return len(self._axes)


class Visualization(objects.Object):
    title = objects.attrproperty("title", "Untitled")
    arrangement = objects.attrproperty("arrangement", Arrangement.Perpendicular)
    
    @property
    def axes(self):
        return objects.attrsof(self).setdefault("axes", collections.OrderedDict())
    
    @property
    def axes_view(self):
        if self.arrangement == Arrangement.Perpendicular:
            return PerpendicularAxisView(self.axes)
        if self.arrangement == Arrangement.Parallel:
            return ParallelAxisView(self.axes)
        return None
    
    @property
    def marks(self):
        return objects.attrsof(self).setdefault("marks", collections.OrderedDict())
    
    @property
    def guides(self):
        return objects.attrsof(self).setdefault("guides", collections.OrderedDict())
    
    @classmethod
    def default_properties(cls, obj):
        show_background = Property(title="Background", type="Boolean", data=True)
        show_shading = Property(title="Shading", type="Boolean", data=False)
        show_overflow = Property(title="Overflow", type="Boolean", data=False)
        show_axes = Property(title="Axes", type="Boolean", data=True)
        show_marks = Property(title="Marks", type="Boolean", data=True)
        show_joins = Property(title="Joins", type="Boolean", data=True)
        show_labels = Property(title="Labels", type="Boolean", data=True)
        show_guides = Property(title="Guides", type="Boolean", data=False)
        
        obj.setdefault("show_background", show_background)
        obj.setdefault("show_shading", show_shading)
        obj.setdefault("show_overflow", show_overflow)
        obj.setdefault("show_axes", show_axes)
        obj.setdefault("show_marks", show_marks)
        obj.setdefault("show_joins", show_joins)
        obj.setdefault("show_labels", show_labels)
        obj.setdefault("show_guides", show_guides)
    
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
    def show_joins(self):
        return accessors.getattr(accessors.getitem(self, "show_joins", None), "data", True)
    
    @property
    def show_labels(self):
        return accessors.getattr(accessors.getitem(self, "show_labels", None), "data", True)
    
    @property
    def show_guides(self):
        return accessors.getattr(accessors.getitem(self, "show_guides", None), "data", True)
    
    def __init__(self, data=None, *args, **kwargs):
        super().__init__(data=data, items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.axis_identifiers = functions.counter(0)
            self.mark_identifiers = functions.counter(0)
            
            self.axis_data_changed = Observable()
            self.mark_data_changed = Observable()
            self.binding_changed = Observable()
        
        objects.attrsof(self).update(kwargs)
        
        self.default_properties(self)
    
    def get_or_create_axis(self, key, *args, **kwargs):
        result = accessors.getitem(self.axes, key, None)
        
        if result is None:
            result = Axis(*args, **kwargs)
            result.data_changed.subscribe(self.axis_data_changed)
            accessors.setitem(self.axes, key, result)
        
        return result
    
    def get_or_create_mark(self, key, *args, **kwargs):
        result = accessors.getitem(self.marks, key, None)
        
        if result is None:
            result = Mark(*args, **kwargs)
            result.data_changed.subscribe(self.mark_data_changed)
            accessors.setitem(self.marks, key, result)
        
        return result
        
    def get_axis_by_index(self, index):
        return next(iter(axis for axis in self.axes.values() if axis.index == index), None)
    
    def create_axis(self, type):
        axis_count = len(self.axes)
        
        def update_axis_indices(axes):
            for index, value in enumerate(axes.values()):
                value.index = index
        
        if axis_count < defaults.DEFAULT_N:
            key = str(self.axis_identifiers())
            self.get_or_create_axis(key, type=type.name)
            update_axis_indices(self.axes)
            return axis_count
        else:
            logger.warn('Exhausted dimensions "{}/{}"'.format(axis_count, defaults.DEFAULT_N))
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
    
    def auto_bind(self, expression, mark):
        type = expressions.typeof(expression)
        
        expression = as_position(expression)
        size = expressions.typeof(expression).size
        
        result = mark["position"].expression
        for i in range(size):
            item = expression[i]
            type = md.typeof(item)
            index = self.create_axis(type)
            
            if index is not None:
                result[index] = item
        
        self.bind_position(mark, result)
    
    def auto_bind_origin(self, expression, mark):
        type = md.typeof(expression)
        index = self.create_axis(type)
        
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
