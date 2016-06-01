import collections
import enum
import logging

from encore import accessors, functions, objects, observables, utilities

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


class Group(objects.Object):
    title = objects.attrproperty("title", "Untitled")
    visible = objects.attrproperty("visible", True)
    
    def __init__(self, items, *args, **kwargs):
        super().__init__(items=items)
        
        with objects.direct_access(self):
            self.data_changed = observables.Observable()
        
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
            self.data_changed = observables.Observable()
        
        objects.attrsof(self).update(kwargs)

    
Axis._codec = objects.Codec(Axis)


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


class SortState(enum.IntEnum):
    Null = 0
    SortAscending = 1
    SortDescending = 2
    
    @classmethod
    def next(cls, value, step=1):
        return cls((value + step) % len(cls))


class GroupState(enum.IntEnum):
    Null = 0
    Group = 1
    
    @classmethod
    def next(cls, value, step=1):
        return cls((value + step) % len(cls))


class Property(objects.Object):
    title = objects.attrproperty("title", "Untitled")
    type = objects.attrproperty("type", None)
    expression = objects.attrproperty("expression", None)
    bound = objects.attrproperty("bound", False)
    sorting = objects.attrproperty("sorting", SortState.Null)
    grouping = objects.attrproperty("grouping", GroupState.Null)
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
            self.data_changed = observables.Observable()
            self.options_changed = observables.Observable()
            
            self.options = kwargs.get("options", None)
            self.default = kwargs.get("default", None)
            self.coerce = kwargs.get("coerce", utilities.identity)
        
        objects.attrsof(self).update(kwargs)
    
    def __getitem__(self, key):
        key = str(key)
        
        return accessors.getitem(objects.itemsof(self), key)
    
    def __setitem__(self, key, value):
        key = str(key)
        
        def _notify(obj, key_, value_):
            self.data_changed(self, (key,) + key_, value_)
        
        old_value, new_value = accessors.getitem(objects.itemsof(self), key, None), value
        if old_value is not None:
            old_value.data_changed.unsubscribe(_notify)
        accessors.setitem(objects.itemsof(self), key, value)
        if new_value is not None:
            new_value.data_changed.subscribe(_notify)
    
    def __delitem__(self, key):
        key = str(key)
        
        old_value = accessors.getitem(objects.itemsof(self), key, None)
        if old_value is not None:
            old_value.data_changed.unsubscribe(_notify)
        accessors.delitem(objects.itemsof(self), key)
    
    def bind(self, expression):
        self.expression = expression
        #self.sorting = SortState.Null
        #self.grouping = GroupState.Null
        self.bound = True
    
    def unbind(self):
        self.expression = None
        #self.sorting = SortState.Null
        #self.grouping = GroupState.Null
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
        position = Property(title="Position", type="Vector", default=None, visible=False)
        for i in range(defaults.DEFAULT_N):
            position[i] = Property(title=str(i), type="Decimal", default=None, visible=False)
        
        size = Property(title="Size", type="Vector", default=None, visible=False)
        for i in range(defaults.DEFAULT_N):
            size[i] = Property(title=str(i), type="Decimal", default=None, visible=False)
        
        offset = Property(title="Offset", type="Vector", default=None, visible=False)
        for i in range(defaults.DEFAULT_N):
            offset[i] = Property(title=str(i), type="Decimal", default=None, visible=False)
        
        color = Property(title="Color", type="Color", options=accessors.getitem(options, "color", None))
        color["primary"] = Property(title="Primary", type="Color", default="#ffffff", visible=False) 
        color["secondary"] = Property(title="Secondary", type="Color", default="#ffffff", visible=False) 
        color["palette"] = Property(title="Palette", type="String", default="brewer.ylgnbu", visible=False)
        color["h"] = Property(title="Hue", type="Decimal", default=1.0, visible=False) 
        color["s"] = Property(title="Saturation", type="Decimal", default=1.0) 
        color["v"] = Property(title="Value", type="Decimal", default=1.0)
        color["a"] = Property(title="Alpha", type="Decimal", default=1.0, visible=False)
        
        shape = Property(title="Shape", type="Shape", options=accessors.getitem(options, "shape", None))
        shape["index"] = Property(title="Index", type="Integer", default=0, visible=False)
        shape["palette"] = Property(title="Palette", type="String", default="geometric.geometric_black", visible=False)
        
        texture = Property(title="Texture", type="Texture", default=None, options=accessors.getitem(options, "texture", None))
        texture["index"] = Property(title="Index", type="Integer", default=0, visible=False)
        texture["palette"] = Property(title="Palette", type="String", default="basic.grid", visible=False)
        
        delta_size = Property(title="Size", type="Decimal", default=0.5)
        delta_offset = Property(title="Motion", type="Decimal", default=0.0)
        
        label = Property(title="Label", type="String", default="")
        
        obj.setdefault("position", position)
        obj.setdefault("size", size)
        obj.setdefault("offset", offset)
        obj.setdefault("color", color)
        obj.setdefault("shape", shape)
        obj.setdefault("texture", texture) 
        obj.setdefault("delta_size", delta_size)
        obj.setdefault("delta_offset", delta_offset)
        obj.setdefault("label", label)
    
    @property
    def position(self):
        return accessors.getattr(accessors.getitem(self, "position", None), "data", None)
    
    @property
    def size(self):
        return accessors.getattr(accessors.getitem(self, "size", None), "data", None)
    
    @property
    def offset(self):
        return accessors.getattr(accessors.getitem(self, "offset", None), "data", None)
    
    @property
    def color(self):
        return accessors.getattr(accessors.getitem(self, "color", None), "data", None)
    
    @property
    def shape(self):
        return accessors.getattr(accessors.getitem(self, "shape", None), "data", None)
    
    @property
    def delta_size(self):
        return accessors.getattr(accessors.getitem(self, "delta_size", None), "data", None)
    
    @property
    def delta_offset(self):
        return accessors.getattr(accessors.getitem(self, "delta_offset", None), "data", None)
    
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
    def default_color_palette(self):
        return accessors.getattr(accessors.getitem(accessors.getitem(self, "color", None), "palette", None), "default")
    
    @default_color_palette.setter
    def default_color_palette(self, value):
        return accessors.setattr(accessors.getitem(accessors.getitem(self, "color", None), "palette", None), "default", value)
    
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
    
    @property
    def sortby(self):
        return list(filter(accessors.attrgetter("bound"), [accessors.getitempath(self, path) for path in self._sortby]))
    
    @property
    def groupby(self):
        return list(filter(accessors.attrgetter("bound"), [accessors.getitempath(self, path) for path in self._groupby]))
    
    def __init__(self, order=None, owner=None, options=None):
        super().__init__(items=collections.OrderedDict())
        
        with objects.direct_access(self):
            self.data_changed = observables.Observable()
            
            self._sortby = list()
            self._groupby = list()
            
            self.total = 0
            self.count = 0
        
        self.rank = order
        self.order = order
        self.owner = owner
        
        self.default_properties(self, options=options)
    
    def __getitem__(self, key):
        return accessors.getitem(objects.itemsof(self), key)
    
    def __setitem__(self, key, value):
        def _notify(obj, key_, value_):
            self.data_changed(self, (key,) + key_, value_)
        
        old_value, new_value = accessors.getitem(objects.itemsof(self), key, None), value
        if old_value is not None:
            old_value.data_changed.unsubscribe(_notify)
        accessors.setitem(objects.itemsof(self), key, value)
        if new_value is not None:
            new_value.data_changed.subscribe(_notify)
    
    def __delitem__(self, key):
        old_value = accessors.getitem(objects.itemsof(self), key, None)
        if old_value is not None:
            old_value.data_changed.unsubscribe(_notify)
        accessors.delitem(objects.itemsof(self), key)

Mark._codec = objects.Codec(Mark, items=Property._codec)


class Guide(objects.Object):
    pass

Guide._codec = objects.Codec(Guide)


class Arrangement(enum.IntEnum):
    Perpendicular = 0
    PerpendicularDisjoint = 1
    Parallel = 2
    Multiple = 3
    
    @classmethod
    def next(cls, value, step=1):
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


class MultipleAxisView(collections.Mapping):
    def __init__(self, axes):
        super().__init__()
        
        self._axes = axes
    
    def __getitem__(self, key):
        return Group(self._axes)
    
    def __iter__(self):
        return iter(["root"])
    
    def __len__(self):
        return 1


def key_to_path(key, sep="."):
    if isinstance(key, str):
        return str(key).split(sep)
    return key


def path_to_key(path, sep="."):
    if isinstance(path, str):
        return path
    return sep.join(map(str, path))


def iter_paths_where(obj, predicate, path=None):
    path = [] if path is None else path
    for key, value in objects.itemsof(obj).items():
        sub_path = path + [key]
        if predicate(value):
            yield sub_path
        yield from iter_paths_where(value, predicate, sub_path)


class FilteredObjectProxy(collections.Mapping):
    def __init__(self, obj, predicate):
        super().__init__()
        
        self._obj = obj
        self._predicate = predicate
    
    def __getattr__(self, key):
        return accessors.getattr(self._obj, key)
    
    def __getitem__(self, key):
        return accessors.getitempath(self._obj, key_to_path(key))
    
    def __iter__(self):
        return iter(path_to_key(path) for path in iter_paths_where(self._obj, self._predicate))
    
    def __len__(self):
        return sum(1 for path in iter_paths_where(self._obj, self._predicate))


class VariablesProxy(collections.Mapping):
    def __init__(self, marks):
        super().__init__()
        
        self._marks = marks
    
    def __getitem__(self, key):
        is_bound = accessors.attrgetter("bound")
        return FilteredObjectProxy(self._marks[key], is_bound)
    
    def __iter__(self):
        return iter(self._marks)
    
    def __len__(self):
        return len(self._marks)


class Visualization(objects.Object):
    title = objects.attrproperty("title", "Untitled")
    arrangement = objects.attrproperty("arrangement", Arrangement.Perpendicular)
    
    @arrangement.setter
    def arrangement(self, value):
        accessors.setitem(objects.attrsof(self), "arrangement", value)
        self.data_changed(self, ("arrangement",), value)
    
    @property
    def variables(self):
        return VariablesProxy(self.marks) 
    
    @property
    def axes(self):
        return objects.attrsof(self).setdefault("axes", collections.OrderedDict())
    
    @property
    def axes_view(self):
        if self.arrangement == Arrangement.Perpendicular:
            return PerpendicularAxisView(self.axes)
        if self.arrangement == Arrangement.PerpendicularDisjoint:
            return PerpendicularAxisView(self.axes)
        if self.arrangement == Arrangement.Parallel:
            return ParallelAxisView(self.axes)
        if self.arrangement == Arrangement.Multiple:
            return MultipleAxisView(self.axes)
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
        show_labels = Property(title="Labels", type="Boolean", data=False)
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
            
            self.selection = set()
            
            self.data_changed = observables.Observable()
            self.axis_data_changed = observables.Observable()
            self.mark_data_changed = observables.Observable()
            self.binding_changed = observables.Observable()
        
        objects.attrsof(self).update(kwargs)
        
        self.default_properties(self)
    
    def get_or_create_axis(self, key, *args, **kwargs):
        result = accessors.getitem(self.axes, key, None)
        
        if result is None:
            result = Axis(*args, **kwargs)
            result.data_changed.subscribe(self.axis_data_changed)
            accessors.setitem(self.axes, key, result)
        
        return result
    
    def remove_axis(self, key):
        accessors.delitem(self.axes, key)
    
    def get_or_create_mark(self, key, *args, **kwargs):
        result = accessors.getitem(self.marks, key, None)
        
        if result is None:
            result = Mark(*args, **kwargs)
            result.data_changed.subscribe(self.mark_data_changed)
            accessors.setitem(self.marks, key, result)
        
        return result
    
    def remove_mark(self, key):
        accessors.delitem(self.marks, key)
    
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
        key = key_to_path(key)
        property = accessors.getitempath(mark, key)
        if property is not None:
            property.bind(expression)
            self.binding_changed(mark)
        else:
            logger.warn('Unknown property "{}".'.format(key))
    
    def unbind(self, mark, key):
        key = key_to_path(key)
        property = accessors.getitempath(mark, key)
        if property is not None:
            property.unbind()
            self.binding_changed(mark)
        else:
            logger.warn('Unknown property "{}".'.format(key))
    
    def sort_by(self, mark, key):
        key = key_to_path(key)
        property = accessors.getitempath(mark, key)
        if property is not None:
            property.sorting = SortState.next(property.sorting)
            if property.sorting == 0:
                mark._sortby.remove(key)
            if property.sorting == 1:
                mark._sortby.append(key)
            self.binding_changed(mark)
        
    def group_by(self, mark, key):
        key = key_to_path(key)
        property = accessors.getitempath(mark, key)
        if property is not None:
            property.grouping = GroupState.next(property.grouping)
            if property.grouping == 0:
                mark._groupby.remove(key)
            if property.grouping == 1:
                mark._groupby.append(key)
            self.binding_changed(mark)
    
    def auto_bind(self, expression, mark):
        type = expressions.typeof(expression)
        
        position = as_position(expression)
        size = expressions.typeof(position).size
        
        for i in range(size):
            item = position[i]
            type = md.typeof(item)
            index = self.create_axis(type)
            
            if index is not None:
                mark["position"][index].bind(item)
        
        self.binding_changed(mark)
        
    def auto_bind_origin(self, expression, mark):
        type = md.typeof(expression)
        index = self.create_axis(type)
        
        if index is not None:
            mark["position"][index].bind(expression)
        
        self.binding_changed(mark)
    
    def auto_bind_axis(self, expression, mark, axis):
        self.bind_position_item(mark, axis.index, expression)
    
    def auto_bind_mark(self, expression, mark):
        type = md.typeof(expression)
        
        if md.is_color_type(type):
            self.bind_color(mark, expression)
            return
        
        if md.is_shape_type(type):
            self.bind_shape(mark, expression)
            return
        
        if md.is_categorical(type) or md.is_numerical(type):
            self.bind_color(mark, expression)
        else:
            self.bind_label(mark, expression)
    
    def bind_position(self, mark, expression):
        mark["position"].bind(expression)
        self.binding_changed(mark)
    
    def bind_position_item(self, mark, key, expression):
        mark["position"][key].bind(expression)
        self.binding_changed(mark)
    
    def bind_color(self, mark, expression):
        mark["color"].bind(expression)
        self.binding_changed(mark)
    
    def bind_color_item(self, mark, key, expression):
        mark["color"][key].bind(expression)
        self.binding_changed(mark)
    
    def bind_shape(self, mark, expression):
        mark["shape"].bind(expression)
        self.binding_changed(mark)
    
    def bind_label(self, mark, expression):
        mark["label"].bind(expression)
        self.binding_changed(mark)
    
    def bind_delta_size(self, mark, expression):
        mark["delta_size"].bind(expression)
        self.binding_changed(mark)
    
    def bind_delta_offset(self, mark, expression):
        mark["delta_offset"].bind(expression)
        self.binding_changed(mark)


Visualization._codec = objects.Codec(Visualization, attrs={
    "axes": objects.Codec(items=Axis._codec),
    "marks": objects.Codec(items=Mark._codec),
    "guides": objects.Codec(items=Guide._codec),
}, items=Property._codec)
