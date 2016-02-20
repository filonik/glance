import enum

from encore import accessors


class Key(enum.IntEnum):
    """ http://doc.qt.io/qt-4.8/qt.html#Key-enum """
    
    Escape = 0x01000000
    Tab = 0x01000001
    Backspace = 0x01000003
    Return = 0x01000004
    Enter = 0x01000005
    
    Left = 0x01000012
    Up = 0x01000013
    Right = 0x01000014
    Down = 0x01000015
    
    _0 = 0x30
    _1 = 0x31
    _2 = 0x32
    _3 = 0x33
    _4 = 0x34
    _5 = 0x35
    _6 = 0x36
    _7 = 0x37
    _8 = 0x38
    _9 = 0x39
    
    Colon = 0x3a
    Semicolon = 0x3b
    Less = 0x3c
    Equal = 0x3d
    Greater = 0x3e
    Question = 0x3f
    At = 0x40
    
    A = 0x41
    B = 0x42
    C = 0x43
    D = 0x44
    E = 0x45
    F = 0x46
    G = 0x47
    H = 0x48
    I = 0x49
    J = 0x4a
    K = 0x4b
    L = 0x4c
    M = 0x4d
    N = 0x4e
    O = 0x4f
    P = 0x50
    Q = 0x51
    R = 0x52
    S = 0x53
    T = 0x54
    U = 0x55
    V = 0x56
    W = 0x57
    X = 0x58
    Y = 0x59
    Z = 0x5a


class KeyState(object):
    def __init__(self, identifier=-1):
        super().__init__()
        
        self.identifier = identifier
    
    def __repr__(self):
        return "{}{{{}}}".format(type(self).__name__, repr(self.__dict__))


class DropState(object):
    def __init__(self, position, identifier=-1, data=None):
        super().__init__()
        
        self.identifier = identifier
        self.position = position
        self.data = data
        self.target = 0

    def __repr__(self):
        return "{}{{{}}}".format(type(self).__name__, repr(self.__dict__))

        
class TouchState(object):
    def __init__(self, position, identifier=-1):
        super().__init__()
        
        self.identifier = identifier
        self.position = position
        self.target = 0
    
    def __repr__(self):
        return "{}{{{}}}".format(type(self).__name__, repr(self.__dict__))


class Input(object):
    @property
    def keys(self):
        return self._keys
    
    @property
    def drops(self):
        return self._drops        
    
    @property
    def touches(self):
        return self._touches
    
    def __init__(self):
        super().__init__()
        
        self._keys = {}
        self._drops = []
        self._touches = {}
        
    def drop(self, value):
        self._drops.append(value)
    
    def key_press(self, value):
        accessors.setitem(self._keys, value.identifier, value)
        
    def key_release(self, value):
        accessors.delitem(self._keys, value.identifier)
    
    def touch_press(self, value):
        accessors.setitem(self._touches, value.identifier, value)
    
    def touch_move(self, value):
        try:
            accessors.getitem(self._touches, value.identifier)
            accessors.setitem(self._touches, value.identifier, value)
        except Exception:
            pass
    
    def touch_release(self, value):
        accessors.delitem(self._touches, value.identifier)
