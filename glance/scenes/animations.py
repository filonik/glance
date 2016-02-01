import numpy as np

# Robert Penner's Easing Equations

def ease_none(t):
    return t


def ease_quad_in(t):
    return t*t


def ease_cubic_in(t):
    return t*t*t


def ease_quart_in(t):
    return t*t*t*t


def ease_quint_in(t):
    return t*t*t*t*t


def ease_quad_out(t):
    t -= 1.0
    return 1.0 - ease_quad_in(t)


def ease_cubic_out(t):
    t -= 1.0
    return 1.0 + ease_cubic_in(t)


def ease_quart_out(t):
    t -= 1.0
    return 1.0 - ease_quart_in(t)


def ease_quint_out(t):
    t -= 1.0
    return 1.0 + ease_quint_in(t)


def ease_sine_in(t):
    return 1.0 - np.cos(t * np.pi/2)


def ease_sine_out(t):
    return np.sin(t * np.pi/2)


def ease_sine_in_out(t):
    return -0.5 * (np.cos(t * np.pi) - 1.0) 


def _ease_bounce_helper(t, c, a):
    if t == 1:
        return c
    elif t < (4/11.0):
        return c*(7.5625*t*t)
    elif t < (8/11.0):
        t -= (6/11.0)
        return -a * (1.0 - (7.5625*t*t + 0.75)) + c
    elif t < (10/11.0):
        t -= (9/11.0)
        return -a * (1.0 - (7.5625*t*t + 0.9375)) + c
    else:
        t -= (21/22.0)
        return -a * (1.0 - (7.5625*t*t + 0.984375)) + c


def ease_bounce_in(t, a=1.70158):
    return 1.0 - _ease_bounce_helper(1.0-t, 1.0, a)


def ease_bounce_out(t, a=1.70158):
    return _ease_bounce_helper(t, 1.0, a);


'''#???
def ease_quad_in_out(t):
    t *= 2.0
    if t < 1.0:
        return +0.5 * ease_quad_in(t)
    else:
        t -= 1
        return -0.5 * ease_quad_out(t) 
'''


def wrap_clamp(t):
    return np.clip(t, 0.0, 1.0)


def wrap_repeat(t):
    return np.mod(t, 1.0)


def wrap_reverse(t):
    return (1.0 - np.mod(t, 1.0)) if int(t) % 2 else np.mod(t, 1.0)


def play(animation):
    import time
    def _play(state):
        animation.play(time.time(), state)
    return _play


def toggle(animation):
    import time
    def _toggle(event):
        animation.reverse(time.time())
    return _toggle


class Animation(object):
    def __init__(self, start, duration=1.0, wrap=wrap_clamp, ease=ease_none, reversed=False):
        self.start = start
        self.duration = duration
        self.wrap = wrap
        self.ease = ease
        self.reversed = reversed
    
    @property
    def end(self):
        return self.start + self.duration
    
    def progress(self, t):
        return t - self.start
    
    def remainder(self, t):
        return self.end - t
    
    def has_started(self, t):
        return 0.0 < self.progress(t)
    
    def has_finished(self, t):
        return self.remainder(t) < 0.0
    
    def is_active(self, t):
        return self.has_started(t) and not self.has_finished(t)
    
    def play(self, t, reversed=False):
        self.start = t - max(0, self.remainder(t))
        self.reversed = reversed

    def reverse(self, t):
        self.start = t - max(0, self.remainder(t))
        self.reversed = not self.reversed

    def __call__(self, t):
        alpha = self.progress(t)/self.duration
        alpha = self.wrap(alpha)
        alpha = 1.0 - alpha if self.reversed else alpha
        alpha = self.ease(alpha)
        return alpha