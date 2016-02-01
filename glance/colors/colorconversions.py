import numpy as np


def astype(obj, type):
    try:
        return obj.astype(type)
    except AttributeError:
        return type(obj)


def _n_chunks(n, seq):
    step = len(seq) // n
    return [seq[i:i+step] for i in range(0, step*n, step)]


def _hex_to_uint8(value):
    return np.array(list(map(lambda s: int(s, 16), _n_chunks(3, value.lstrip('#')))) + [255], dtype=np.uint8)


def _uint8_to_hex(value):
    return '#%02x%02x%02x' % tuple(value[:3])


def _float_to_uint8(value):
    return np.array(list(map(lambda f: f*255.0, value)), dtype=np.uint8)


def _uint8_to_float(value):
    return np.array(list(map(lambda i: i/255.0, value)), dtype=np.float32)


def save_hex(value):
    return _uint8_to_hex(_float_to_uint8(value))


def load_hex(value):
    return _uint8_to_float(_hex_to_uint8(value))


def rgb_to_hsv(r, g, b):
    maxc = np.maximum(r, np.maximum(g, b))
    minc = np.minimum(r, np.minimum(g, b))

    v = maxc

    minc_eq_maxc = np.equal(minc, maxc)

    # compute the difference, but reset zeros to ones to avoid divide by zeros later.
    ones = np.ones_like(r)
    maxc_minus_minc = np.choose(minc_eq_maxc, (maxc-minc, ones))

    s = (maxc-minc) / np.maximum(ones,maxc)
    rc = (maxc-r) / maxc_minus_minc
    gc = (maxc-g) / maxc_minus_minc
    bc = (maxc-b) / maxc_minus_minc

    maxc_is_r = np.equal(maxc, r)
    maxc_is_g = np.equal(maxc, g)
    maxc_is_b = np.equal(maxc, b)

    h = np.zeros_like(r)
    h = np.choose(maxc_is_b, (h, gc-rc+4.0))
    h = np.choose(maxc_is_g, (h, rc-bc+2.0))
    h = np.choose(maxc_is_r, (h, bc-gc))

    h = np.mod(h/6.0, 1.0)

    return (h, s, v)


def hsv_to_rgb(h, s, v):
    h = np.clip(h, 0.0, 1.0)
    s = np.clip(s, 0.0, 1.0)
    v = np.clip(v, 0.0, 1.0)

    if s == 0.0:
        return v, v, v
    
    i = astype(h*5.999999, int)
    f = (h*6.0) - i

    p = v*(1.0 - s)
    q = v*(1.0 - s*f)
    t = v*(1.0 - s*(1.0-f))

    r = np.choose(i, [v, q, p, p, t, v])
    g = np.choose(i, [t, v, v, q, p, p])
    b = np.choose(i, [p, p, t, v, v, q])
    
    return (r, g, b)