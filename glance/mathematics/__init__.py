from .mathematics import *

from . import defaults

components = next_multiple(defaults.CHUNK_SIZE)

def configure(n=defaults.DEFAULT_N, dtype=defaults.DEFAULT_DTYPE):
    defaults.DEFAULT_N = n
    defaults.DEFAULT_M = components(n + 1)
    defaults.DEFAULT_DTYPE = dtype

