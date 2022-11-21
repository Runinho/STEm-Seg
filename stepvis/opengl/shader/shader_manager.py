from collections import defaultdict
from typing import TypeVar

from stepvis.opengl.shader.base import Shader

shaders = defaultdict(dict)

S = TypeVar("S", bound=Shader)

def freeze(obj):
    if type(obj) is dict:
        return frozenset({k: freeze(v) for k, v in obj.items()})
    if type(obj) is list:
        return tuple([freeze(x) for x in obj])
    return obj

# Singelton to load shader
def get_shader(cls: S, parent, kwargs) -> S:
    kwargs_hashable = freeze(kwargs)
    global shaders
    if (cls, kwargs_hashable) in shaders[parent]:
        return shaders[parent][(cls, kwargs_hashable)]
    else:
        shader = cls(parent, **kwargs)
        shaders[parent][(cls, kwargs_hashable)] = shader
        return shader