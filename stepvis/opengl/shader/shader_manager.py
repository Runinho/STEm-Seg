from collections import defaultdict
from typing import TypeVar

from stepvis.opengl.shader.base import Shader

shaders = defaultdict(dict)

S = TypeVar("S", bound=Shader)


# Singelton to load shader
def get_shader(cls: S, parent) -> S:
    global shaders
    if cls in shaders[parent]:
        return shaders[parent][cls]
    else:
        shader = cls(parent)
        shaders[parent][cls] = shader
        return shader