# base class for renderer
from stepvis.opengl.shader.shader_manager import get_shader


class Renderer():
    def __init__(self, parent, shader_cls):
        self.parent = parent
        self.shader = get_shader(shader_cls, parent)

    def load(self):
        self.shader.load()