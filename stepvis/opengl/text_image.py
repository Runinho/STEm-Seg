# renders a text on a square
from PyQt5.QtGui import QImage

from stepvis.opengl.image import ImageRenderer
from stepvis.opengl.shader.label_texture import LabelTextureShader


class TextImageRenderer(ImageRenderer):
    def  __init__(self, parent, color, text, angle=45, texture_size=128, scale=1, position=None, shader=LabelTextureShader):
        super().__init__(parent,
                         image=None,
                         position=position,
                         scale=scale,
                         shader=shader,
                         shader_kwargs={"color":color, "text":text, "angle": angle,
                                        "texture_size": texture_size})
        assert len(color) == 4, "expect r,g,b,a value as tuple"
        self.alpha = -1 # use the alpha as provided by the texture

    def load_image(self, image: QImage):
        # we do not need this (its in the shader)
        pass

    def get_texture_id(self):
        return self.shader.framebuffer.texture()
