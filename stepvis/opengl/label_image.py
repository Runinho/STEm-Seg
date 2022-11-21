# render image with a cmap
from stepvis.opengl.image import ImageRenderer

from stepvis.opengl.shader.label_image_shader import LabelImageShader


class LabelImageRenderer(ImageRenderer):
    def __init__(self, parent, image):
        super().__init__(parent, image, LabelImageShader)