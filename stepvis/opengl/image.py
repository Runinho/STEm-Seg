# Render a image
from OpenGL.raw.GL.ARB.internalformat_query2 import GL_TEXTURE_2D
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_TEXTURE_WRAP_S, GL_REPEAT, GL_TEXTURE_WRAP_T, \
    GL_TRIANGLES
from PySide6.QtGui import QImage, QVector4D
from PySide6.QtOpenGL import QOpenGLTexture, QOpenGLFramebufferObject

from stepvis.opengl.base import Renderer
from stepvis.opengl.shader.image_shader import ImageShader


class ImageRenderer(Renderer):
    def __init__(self, parent, image: QImage, shader=ImageShader, shader_kwargs={}):
        super().__init__(parent, shader, shader_kwargs=shader_kwargs)
        self.image = image
        self.position = QVector4D(0, 0, 5, 1)
        self.size = (1, 1)
        self.alpha = 1
        self.loaded = False

    def set_alpha(self, alpha):
        self.alpha = alpha

    def load_image(self, image: QImage):
        # Setup texture
        self.size = image.size().toTuple()
        self.texture = QOpenGLTexture(image)
        self.texture.setMinificationFilter(QOpenGLTexture.LinearMipMapLinear)
        self.texture.setMagnificationFilter(QOpenGLTexture.Linear)

        self.drawRect = image.rect()
        self.drawRectSize = self.drawRect.size()

    def load(self):
        self.shader.load()
        self.load_image(self.image)
        self.loaded = True

    def update_shader(self):
        # set all the shader properties
        self.shader.set_position(self.position)
        self.shader.set_size(self.size)
        self.shader.set_alpha(self.alpha)

    def get_texture_id(self):
        return self.texture.textureId()

    def draw_texture(self, f):
        self.shader.draw_with_texture(f, self.get_texture_id())

    def render(self, f, projection_matrix):
        if not self.loaded:
            self.load()
        if self.shader.program is not None:
            self.shader.program.bind()
            self.shader.set_projection(projection_matrix)
            self.update_shader()
            self.draw_texture(f)
            self.shader.program.release()


