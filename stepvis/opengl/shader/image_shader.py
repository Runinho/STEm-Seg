# shader used to render an image
import numpy as np
from OpenGL.raw.GL.ARB.internalformat_query2 import GL_TEXTURE_2D
from OpenGL.raw.GL.ARB.vertex_shader import GL_FLOAT
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_TEXTURE_WRAP_S, GL_REPEAT, GL_TEXTURE_WRAP_T, \
    GL_TRIANGLES, GL_TEXTURE_MIN_FILTER, GL_NEAREST, GL_TEXTURE_MAG_FILTER, GL_LINEAR
from OpenGL.raw.GL.VERSION.GL_1_3 import GL_TEXTURE0
from PySide6.QtGui import QVector4D, QVector2D
from PySide6.QtOpenGL import QOpenGLBuffer, QOpenGLVertexArrayObject
from shiboken6 import VoidPtr

from stepvis.opengl.shader.base import Shader

vertex_shader = """
#version 330 core
layout (location = 0) in vec2 vPosition;

uniform mat4 projection; // projection matrix
uniform vec4 position;   // center position of the image
uniform vec2 size;       // width and height of the image

out vec2 TexCoord;

void main()
{
    gl_Position = projection *
                  (vec4((vPosition.x - 0.5) * size.x + position.x,
                        (vPosition.y - 0.5) * size.y + position.x,
                         position.z, 1.0));
    TexCoord = vec2(vPosition.x, vPosition.y);
}
"""

fragment_shader = """
#version 330 core
in vec2 TexCoord;
uniform sampler2D ourTexture;
uniform vec2 alpha;
out vec4 FragColor;
void main()
{
    FragColor = texture(ourTexture, TexCoord);
    FragColor.a = alpha.x;
} 
"""


class ImageShader(Shader):
    def __init__(self, parent, vertex_shader=vertex_shader, fragment_shader=fragment_shader):
        super().__init__(parent, vertex_shader, fragment_shader)

        self.vertices = np.array([[0.0,1.0],
                                  [1.0,1.0],
                                  [1.0,0.0],
                                  [0.0,1.0],
                                  [1.0,0.0],
                                  [0.0,0.0]], dtype=np.float32)

    def load(self) -> bool:
        if super().load():
            # load vertices
            # load polygons into opengl
            # Create and bind VBO
            self.vbo = QOpenGLBuffer()
            self.vbo.create()
            self.vbo.bind()
            # Allocate VBO, and copy in data
            vertices_data = self.vertices.tobytes()
            # self.vbo.allocate( data_to_initialize , data_size_to_allocate )
            self.vbo.allocate(VoidPtr(vertices_data), 4 * self.vertices.size)

            # Setup VAO
            self.vao = QOpenGLVertexArrayObject()
            vao_binder = QOpenGLVertexArrayObject.Binder(self.vao)
            # configure vertex attribute 0
            self.posAttr = self.program.attributeLocation("vPosition")
            self.program.setAttributeBuffer(0, GL_FLOAT, 0, 2)
            self.program.enableAttributeArray(self.posAttr)

            self.position = self.program.uniformLocation("position")
            self.projection = self.program.uniformLocation("projection")
            self.size = self.program.uniformLocation("size")
            # transparency
            self.alpha = self.program.uniformLocation("alpha")

            self.utexture = self.program.uniformLocation("ourTexture")

            # Release VBO
            self.vbo.release()
            vao_binder.release()
            return True
        return False

    def set_projection(self, projection):
        self.program.setUniformValue(self.projection, projection)

    def set_position(self, position):
        self.program.setUniformValue(self.position, position)

    def set_size(self, size):
        self.program.setUniformValue(self.size, QVector2D(*size))

    def set_alpha(self, alpha):
        # sadly the float type is not working, so we use the 2vec as workaround
        self.program.setUniformValue(self.alpha, QVector2D(alpha, 0))

    def draw_with_texture(self, f, texture):
        vao_binder = QOpenGLVertexArrayObject.Binder(self.vao)
        self.program.setUniformValue1i(self.utexture, 0)
        f.glActiveTexture(GL_TEXTURE0)
        f.glBindTexture(GL_TEXTURE_2D, texture)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        f.glDrawArrays(GL_TRIANGLES, 0, self.vertices.shape[0])
        vao_binder.release()


