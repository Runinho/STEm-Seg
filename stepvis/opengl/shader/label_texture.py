# renders a text with solid background that can be used as texture for the mask image
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_COLOR_BUFFER_BIT
from PyQt5.QtCore import QRect, QPoint, Qt
from PyQt5.QtGui import QPainter, QPen, QOpenGLFramebufferObject, QOpenGLPaintDevice

from stepvis.opengl.shader.image_shader import ImageShader


fragment_shader = """
#version 330 core
in vec2 TexCoord;
uniform sampler2D labelTexture;
uniform vec2 alpha;
uniform vec3 cmap[32]; // 32 images
out vec4 FragColor;
void main()
{
    // flip y cord
    FragColor = texture(labelTexture, vec2(TexCoord.x, TexCoord.y * -1 + 1));
    FragColor.a = alpha.x;
} 
"""

class LabelTextureShader(ImageShader):
    def __init__(self, parent, color, text, texture_size=128, angle=45):
        super().__init__(parent, fragment_shader=fragment_shader)
        self.texture_size = texture_size
        self.initialized = False
        self.color = color
        self.text = text
        self.angle = angle

    def load(self) ->bool:
        if super().load():
            self.drawRect = QRect(0, 0, self.texture_size, self.texture_size)
            self.drawRectSize = self.drawRect.size()
            self.framebuffer = QOpenGLFramebufferObject(self.drawRectSize)

    def initialize(self, f):
        if not self.initialized:
            # render the framebuffer once
            half_rect_size = int(self.texture_size/2)
            self.framebuffer.bind()
            device = QOpenGLPaintDevice(self.drawRectSize)
            painter = QPainter()
            painter.begin(device)
            f.glClearColor(*self.color)
            f.glClear(GL_COLOR_BUFFER_BIT)
            # painter.drawTiledPixmap(self.drawRect, QPixmap(":/qt-project.org/qmessagebox/images/qtlogo-64.png"))
            painter.setPen(QPen(Qt.black, 20))
            painter.setBrush(Qt.black)
            #painter.drawEllipse(0, 0, 20, 40)
            #painter.drawEllipse(100, 0, 200, 400)
            # update font size
            QFont = font = painter.font()
            font.setPointSize(font.pointSize() * 2.5)
            # translate because we want to rotate around the center of the image
            painter.translate(half_rect_size, half_rect_size);
            painter.rotate(self.angle)
            painter.setFont(font)
            painter.drawText(QRect(-half_rect_size, -half_rect_size, self.texture_size, self.texture_size), Qt.AlignVCenter | Qt.AlignHCenter, self.text)
            f.glClearColor(1, 1, 1, 1)
            painter.end()
            self.framebuffer.release()
            self.initialized = True


    def draw_with_texture(self, f, texture):
        self.set_size((self.texture_size, self.texture_size))
        self.initialize(f)

        super().draw_with_texture(f, self.framebuffer.texture())