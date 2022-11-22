"""OpenGl visualization renderer, renders the 3D Viewport"""
import OpenGL.GL as gl
from PyQt5.QtGui import QOpenGLContext, \
    QMouseEvent, QMatrix4x4
from PyQt5.QtWidgets import QOpenGLWidget

t = 0

class PreviewOpenGL(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.context = QOpenGLContext(self)
        self.setMinimumWidth(200)
        self.zoom = 1
        self.pos = (0,0)

        self.to_render = None

        # track mouse
        self.setMouseTracking(True)

        if not self.context.create():
            raise Exception("Unable to create GL context")

    def initializeGL(self):
        # Set up the rendering context, load shaders and other resources, etc.:
        f = gl
        f.glClearColor(1.0, 1.0, 1.0, 1.0);
        f.glEnable(gl.GL_MULTISAMPLE)
        f.glEnable(gl.GL_BLEND)
        f.glBlendFunc(gl.GL_SRC_ALPHA, gl.GL_ONE_MINUS_SRC_ALPHA)

        #self.play()

    def resizeGL(self, w, h):
        # Update projection matrix and other size related settings:
        f = gl
        retina_scale = self.devicePixelRatio()
        f.glViewport(0, 0, int(self.width() * retina_scale * self.zoom), int(self.height() * retina_scale * self.zoom))
        #self.resizeViewport()

    def mouseMoveEvent(self, e:QMouseEvent):
        # check if left button is pressed
        self.pos = e.pos()
        #self.update()

    def projection_matrix(self, zoom, offset, rotation):
        projection = QMatrix4x4()
        # projection matrix for opengl
        projection.perspective(45, self.width() / self.height(), 1, 2000.0)
        #projection.ortho(-zoom[0] + offset[0], zoom[0] + offset[0], -zoom[1] + offset[1], zoom[1] + offset[1], -1, 1)
        projection.rotate(360, 0, 1, 0)
        projection.rotate(180, 1, 0, 0)
        projection.translate(0, 0, 500)
        # print(projection)
        return projection

    def paintGL(self):
        print(f"rendering {self.pos}")
        # Draw the scene:
        f = gl
        f.glClearColor(1,1,1,1)
        f.glClear(gl.GL_COLOR_BUFFER_BIT)
        f.glEnable(gl.GL_MULTISAMPLE)
        projection_matrix = self.projection_matrix((2, 2), (0, 0), self.pos)

        if self.to_render != None:
            self.to_render.render(f, projection_matrix)

    def time_changed(self, t):
        print(f"time changed {t}")

