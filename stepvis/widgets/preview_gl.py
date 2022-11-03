"""OpenGl visualization renderer"""
import time
from threading import Timer

import numpy as np
from OpenGL.GL import *
from PySide6.QtCore import Slot, QTimer, QPoint
from PySide6.QtGui import QOpenGLContext, QNativeGestureEvent, Qt, \
    QWheelEvent, QCursor, QMouseEvent, QImage, QMatrix4x4
from PySide6.QtOpenGLWidgets import QOpenGLWidget

from stepvis.opengl.image import ImageRenderer

t = 0

class PreviewOpenGL(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.context = QOpenGLContext(self)
        self.setMinimumWidth(200)
        self.zoom = 1
        self.pos = (0,0)

        #image = QImage("/home/rune/projects/dreambooth/data/output/moneyboy/mb_space_upres.png")
        #self.example_image = ImageRenderer(self, image)
        self.to_render = None

        # track mouse
        self.setMouseTracking(True)

        if not self.context.create():
            raise Exception("Unable to create GL context")

    def initializeGL(self):
        # Set up the rendering context, load shaders and other resources, etc.:
        f = self.context.functions()
        f.glClearColor(1.0, 1.0, 1.0, 1.0);
        f.glEnable(GL_MULTISAMPLE)
        f.glEnable(GL_BLEND)
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        #self.play()

    def resizeGL(self, w, h):
        # Update projection matrix and other size related settings:
        #f = self.context.functions()
        retina_scale = self.devicePixelRatio()
        glViewport(0, 0, int(self.width() * retina_scale * self.zoom), int(self.height() * retina_scale * self.zoom))
        #self.resizeViewport()

    def mouseMoveEvent(self, e:QMouseEvent):
        # check if left button is pressed
        self.pos = e.pos().toTuple()
        self.update()

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
        f = self.context.functions()
        glClearColor(1,1,1,1)
        glClear(GL_COLOR_BUFFER_BIT)
        f.glEnable(GL_MULTISAMPLE)
        projection_matrix = self.projection_matrix((2, 2), (0, 0), self.pos)

        if self.to_render != None:
            self.to_render.render(f, projection_matrix)

    def time_changed(self, t):
        print(f"time changed {t}")

