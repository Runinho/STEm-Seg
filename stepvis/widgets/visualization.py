# handel the main visualization tab with its view config manages e.g the timeline
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QSlider, QVBoxLayout

from stepvis.inference_loader import InferenceDataProvider
from stepvis.opengl.sequence import ImageSequenceRender
from stepvis.opengl.text_image import TextImageRenderer
from stepvis.widgets.preview_gl import PreviewOpenGL


class VisualizationWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        #  create widgets
        self.setMinimumWidth(512)
        self.setMinimumHeight(512)

        # self.settings = Settings()
        #self.preview = PreviewMatplotlib()
        self.preview = PreviewOpenGL()
        self.timeline = QSlider(orientation=Qt.Horizontal)
        self.timeline.setMaximum(500)
        # self.settings.render.connect(self.preview.on_change)
        self.preview.to_render = TextImageRenderer(self.preview, (1, 1, 1, 1), "please select \ndata source", 0, 256)

        #  Create layout

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.preview, stretch=1)
        vlayout.addWidget(self.timeline)
        # hlayout.addWidget(self.settings)

        self.setLayout(vlayout)


    def sequence_changed(self, sequence):
        self.sequence = sequence

        #load online
        self.preview.to_render = ImageSequenceRender(self.preview,
                                                     sequence=sequence)
        self.timeline.valueChanged.connect(self.preview.to_render.time_changed)
        # maximum includes the biggest value. so we have to subtract one
        self.timeline.setMaximum(len(self.sequence) -1)
        self.timeline.update()