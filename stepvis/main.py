
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat, Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout, QSlider, QVBoxLayout, )

from stemseg.config import cfg
from stemseg.utils import RepoPaths
from stepvis.inference_loader import InferenceDataProvider, InferenceSplitType
from stepvis.opengl.sequence import ImageSequenceRender
from stepvis.widgets.preview_gl import PreviewOpenGL
# from stepvis.widgets.settings import Settings


class PlotWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        #  create widgets
        self.setMinimumWidth(512)
        self.setMinimumHeight(512)

        # self.settings = Settings()
        #self.preview = PreviewMatplotlib()
        self.preview = PreviewOpenGL()
        self.timeline = QSlider(orientation=Qt.Horizontal)
        # self.settings.render.connect(self.preview.on_change)

        #  Create layout

        vlayout = QVBoxLayout()
        vlayout.addWidget(self.preview, stretch=1)
        vlayout.addWidget(self.timeline)
        # hlayout.addWidget(self.settings)

        self.setLayout(vlayout)
        self.setLayout(vlayout)

        # laod the data
        # TODO: add option to load the sequence at will
        cfg_path = RepoPaths.configs_dir() / "kitti_step_2.yaml"
        cfg.merge_from_file(cfg_path)
        self.data_provider = InferenceDataProvider(InferenceSplitType.VAL)
        self.preview.to_render = ImageSequenceRender(self.preview,
                                                     self.data_provider.sequence_providers[0])
        self.timeline.valueChanged.connect(self.preview.to_render.time_changed)

def sigint_handler(*args):
    print("closing qt because of sigint")
    QApplication.quit()


if __name__ == "__main__":
    signal.signal(signal.SIGINT, sigint_handler)

    app = QApplication(sys.argv)

    #some opengl stuff
    format = QSurfaceFormat()
    format.setDepthBufferSize(24)
    format.setStencilBufferSize(8)
    format.setVersion(3, 2)
    format.setProfile(QSurfaceFormat.CoreProfile)
    format.setSamples(16)
    QSurfaceFormat.setDefaultFormat(format)

    w = PlotWidget()
    w.show()

    # we need to run the interpreter from time to time to recive the keyboard interrupt
    # see: https://stackoverflow.com/a/4939113/18724786
    timer = QTimer()
    timer.start(500)  # You may change this if you wish.
    timer.timeout.connect(lambda: None)  # Let the interpreter run each 500 ms.
    # Your code here.

    sys.exit(app.exec())

