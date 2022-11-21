
import signal
import sys

from PySide6.QtCore import QTimer
from PySide6.QtGui import QSurfaceFormat, Qt
from PySide6.QtWidgets import (
    QApplication,
    QWidget,
    QHBoxLayout, QSlider, QVBoxLayout, QTabWidget, )


from stepvis.widgets.data_source_config import DataSourceConfigWidget
from stepvis.widgets.visualization import VisualizationWidget


# from stepvis.widgets.settings import Settings


class PlotWidget(QTabWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.visualization = VisualizationWidget(self)
        self.data_source_config = DataSourceConfigWidget(self, self.visualization.sequence_changed)

        self.addTab(self.visualization, "view")
        self.addTab(self.data_source_config, "data source")


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

