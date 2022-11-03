# image sequence
from typing import List

from PySide6.QtGui import QImage

from stepvis.inference_loader import InferenceDataProviderSequence
from stepvis.opengl.image import ImageRenderer
from stepvis.opengl.label_image import LabelImageRenderer


def numpy2qiamge(img):
    h, w, _ = img.shape
    return QImage(img.data, w, h, 3 * w, QImage.Format_BGR888)

class ImageSequenceRender:
    def __init__(self, parent, sequence:InferenceDataProviderSequence):
        self.sequence = sequence
        self.parent = parent
        self.load_data()
        self.t = 0

        self.loaded = False

    def load_data(self):
        images = self.sequence.get_images()
        outputs = self.sequence.get_outputs()

        self.image_renderer = list([ImageRenderer(self.parent, numpy2qiamge(img)) for img in images])
        self.mask_renderer = list([LabelImageRenderer(self.parent, numpy2qiamge(img)) for img in outputs])

    def load(self):
        for renderer in self.image_renderer + self.mask_renderer:
            renderer.load()
        self.loaded = True

    def render(self, f, projection_matrix):
        if not self.loaded:
            self.load()
        self.image_renderer[self.t].render(f, projection_matrix)
        mask_renderer =  self.mask_renderer[self.t]
        mask_renderer.set_alpha(0.5)
        mask_renderer.render(f, projection_matrix)

        # increase image index
        #self.t = (self.t + 1) % len(self.renderer)

    def time_changed(self, t):
        self.t = t
        self.parent.update()
