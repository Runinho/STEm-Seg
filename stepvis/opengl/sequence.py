# image sequence

from OpenGL.raw.GL.VERSION.GL_1_0 import GL_SRC_ALPHA, GL_BLEND, GL_ONE_MINUS_SRC_ALPHA

from stepvis.inference_loader import OnlineInferenceDataProviderSequence
from stepvis.opengl import numpy2qimage
from stepvis.opengl.image import ImageRenderer
from stepvis.opengl.label_image_texture import LabelImageTextureRenderer
from stepvis.opengl.instance import InstanceRenderer
from stepvis.opengl.text_image import TextImageRenderer


class ImageSequenceRender:
    def __init__(self, parent, sequence:OnlineInferenceDataProviderSequence):
        self.sequence = sequence
        self.parent = parent
        self.load_data()
        self.t = 0

        self.loaded = False

    def load_data(self):
        images = self.sequence.get_images()
        outputs = self.sequence.get_outputs()

        self.image_renderer = list([ImageRenderer(self.parent, numpy2qimage(img)) for img in images])
        self.mask_renderer = list([LabelImageTextureRenderer(self.parent, numpy2qimage(img), self.sequence.categories) for img in outputs])
        self.instance_renderer = list([InstanceRenderer(self.parent, img) for img in outputs])
        self.test = TextImageRenderer(self.parent, (1, 0, 0, 1), "LOL")

    def load(self):
        for renderer in self.image_renderer + self.mask_renderer + self.instance_renderer:
            renderer.load()
        self.loaded = True
        self.test.load()

    def render(self, f, projection_matrix):
        if not self.loaded:
            self.load()
        f.glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        f.glEnable(GL_BLEND)
        self.image_renderer[self.t].render(f, projection_matrix)
        def render_overlay(overlay, alpha):
            overlay =  overlay[self.t]
            overlay.set_alpha(alpha)
            overlay.render(f, projection_matrix)

        render_overlay(self.instance_renderer, 0.5)

        self.test.render(f, projection_matrix)
        # increase image index
        #self.t = (self.t + 1) % len(self.image_renderer)

    def time_changed(self, t):
        print(t)
        self.t = t
        self.parent.update()
