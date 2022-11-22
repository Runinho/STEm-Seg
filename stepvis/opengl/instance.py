# render the instance overlay
import numpy as np

from stepvis.opengl import numpy2qimage
from stepvis.opengl.image import ImageRenderer
from stepvis.opengl.label_image import LabelImageRenderer
from stepvis.opengl.label_image_texture import LabelImageTextureRenderer
from stepvis.opengl.shader.label_image_shader import get_default_cmap
from stepvis.opengl.text_image import TextImageRenderer

CHANNEL = 1

def instances_to_dict(image):
   labels = np.unique(image[:, :, 2])
   return {k: f"id: {k}" for k in labels}

class InstanceRenderer(LabelImageRenderer):
    def __init__(self, parent, image):
        super().__init__(parent, numpy2qimage(image))

        # setup label positions
        id_img = image[:,:,2]
        self.text_labels = []
        cmap = get_default_cmap()
        for id in np.unique(id_img):
            if id > 0:
                mask = id == id_img
                coords = np.array(np.where(mask))
                mean_pos = coords.mean(axis=1)

                # TODO: also print the class :)
                color = cmap[id % len(cmap)]
                color = (*color, 0.5)
                position = (mean_pos - np.array(id_img.shape) * 0.5)[[1, 0]]
                text = f"ID:{id}"

                renderer = TextImageRenderer(self.parent, text=text, color=color,
                                             position=position, angle=0, scale=0.25, texture_size=(128, 64))
                self.text_labels.append(renderer)

        #TODO: create text Textures

        #self.text_labels = [TextImageRenderer(self.parent, (1, 0, 0, 1), "top-left", position=(-id_img.shape[1]/2, -id_img.shape[0]/2)), ]

    def render(self, f, projection_matrix):
        super().render(f, projection_matrix)
        # mmh h
        # TODO: display text textures
        for text in self.text_labels:
            text.render(f, projection_matrix)