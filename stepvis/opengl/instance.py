# render the instance overlay
import numpy as np

from stepvis.opengl import numpy2qimage
from stepvis.opengl.image import ImageRenderer
from stepvis.opengl.label_image import LabelImageRenderer
from stepvis.opengl.label_image_texture import LabelImageTextureRenderer

CHANNEL = 1

def instances_to_dict(image):
   labels = np.unique(image[:, :, 2])
   return {k: f"id: {k}" for k in labels}

class InstanceRenderer(LabelImageRenderer):
    def __init__(self, parent, image):
        super().__init__(parent, numpy2qimage(image))

        # setup label positions
        id_img = image[:,:,2]
        labels = []  # string, position
        for id in np.unique(id_img):
            if id > 0:
                mask = id == id_img
                coords = np.array(np.where(mask))
                mean_pos = coords.mean(axis=1)

                # TODO: also print the class :)
                labels.append((f"ID: {id}", mean_pos))

        #TODO: create text Textures

        #self.text_labels = [TextImageRenderer(self.parent, (1, 0, 0, 1), "top-left", position=(-id_img.shape[1]/2, -id_img.shape[0]/2)), ]

    def render(self, f, projection_matrix):
        super().render(f, projection_matrix)
        # mmh h
        # TODO: display text textures