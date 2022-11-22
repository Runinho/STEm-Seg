from typing import Dict

from stepvis.opengl.image import ImageRenderer
from stepvis.opengl.text_image import TextImageRenderer
from stepvis.opengl.shader.label_image_shader import get_default_cmap
from stepvis.opengl.shader.label_image_texture_shader import LabelImageTextureShader


class LabelImageTextureRenderer(ImageRenderer):
    def __init__(self, parent, image, categories: Dict[int, str], channel=0):
        super().__init__(parent,
                         image,
                         shader=LabelImageTextureShader,
                         shader_kwargs={"category_ids": list(categories.keys())})
        self.categories = categories
        # on which channel to switch
        self.channel = channel

    def load(self):
        super().load()
        cmap = get_default_cmap()
        self.textures = {k: TextImageRenderer(self.parent, (*cmap[k%len(cmap)], 1), text)
                              for (k, text) in self.categories.items()}
        for texture in self.textures.values():
            texture.load()


    def draw_texture(self, f):
        # initialize
        for texture in self.textures.values():
            texture.shader.initialize(f)
        ids = {k: texture.get_texture_id() for k, texture in self.textures.items()}
        self.shader.draw_with_texture(f, super().get_texture_id(), ids)

    def update_shader(self):
        super().update_shader()
        self.shader.set_channel(self.channel)
