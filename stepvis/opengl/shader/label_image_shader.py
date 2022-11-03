# shader to render label annotations
from typing import Sequence

import matplotlib
import numpy as np
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST, \
    GL_TEXTURE_MAG_FILTER
from PySide6.QtGui import QVector3D
from shiboken6 import VoidPtr

from stepvis.opengl.shader.image_shader import ImageShader


def get_default_cmap():
    colors = list(matplotlib.cm.get_cmap("Dark2").colors) + \
             list(matplotlib.cm.get_cmap("Set1").colors) + \
             list(matplotlib.cm.get_cmap("Set2").colors) + \
             list(matplotlib.cm.get_cmap("Accent").colors)
    colors = colors[:32]
    return colors

# TODO: maybe hardcode the cmap into the shader
fragment_shader = """
#version 330 core
in vec2 TexCoord;
uniform sampler2D ourTexture;
uniform vec2 alpha;
out vec4 FragColor;
void main()
{
    FragColor = texture(ourTexture, TexCoord);
    FragColor.a = alpha.x;
    // we use the rgb channel to look up the color in the colormap
    // convert to int
    int index = (int) (FragColor.b * 256);
    switch(index % 32){
        //case x:
        //    FragColor.rgb = vec3(*c);
        //    break;
        """ + \
         "".join([f"case {i}: \n FragColor.rgb = vec3({','.join([str(c) for c in colors])}); break;\n"
                  for i, colors in enumerate(get_default_cmap())])\
        + """
    }
    //FragColor.b = FragColor.b * 10;
    // FragColor.b = 1;
    // FragColor.r = 1;
    // FragColor.g = 1;
    //FragColor.rgb = FragColor.rgb * 10;
    //FragColor.rgb = cmap[index%32];
} 
"""

class LabelImageShader(ImageShader):
    def __init__(self, parent):
        super().__init__(parent, fragment_shader=fragment_shader)

    def draw_with_texture(self, f,  texture):
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        super().draw_with_texture(f, texture)

    def load(self):
        if super().load():
            self.cmap = self.program.uniformLocation("cmap")