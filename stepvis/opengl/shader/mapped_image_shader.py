# NOTE: this class is currently not used
# render the image but map the values from one channel to colors
# usefull for visualizing the instance ids
from typing import Tuple, Dict

from PySide6.QtGui import QVector4D

from stepvis.opengl.shader.image_shader import ImageShader

MAX_CMAP = 32
fragment_shader = """
#version 330 core
in vec2 TexCoord;
uniform sampler2D ourTexture;
uniform vec2 alpha;
uniform int numColors;
uniform vec4 cmap["""+MAX_CMAP+"""];
out vec4 FragColor;
void main()
{
    FragColor = texture(ourTexture, TexCoord);
    index = FragColor.r; // this is used to determine the number in the cmap
    FragColor = vec4(0,0,0,0);
    for(int i=0; i < numColors; i++){
        if(index == cmap[i].a){
            FragColor.rgb = cmap[i].rgb;
            FragColor.a = alpha.x;
        }
    }  
    
} 
"""

class MappedImageShader(ImageShader):
    def __init__(self, parent):
        super().__init__(parent, fragment_shader=fragment_shader)

    def load(self) -> bool:
        if super().load():
            self.num_colors = self.program.uniformLocation("numColors");
            self.cmap = self.program.uniformLocation("cmap")

    def set_cmap(self, map:Dict[int, Tuple[float, float, float]]):
        """map the values to colors"""
        num_colors = len(map.keys())
        assert len(map.keys()) <= MAX_CMAP

        self.program.setUniformValueArray(self.cmap, [QVector4D(*c, v) for v, c in map.items()])
        self.program.setUniformValue(self.num_colors, num_colors)