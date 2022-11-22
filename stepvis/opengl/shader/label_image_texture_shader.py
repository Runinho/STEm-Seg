# renders a label image with textures
# same as label_image but with the textures instead of colors
import OpenGL.raw.GL.VERSION.GL_1_3 as gl_constants
from OpenGL.raw.GL.VERSION.GL_1_0 import GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT, \
    GL_TEXTURE_WRAP_T, GL_TEXTURE_MIN_FILTER, GL_TEXTURE_MAG_FILTER, GL_LINEAR, GL_TRIANGLES
from PyQt5.QtGui import QOpenGLVertexArrayObject

from stepvis.opengl.shader.image_shader import ImageShader
from stepvis.opengl.shader.label_image_shader import LabelImageShader


def create_fragment_shader(category_ids):

    texture_def = "\n".join(f"uniform sampler2D texture{i};" for i in category_ids)
    texture_cases ="".join([f"case {i}: \n FragColor.rgb = texture(texture{i}, c).rgb; break;\n"
                            for i in category_ids])

    fragment_shader = """
    #version 330 core
    
    #define PI 3.1415926538
    
    in vec2 TexCoord;
    uniform sampler2D labelTexture;
    """ + texture_def + """
    uniform vec2 alpha;
    uniform vec2 size;
    uniform int channel;
    out vec4 FragColor;
    void main()
    {
        FragColor = texture(labelTexture, TexCoord);
        // we use the rgb channel `channel` to look up the color in the colormap
        float raw_index = 0;
        // we use brg format
        switch(channel){
            case 0:
                raw_index = FragColor.b;
                break;
            case 1:
                raw_index = FragColor.r;
                break;
            case 2:
                raw_index = FragColor.g;
                break;
        } 
        // convert to int
        int index = (int) (raw_index * 256);
        int repeats = 30; // repeats over the whole image
        vec2 c = TexCoord;
        c.y *= (size.y/size.x);
        c = vec2(c.x * repeats, c.y * repeats);
        
        // rotate the image
        float a = ((float)index * ((90)/20) - 45)*(PI/180);
        vec2 t = c;
        c.x = cos(a) * t.x - sin(a) * t.y;
        c.y = sin(a) * t.x +  cos(a) * t.y;
        
        // flip image
        c.y = c.y * -1;
       
        // clamp to 0..1  
        c.x = mod(c.x, 1);
        c.y = mod(c.y, 1);
        
        switch(index){
            //case x:
            //    FragColor = texture(textureX, c);
            //    break;
            """ + texture_cases + """
        }
        // set alpha
        FragColor.a = alpha.x;
    } 
    """
    print(fragment_shader)
    return fragment_shader

class LabelImageTextureShader(LabelImageShader):
    def __init__(self, parent, category_ids):
        super().__init__(parent, fragment_shader=create_fragment_shader(category_ids))
        self.category_ids = category_ids
        # opengl only supports 32 textures we need 1 for the label data itself
        assert self.num_categories() < 31

    def num_categories(self) -> int:
        return len(self.category_ids)

    def load(self) -> bool:
        if super().load():
            self.textures = {i: self.program.uniformLocation(f"texture{i}")
                             for i in self.category_ids}
            self.label_texture = self.program.uniformLocation("labelTexture")
            self.channel = self.program.uniformLocation("channel")
            return True
        return False

    def set_channel(self, new_channel):
        self.program.setUniformValue(self.channel, new_channel)

    def draw_with_texture(self, f, texture, textureIds):
        vao_binder = QOpenGLVertexArrayObject.Binder(self.vao)
        self.program.setUniformValue(self.label_texture, 0)
        f.glActiveTexture(gl_constants.GL_TEXTURE0)
        f.glBindTexture(GL_TEXTURE_2D, texture)

        # set all the textures
        for i, (k, t) in enumerate(self.textures.items()):
            self.program.setUniformValue(t, i+1)
            f.glActiveTexture(getattr(gl_constants, f"GL_TEXTURE{i + 1}"))
            f.glBindTexture(GL_TEXTURE_2D, textureIds[k])

        # set the texture that decides witch texture to render

        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        f.glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
        f.glDrawArrays(GL_TRIANGLES, 0, self.vertices.shape[0])
        vao_binder.release()
