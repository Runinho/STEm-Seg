from PySide6.QtOpenGL import QOpenGLShaderProgram, QOpenGLShader


class Shader:
    def __init__(self, parent, vertex_shader, fragment_shader):
        self.parent = parent
        self.vertex_shader = vertex_shader
        self.fragment_shader = fragment_shader
        self.program = None

    def load_program(self):
        self.program = QOpenGLShaderProgram(self.parent)
        self.program.addShaderFromSourceCode(QOpenGLShader.Vertex, self.vertex_shader)
        self.program.addShaderFromSourceCode(QOpenGLShader.Fragment, self.fragment_shader)
        self.program.link()

    def load(self) -> bool:
        """
        load shader code and initalizer required buffers
        Returns:
            if the program was loaded. False if it was already initialized

        """
        if self.program is None:
            self.load_program()
            return True
        return False