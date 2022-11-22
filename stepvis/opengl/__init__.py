from PySide6.QtGui import QImage


# small helper
def numpy2qimage(img):
    h, w, _ = img.shape
    return QImage(img.data, w, h, 3 * w, QImage.Format_BGR888)
