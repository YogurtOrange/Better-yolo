"""Resizable image display with a stable layout hint and an original pixmap."""

from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QLabel, QSizePolicy

from gui.ui_scale import ui_pixels


class ImagePreview(QLabel):
    """Fit pixels to the allocated viewport without fixing its geometry."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self._source_pixmap = QPixmap()
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setWordWrap(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(240, 180)

    def sizeHint(self):
        # QLabel normally reports the pixmap size. Feeding a resized pixmap
        # into a splitter/scroll area must not change that layout's preference.
        return QSize(ui_pixels(480), ui_pixels(360))

    def minimumSizeHint(self):
        return QSize(ui_pixels(240), ui_pixels(180))

    def setPixmap(self, pixmap):
        self._source_pixmap = QPixmap(pixmap)
        self._refresh_pixmap()

    def setText(self, text):
        self._source_pixmap = QPixmap()
        super().setText(text)

    def clear(self):
        self._source_pixmap = QPixmap()
        super().clear()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_pixmap()

    def _refresh_pixmap(self):
        if self._source_pixmap.isNull():
            return
        target = self.contentsRect().size()
        if target.isEmpty():
            return
        ratio = self.devicePixelRatioF()
        fitted = self._source_pixmap.scaled(
            target * ratio,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        fitted.setDevicePixelRatio(ratio)
        super().setPixmap(fitted)
