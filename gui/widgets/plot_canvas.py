"""A Qt plot canvas whose preferred size cannot grow with its rendered image."""

from PyQt6.QtCore import QSize
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from gui.ui_scale import ui_pixels


class PlotCanvas(FigureCanvasQTAgg):
    def __init__(self, figure):
        # FigureCanvasQT.sizeHint() uses the figure's *current* pixel size.
        # Inside a resizable scroll area/splitter that size changes with layout,
        # feeding the rendered size back into the next layout calculation.
        # Keep the original logical preference, independent of render geometry
        # and of the display's device pixel ratio.
        self._logical_size_hint = QSize(
            round(figure.get_figwidth() * figure.dpi),
            round(figure.get_figheight() * figure.dpi),
        )
        super().__init__(figure)

    def sizeHint(self):
        return QSize(
            ui_pixels(self._logical_size_hint.width()),
            ui_pixels(self._logical_size_hint.height()),
        )
