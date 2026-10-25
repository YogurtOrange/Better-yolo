"""Regression coverage for plot sizing inside resizable Qt pages."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QEvent, QSize, Qt
from PyQt6.QtWidgets import (
    QApplication, QHBoxLayout, QLabel, QScrollArea, QSplitter,
    QStackedWidget, QTabWidget, QVBoxLayout, QWidget,
)
from matplotlib.figure import Figure

from gui.ui_scale import UiScaleManager
from gui.widgets.plot_canvas import PlotCanvas


class PlotCanvasTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.manager = UiScaleManager(self.app, 'QWidget { font-size: 14px; }')
        self.root = QWidget()
        self.root.resize(1200, 800)
        layout = QVBoxLayout(self.root)
        self.stack = QStackedWidget()
        layout.addWidget(self.stack)
        self.stack.addWidget(QWidget())
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        page = QWidget()
        page_layout = QHBoxLayout(page)
        page_layout.setContentsMargins(20, 20, 20, 20)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        config = QWidget()
        config.setMinimumWidth(200)
        QVBoxLayout(config).addWidget(QLabel('Training configuration'))
        splitter.addWidget(config)
        self.tabs = QTabWidget()
        self.canvases = []
        for title in ('Loss', 'mAP'):
            tab = QWidget()
            figure = Figure(figsize=(8, 6), dpi=100)
            figure.add_subplot(111).plot([1, 2, 3], [1, 0.5, 0.2])
            canvas = PlotCanvas(figure)
            canvas.setMinimumSize(400, 300)
            QVBoxLayout(tab).addWidget(canvas)
            self.tabs.addTab(tab, title)
            self.canvases.append(canvas)
        splitter.addWidget(self.tabs)
        splitter.setSizes([400, 800])
        page_layout.addWidget(splitter)
        scroll.setWidget(page)
        self.stack.addWidget(scroll)
        self.manager.set_scale(100)
        self.root.show()
        self.app.processEvents()

    def tearDown(self):
        self.app.removeEventFilter(self.manager)
        self.root.close()
        self.root.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.manager.deleteLater()
        self.app.processEvents()

    def test_preferred_size_does_not_follow_rendered_image(self):
        canvas = self.canvases[0]
        self.assertEqual(canvas.sizeHint(), QSize(800, 600))
        # The base Matplotlib canvas would now return 4000 x 2000, starting
        # the scroll-area/layout feedback loop reported by the user.
        canvas.figure.set_size_inches(40, 20, forward=False)
        self.assertEqual(canvas.sizeHint(), QSize(800, 600))
        self.manager.set_scale(150)
        self.assertEqual(canvas.sizeHint(), QSize(1200, 900))
        self.manager.set_scale(100)
        self.assertEqual(canvas.sizeHint(), QSize(800, 600))

    def test_repeated_zoom_and_page_switches_keep_renderer_bounded(self):
        from matplotlib.backends.backend_agg import RendererAgg
        original_init = RendererAgg.__init__
        renders = []
        oversized = []
        original_resize = PlotCanvas.resizeEvent
        original_paint = PlotCanvas.paintEvent

        def checked_resize(canvas, event):
            size = event.size()
            if max(size.width(), size.height()) * canvas.device_pixel_ratio > 4096:
                oversized.append((size.width(), size.height()))
                return
            original_resize(canvas, event)

        def checked_paint(canvas, event):
            if max(canvas.width(), canvas.height()) * canvas.device_pixel_ratio > 4096:
                return
            original_paint(canvas, event)

        def checked_init(renderer, width, height, dpi):
            renders.append((width, height))
            if max(width, height) > 4096:
                # Never allocate an enormous buffer even if this test regresses.
                oversized.append((width, height))
                raise ValueError(f'Unexpected oversized test plot: {width}x{height}')
            original_init(renderer, width, height, dpi)

        with patch.object(RendererAgg, '__init__', checked_init), \
                patch.object(PlotCanvas, 'resizeEvent', checked_resize), \
                patch.object(PlotCanvas, 'paintEvent', checked_paint):
            for cycle in range(3):
                for percent in (75, 100, 125, 150, 175, 200, 100):
                    self.stack.setCurrentIndex(0)
                    self.manager.set_scale(percent)
                    self.stack.setCurrentIndex(1)
                    self.tabs.setCurrentIndex(cycle % 2)
                    for _ in range(3):
                        self.app.processEvents()
                    for canvas in self.canvases:
                        self.assertLess(canvas.width(), 4096)
                        self.assertLess(canvas.height(), 4096)
                        self.assertEqual(canvas.sizeHint(), QSize(8 * percent, 6 * percent))
                    visible = self.canvases[self.tabs.currentIndex()]
                    visible.draw()
                    self.assertAlmostEqual(
                        visible.figure.bbox.width, visible.width() * visible.device_pixel_ratio,
                    )
        self.assertTrue(renders)
        self.assertFalse(oversized)


if __name__ == '__main__':
    unittest.main()
