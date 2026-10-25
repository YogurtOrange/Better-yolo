"""Run with QT_QPA_PLATFORM=offscreen python -m unittest discover -s tests."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import unittest

from PyQt6.QtCore import QCoreApplication, QEvent, QSize, Qt
from PyQt6.QtWidgets import (
    QApplication, QDialog, QLabel, QListWidget, QPushButton, QVBoxLayout, QWidget,
)

from gui.ui_scale import UiScaleManager, scale_stylesheet, ui_font
from gui.widgets.flow_layout import FlowLayout


class UiScaleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.manager = UiScaleManager(self.app, 'QWidget { font-size: 14px; }')
        self.root = QWidget()
        self.layout = QVBoxLayout(self.root)
        self.layout.setContentsMargins(12, 14, 12, 14)
        self.layout.setSpacing(8)
        self.label = QLabel('测试文字')
        self.label.setStyleSheet('font-size: 16px; padding: 4px;')
        self.button = QPushButton('确认')
        self.button.setFixedSize(120, 36)
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.button)
        self.manager.set_scale(100)
        self.root.show()
        self.app.processEvents()

    def tearDown(self):
        self.app.removeEventFilter(self.manager)
        for widget in self.app.topLevelWidgets():
            widget.close()
            widget.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.manager.deleteLater()
        self.app.processEvents()

    def test_font_controls_and_spacing_scale_together(self):
        self.manager.set_scale(150)
        self.assertEqual(self.button.size(), QSize(180, 54))
        self.assertEqual(self.label.font().pixelSize(), 24)
        self.assertEqual(self.layout.contentsMargins().left(), 18)
        self.assertEqual(self.layout.spacing(), 12)
        self.assertEqual(ui_font('sans-serif', 10).pointSizeF(), 15)

    def test_round_trips_do_not_accumulate(self):
        for percent in (125, 200, 75, 150, 100, 125, 100):
            self.manager.set_scale(percent)
        self.assertEqual(self.button.size(), QSize(120, 36))
        self.assertEqual(self.label.styleSheet(), 'font-size: 16px; padding: 4px;')
        self.assertEqual(self.layout.spacing(), 8)
        self.assertEqual(self.layout.contentsMargins().top(), 14)

    def test_styles_changed_after_zoom_are_scaled_once(self):
        self.manager.set_scale(150)
        self.label.setStyleSheet('font-size: 20px; padding: 6px;')
        self.assertIn('font-size: 30px', self.label.styleSheet())
        self.manager.set_scale(100)
        self.assertEqual(self.label.styleSheet(), 'font-size: 20px; padding: 6px;')

    def test_baselines_survive_loss_of_python_wrapper_caches(self):
        self.manager.set_scale(150)
        # A new Python wrapper has no entry in the manager's weak caches, but
        # the underlying Qt object must still retain its original metrics.
        self.manager._widgets.clear()
        self.manager._layouts.clear()
        self.manager.set_scale(200)
        self.assertEqual(self.button.size(), QSize(240, 72))
        self.assertEqual(self.label.font().pixelSize(), 32)
        self.assertEqual(self.layout.contentsMargins().left(), 24)
        self.assertEqual(self.layout.spacing(), 16)
        self.manager.set_scale(100)
        self.assertEqual(self.button.size(), QSize(120, 36))
        self.assertEqual(self.label.styleSheet(), 'font-size: 16px; padding: 4px;')

    def test_updated_inline_style_survives_wrapper_cache_loss(self):
        self.manager.set_scale(150)
        self.label.setStyleSheet('font-size: 20px; padding: 6px;')
        self.manager._widgets.clear()
        self.manager._layouts.clear()
        self.manager.set_scale(200)
        self.assertIn('font-size: 40px', self.label.styleSheet())
        self.manager.set_scale(100)
        self.assertEqual(self.label.styleSheet(), 'font-size: 20px; padding: 6px;')

    def test_new_dialog_uses_current_scale_before_display(self):
        self.manager.set_scale(150)
        dialog = QDialog(self.root)
        dialog.setFixedSize(320, 160)
        layout = QVBoxLayout(dialog)
        button = QPushButton('完成')
        button.setFixedHeight(30)
        button.setStyleSheet('font-size: 16px;')
        layout.addWidget(button)
        dialog.show()
        self.app.processEvents()
        self.assertEqual(dialog.size(), QSize(480, 240))
        self.assertEqual(button.height(), 45)
        self.assertIn('24px', button.styleSheet())

    def test_thumbnail_grid_and_icons_scale(self):
        view = QListWidget(self.root)
        view.setIconSize(QSize(160, 160))
        view.setGridSize(QSize(180, 200))
        view.setSpacing(16)
        self.layout.addWidget(view)
        self.manager.set_scale(150)
        self.assertEqual(view.iconSize(), QSize(240, 240))
        self.assertEqual(view.gridSize(), QSize(270, 300))
        self.assertEqual(view.spacing(), 24)

    def test_zoom_limits(self):
        self.manager.set_scale(500)
        self.assertEqual(self.manager.percent, 200)
        self.manager.set_scale(20)
        self.assertEqual(self.manager.percent, 75)

    def test_wrapping_toolbar_keeps_buttons_inside(self):
        panel = QWidget(self.root)
        flow = FlowLayout(panel, spacing=10)
        flow.setContentsMargins(0, 0, 0, 0)
        for _ in range(3):
            button = QPushButton('工具', panel)
            button.setFixedSize(100, 36)
            flow.addWidget(button)
        self.layout.addWidget(panel)
        panel.show()
        self.app.processEvents()
        self.assertEqual(flow.heightForWidth(220), 82)
        self.assertEqual(flow.heightForWidth(110), 128)

    def test_stylesheet_units_preserve_colors_and_percentages(self):
        source = 'padding: -2px 8px; color: #123456; width: 50%;'
        self.assertEqual(scale_stylesheet(source, 1.5), 'padding: -3px 12px; color: #123456; width: 50%;')


if __name__ == '__main__':
    unittest.main()
