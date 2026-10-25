"""Display regressions for the results (04) and inference (05) pages."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from PyQt6.QtCore import QCoreApplication, QEvent, QSize
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication, QScrollArea, QStackedWidget, QVBoxLayout, QWidget
import numpy as np

# Replace only the database entry: restoring all of sys.modules after importing
# cv2 would unload Python modules while leaving its native extension loaded.
_original_database = sys.modules.get('models.database')
sys.modules['models.database'] = types.SimpleNamespace(db=Mock())
try:
    from gui.pages.test_page import TestPage as InferencePage
    from gui.pages.result_page import ResultPage
finally:
    if _original_database is None:
        sys.modules.pop('models.database', None)
    else:
        sys.modules['models.database'] = _original_database
from gui.styles import get_full_stylesheet
from gui.ui_scale import UiScaleManager
from gui.widgets.image_preview import ImagePreview


class ResultAndInferenceLayoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.manager = UiScaleManager(self.app, get_full_stylesheet())
        self.root = QWidget()
        self.root.setObjectName('workspaceRoot')
        self.root.resize(1600, 900)
        self.stack = QStackedWidget()
        QVBoxLayout(self.root).addWidget(self.stack)
        self.result = ResultPage()
        # Production startup cleanup must never run against real output files.
        with patch.object(InferencePage, '_clear_test_outputs_on_startup'):
            self.inference = InferencePage()
        for page in (self.result, self.inference):
            area = QScrollArea()
            area.setWidgetResizable(True)
            area.setWidget(page)
            self.stack.addWidget(area)
        self.manager.set_scale(100)
        self.root.show()
        self.pump()

    def tearDown(self):
        self.inference.video_player.clear()
        self.app.removeEventFilter(self.manager)
        self.root.close()
        self.root.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.manager.deleteLater()
        self.app.processEvents()

    def pump(self):
        for _ in range(4):
            self.app.processEvents()

    def assert_preview_fits(self, viewer, preview):
        self.assertTrue(viewer.rect().contains(preview.geometry()),
                        (viewer.size(), preview.geometry()))
        pixmap = preview.pixmap()
        if pixmap is not None and not pixmap.isNull():
            displayed = pixmap.deviceIndependentSize()
            self.assertLessEqual(displayed.width(), preview.contentsRect().width() + 1)
            self.assertLessEqual(displayed.height(), preview.contentsRect().height() + 1)

    def test_result_image_remains_visible_and_resizes(self):
        self.stack.setCurrentIndex(0)
        with tempfile.TemporaryDirectory() as folder:
            path = str(Path(folder) / 'result.png')
            source = QPixmap(2400, 1600)
            source.fill()
            self.assertTrue(source.save(path))
            self.result.display_image(path)
            for percent in (100, 130, 150, 200, 75, 130):
                self.manager.set_scale(percent)
                self.root.resize(1400 if percent == 200 else 1600, 900)
                self.pump()
                self.assertFalse(self.result.image_label.pixmap().isNull())
                self.assert_preview_fits(self.result.image_container, self.result.image_label)
                self.assertEqual(self.result.image_label.sizeHint(), QSize(round(4.8 * percent), round(3.6 * percent)))
        self.result.set_project(None)
        self.assertTrue(self.result.image_label.pixmap().isNull())

    def test_inference_columns_and_both_previews_survive_zoom(self):
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        self.inference.image_viewer.show_array(frame)
        self.inference.video_player.display_frame(frame)
        for percent in (100, 130, 150, 200, 75, 130):
            self.stack.setCurrentIndex(0)
            self.manager.set_scale(percent)
            self.stack.setCurrentIndex(1)
            self.pump()
            split = self.inference.main_splitter
            self.assertFalse(split.childrenCollapsible())
            self.assertGreaterEqual(split.sizes()[0], self.inference.config_panel.minimumWidth())
            self.assertGreaterEqual(split.sizes()[1], self.inference.visualization_panel.minimumWidth())
            self.assertFalse(split.widget(0).geometry().intersects(split.widget(1).geometry()))
            self.assertEqual(self.inference.config_scroll.horizontalScrollBar().maximum(), 0)
            for tab in (0, 1):
                self.inference.result_tabs.setCurrentIndex(tab)
                self.pump()
                viewer = self.inference.image_viewer if tab == 0 else self.inference.video_player
                label = viewer.image_label if tab == 0 else viewer.video_label
                self.assertTrue(viewer.isVisible())
                self.assertFalse(label.pixmap().isNull())
                self.assert_preview_fits(viewer, label)
            # A dragged splitter must keep the configuration controls reachable.
            split.moveSplitter(0, 1)
            self.pump()
            self.assertGreaterEqual(split.sizes()[0], self.inference.config_panel.minimumWidth())

    def test_preview_hint_does_not_depend_on_source_image(self):
        preview = ImagePreview(parent=self.root)
        hint = preview.sizeHint()
        pixmap = QPixmap(2400, 1600)
        pixmap.fill()
        preview.setPixmap(pixmap)
        self.assertEqual(preview.sizeHint(), hint)
        self.manager.set_scale(130)
        self.assertEqual(preview.sizeHint(), QSize(624, 468))

    def test_export_buttons_stay_inside_panel_when_wrapped(self):
        self.stack.setCurrentIndex(0)
        buttons = (self.result.btn_export_onnx, self.result.btn_export_tensorrt,
                   self.result.btn_export_torchscript, self.result.btn_export_pt,
                   self.result.btn_export_folder)
        for percent in (100, 130, 200):
            self.manager.set_scale(percent)
            self.root.resize(1000, 800)
            self.pump()
            for button in buttons:
                self.assertTrue(button.parentWidget().rect().contains(button.geometry()),
                                (percent, button.text(), button.geometry(), button.parentWidget().size()))


if __name__ == '__main__':
    unittest.main()
