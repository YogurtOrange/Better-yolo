"""Regression tests for empty initial classes in AnnotatePage and AnnotatePage <-> AiLabelPage class sync."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import QApplication, QColorDialog, QInputDialog, QMessageBox

import config.app_info as app_info
import core.qwen_annotator as qwen_mod
import gui.pages.ai_label_page as ai_label_mod
import gui.pages.annotate_page as annotate_mod
from gui.pages.ai_label_page import AiLabelPage
from gui.pages.annotate_page import AnnotatePage
from models.database import Database


class ClassSyncAndEmptyPresetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.temp_db = Database(str(self.root / "sync_test.db"))

        self._patches = [
            patch.object(annotate_mod, 'db', self.temp_db),
            patch.object(ai_label_mod, 'db', self.temp_db),
            patch.object(qwen_mod, 'db', self.temp_db),
        ]
        for p in self._patches:
            p.start()

        self.project_id = self.temp_db.create_project("sync_proj", project_type="detect", classes=[])
        proj = self.temp_db.get_project(self.project_id)
        if proj['storage_path'] and os.path.exists(proj['storage_path']):
            shutil.rmtree(proj['storage_path'], ignore_errors=True)
        self.proj_storage = self.root / "proj_dir"
        self.proj_storage.mkdir(parents=True, exist_ok=True)
        self.temp_db.update_project(self.project_id, storage_path=str(self.proj_storage))

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.tmpdir.cleanup()

    def _wait_project_load(self, annotate_page: AnnotatePage):
        if hasattr(annotate_page, 'load_thread'):
            annotate_page.load_thread.wait(3000)
        for _ in range(10):
            QCoreApplication.processEvents()

    def test_annotate_page_has_no_preset_classes_by_default(self):
        """Opening 02 AnnotatePage on a new project must NOT preset person/car."""
        annotate_page = AnnotatePage()
        annotate_page.set_project(self.project_id)
        self._wait_project_load(annotate_page)

        self.assertEqual(annotate_page.classes, [])
        self.assertEqual(annotate_page.class_list.count(), 0)
        annotate_page.deleteLater()

    def test_manual_classes_sync_to_ai_page_with_empty_descriptions(self):
        """Adding, renaming, and deleting classes in 02 AnnotatePage syncs labels to AiLabelPage without generating descriptions."""
        annotate_page = AnnotatePage()
        ai_page = AiLabelPage()

        # Wire bidirectional signals just like MainWindow
        annotate_page.classes_changed.connect(
            lambda info=None: ai_page.on_external_project_classes_changed(
                annotate_page.current_project_id, info
            )
        )
        ai_page.classes_changed.connect(annotate_page.refresh_classes)

        annotate_page.set_project(self.project_id)
        self._wait_project_load(annotate_page)

        # Case 1: Add two classes in 02 AnnotatePage BEFORE AiLabelPage even opens the project
        with patch.object(QInputDialog, 'getText', side_effect=[("defect_a", True), ("defect_b", True)]), \
                patch.object(QColorDialog, 'getColor', return_value=QColor("#FF5500")):
            annotate_page.add_class()
            annotate_page.add_class()

        # Switch to AiLabelPage -> both labels appear in order with empty descriptions
        ai_page.set_project(self.project_id)
        self.assertEqual(ai_page.category_table.rowCount(), 2)
        self.assertEqual(ai_page.category_table.item(0, 1).text(), "defect_a")
        self.assertEqual(ai_page.category_table.item(0, 2).text(), "")
        self.assertEqual(ai_page.category_table.item(1, 1).text(), "defect_b")
        self.assertEqual(ai_page.category_table.item(1, 2).text(), "")

        # Detection prompt should be empty until descriptions are filled,
        # whereas generation prompt must include the empty-desc label placeholders!
        self.assertEqual(ai_page.build_prompt_text(include_empty_desc=False), "")
        gen_prompt = ai_page.build_prompt_text(include_empty_desc=True)
        self.assertIn("defect_a：", gen_prompt)
        self.assertIn("defect_b：", gen_prompt)

        # Simulate AI filling descriptions via fill_table_from_prompt
        ai_page.fill_table_from_prompt(
            "defect_a：金属表面的圆形凹坑缺陷\ndefect_b：细长条状表面划痕"
        )
        self.assertEqual(ai_page.category_table.item(0, 2).text(), "金属表面的圆形凹坑缺陷")
        self.assertEqual(ai_page.category_table.item(1, 2).text(), "细长条状表面划痕")

        # Case 2: Add a third class while AiLabelPage already has the project loaded -> real-time sync, empty desc
        with patch.object(QInputDialog, 'getText', return_value=("defect_c", True)), \
                patch.object(QColorDialog, 'getColor', return_value=QColor("#00AAFF")):
            annotate_page.add_class()

        self.assertEqual(ai_page.category_table.rowCount(), 3)
        self.assertEqual(ai_page.category_table.item(2, 1).text(), "defect_c")
        self.assertEqual(ai_page.category_table.item(2, 2).text(), "")

        # Case 3: Rename defect_a -> pit_defect in 02 AnnotatePage -> preserves existing AI description!
        first_item = annotate_page.class_list.item(0)
        with patch.object(QInputDialog, 'getText', return_value=("pit_defect", True)), \
                patch.object(QColorDialog, 'getColor', return_value=QColor("#FF5500")):
            annotate_page.edit_class(first_item)

        self.assertEqual(ai_page.category_table.item(0, 1).text(), "pit_defect")
        self.assertEqual(ai_page.category_table.item(0, 2).text(), "金属表面的圆形凹坑缺陷")

        # Case 4: Delete defect_b in 02 AnnotatePage -> removes row from AiLabelPage
        second_item = annotate_page.class_list.item(1)
        with patch.object(QMessageBox, 'question', return_value=QMessageBox.StandardButton.Yes):
            annotate_page.delete_class(second_item)

        remaining_labels = [
            ai_page.category_table.item(r, 1).text()
            for r in range(ai_page.category_table.rowCount())
        ]
        self.assertEqual(remaining_labels, ["pit_defect", "defect_c"])

        annotate_page.deleteLater()
        ai_page.deleteLater()

    def test_canvas_renders_class_label_without_writing_name_to_txt(self):
        """Drawing/displaying boxes shows class name on canvas UI only, never in exported YOLO .txt."""
        from PIL import Image
        from PyQt6.QtCore import QPoint
        from PyQt6.QtGui import QImage, QPainter
        from PyQt6.QtWidgets import QFileDialog

        img_path = self.proj_storage / "canvas_test.png"
        Image.new("RGB", (300, 200), color=(255, 255, 255)).save(img_path, format="PNG")
        img_id = self.temp_db.add_image(
            self.project_id, "canvas_test.png", str(img_path), width=300, height=200
        )
        cls_list = [{"id": 0, "name": "焊缝气孔", "color": "#E11D48"}]
        self.temp_db.update_project(self.project_id, classes=cls_list)
        self.temp_db.add_annotation(
            img_id, self.project_id, 0, "焊缝气孔", "bbox",
            {"x": 40, "y": 50, "width": 100, "height": 80},
        )

        page = AnnotatePage()
        page.set_project(self.project_id)
        self._wait_project_load(page)
        page.resize(900, 600)
        page.load_image(img_id)

        # Verify canvas has class_names mapped and renders both saved box label and live-drawing preview
        self.assertEqual(page.canvas.class_names.get(0), "焊缝气孔")

        baseline = QImage(600, 400, QImage.Format.Format_ARGB32)
        baseline.fill(0xFFFFFFFF)
        rendered = QImage(600, 400, QImage.Format.Format_ARGB32)
        rendered.fill(0xFFFFFFFF)

        p = QPainter(rendered)
        page.canvas.draw_class_label(p, QPoint(50, 60), "焊缝气孔", QColor("#E11D48"), is_selected=True)
        p.end()
        self.assertNotEqual(rendered.bits().asstring(rendered.sizeInBytes()),
                            baseline.bits().asstring(baseline.sizeInBytes()))

        # Trigger full paint path including active rectangle drawing preview
        page.canvas.drawing = True
        page.canvas.current_tool = 'rectangle'
        page.canvas.start_point = QPoint(80, 80)
        page.canvas.current_point = QPoint(180, 160)
        pixmap = page.canvas.grab()
        self.assertFalse(pixmap.isNull())

        # Export YOLO .txt and verify ONLY numeric class_id + coords are written (no class name in .txt)
        out_dir = self.root / "export_check"
        out_dir.mkdir()
        page.export_format.setCurrentText("YOLO格式")
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(out_dir)), \
                patch.object(QMessageBox, 'information'):
            page.export_annotations()

        txt_file = out_dir / "sync_proj_annotations" / "labels" / "canvas_test.txt"
        txt_content = txt_file.read_text(encoding="utf-8").strip()
        self.assertNotIn("焊缝气孔", txt_content)
        tokens = txt_content.split()
        self.assertEqual(len(tokens), 5)
        self.assertEqual(tokens[0], "0")
        page.deleteLater()

    def test_single_line_semicolon_prompt_updates_all_categories(self):
        """Regression for single-line semicolon-joined Qwen prompt output updating all 3/3 categories."""
        ai_page = AiLabelPage()
        ai_page.set_project(self.project_id)
        for idx, name in enumerate(("tiger", "monkey", "bird"), 1):
            ai_page.category_table.insertRow(idx - 1)
            ai_page.category_table.setItem(idx - 1, 0, ai_label_mod.QTableWidgetItem(str(idx)))
            ai_page.category_table.setItem(idx - 1, 1, ai_label_mod.QTableWidgetItem(name))
            ai_page.category_table.setItem(idx - 1, 2, ai_label_mod.QTableWidgetItem(""))

        raw_single_line = (
            "tiger：'tiger'指靶纸上印有的橙黑相间条纹虎形图案（含条纹边缘），不含网格线；"
            "monkey：'monkey'指靶纸上印有的深灰褐色灵长类动物图案（与虎区分），不含背景灌木；"
            "bird：'bird'指靶纸上印有的大型走禽类图案（形似鸸鹋），不含地面裂纹。"
            "若目标被部分遮挡，仅框出可见部分的完整外轮廓。"
        )
        ai_page.fill_table_from_prompt(raw_single_line)

        self.assertIn("橙黑相间条纹虎形图案", ai_page.category_table.item(0, 2).text())
        self.assertNotIn("monkey：", ai_page.category_table.item(0, 2).text())
        self.assertIn("深灰褐色灵长类动物图案", ai_page.category_table.item(1, 2).text())
        self.assertIn("大型走禽类图案", ai_page.category_table.item(2, 2).text())
        self.assertNotIn("若目标被部分遮挡", ai_page.category_table.item(2, 2).text())
        self.assertEqual(
            ai_page.te_extra_rules.toPlainText().strip(),
            "若目标被部分遮挡，仅框出可见部分的完整外轮廓。",
        )
        ai_page.deleteLater()

    def test_version_info_synced(self):
        import re
        root_dir = Path(__file__).parent.parent
        changelog = (root_dir / "CHANGELOG.md").read_text(encoding="utf-8")
        m = re.search(r"^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})", changelog, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(app_info.APP_VERSION, m.group(1))
        self.assertEqual(app_info.LAST_UPDATED, m.group(2))
        self.assertRegex(app_info.APP_VERSION, r"^\d+\.\d+\.\d+$")

        readme = (root_dir / "README.md").read_text(encoding="utf-8")
        self.assertIn(f"Version-v{app_info.APP_VERSION}-", readme)
        self.assertIn(f"`v{app_info.APP_VERSION}`", readme)


if __name__ == '__main__':
    unittest.main()
