"""Regression tests covering the 5 Quick Wins across DB, importers, exports, auto-labeler, and settings."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PyQt6.QtCore import QSettings
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

import core.annotation_importer as ann_importer_mod
import core.auto_labeler as auto_labeler_mod
import core.import_manager as import_mgr_mod
from core.annotation_importer import AnnotationImporter
from core.auto_labeler import AutoLabeler, BatchLabelingThread
from core.import_manager import ImportManager
from core.model_manager import ModelManager, get_pretrained_dir
import gui.pages.annotate_page as annotate_page_mod
from gui.pages.annotate_page import AnnotatePage
from gui.pages.import_page import ImageBatchImportThread
from gui.pages.settings_page import SettingsPage
from models.database import Database


def _create_rgb_image(path: Path, color=(20, 120, 220), size=(200, 100)):
    path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", size, color=color)
    img.save(path, format="PNG")


class QuickWinsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.temp_db = Database(str(self.root / "test.db"))

        # Redirect module-level db references to the isolated test DB
        self._patches = [
            patch.object(import_mgr_mod, 'db', self.temp_db),
            patch.object(ann_importer_mod, 'db', self.temp_db),
            patch.object(auto_labeler_mod, 'db', self.temp_db),
            patch.object(annotate_page_mod, 'db', self.temp_db),
        ]
        for p in self._patches:
            p.start()

        # Override project storage_path inside temp directory
        self.project_id = self.temp_db.create_project("qw_proj", project_type="segment")
        proj_info = self.temp_db.get_project(self.project_id)
        created_storage = proj_info['storage_path']
        if created_storage and os.path.exists(created_storage):
            shutil.rmtree(created_storage, ignore_errors=True)
        self.proj_storage = self.root / "project_storage"
        self.proj_storage.mkdir(parents=True, exist_ok=True)
        self.temp_db.update_project(self.project_id, storage_path=str(self.proj_storage))

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.tmpdir.cleanup()

    def test_1_import_hash_deduplication_and_async_thread(self):
        """Quick Win 1: MD5 deduplication (including legacy backfill), delete_image DB sync, and async QThread."""
        src_dir = self.root / "raw_images"
        img_a = src_dir / "a.png"
        img_a_dup = src_dir / "a_copy.png"
        img_b = src_dir / "b.png"
        _create_rgb_image(img_a, color=(255, 0, 0))
        _create_rgb_image(img_a_dup, color=(255, 0, 0))  # Identical content, different filename
        _create_rgb_image(img_b, color=(0, 255, 0))

        # Simulate a legacy image imported before file_hash existed (file_hash=None)
        legacy_dst = self.proj_storage / "images" / "legacy_a.png"
        _create_rgb_image(legacy_dst, color=(255, 0, 0))
        legacy_id = self.temp_db.add_image(
            project_id=self.project_id,
            filename="legacy_a.png",
            storage_path=str(legacy_dst),
            width=200,
            height=100,
            file_hash=None,
        )

        # Run async folder import thread: both a.png and a_copy.png match legacy_a.png's hash, only b.png should import
        results = []
        worker = ImageBatchImportThread(project_id=self.project_id, folder_path=str(src_dir))
        worker.finished.connect(lambda ok, msg, imp, skp, lbl: results.append((ok, imp, skp)))
        worker.run()

        self.assertEqual(results, [(True, 1, 2)])
        # Legacy record should now have its file_hash backfilled
        self.assertIsNotNone(self.temp_db.get_image(legacy_id)['file_hash'])

        # ImportManager.delete_image removes both file and DB row
        mgr = ImportManager(self.project_id)
        self.assertTrue(mgr.delete_image(legacy_id))
        self.assertIsNone(self.temp_db.get_image(legacy_id))
        self.assertFalse(legacy_dst.exists())

    def test_2_annotation_importer_auto_image_import_and_coco_multi_box(self):
        """Quick Win 2: Auto-import missing images + YOLO classes.txt + COCO multi-object grouping + VOC classes."""
        # 2a) COCO multi-object import without pre-importing images
        coco_dir = self.root / "coco_ds"
        _create_rgb_image(coco_dir / "images" / "sample1.png", color=(10, 20, 30), size=(200, 100))
        coco_json = {
            "images": [{"id": 10, "file_name": "sample1.png", "width": 200, "height": 100}],
            "categories": [
                {"id": 1, "name": "cat"},
                {"id": 2, "name": "dog"},
            ],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 10,
                    "category_id": 1,
                    "bbox": [10, 10, 50, 40],
                    "segmentation": [[10, 10, 60, 10, 60, 50, 10, 50]],
                },
                {
                    "id": 2,
                    "image_id": 10,
                    "category_id": 2,
                    "bbox": [80, 20, 40, 30],
                    "segmentation": [[80, 20, 120, 20, 120, 50, 80, 50]],
                },
            ],
        }
        coco_file = coco_dir / "annotations.json"
        coco_file.write_text(json.dumps(coco_json), encoding="utf-8")

        importer = AnnotationImporter(self.project_id)
        imported, skipped = importer.import_coco_annotations(str(coco_file), overwrite=False)
        self.assertEqual((imported, skipped), (2, 0))

        proj_imgs = self.temp_db.get_project_images(self.project_id)
        self.assertEqual(len(proj_imgs), 1)
        anns = self.temp_db.get_image_annotations(proj_imgs[0]['id'])
        self.assertEqual(len(anns), 2)
        self.assertEqual(anns[0]['type'], 'polygon')
        self.assertIsInstance(anns[0]['data']['points'][0], dict)
        self.assertIn('x', anns[0]['data']['points'][0])
        self.assertEqual({a['class_name'] for a in anns}, {"cat", "dog"})
        self.assertEqual({a['class_id'] for a in anns}, {0, 1})

        # 2b) VOC import with JPEGImages auto-discovery & distinct class IDs
        voc_root = self.root / "voc_ds"
        _create_rgb_image(voc_root / "JPEGImages" / "voc1.png", color=(40, 50, 60), size=(200, 100))
        voc_anns_dir = voc_root / "Annotations"
        voc_anns_dir.mkdir(parents=True, exist_ok=True)
        (voc_anns_dir / "voc1.xml").write_text(
            """<annotation>
                <filename>voc1.png</filename>
                <size><width>200</width><height>100</height></size>
                <object><name>bird</name><bndbox><xmin>5</xmin><ymin>5</ymin><xmax>45</xmax><ymax>35</ymax></bndbox></object>
            </annotation>""",
            encoding="utf-8",
        )
        imp_voc, skp_voc = AnnotationImporter(self.project_id).import_voc_annotations(str(voc_anns_dir))
        self.assertEqual((imp_voc, skp_voc), (1, 0))
        classes = json.loads(self.temp_db.get_project(self.project_id)['classes'])
        bird_cls = next(c for c in classes if c['name'] == 'bird')
        self.assertEqual(bird_cls['id'], 2)

    def test_3_polygon_annotations_export_and_coco_dataset_export(self):
        """Quick Win 3: export_annotations exports 'polygon' properly in YOLO format, and export_dataset honors COCO format."""
        img_path = self.proj_storage / "images" / "seg.png"
        _create_rgb_image(img_path, color=(100, 150, 200), size=(200, 100))
        img_id = self.temp_db.add_image(
            project_id=self.project_id,
            filename="seg.png",
            storage_path=str(img_path),
            width=200,
            height=100,
        )
        self.temp_db.update_project(
            self.project_id,
            classes=[{"id": 0, "name": "leaf", "color": "#10B981"}],
        )
        self.temp_db.add_annotation(
            image_id=img_id,
            project_id=self.project_id,
            class_id=0,
            class_name="leaf",
            annotation_type="polygon",
            data={"points": [{"x": 20, "y": 10}, {"x": 120, "y": 10}, {"x": 120, "y": 60}]},
        )

        page = AnnotatePage()
        page.current_project_id = self.project_id
        page.classes = [{"id": 0, "name": "leaf", "color": "#10B981"}]

        out_yolo_dir = self.root / "out_yolo"
        out_yolo_dir.mkdir()
        page.export_format.setCurrentText("YOLO格式")
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(out_yolo_dir)), \
                patch.object(QMessageBox, 'information'):
            page.export_annotations()

        label_txt = out_yolo_dir / "qw_proj_annotations" / "labels" / "seg.txt"
        self.assertTrue(label_txt.exists())
        content = label_txt.read_text(encoding="utf-8").strip()
        self.assertEqual(content, "0 0.100000 0.100000 0.600000 0.100000 0.600000 0.600000")

        out_coco_ds = self.root / "out_coco_ds"
        out_coco_ds.mkdir()
        page.export_format.setCurrentText("COCO格式")
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(out_coco_ds)), \
                patch.object(QMessageBox, 'information'):
            page.export_dataset()

        coco_ds_root = out_coco_ds / "qw_proj"
        self.assertTrue((coco_ds_root / "images" / "seg.png").exists())
        coco_out = json.loads((coco_ds_root / "annotations.json").read_text(encoding="utf-8"))
        self.assertEqual(len(coco_out["annotations"]), 1)
        self.assertEqual(coco_out["annotations"][0]["segmentation"], [[20.0, 10.0, 120.0, 10.0, 120.0, 60.0]])
        page.deleteLater()

    def test_4_auto_labeler_respects_lock_and_resolves_real_class_names(self):
        """Quick Win 4: AutoLabeler uses real model/project class names instead of class_0 and skips locked=1 images."""
        img_unlocked_path = self.proj_storage / "images" / "u.png"
        img_locked_path = self.proj_storage / "images" / "l.png"
        _create_rgb_image(img_unlocked_path)
        _create_rgb_image(img_locked_path)

        u_id = self.temp_db.add_image(self.project_id, "u.png", str(img_unlocked_path), width=200, height=100)
        l_id = self.temp_db.add_image(self.project_id, "l.png", str(img_locked_path), width=200, height=100)
        self.temp_db.set_image_locked(l_id, True)
        self.temp_db.add_annotation(
            l_id, self.project_id, 0, "human_verified", "bbox", {"x": 1, "y": 2, "width": 30, "height": 40}
        )

        labeler = AutoLabeler("yolov8n", ModelManager())
        fake_preds = [{
            'image_id': u_id,
            'type': 'bbox',
            'class_id': 0,
            'model_class_name': 'helmet',
            'confidence': 0.87654,
            'data': {'x': 10, 'y': 20, 'width': 50, 'height': 60},
        }]

        with patch.object(labeler, 'process_single_image', return_value=fake_preds):
            images = self.temp_db.get_project_images(self.project_id)
            thread = BatchLabelingThread(labeler, images, {'overwrite_labels': True})
            completed_msgs = []
            thread.batch_completed.connect(lambda ok, msg: completed_msgs.append((ok, msg)))
            thread.run()

        self.assertTrue(completed_msgs and completed_msgs[0][0])
        self.assertIn("跳过人工锁定图片 1 张", completed_msgs[0][1])

        # Unlocked image got real class_name 'helmet' and source='yolo'
        u_anns = self.temp_db.get_image_annotations(u_id)
        self.assertEqual(len(u_anns), 1)
        self.assertEqual(u_anns[0]['class_name'], 'helmet')
        self.assertEqual(u_anns[0]['attributes']['source'], 'yolo')

        # Locked image preserved its human annotation untouched
        l_anns = self.temp_db.get_image_annotations(l_id)
        self.assertEqual(len(l_anns), 1)
        self.assertEqual(l_anns[0]['class_name'], 'human_verified')

    def test_5_settings_reset_shortcuts_and_pretrained_dir_linkage(self):
        """Quick Win 5: Resetting settings saves Q/E shortcuts (not W/V) and ModelManager reads pretrained_path."""
        settings = QSettings("Better-YOLO", "Settings")
        orig_rect = settings.value("rect_tool_shortcut")
        orig_move = settings.value("move_tool_shortcut")
        orig_pretrained = settings.value("pretrained_path")

        try:
            custom_pretrained = self.root / "custom_weights"
            settings.setValue("pretrained_path", str(custom_pretrained))
            mgr = ModelManager()
            self.assertEqual(get_pretrained_dir(), custom_pretrained)
            self.assertEqual(mgr.pretrained_dir, custom_pretrained)
            self.assertEqual(mgr.get_model_path("YOLOv11", "n", "detect"), custom_pretrained / "yolo11n.pt")

            page = SettingsPage()
            saved_events = []
            page.settings_saved.connect(lambda: saved_events.append(True))
            with patch.object(QMessageBox, 'information'):
                page.reset_settings()

            self.assertEqual(settings.value("rect_tool_shortcut"), "Q")
            self.assertEqual(settings.value("move_tool_shortcut"), "E")
            self.assertTrue(saved_events)
            page.deleteLater()
        finally:
            if orig_rect is None:
                settings.remove("rect_tool_shortcut")
            else:
                settings.setValue("rect_tool_shortcut", orig_rect)
            if orig_move is None:
                settings.remove("move_tool_shortcut")
            else:
                settings.setValue("move_tool_shortcut", orig_move)
            if orig_pretrained is None:
                settings.remove("pretrained_path")
            else:
                settings.setValue("pretrained_path", orig_pretrained)


if __name__ == '__main__':
    unittest.main()
