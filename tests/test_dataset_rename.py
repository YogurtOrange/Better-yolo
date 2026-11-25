"""Unit tests for one-click dataset image & label renaming (v1.2.0)."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

import config.app_info as app_info
import core.dataset_renamer as renamer_mod
import gui.pages.annotate_page as annotate_mod
import gui.pages.dataset_rename_dialog as dialog_mod
import gui.pages.import_page as import_mod
from core.dataset_renamer import (
    preview_external_folder_rename,
    preview_project_rename,
    rename_external_folder_dataset,
    rename_project_dataset,
)
from gui.pages.annotate_page import AnnotatePage
from gui.pages.dataset_rename_dialog import DatasetRenameDialog
from gui.pages.import_page import ImportPage
from models.database import Database


class DatasetRenameTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.temp_db = Database(str(self.root / "rename_test.db"))

        self._patches = [
            patch.object(renamer_mod, 'db', self.temp_db),
            patch.object(dialog_mod, 'db', self.temp_db),
            patch.object(import_mod, 'db', self.temp_db),
            patch.object(annotate_mod, 'db', self.temp_db),
        ]
        for p in self._patches:
            p.start()

        self.project_id = self.temp_db.create_project(
            "rename_proj",
            project_type="detect",
            classes=[{"id": 0, "name": "target", "color": "#EF4444"}],
        )
        proj = self.temp_db.get_project(self.project_id)
        if proj['storage_path'] and os.path.exists(proj['storage_path']):
            shutil.rmtree(proj['storage_path'], ignore_errors=True)
        self.proj_storage = self.root / "proj_dir"
        (self.proj_storage / "images").mkdir(parents=True, exist_ok=True)
        self.temp_db.update_project(self.project_id, storage_path=str(self.proj_storage))

    def tearDown(self):
        for p in reversed(self._patches):
            p.stop()
        self.tmpdir.cleanup()

    def _create_img(self, path: Path, color=(120, 130, 140)):
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 80), color=color).save(path)

    def test_project_rename_default_img_001_and_custom_start_index_collision_safe(self):
        """Project rename defaults to img_001.., supports custom start number, avoids collision on re-rename,
        syncs ai_progress.json, and exports matching img_xxx.txt files."""
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        raw_names = ["DSC_0099.jpg", "DSC_0002.png", "DSC_0015.jpg"]
        img_ids = []
        expected_bytes = []
        for name, col in zip(raw_names, colors):
            sp = self.proj_storage / "images" / f"20261125_{name}"
            self._create_img(sp, color=col)
            expected_bytes.append(sp.read_bytes())
            iid = self.temp_db.add_image(self.project_id, name, str(sp), width=100, height=80)
            img_ids.append(iid)

        # Add annotation on first two images
        self.temp_db.add_annotation(
            img_ids[0], self.project_id, 0, "target", "bbox",
            {"x": 10, "y": 10, "width": 40, "height": 30},
        )
        self.temp_db.add_annotation(
            img_ids[1], self.project_id, 0, "target", "bbox",
            {"x": 20, "y": 15, "width": 50, "height": 40},
        )

        # Save AI breakpoint referencing original names
        bp_file = self.proj_storage / "ai_progress.json"
        bp_file.write_text(
            json.dumps({"completed": ["DSC_0099.jpg", "DSC_0002.png"], "task_signature": "sig1"}),
            encoding="utf-8",
        )

        # 1) Default rename: prefix='img_', start_index=1 -> img_001.jpg, img_002.png, img_003.jpg
        res1 = rename_project_dataset(self.project_id)
        self.assertEqual(res1['renamed_images'], 3)
        self.assertEqual(res1['synced_db_labels'], 2)

        db_imgs = self.temp_db.get_project_images(self.project_id)
        self.assertEqual([i['filename'] for i in db_imgs], ["img_001.jpg", "img_002.png", "img_003.jpg"])
        for i, orig_bytes in zip(db_imgs, expected_bytes):
            p = Path(i['storage_path'])
            self.assertTrue(p.is_file())
            self.assertEqual(p.name, i['filename'])
            self.assertEqual(p.read_bytes(), orig_bytes)

        # AI breakpoint updated to new filenames
        bp_after = json.loads(bp_file.read_text(encoding="utf-8"))
        self.assertEqual(bp_after['completed'], ["img_001.jpg", "img_002.png"])
        self.assertEqual(bp_after['task_signature'], "sig1")

        # 2) Collision-safe re-rename starting from 2 (img_001..003 -> img_002..004) without overwriting!
        res2 = rename_project_dataset(self.project_id, prefix="img_", start_index=2, min_digits=3)
        self.assertEqual(res2['first_name'], "img_002.jpg")
        self.assertEqual(res2['last_name'], "img_004.jpg")
        db_imgs_2 = self.temp_db.get_project_images(self.project_id)
        for i, orig_bytes in zip(db_imgs_2, expected_bytes):
            self.assertEqual(Path(i['storage_path']).read_bytes(), orig_bytes)

        # 3) Export YOLO annotations from AnnotatePage -> produces img_002.txt and img_003.txt!
        page = AnnotatePage()
        page.current_project_id = self.project_id
        page.classes = [{"id": 0, "name": "target", "color": "#EF4444"}]
        export_root = self.root / "exported"
        export_root.mkdir()
        page.export_format.setCurrentText("YOLO格式")
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(export_root)), \
                patch.object(QMessageBox, 'information'):
            page.export_annotations()
        labels_out = sorted(p.name for p in (export_root / "rename_proj_annotations" / "labels").glob("*.txt"))
        self.assertEqual(labels_out, ["img_002.txt", "img_003.txt"])
        page.deleteLater()

    def test_external_folder_yolo_split_and_flat_with_classes_txt_protected(self):
        """External folder rename pairs images with labels in both images/+labels/ and flat layouts,
        never renames classes.txt, and honors custom start_index."""
        # Layout A: YOLO images/train + labels/train + classes.txt
        ds_yolo = self.root / "yolo_ds"
        (ds_yolo / "images" / "train").mkdir(parents=True)
        (ds_yolo / "labels" / "train").mkdir(parents=True)
        (ds_yolo / "classes.txt").write_text("tiger\nmonkey\n", encoding="utf-8")
        (ds_yolo / "labels" / "classes.txt").write_text("tiger\nmonkey\n", encoding="utf-8")

        for stem, box_line, col in [
            ("frame_10", "0 0.5 0.5 0.2 0.2\n", (10, 20, 30)),
            ("frame_2", "1 0.3 0.3 0.1 0.1\n", (40, 50, 60)),
        ]:
            self._create_img(ds_yolo / "images" / "train" / f"{stem}.jpg", color=col)
            (ds_yolo / "labels" / "train" / f"{stem}.txt").write_text(box_line, encoding="utf-8")

        # Natural sort puts frame_2 BEFORE frame_10! Start from 50 -> img_050, img_051
        res = rename_external_folder_dataset(
            str(ds_yolo), prefix="img_", start_index=50, min_digits=3, sort_mode="natural_name"
        )
        self.assertEqual(res['renamed_images'], 2)
        self.assertEqual(res['renamed_label_files'], 2)

        # frame_2 became img_050, frame_10 became img_051
        self.assertTrue((ds_yolo / "images" / "train" / "img_050.jpg").is_file())
        self.assertTrue((ds_yolo / "images" / "train" / "img_051.jpg").is_file())
        self.assertEqual(
            (ds_yolo / "labels" / "train" / "img_050.txt").read_text(encoding="utf-8"),
            "1 0.3 0.3 0.1 0.1\n",
        )
        self.assertEqual(
            (ds_yolo / "labels" / "train" / "img_051.txt").read_text(encoding="utf-8"),
            "0 0.5 0.5 0.2 0.2\n",
        )
        # classes.txt must remain completely untouched!
        self.assertTrue((ds_yolo / "classes.txt").is_file())
        self.assertTrue((ds_yolo / "labels" / "classes.txt").is_file())

    def test_dialog_defaults_and_entry_buttons_on_import_and_annotate_pages(self):
        """DatasetRenameDialog defaults to img_ + 1 + 3 digits (img_001), and both ImportPage & AnnotatePage expose entry buttons."""
        sp = self.proj_storage / "images" / "raw_a.jpg"
        self._create_img(sp)
        self.temp_db.add_image(self.project_id, "raw_a.jpg", str(sp), width=100, height=80)

        dlg = DatasetRenameDialog(None, project_id=self.project_id)
        self.assertEqual(dlg.le_prefix.text(), "img_")
        self.assertEqual(dlg.sb_start_index.value(), 1)
        self.assertEqual(dlg.sb_min_digits.value(), 3)
        self.assertEqual(dlg.preview_list.count(), 1)
        self.assertIn("img_001.jpg", dlg.preview_list.item(0).text())

        # Change start index to 18 -> preview immediately updates to img_018.jpg!
        dlg.sb_start_index.setValue(18)
        self.assertIn("img_018.jpg", dlg.preview_list.item(0).text())
        dlg.deleteLater()

        imp_page = ImportPage()
        ann_page = AnnotatePage()
        self.assertTrue(hasattr(imp_page, 'btn_rename_dataset'))
        self.assertTrue(hasattr(ann_page, 'btn_rename_dataset'))
        imp_page.deleteLater()
        ann_page.deleteLater()

        self.assertEqual(app_info.APP_VERSION, "1.3.1")

    def test_export_dataset_dialog_and_default_server_structures(self):
        """Export options live inside DatasetExportDialog (not cluttering right panel),
        and export_dataset supports both default structure and server structure with 8:2 and custom ratios."""
        from PyQt6.QtCore import QSettings
        from gui.pages.dataset_export_dialog import DatasetExportDialog

        QSettings("Better-YOLO", "ExportDataset").clear()

        for idx in range(1, 11):
            fname = f"img_{idx:03d}.png"
            sp = self.proj_storage / "images" / fname
            self._create_img(sp, color=(idx * 10, 50, 100))
            iid = self.temp_db.add_image(self.project_id, fname, str(sp), width=100, height=80)
            self.temp_db.add_annotation(
                iid, self.project_id, 0, "target", "bbox",
                {"x": 10, "y": 10, "width": 30, "height": 30},
            )

        page = AnnotatePage()
        page.current_project_id = self.project_id
        page.classes = [{"id": 0, "name": "target", "color": "#EF4444"}]

        # Right panel no longer embeds yolo_split_widget; options live in DatasetExportDialog!
        self.assertFalse(hasattr(page, 'yolo_split_widget'))
        self.assertTrue(page.export_format.isHidden())

        dlg = DatasetExportDialog(page, mode="dataset", image_count=10)
        cfg = dlg.get_config()
        self.assertEqual(cfg["export_format"], "YOLO格式")
        self.assertEqual(cfg["structure_mode"], "default")
        self.assertEqual(cfg["train_ratio"], 8)
        self.assertEqual(cfg["val_ratio"], 2)
        self.assertIn("预计训练集 8 张，验证集 2 张", dlg.lbl_split_summary.text())
        dlg.rb_server_struct.setChecked(True)
        self.assertEqual(dlg.get_structure_mode(), "server")
        self.assertIn("├── train", dlg.lbl_tree_preview.text())
        dlg.deleteLater()

        out_dir = self.root / "split_exports"
        out_dir.mkdir()
        ds_dir = out_dir / "rename_proj"

        # 1) Default structure (8:2 on 10 images -> 8 train, 2 val)
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(out_dir)), \
                patch.object(QMessageBox, 'information'):
            page.export_dataset(
                export_format="YOLO格式",
                structure_mode="default",
                train_ratio=8,
                val_ratio=2,
                shuffle=False,
            )

        self.assertEqual(len(list((ds_dir / "images" / "train").glob("*.png"))), 8)
        self.assertEqual(len(list((ds_dir / "images" / "val").glob("*.png"))), 2)
        self.assertEqual(len(list((ds_dir / "labels" / "train").glob("*.txt"))), 8)
        self.assertEqual(len(list((ds_dir / "labels" / "val").glob("*.txt"))), 2)
        yaml_default = (ds_dir / "data.yaml").read_text(encoding="utf-8")
        self.assertIn("train: images/train", yaml_default)
        self.assertIn("val: images/val", yaml_default)

        # 2) Server structure with custom 7:3 ratio into the same target dir -> cleanly switches tree!
        with patch.object(QFileDialog, 'getExistingDirectory', return_value=str(out_dir)), \
                patch.object(QMessageBox, 'information'):
            page.export_dataset(
                export_format="YOLO格式",
                structure_mode="server",
                train_ratio=7,
                val_ratio=3,
                shuffle=False,
            )

        self.assertFalse((ds_dir / "images").exists())
        self.assertFalse((ds_dir / "labels").exists())
        self.assertEqual(len(list((ds_dir / "train" / "images").glob("*.png"))), 7)
        self.assertEqual(len(list((ds_dir / "train" / "labels").glob("*.txt"))), 7)
        self.assertEqual(len(list((ds_dir / "val" / "images").glob("*.png"))), 3)
        self.assertEqual(len(list((ds_dir / "val" / "labels").glob("*.txt"))), 3)
        yaml_server = (ds_dir / "data.yaml").read_text(encoding="utf-8")
        self.assertIn("train: train/images", yaml_server)
        self.assertIn("val: val/images", yaml_server)
        self.assertNotIn("path:", yaml_server)

        page.deleteLater()


if __name__ == '__main__':
    unittest.main()
