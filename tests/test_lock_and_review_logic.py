"""Unit tests for Plan A image locking (🔒) and AI co-labeling refinement (✏️) logic."""

import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image
from PyQt6.QtCore import QCoreApplication, QEvent, QPoint, QPointF, Qt
from PyQt6.QtGui import QKeyEvent, QMouseEvent
from PyQt6.QtWidgets import QApplication, QMessageBox

import config.app_info as app_info
import core.qwen_annotator as qwen_mod
import gui.pages.ai_label_page as ai_label_mod
import gui.pages.annotate_page as annotate_mod
from core.qwen_annotator import compute_image_ai_modified, remove_image_from_breakpoint
from gui.pages.ai_label_page import AiLabelPage, ModifiedImagesDialog
from gui.pages.annotate_page import AnnotatePage
from gui.pages.settings_page import SettingsPage
from models.database import Database


class LockAndReviewLogicTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tmpdir.name)
        self.temp_db = Database(str(self.root / "review_test.db"))

        self._patches = [
            patch.object(annotate_mod, 'db', self.temp_db),
            patch.object(ai_label_mod, 'db', self.temp_db),
            patch.object(qwen_mod, 'db', self.temp_db),
        ]
        for p in self._patches:
            p.start()

        self.classes = [
            {"id": 0, "name": "tiger", "color": "#EF4444"},
            {"id": 1, "name": "monkey", "color": "#10B981"},
        ]
        self.project_id = self.temp_db.create_project(
            "review_proj", project_type="detect", classes=self.classes
        )
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

    def _add_dummy_image(self, filename: str) -> int:
        img_path = self.proj_storage / filename
        Image.new("RGB", (400, 300), color=(240, 240, 240)).save(img_path, format="PNG")
        return self.temp_db.add_image(
            self.project_id, filename, str(img_path), width=400, height=300
        )

    def _wait_project_load(self, page: AnnotatePage):
        if hasattr(page, 'load_thread'):
            page.load_thread.wait(3000)
        for _ in range(10):
            QCoreApplication.processEvents()
        if page.load_worker and page.load_worker.isRunning():
            page.load_worker.wait(3000)

    def test_pure_manual_labeling_locks_without_modified_flag(self):
        """Pure manual labeling (no ai_original) sets locked=1, modified=0 and never pollutes ModifiedImagesDialog."""
        img_id = self._add_dummy_image("manual_01.png")

        page = AnnotatePage()
        page.set_project(self.project_id)
        self._wait_project_load(page)
        page.load_image(img_id)

        page.on_annotation_created({
            'type': 'bbox',
            'class_id': 0,
            'data': {'x': 20.0, 'y': 30.0, 'width': 100.0, 'height': 80.0},
        })

        img_row = self.temp_db.get_image(img_id)
        self.assertEqual(img_row['status'], 'annotated')
        self.assertEqual(img_row['locked'], 1)
        self.assertEqual(img_row['modified'], 0)
        self.assertEqual(page.image_list.item(0).text(), "✓ 🔒 manual_01.png")

        # ModifiedImagesDialog must remain empty for pure manual images
        dlg = ModifiedImagesDialog(None, self.project_id)
        self.assertEqual(dlg.image_list.count(), 0)
        dlg.deleteLater()

        # Deleting the only manual box resets pure manual image back to pending & unlocked
        ann_id = page.annotations[0]['id']
        page.on_annotation_deleted(ann_id)
        img_row_after = self.temp_db.get_image(img_id)
        self.assertEqual(img_row_after['status'], 'pending')
        self.assertEqual(img_row_after['locked'], 0)
        self.assertEqual(img_row_after['modified'], 0)
        self.assertEqual(page.image_list.item(0).text(), "○ manual_01.png")
        page.deleteLater()

    def test_ai_image_confirm_edit_undo_and_unlock_loop(self):
        """AI image workflow:
        1) Initial AI prediction -> ✓ (status=annotated, locked=0, modified=0)
        2) Single click in move mode without dragging -> does NOT lock or mark modified
        3) Selecting/deselecting box -> btn_lock stays enabled with whole-image text
        4) Pressing L / toggle lock without edits -> ✓ 🔒 (locked=1, modified=0)
        5) Moving box / changing class -> ✓ ✏️🔒 (locked=1, modified=1)
        6) Ctrl+Z undo back to AI snapshot -> ✓ 🔒 (modified automatically reverts to 0!)
        7) Unlocking -> locked=0, modified=0, clears per-box lock, removes filename from ai_progress.json
        """
        img_id = self._add_dummy_image("ai_01.png")
        ann_id = self.temp_db.add_annotation(
            img_id, self.project_id, 0, "tiger", "bbox",
            {"x": 50.0, "y": 60.0, "width": 120.0, "height": 90.0},
            {"source": "ai", "confidence": 0.95},
        )
        self.temp_db.set_image_ai_original(img_id, [
            {"class_name": "tiger", "bbox": [50.0, 60.0, 170.0, 150.0]}
        ])
        progress_path = self.proj_storage / "ai_progress.json"
        progress_path.write_text(
            json.dumps({"completed": ["ai_01.png", "other.png"]}),
            encoding="utf-8",
        )

        page = AnnotatePage()
        page.resize(1000, 700)
        page.set_project(self.project_id)
        self._wait_project_load(page)
        page.load_image(img_id)

        # 1) Initial AI prediction is '待复核' (✓, unlocked, unmodified)
        self.assertEqual(page.image_list.item(0).text(), "✓ ai_01.png")
        self.assertTrue(page.btn_lock.isEnabled())
        self.assertEqual(page.btn_lock.text(), "🔒 锁定本图")

        # 2) Switch to move mode and click inside the box WITHOUT moving -> must NOT lock!
        page.set_tool('move')
        box_widget_pt = page.canvas.image_to_widget(80, 90)
        press_ev = QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(box_widget_pt),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        release_ev = QMouseEvent(
            QEvent.Type.MouseButtonRelease,
            QPointF(box_widget_pt),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
        )
        page.canvas.mousePressEvent(press_ev)
        page.canvas.mouseReleaseEvent(release_ev)

        self.assertEqual(page.canvas.selected_annotation_id, ann_id)
        self.assertEqual(self.temp_db.get_image(img_id)['locked'], 0)
        self.assertEqual(self.temp_db.get_image(img_id)['modified'], 0)
        # 3) Selecting or clearing attribute panel never hijacks or disables btn_lock
        self.assertTrue(page.btn_lock.isEnabled())
        self.assertEqual(page.btn_lock.text(), "🔒 锁定本图")
        page.clear_attribute_panel()
        self.assertTrue(page.btn_lock.isEnabled())
        self.assertEqual(page.btn_lock.text(), "🔒 锁定本图")

        # 4) Press 'L' key to confirm AI prediction without edits -> ✓ 🔒 (locked=1, modified=0)
        key_l = QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_L, Qt.KeyboardModifier.NoModifier, "l")
        page.keyPressEvent(key_l)
        img_row = self.temp_db.get_image(img_id)
        self.assertEqual(img_row['locked'], 1)
        self.assertEqual(img_row['modified'], 0)
        self.assertEqual(page.btn_lock.text(), "🔓 解锁本图")
        self.assertEqual(page.image_list.item(0).text(), "✓ 🔒 ai_01.png")

        # 5) Substantively move the box -> ✓ ✏️🔒 (locked=1, modified=1)
        page.on_annotation_modified(
            ann_id,
            {"x": 100.0, "y": 95.0, "width": 120.0, "height": 90.0},
        )
        img_row = self.temp_db.get_image(img_id)
        self.assertEqual(img_row['locked'], 1)
        self.assertEqual(img_row['modified'], 1)
        self.assertEqual(page.image_list.item(0).text(), "✓ ✏️🔒 ai_01.png")

        # 6) Undo (Ctrl+Z) restores original box coordinates AND clears modified back to 0!
        page.undo()
        img_row = self.temp_db.get_image(img_id)
        self.assertEqual(img_row['locked'], 1)
        self.assertEqual(img_row['modified'], 0)
        self.assertEqual(page.image_list.item(0).text(), "✓ 🔒 ai_01.png")
        restored_ann = self.temp_db.get_annotation(ann_id)
        self.assertEqual(restored_ann['data']['x'], 50.0)

        # 6b) Changing class via apply_annotation_changes marks modified=1, and undo reverts it!
        page.canvas.selected_annotation_id = ann_id
        page.class_list.setCurrentRow(1)  # monkey
        with patch.object(QMessageBox, 'information'):
            page.apply_annotation_changes()
        self.assertEqual(self.temp_db.get_image(img_id)['modified'], 1)
        self.assertEqual(self.temp_db.get_annotation(ann_id)['class_name'], 'monkey')

        page.undo()
        self.assertEqual(self.temp_db.get_image(img_id)['modified'], 0)
        self.assertEqual(self.temp_db.get_annotation(ann_id)['class_name'], 'tiger')

        # 7) Unlock image -> clears image lock, clears per-box lock, and removes ai_01.png from breakpoint!
        # First modify once so per-box lock & modified=1 are active
        page.on_annotation_modified(
            ann_id,
            {"x": 110.0, "y": 100.0, "width": 120.0, "height": 90.0},
        )
        self.assertTrue(self.temp_db.get_annotation(ann_id)['attributes'].get('locked'))
        page.on_toggle_lock()  # unlock

        unlocked_img = self.temp_db.get_image(img_id)
        self.assertEqual(unlocked_img['locked'], 0)
        self.assertEqual(unlocked_img['modified'], 0)
        self.assertNotIn('locked', self.temp_db.get_annotation(ann_id)['attributes'])
        bp_after = json.loads(progress_path.read_text(encoding="utf-8"))
        self.assertEqual(bp_after["completed"], ["other.png"])
        page.deleteLater()

    def test_negative_sample_and_filters_workflow(self):
        """Clearing AI false positives (via delete or mark negative sample) locks image with modified=1,
        and image_filter accurately filters 待复核 / 已锁定 / 已复标修正."""
        img_unlabeled = self._add_dummy_image("01_unlabeled.png")
        img_pending_review = self._add_dummy_image("02_ai_pending.png")
        img_confirmed = self._add_dummy_image("03_confirmed.png")
        img_false_pos = self._add_dummy_image("04_false_pos.png")

        # 02: AI labeled, unreviewed
        self.temp_db.add_annotation(
            img_pending_review, self.project_id, 0, "tiger", "bbox",
            {"x": 10, "y": 10, "width": 50, "height": 50}, {"source": "ai"}
        )
        self.temp_db.set_image_ai_original(
            img_pending_review, [{"class_name": "tiger", "bbox": [10, 10, 60, 60]}]
        )

        # 03: AI labeled, confirmed locked without changes
        self.temp_db.add_annotation(
            img_confirmed, self.project_id, 0, "tiger", "bbox",
            {"x": 20, "y": 20, "width": 60, "height": 60}, {"source": "ai"}
        )
        self.temp_db.set_image_ai_original(
            img_confirmed, [{"class_name": "tiger", "bbox": [20, 20, 80, 80]}]
        )
        self.temp_db.set_image_locked(img_confirmed, True)

        # 04: AI falsely detected a box on background -> user deletes the false-positive box
        fp_ann_id = self.temp_db.add_annotation(
            img_false_pos, self.project_id, 0, "tiger", "bbox",
            {"x": 30, "y": 30, "width": 70, "height": 70}, {"source": "ai"}
        )
        self.temp_db.set_image_ai_original(
            img_false_pos, [{"class_name": "tiger", "bbox": [30, 30, 100, 100]}]
        )

        page = AnnotatePage()
        page.set_project(self.project_id)
        self._wait_project_load(page)
        page.load_image(img_false_pos)
        page.on_annotation_deleted(fp_ann_id)

        # Deleting the last false-positive AI box keeps status='annotated' (negative sample) + locked=1 + modified=1!
        fp_row = self.temp_db.get_image(img_false_pos)
        self.assertEqual(fp_row['status'], 'annotated')
        self.assertEqual(fp_row['locked'], 1)
        self.assertEqual(fp_row['modified'], 1)
        self.assertTrue(compute_image_ai_modified(img_false_pos))

        def visible_names(filter_name: str):
            page.image_filter.setCurrentText(filter_name)
            return [
                page.images[i]['filename']
                for i in range(page.image_list.count())
                if not page.image_list.item(i).isHidden()
            ]

        self.assertEqual(visible_names("未标注"), ["01_unlabeled.png"])
        self.assertEqual(visible_names("待复核"), ["02_ai_pending.png"])
        self.assertEqual(visible_names("已锁定"), ["03_confirmed.png", "04_false_pos.png"])
        self.assertEqual(visible_names("已复标修正"), ["04_false_pos.png"])

        # SettingsPage exposes lock_image_shortcut defaulting to L
        settings_page = SettingsPage()
        self.assertTrue(hasattr(settings_page, 'lock_image_shortcut'))
        settings_page.deleteLater()

        # AiLabelPage stats show both locked and modified counts
        ai_page = AiLabelPage()
        ai_page.set_project(self.project_id)
        stats_txt = ai_page.project_stats_label.text()
        self.assertIn("已锁定确认 2 张", stats_txt)
        self.assertIn("复标纠错 1 张", stats_txt)
        ai_page.deleteLater()
        page.deleteLater()

    def test_iterative_reannotation_resets_stale_breakpoint_for_unlocked_images(self):
        """Regression test for Round 1 -> Human corrects & locks subset -> Prompt updated / Overwrite -> Round 2:
        AI must skip locked images and re-annotate all unlocked images instead of reporting '待处理数量: 0'."""
        img_locked = self._add_dummy_image("locked_fixed.png")
        img_unlocked_1 = self._add_dummy_image("unlocked_01.png")
        img_unlocked_2 = self._add_dummy_image("unlocked_02.png")

        # Simulate Round 1 finished all 3 images (legacy ai_progress.json without task_signature)
        for iid in (img_locked, img_unlocked_1, img_unlocked_2):
            self.temp_db.add_annotation(
                iid, self.project_id, 0, "tiger", "bbox",
                {"x": 10.0, "y": 10.0, "width": 80.0, "height": 80.0},
                {"source": "ai", "confidence": 0.9},
            )
            self.temp_db.set_image_ai_original(
                iid, [{"class_name": "tiger", "bbox": [10.0, 10.0, 90.0, 90.0]}]
            )
        # Human corrected & locked img_locked
        self.temp_db.update_annotation(
            self.temp_db.get_image_annotations(img_locked)[0]['id'],
            data={"x": 15.0, "y": 15.0, "width": 70.0, "height": 70.0},
        )
        self.temp_db.set_image_locked(img_locked, True)
        self.temp_db.set_image_modified(img_locked, True)

        progress_file = self.proj_storage / "ai_progress.json"
        progress_file.write_text(
            json.dumps({"completed": ["locked_fixed.png", "unlocked_01.png", "unlocked_02.png"]}),
            encoding="utf-8",
        )

        # Case A: Legacy breakpoint + overwrite=True -> automatically re-annotates the 2 unlocked images, skips locked
        logs = []
        fake_info = {"width": 400, "height": 300}

        def fake_detect(_client, _model, image_path, *_args, **_kwargs):
            if image_path.name == "unlocked_01.png":
                return ([{"class_name": "monkey", "bbox": [40.0, 50.0, 140.0, 150.0], "confidence": 0.98}], fake_info)
            # Simulate Round 2 filtering out false positive -> 0 boxes on unlocked_02.png
            return ([], fake_info)

        with patch.object(qwen_mod, "make_client", return_value=object()), \
                patch.object(qwen_mod, "detect_single", side_effect=fake_detect):
            qwen_mod.run_ai_annotation(
                {
                    "project_id": self.project_id,
                    "images": self.temp_db.get_project_images(self.project_id),
                    "project_dir": str(self.proj_storage),
                    "prompt": "tiger：老虎图案\nmonkey：猴子图案",
                    "negative": "靶板、网格线",
                    "api_key": "dummy-key",
                    "zoom_refine": False,
                    "overwrite": True,
                },
                log=logs.append,
            )

        # Locked image keeps its human-corrected tiger box untouched
        locked_anns = self.temp_db.get_image_annotations(img_locked)
        self.assertEqual(len(locked_anns), 1)
        self.assertEqual(locked_anns[0]['class_name'], 'tiger')
        self.assertEqual(locked_anns[0]['data']['x'], 15.0)

        # Unlocked image 1 was updated to monkey box
        u1_anns = self.temp_db.get_image_annotations(img_unlocked_1)
        self.assertEqual(len(u1_anns), 1)
        self.assertEqual(u1_anns[0]['class_name'], 'monkey')

        # Unlocked image 2 had its false-positive box cleared to 0 boxes and stays status='annotated'
        u2_anns = self.temp_db.get_image_annotations(img_unlocked_2)
        self.assertEqual(len(u2_anns), 0)
        self.assertEqual(self.temp_db.get_image(img_unlocked_2)['status'], 'annotated')

        # Case B: Round 3 with overwrite=False, when prompt/negative changes -> task_signature mismatch auto-reruns unlocked!
        calls = []

        def fake_detect_r3(_client, _model, image_path, *_args, **_kwargs):
            calls.append(image_path.name)
            return ([{"class_name": "tiger", "bbox": [25.0, 25.0, 95.0, 95.0], "confidence": 0.99}], fake_info)

        with patch.object(qwen_mod, "make_client", return_value=object()), \
                patch.object(qwen_mod, "detect_single", side_effect=fake_detect_r3):
            qwen_mod.run_ai_annotation(
                {
                    "project_id": self.project_id,
                    "images": self.temp_db.get_project_images(self.project_id),
                    "project_dir": str(self.proj_storage),
                    "prompt": "tiger：新特征描述的老虎图案\nmonkey：猴子图案",
                    "negative": "靶板、网格线、鞋尖",
                    "api_key": "dummy-key",
                    "zoom_refine": False,
                    "overwrite": False,
                },
                log=logs.append,
            )

        self.assertEqual(sorted(calls), ["unlocked_01.png", "unlocked_02.png"])

        # Case C: Same prompt & overwrite=False after partial trial run (limit=1 -> limit=0) still resumes from breakpoint!
        progress_file.unlink()
        calls_trial = []
        with patch.object(qwen_mod, "make_client", return_value=object()), \
                patch.object(qwen_mod, "detect_single", side_effect=lambda _c, _m, p, *a, **kw: (calls_trial.append(p.name) or [], fake_info)):
            common_params = {
                "project_id": self.project_id,
                "images": self.temp_db.get_project_images(self.project_id),
                "project_dir": str(self.proj_storage),
                "prompt": "tiger：稳定提示词\nmonkey：稳定提示词",
                "negative": "靶板",
                "api_key": "dummy-key",
                "zoom_refine": False,
                "overwrite": False,
            }
            qwen_mod.run_ai_annotation({**common_params, "limit": 1}, log=logs.append)
            self.assertEqual(calls_trial, ["unlocked_01.png"])
            calls_trial.clear()
            qwen_mod.run_ai_annotation({**common_params, "limit": 0}, log=logs.append)
            self.assertEqual(calls_trial, ["unlocked_02.png"])

        # Case D: AiLabelPage.on_prompt_generated automatically clears old breakpoint, and btn_clear_log clears log_edit
        ai_page = AiLabelPage()
        ai_page.set_project(self.project_id)
        self.assertTrue(progress_file.exists())
        ai_page.on_prompt_generated("tiger：更新后的老虎描述\nmonkey：更新后的猴子描述", "靶板、阴影")
        self.assertFalse(progress_file.exists())
        self.assertNotEqual(ai_page.log_edit.toPlainText().strip(), "")
        ai_page.btn_clear_log.click()
        self.assertEqual(ai_page.log_edit.toPlainText().strip(), "")
        ai_page.deleteLater()

    def test_version_synced_with_changelog(self):
        import re
        changelog = (Path(__file__).parent.parent / "CHANGELOG.md").read_text(encoding="utf-8")
        m = re.search(r"^## \[([^\]]+)\] - (\d{4}-\d{2}-\d{2})", changelog, re.M)
        self.assertIsNotNone(m)
        self.assertEqual(app_info.APP_VERSION, m.group(1))
        self.assertEqual(app_info.LAST_UPDATED, m.group(2))


if __name__ == '__main__':
    unittest.main()
