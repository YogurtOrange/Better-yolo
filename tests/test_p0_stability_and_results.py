"""P0 stability and UX tests: SQLite WAL/FK, ResultPage runs & async export, ImportPage preview."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import sqlite3
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

from PyQt6.QtCore import QCoreApplication, QEvent
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import QApplication

from models.database import Database
from gui.pages.result_page import (
    ResultPage,
    ModelExportThread,
    discover_run_directories,
    load_pixmap_unicode_safe,
)


class DatabaseConcurrencyAndIntegrityTests(unittest.TestCase):
    def test_wal_busy_timeout_and_foreign_keys_enabled(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "test.db")
            database = Database(db_path=db_path)

            with database.get_connection() as conn:
                journal_mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
                foreign_keys = conn.execute("PRAGMA foreign_keys").fetchone()[0]
                busy_timeout = conn.execute("PRAGMA busy_timeout").fetchone()[0]

            self.assertEqual(journal_mode.lower(), "wal")
            self.assertEqual(foreign_keys, 1)
            self.assertGreaterEqual(busy_timeout, 30000)

    def test_foreign_key_cascade_deletes_images_and_annotations(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "cascade.db")
            database = Database(db_path=db_path)
            pid = database.create_project("demo_fk")
            img_id = database.add_image(pid, "a.jpg", str(Path(tmpdir) / "a.jpg"))
            database.add_annotation(img_id, pid, 0, "cat", "bbox", {"x": 1, "y": 2, "w": 3, "h": 4})

            # Deleting project must cascade to images and annotations
            self.assertTrue(database.delete_project(pid))
            self.assertEqual(database.get_project_images(pid), [])
            self.assertEqual(database.get_image_annotations(img_id), [])

            # Invalid project_id must be rejected by foreign key constraint
            with self.assertRaises(sqlite3.IntegrityError):
                database.add_image(999999, "orphan.jpg", str(Path(tmpdir) / "orphan.jpg"))

    def test_multithreaded_concurrent_writes_succeed_without_lock_errors(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            db_path = str(Path(tmpdir) / "concurrent.db")
            database = Database(db_path=db_path)
            pid = database.create_project("concurrent_proj")

            errors = []

            def worker(worker_idx: int):
                try:
                    for i in range(15):
                        img_id = database.add_image(
                            pid,
                            f"w{worker_idx}_{i}.jpg",
                            str(Path(tmpdir) / f"w{worker_idx}_{i}.jpg"),
                        )
                        database.add_annotation(
                            img_id,
                            pid,
                            0,
                            "obj",
                            "bbox",
                            {"x": i, "y": i, "w": 10, "h": 10},
                        )
                except Exception as exc:
                    errors.append(exc)

            threads = [threading.Thread(target=worker, args=(idx,)) for idx in range(8)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            self.assertEqual(errors, [])
            self.assertEqual(len(database.get_project_images(pid)), 8 * 15)


class ResultPageP0Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def pump(self):
        for _ in range(5):
            self.app.processEvents()

    def _create_run(self, runs_root: Path, rel_name: str, mtime: float, map50: float, img_name: str):
        run_dir = runs_root / rel_name
        weights_dir = run_dir / "weights"
        weights_dir.mkdir(parents=True, exist_ok=True)
        best_pt = weights_dir / "best.pt"
        best_pt.write_bytes(b"fake-pt-weights")

        csv_path = run_dir / "results.csv"
        csv_path.write_text(
            "  epoch,   metrics/precision(B),   metrics/recall(B),   metrics/mAP50(B),   metrics/mAP50-95(B)\n"
            f"  1,       0.8123,                 0.7456,              {map50:.4f},         0.5123\n",
            encoding="utf-8",
        )

        img_path = run_dir / img_name
        pix = QPixmap(64, 48)
        pix.fill()
        self.assertTrue(pix.save(str(img_path)))

        for target in (best_pt, csv_path, img_path, run_dir):
            os.utime(target, (mtime, mtime))
        return run_dir

    def test_run_selector_sorts_by_mtime_desc_and_switches_metrics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "runs"
            now = time.time()
            # Intentionally name older run lexicographically greater than newer run
            self._create_run(runs_root, "detect/exp_7_old_z", now - 300, 0.5500, "old.png")
            self._create_run(runs_root, "detect/exp_7_new_a", now - 10, 0.9125, "结果图_中文.png")

            runs, fallback = discover_run_directories(str(runs_root), project_id=7)
            self.assertFalse(fallback)
            self.assertEqual(len(runs), 2)
            self.assertTrue(runs[0]["rel_path"].endswith("exp_7_new_a"))

            page = ResultPage()
            try:
                page.runs_dir = str(runs_root)
                page.current_project_id = 7
                page.scan_runs_directory(show_dialogs=False)
                self.pump()

                self.assertEqual(page.run_selector.count(), 2)
                self.assertEqual(page.metric_labels["mAP50"].text(), "0.9125")
                self.assertFalse(page.image_label.pixmap().isNull())

                # Switch to older run and verify metrics update
                page.run_selector.setCurrentIndex(1)
                self.pump()
                self.assertEqual(page.metric_labels["mAP50"].text(), "0.5500")
            finally:
                page.close()
                page.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)

    def test_unicode_safe_image_loader_handles_chinese_paths(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            chinese_dir = Path(tmpdir) / "中文目录"
            chinese_dir.mkdir()
            img_path = chinese_dir / "训练混淆矩阵.png"
            pix = QPixmap(120, 80)
            pix.fill()
            self.assertTrue(pix.save(str(img_path)))

            loaded = load_pixmap_unicode_safe(str(img_path))
            self.assertFalse(loaded.isNull())
            self.assertEqual(loaded.width(), 120)
            self.assertEqual(loaded.height(), 80)

    def test_background_export_thread_exports_pt_and_folder(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runs_root = Path(tmpdir) / "runs"
            run_dir = self._create_run(runs_root, "exp_3", time.time(), 0.8800, "curve.png")
            out_pt = str(Path(tmpdir) / "exported_best.pt")
            out_folder_parent = str(Path(tmpdir) / "exported_runs")
            os.makedirs(out_folder_parent, exist_ok=True)

            results = []
            worker_pt = ModelExportThread("pt", str(run_dir), out_pt)
            worker_pt.export_finished.connect(lambda ok, msg: results.append((ok, msg)))
            worker_pt.start()
            self.assertTrue(worker_pt.wait(5000))
            self.pump()
            self.assertEqual(len(results), 1)
            self.assertTrue(results[0][0], results[0][1])
            self.assertTrue(os.path.exists(out_pt))

            folder_results = []
            worker_folder = ModelExportThread("folder", str(run_dir), out_folder_parent, run_name="exp_3")
            worker_folder.export_finished.connect(lambda ok, msg: folder_results.append((ok, msg)))
            worker_folder.start()
            self.assertTrue(worker_folder.wait(5000))
            self.pump()
            self.assertEqual(len(folder_results), 1)
            self.assertTrue(folder_results[0][0], folder_results[0][1])
            self.assertTrue((Path(out_folder_parent) / "exp_3" / "weights" / "best.pt").exists())


class ImportImagePreviewDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_import_image_preview_dialog_displays_and_navigates(self):
        from gui.pages.import_page import ImportImagePreviewDialog

        with tempfile.TemporaryDirectory() as tmpdir:
            p1 = Path(tmpdir) / "图一.png"
            p2 = Path(tmpdir) / "图二.png"
            for p, size in ((p1, (100, 60)), (p2, (160, 90))):
                pix = QPixmap(*size)
                pix.fill()
                self.assertTrue(pix.save(str(p)))

            images = [
                {"id": 1, "filename": "图一.png", "storage_path": str(p1), "width": 100, "height": 60, "size": 1024, "status": "pending"},
                {"id": 2, "filename": "图二.png", "storage_path": str(p2), "width": 160, "height": 90, "size": 2048, "status": "annotated"},
            ]
            dialog = ImportImagePreviewDialog(images, start_index=0)
            try:
                self.assertIn("图一.png", dialog.meta_label.text())
                self.assertFalse(dialog.preview.pixmap().isNull())
                dialog.show_next()
                self.assertIn("图二.png", dialog.meta_label.text())
                self.assertIn("已标注", dialog.meta_label.text())
                dialog.show_prev()
                self.assertIn("图一.png", dialog.meta_label.text())
            finally:
                dialog.close()
                dialog.deleteLater()
                QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)


if __name__ == "__main__":
    unittest.main()
