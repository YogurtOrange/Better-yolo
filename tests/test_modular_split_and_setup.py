"""Tests for modular annotate_page split and one-click environment setup script."""

import os
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path
import sys
import tempfile
import unittest

from scripts.setup_env import (
    build_install_commands,
    recommend_cuda_tag,
    resolve_mirror_url,
    run_environment_checks,
)


class ModularAnnotateSplitTests(unittest.TestCase):
    def test_split_modules_and_backward_compatible_reexports(self):
        from core.sam_manager import (
            SAM3_DOWNLOAD_URL,
            SAMInferenceWorker,
            SAMMemoryPredictorManager,
            SAMModelManager,
        )
        from gui.canvas import AnnotationCanvas
        from gui.pages.annotate_workers import (
            NEGATIVE_SAMPLE_CLASS_ID,
            AnnotateImageLoadWorker,
            RandomSampleDeleteWorker,
        )
        from gui.widgets.sam_memory_dialog import SAMMemoryObjectsDialog
        import gui.pages.annotate_page as annotate_page

        self.assertIs(annotate_page.AnnotationCanvas, AnnotationCanvas)
        self.assertIs(annotate_page.SAMModelManager, SAMModelManager)
        self.assertIs(annotate_page.SAMMemoryPredictorManager, SAMMemoryPredictorManager)
        self.assertIs(annotate_page.SAMInferenceWorker, SAMInferenceWorker)
        self.assertIs(annotate_page.SAMMemoryObjectsDialog, SAMMemoryObjectsDialog)
        self.assertIs(annotate_page.AnnotateImageLoadWorker, AnnotateImageLoadWorker)
        self.assertIs(annotate_page.RandomSampleDeleteWorker, RandomSampleDeleteWorker)
        self.assertEqual(annotate_page.SAM3_DOWNLOAD_URL, SAM3_DOWNLOAD_URL)
        self.assertEqual(annotate_page.NEGATIVE_SAMPLE_CLASS_ID, NEGATIVE_SAMPLE_CLASS_ID)

    def test_annotate_page_file_size_significantly_reduced(self):
        root = Path(__file__).resolve().parent.parent
        annotate_lines = len((root / "gui/pages/annotate_page.py").read_text(encoding="utf-8").splitlines())
        canvas_lines = len((root / "gui/canvas/annotation_canvas.py").read_text(encoding="utf-8").splitlines())
        self.assertLess(annotate_lines, 4500)
        self.assertGreater(canvas_lines, 1800)


class SetupEnvScriptTests(unittest.TestCase):
    def test_recommend_cuda_tag_mapping(self):
        self.assertEqual(recommend_cuda_tag("13.0"), "cu128")
        self.assertEqual(recommend_cuda_tag("12.8"), "cu128")
        self.assertEqual(recommend_cuda_tag("12.6"), "cu126")
        self.assertEqual(recommend_cuda_tag("12.4"), "cu124")
        self.assertEqual(recommend_cuda_tag("12.1"), "cu121")
        self.assertEqual(recommend_cuda_tag("11.8"), "cu118")
        self.assertEqual(recommend_cuda_tag(None), "cpu")

    def test_build_install_commands_gpu_and_cpu_variants(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            req = Path(tmpdir) / "requirements.txt"
            req.write_text(
                "# sample\nPyQt6==6.11.0\ntorch==2.10.0\ntorchvision==0.25.0\nonnxruntime-gpu==1.22.0\n",
                encoding="utf-8",
            )

            gpu_cmds = build_install_commands(
                python_exe=sys.executable,
                device="auto",
                mirror="tuna",
                requirements_path=req,
                gpu_info={"has_gpu": True, "gpu_name": "RTX 4090", "cuda_version": "12.8"},
            )
            self.assertEqual(len(gpu_cmds), 2)
            self.assertIn("https://download.pytorch.org/whl/cu128", gpu_cmds[0])
            self.assertIn("onnxruntime-gpu==1.22.0", gpu_cmds[1])
            self.assertIn(resolve_mirror_url("tuna"), gpu_cmds[1])

            cpu_cmds = build_install_commands(
                python_exe=sys.executable,
                device="cpu",
                requirements_path=req,
                gpu_info={"has_gpu": True, "gpu_name": "RTX 4090", "cuda_version": "12.8"},
            )
            self.assertEqual(len(cpu_cmds), 2)
            self.assertIn("https://download.pytorch.org/whl/cpu", cpu_cmds[0])
            self.assertIn("onnxruntime==1.23.2", cpu_cmds[1])

    def test_run_environment_checks_returns_structured_items(self):
        items = run_environment_checks()
        names = {item.name for item in items}
        self.assertIn("Python 版本", names)
        self.assertIn("PyQt6 (桌面界面)", names)


if __name__ == "__main__":
    unittest.main()
