# -*- coding: utf-8 -*-
"""标注页面后台工作线程：缩略图加载与随机抽样删除。"""

import os
import random
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from PyQt6.QtGui import QColor, QImage, QPixmap

from gui.styles import COLORS
from models.database import db as default_db

NEGATIVE_SAMPLE_CLASS_ID = -1


class AnnotateImageLoadWorker(QThread):
    """标注页面图片加载工作线程"""

    image_loaded = pyqtSignal(int, object)  # 索引, 缩略图
    finished_loading = pyqtSignal()

    def __init__(self, images: List[Dict]):
        super().__init__()
        self.images = images
        self._is_running = True

    def run(self):
        """在后台线程中加载图片（兼容中文路径）"""
        for i, image_data in enumerate(self.images):
            if not self._is_running:
                break

            storage_path = image_data.get("storage_path", "")
            pixmap = None

            if storage_path and os.path.exists(storage_path):
                try:
                    raw = np.fromfile(storage_path, dtype=np.uint8)
                    img = cv2.imdecode(raw, cv2.IMREAD_COLOR) if raw.size > 0 else None
                    if img is not None:
                        img = cv2.resize(img, (80, 80))
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        h, w, ch = img.shape
                        bytes_per_line = ch * w
                        qt_image = QImage(
                            img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888
                        ).copy()
                        pixmap = QPixmap.fromImage(qt_image)
                except Exception:
                    pass

            if pixmap is None or pixmap.isNull():
                pixmap = QPixmap(80, 80)
                pixmap.fill(QColor(COLORS["sidebar"]))

            self.image_loaded.emit(i, pixmap)

            if i % 10 == 0:
                self.msleep(1)

        self.finished_loading.emit()

    def stop(self):
        self._is_running = False


class RandomSampleDeleteWorker(QThread):
    """随机删除样本工作线程"""

    progress_updated = pyqtSignal(int, int, str)  # 当前进度, 总数, 文件名
    delete_finished = pyqtSignal(object)  # 删除结果
    delete_failed = pyqtSignal(str)

    def __init__(
        self,
        project_id: int,
        target_class_id: int,
        delete_count: int,
        current_image_id: Optional[int],
        db_instance=None,
    ):
        super().__init__()
        self.project_id = project_id
        self.target_class_id = target_class_id
        self.delete_count = delete_count
        self.current_image_id = current_image_id
        self._db = db_instance if db_instance is not None else default_db
        self._is_running = True

    def run(self):
        try:
            if self.target_class_id == NEGATIVE_SAMPLE_CLASS_ID:
                candidate_images = self._db.get_negative_sample_images(
                    self.project_id, annotated_only=True
                )
            else:
                candidate_images = self._db.get_project_images_by_class(
                    self.project_id, self.target_class_id
                )

            available_count = len(candidate_images)
            total_count = min(self.delete_count, available_count)
            if total_count <= 0:
                self.delete_finished.emit(
                    {
                        "deleted_count": 0,
                        "failed_files": [],
                        "current_image_deleted": False,
                        "canceled": False,
                        "available_count": available_count,
                        "total_count": 0,
                    }
                )
                return

            target_images = random.sample(candidate_images, total_count)
            deleted_count = 0
            failed_files = []
            current_image_deleted = False

            for index, image in enumerate(target_images, start=1):
                if not self._is_running:
                    self.delete_finished.emit(
                        {
                            "deleted_count": deleted_count,
                            "failed_files": failed_files,
                            "current_image_deleted": current_image_deleted,
                            "canceled": True,
                            "available_count": available_count,
                            "total_count": total_count,
                        }
                    )
                    return

                filename = image.get("filename", str(image.get("id")))
                self.progress_updated.emit(index - 1, total_count, filename)

                if image["id"] == self.current_image_id:
                    current_image_deleted = True

                if self._db.delete_image(image["id"]):
                    deleted_count += 1
                else:
                    failed_files.append(filename)

                self.progress_updated.emit(index, total_count, filename)

            self.delete_finished.emit(
                {
                    "deleted_count": deleted_count,
                    "failed_files": failed_files,
                    "current_image_deleted": current_image_deleted,
                    "canceled": False,
                    "available_count": available_count,
                    "total_count": total_count,
                }
            )
        except Exception as e:
            self.delete_failed.emit(str(e))

    def stop(self):
        self._is_running = False
