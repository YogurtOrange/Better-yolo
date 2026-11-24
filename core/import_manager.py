# -*- coding: utf-8 -*-
"""
导入管理器
处理图像导入、视频抽帧、去重与图像管理功能
"""

import shutil
import hashlib
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Callable
from datetime import datetime
import cv2
from PIL import Image

from models.database import db


class ImportManager:
    """导入管理器类"""

    # 支持的图像格式
    SUPPORTED_IMAGE_FORMATS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif'}

    # 支持的视频格式
    SUPPORTED_VIDEO_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

    def __init__(self, project_id: int):
        """
        初始化导入管理器

        Args:
            project_id: 项目ID
        """
        self.project_id = project_id
        self.project = db.get_project(project_id)
        if not self.project:
            raise ValueError(f"项目 {project_id} 不存在")

        # 确保项目存储目录存在
        self.storage_path = Path(self.project['storage_path'])
        self.storage_path.mkdir(parents=True, exist_ok=True)

        # 创建images子目录
        self.images_path = self.storage_path / "images"
        self.images_path.mkdir(exist_ok=True)

        # 项目内已有图像哈希缓存: {md5_hex: image_record}
        self._hash_to_image: Optional[Dict[str, Dict]] = None

    def _ensure_hash_index(self):
        """惰性加载项目内已有图像哈希索引，并为旧数据自动回填缺失的 file_hash"""
        if self._hash_to_image is not None:
            return

        self._hash_to_image = {}
        existing_images = db.get_project_images(self.project_id)
        for img in existing_images:
            file_hash = img.get('file_hash')
            storage_path = img.get('storage_path')
            if not file_hash and storage_path and Path(storage_path).is_file():
                try:
                    file_hash = self._calculate_file_hash(storage_path)
                    db.update_image_hash(img['id'], file_hash)
                    img['file_hash'] = file_hash
                except Exception:
                    file_hash = None
            if file_hash:
                self._hash_to_image[file_hash] = img

    def import_folder(self, folder_path: str,
                     progress_callback: Callable[[int, str], None] = None) -> Tuple[int, int]:
        """
        导入文件夹中的所有图像

        Args:
            folder_path: 文件夹路径
            progress_callback: 进度回调函数，参数为(进度百分比, 状态信息)

        Returns:
            (成功导入数量, 跳过数量)
        """
        folder_path = Path(folder_path)
        if not folder_path.exists() or not folder_path.is_dir():
            raise ValueError(f"无效的文件夹路径: {folder_path}")

        # 获取所有图像文件
        image_files = [
            p for p in folder_path.rglob("*")
            if p.is_file() and p.suffix.lower() in self.SUPPORTED_IMAGE_FORMATS
        ]

        # 去重并排序
        image_files = sorted(set(image_files))

        if not image_files:
            return 0, 0

        self._ensure_hash_index()

        total = len(image_files)
        imported = 0
        skipped = 0

        for i, image_file in enumerate(image_files):
            try:
                progress = int((i / total) * 100)
                if progress_callback:
                    progress_callback(progress, f"正在导入: {image_file.name}")

                if self.import_single_image(str(image_file)):
                    imported += 1
                else:
                    skipped += 1

            except Exception as e:
                print(f"导入图像失败 {image_file}: {e}")
                skipped += 1

        if progress_callback:
            progress_callback(100, f"导入完成: 成功 {imported}, 跳过 {skipped}")

        return imported, skipped

    def import_images(self, file_paths: List[str],
                     progress_callback: Callable[[int, str], None] = None) -> Tuple[int, int]:
        """
        导入多张图像

        Args:
            file_paths: 图像文件路径列表
            progress_callback: 进度回调函数

        Returns:
            (成功导入数量, 跳过数量)
        """
        if not file_paths:
            return 0, 0

        self._ensure_hash_index()

        total = len(file_paths)
        imported = 0
        skipped = 0

        for i, file_path in enumerate(file_paths):
            try:
                progress = int((i / total) * 100)
                if progress_callback:
                    progress_callback(progress, f"正在导入: {Path(file_path).name}")

                if self.import_single_image(file_path):
                    imported += 1
                else:
                    skipped += 1

            except Exception as e:
                print(f"导入图像失败 {file_path}: {e}")
                skipped += 1

        if progress_callback:
            progress_callback(100, f"导入完成: 成功 {imported}, 跳过 {skipped}")

        return imported, skipped

    def _import_single_image_internal(self, file_path: str) -> Tuple[Optional[Dict], bool]:
        """
        内部单图导入逻辑

        Returns:
            (图像数据库记录, 是否为本次新导入)
        """
        path = Path(file_path)

        if not path.exists() or not path.is_file():
            return None, False

        if path.suffix.lower() not in self.SUPPORTED_IMAGE_FORMATS:
            return None, False

        self._ensure_hash_index()

        file_hash = self._calculate_file_hash(str(path))
        if self._check_duplicate(file_hash):
            existing = self._hash_to_image.get(file_hash)
            print(f"图像已存在，跳过重复内容: {path.name}")
            return existing, False

        try:
            image_info = self._get_image_info(str(path))
            if not image_info:
                return None, False

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            target_filename = f"{timestamp}_{path.name}"
            target_path = self.images_path / target_filename

            shutil.copy2(str(path), str(target_path))

            image_id = db.add_image(
                project_id=self.project_id,
                filename=path.name,
                storage_path=str(target_path),
                width=image_info['width'],
                height=image_info['height'],
                size=image_info['size'],
                image_format=image_info['format'],
                original_path=str(path),
                file_hash=file_hash,
            )

            record = {
                'id': image_id,
                'project_id': self.project_id,
                'filename': path.name,
                'original_path': str(path),
                'storage_path': str(target_path),
                'width': image_info['width'],
                'height': image_info['height'],
                'size': image_info['size'],
                'format': image_info['format'],
                'status': 'pending',
                'file_hash': file_hash,
            }
            self._hash_to_image[file_hash] = record
            return record, True

        except Exception as e:
            print(f"导入图像失败 {path}: {e}")
            return None, False

    def import_single_image(self, file_path: str) -> bool:
        """
        导入单张图像（若项目内已有相同哈希文件则跳过并返回False）
        """
        _, newly_imported = self._import_single_image_internal(file_path)
        return newly_imported

    def import_and_get_image(self, file_path: str) -> Optional[Dict]:
        """
        导入单张图像并返回其数据库记录；若同内容图片已存在，则直接返回已有记录
        """
        record, _ = self._import_single_image_internal(file_path)
        return record

    def import_video(self, video_path: str, frame_interval: int = 1,
                    progress_callback: Callable[[int, str], None] = None) -> Tuple[int, int]:
        """
        从视频中抽取帧导入

        Args:
            video_path: 视频文件路径
            frame_interval: 抽帧间隔（每隔多少帧抽取一帧）
            progress_callback: 进度回调函数

        Returns:
            (成功导入数量, 跳过数量)
        """
        video_path = Path(video_path)
        if not video_path.exists():
            raise ValueError(f"视频文件不存在: {video_path}")

        if video_path.suffix.lower() not in self.SUPPORTED_VIDEO_FORMATS:
            raise ValueError(f"不支持的视频格式: {video_path.suffix}")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")

        self._ensure_hash_index()

        total_frames = max(1, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        frame_interval = max(1, int(frame_interval))

        imported = 0
        skipped = 0
        frame_count = 0

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % frame_interval == 0:
                    try:
                        progress = min(99, int((frame_count / total_frames) * 100))
                        if progress_callback:
                            progress_callback(progress, f"正在抽取帧 {frame_count}/{total_frames}")

                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                        target_filename = f"{timestamp}_frame_{frame_count:06d}.jpg"
                        target_path = self.images_path / target_filename

                        cv2.imwrite(str(target_path), frame)

                        file_hash = self._calculate_file_hash(str(target_path))
                        if self._check_duplicate(file_hash):
                            target_path.unlink(missing_ok=True)
                            skipped += 1
                        else:
                            height, width = frame.shape[:2]
                            size = target_path.stat().st_size

                            image_id = db.add_image(
                                project_id=self.project_id,
                                filename=target_filename,
                                storage_path=str(target_path),
                                width=width,
                                height=height,
                                size=size,
                                image_format='jpg',
                                original_path=str(video_path),
                                file_hash=file_hash,
                            )
                            self._hash_to_image[file_hash] = {
                                'id': image_id,
                                'project_id': self.project_id,
                                'filename': target_filename,
                                'storage_path': str(target_path),
                                'width': width,
                                'height': height,
                                'size': size,
                                'format': 'jpg',
                                'original_path': str(video_path),
                                'file_hash': file_hash,
                            }
                            imported += 1

                    except Exception as e:
                        print(f"保存帧失败 {frame_count}: {e}")
                        skipped += 1

                frame_count += 1

        finally:
            cap.release()

        if progress_callback:
            progress_callback(100, f"视频导入完成: 成功 {imported}, 跳过 {skipped}")

        return imported, skipped

    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件MD5哈希值（用于去重）"""
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()

    def _check_duplicate(self, file_hash: str) -> bool:
        """检查文件哈希是否已存在于当前项目中"""
        if not file_hash:
            return False
        self._ensure_hash_index()
        return file_hash in self._hash_to_image

    def _get_image_info(self, file_path: str) -> Optional[Dict]:
        """获取图像信息"""
        try:
            with Image.open(file_path) as img:
                width, height = img.size
                image_format = img.format.lower() if img.format else 'unknown'

            size = Path(file_path).stat().st_size

            return {
                'width': width,
                'height': height,
                'size': size,
                'format': image_format
            }

        except Exception as e:
            print(f"获取图像信息失败 {file_path}: {e}")
            return None

    def get_project_images(self) -> List[Dict]:
        """获取项目中的所有图像"""
        return db.get_project_images(self.project_id)

    def delete_image(self, image_id: int) -> bool:
        """删除图像（同步清理磁盘文件、标注与数据库记录）"""
        try:
            image_info = db.get_image(image_id)
            if not image_info or image_info.get('project_id') != self.project_id:
                return False

            file_hash = image_info.get('file_hash')
            deleted = db.delete_image(image_id)
            if deleted and self._hash_to_image is not None and file_hash:
                self._hash_to_image.pop(file_hash, None)
            return deleted

        except Exception as e:
            print(f"删除图像失败 {image_id}: {e}")
            return False
