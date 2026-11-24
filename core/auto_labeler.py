# -*- coding: utf-8 -*-
"""
自动打标签核心逻辑模块
负责使用YOLO模型自动生成标注
"""

import json
import os
import random
import re
import time
from typing import Dict, List, Optional
from PyQt6.QtCore import QThread, pyqtSignal, QObject

from core.model_manager import model_manager, get_pretrained_dir
from models.database import db


def _auto_class_color(class_id: int) -> str:
    palette = [
        "#EF4444", "#3B82F6", "#10B981", "#F59E0B",
        "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16",
        "#F97316", "#6366F1",
    ]
    if 0 <= class_id < len(palette):
        return palette[class_id]
    rng = random.Random(class_id)
    return f"#{rng.randint(40, 220):02x}{rng.randint(40, 220):02x}{rng.randint(40, 220):02x}"


class AutoLabeler:
    """自动打标签器"""

    def __init__(self, model_path: str, model_manager):
        self.model_path = model_path
        self.model_manager = model_manager
        self.current_model = None
        self.model_info = {}
        self.class_mappings = {}
        self.model_task = 'detect'
        self._project_classes_cache: Dict[int, List[Dict]] = {}

    def load_model(self, config: Dict) -> bool:
        """加载模型"""
        try:
            model_version = config['model_version']
            model_size = config['model_size']
            model_source = config['model_source']
            model_task = config.get('model_task', 'detect')
            custom_model_path = config.get('custom_model_path', '')

            print("=" * 50)
            print("AutoLabeler.load_model 开始加载模型:")
            print(f"  模型版本: {model_version}")
            print(f"  模型大小: {model_size}")
            print(f"  模型来源: {model_source}")
            print(f"  任务类型: {model_task}")
            if model_source == 'custom':
                print(f"  自定义模型路径: {custom_model_path}")
            print("=" * 50)

            if model_source == 'official':
                self.current_model = model_manager.load_model(
                    model_version, model_size, model_task
                )
            else:
                self.current_model = model_manager.load_custom_model(custom_model_path)

            if self.current_model:
                self.model_info = model_manager.get_model_info(self.current_model)
                self.class_mappings = config.get('class_mappings', {})
                self.model_task = model_task

                model_name = (
                    f"{model_version}-{model_size}-{model_task}"
                    if model_source == 'official'
                    else os.path.basename(custom_model_path)
                )
                print("=" * 50)
                print("AutoLabeler.load_model 模型加载成功:")
                print(f"  模型名称: {model_name}")
                print(f"  任务类型: {self.model_task}")
                print(f"  模型任务 (model.task): {self.model_info.get('task', 'unknown')}")
                print(f"  类别数量: {self.model_info.get('nc', 0)}")
                print("=" * 50)
                return True
            else:
                print("=" * 50)
                print("AutoLabeler.load_model 模型加载失败!")
                print("=" * 50)
            return False
        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _get_model_class_name(self, result: object, raw_class_id: int) -> Optional[str]:
        """从推理结果或模型元信息中提取原始类别名称"""
        names = getattr(result, 'names', None)
        if isinstance(names, dict) and raw_class_id in names:
            return str(names[raw_class_id])
        if isinstance(names, list) and 0 <= raw_class_id < len(names):
            return str(names[raw_class_id])
        info_names = self.model_info.get('names', {})
        if isinstance(info_names, dict) and raw_class_id in info_names:
            return str(info_names[raw_class_id])
        return None

    def _resolve_project_class_name(
        self,
        project_id: int,
        class_id: int,
        model_class_name: Optional[str] = None,
    ) -> str:
        """根据项目已有类别表解析或自动注册类别名称，避免写入无意义的 class_0"""
        if project_id not in self._project_classes_cache:
            project = db.get_project(project_id)
            raw_classes = project.get('classes', '[]') if project else '[]'
            if isinstance(raw_classes, str):
                try:
                    classes = json.loads(raw_classes) if raw_classes else []
                except Exception:
                    classes = []
            elif isinstance(raw_classes, list):
                classes = list(raw_classes)
            else:
                classes = []
            self._project_classes_cache[project_id] = classes

        classes = self._project_classes_cache[project_id]
        for cls in classes:
            if cls.get('id') == class_id:
                return cls.get('name') or model_class_name or f"class_{class_id}"

        resolved_name = (model_class_name or "").strip() or f"class_{class_id}"
        # 防止同项目不同ID重名
        existing_names = {c.get('name') for c in classes}
        if resolved_name in existing_names:
            resolved_name = f"{resolved_name}_{class_id}"

        classes.append({
            'id': class_id,
            'name': resolved_name,
            'color': _auto_class_color(class_id),
        })
        if project_id:
            db.update_project(project_id, classes=classes)
        return resolved_name

    def process_single_image(self, image_path: str, image_id: int, config: Dict) -> List[Dict]:
        """处理单张图像"""
        if not self.current_model:
            return []

        conf_threshold = config.get('conf_threshold', 0.5)
        iou_threshold = config.get('iou_threshold', 0.45)

        result = model_manager.infer(
            self.current_model, image_path, conf_threshold, iou_threshold
        )

        if not result:
            return []

        return self._generate_annotations(result, image_id, config)

    def _generate_annotations(self, result: object, image_id: int, config: Dict) -> List[Dict]:
        """根据推理结果生成标注"""
        annotations = []
        model_task = getattr(self, 'model_task', 'detect')

        for i, box in enumerate(result.boxes):
            raw_class_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            width = x2 - x1
            height = y2 - y1

            mapped_class_id = self._map_class_id(raw_class_id)
            raw_class_name = self._get_model_class_name(result, raw_class_id)

            if model_task == 'segment' and hasattr(result, 'masks') and result.masks:
                if hasattr(result.masks, 'xy') and result.masks.xy is not None and i < len(result.masks.xy):
                    polygons = result.masks.xy[i]
                    polygons_list = polygons.tolist() if hasattr(polygons, 'tolist') else polygons

                    points = []
                    for point in polygons_list:
                        if hasattr(point, 'tolist'):
                            point = point.tolist()
                        if isinstance(point, (list, tuple)) and len(point) >= 2:
                            points.append({'x': float(point[0]), 'y': float(point[1])})

                    if points:
                        annotation = {
                            'image_id': image_id,
                            'type': 'polygon',
                            'class_id': mapped_class_id,
                            'model_class_name': raw_class_name,
                            'confidence': conf,
                            'data': {'points': points},
                        }
                    else:
                        annotation = {
                            'image_id': image_id,
                            'type': 'bbox',
                            'class_id': mapped_class_id,
                            'model_class_name': raw_class_name,
                            'confidence': conf,
                            'data': {'x': x1, 'y': y1, 'width': width, 'height': height},
                        }
                else:
                    annotation = {
                        'image_id': image_id,
                        'type': 'bbox',
                        'class_id': mapped_class_id,
                        'model_class_name': raw_class_name,
                        'confidence': conf,
                        'data': {'x': x1, 'y': y1, 'width': width, 'height': height},
                    }
            else:
                annotation = {
                    'image_id': image_id,
                    'type': 'bbox',
                    'class_id': mapped_class_id,
                    'model_class_name': raw_class_name,
                    'confidence': conf,
                    'data': {'x': x1, 'y': y1, 'width': width, 'height': height},
                }

            annotations.append(annotation)

        return annotations

    def _map_class_id(self, class_id: int) -> int:
        """映射类别ID"""
        if class_id in self.class_mappings:
            return self.class_mappings[class_id]
        return class_id

    def process_class_id(self, class_id: int, project_classes: list, enable_mapping: bool = False) -> int:
        """处理类别ID，根据映射设置和项目类别列表"""
        if enable_mapping:
            return self._map_class_id(class_id)
        return class_id

    def save_annotations(
        self,
        annotations: List[Dict],
        image_id: int,
        overwrite: bool = False,
        ignore_lock: bool = False,
    ) -> int:
        """保存标注（默认跳过已人工锁定的图片，并自动解析项目/模型真实类别名）"""
        image_info = self.get_image_info(image_id)
        if not image_info:
            return 0

        if image_info.get('locked') and not ignore_lock:
            return 0

        project_id = image_info.get('project_id', 0)

        if overwrite:
            # 覆盖模式下仅删除未锁定的旧标注，保护人工锁定的单框
            existing = db.get_image_annotations(image_id)
            for ann in existing:
                attrs = ann.get('attributes') or {}
                if not attrs.get('locked'):
                    db.delete_annotation(ann['id'])

        saved_count = 0
        for annotation in annotations:
            class_id = annotation.get('class_id', 0)
            class_name = self._resolve_project_class_name(
                project_id,
                class_id,
                annotation.get('model_class_name'),
            )
            conf = round(float(annotation.get('confidence', 1.0)), 4)

            db.add_annotation(
                image_id,
                project_id,
                class_id,
                class_name,
                annotation.get('type', 'bbox'),
                annotation.get('data', {}),
                {'source': 'yolo', 'confidence': conf},
            )
            saved_count += 1

        return saved_count

    def get_image_info(self, image_id: int) -> Optional[Dict]:
        """获取图像信息"""
        return db.get_image(image_id)

    def get_unlabeled_images(self, project_id: int) -> List[Dict]:
        """获取未标注且未锁定的图像"""
        images = db.get_project_images(project_id)
        unlabeled = []
        for image in images:
            if image.get('locked'):
                continue
            if image.get('status') != 'annotated' and not db.get_image_annotations(image['id']):
                unlabeled.append(image)
        return unlabeled

    def get_all_images(self, project_id: int) -> List[Dict]:
        """获取所有图像"""
        return db.get_project_images(project_id)

    def process_image(self, image_path: str, conf_threshold: float, iou_threshold: float, class_mapping: dict) -> List[Dict]:
        """处理单张图像（兼容旧接口）"""
        try:
            if not self.current_model:
                if os.path.exists(self.model_path):
                    self.current_model = self.model_manager.load_custom_model(self.model_path)
                else:
                    match = re.match(r'(yolov?)(\d+)([a-z]+)', self.model_path, re.I)
                    if match:
                        version_num = match.group(2)
                        size = match.group(3)
                        version = f"YOLOv{version_num}"
                        self.current_model = self.model_manager.load_model(version, size)
                    if not self.current_model:
                        from ultralytics import YOLO

                        pretrained_dir = str(get_pretrained_dir())
                        original_hub_dir = os.environ.get('YOLO_HUB_DIR')
                        os.environ['YOLO_HUB_DIR'] = pretrained_dir
                        try:
                            self.current_model = YOLO(self.model_path)
                        finally:
                            if original_hub_dir:
                                os.environ['YOLO_HUB_DIR'] = original_hub_dir
                            else:
                                os.environ.pop('YOLO_HUB_DIR', None)

            if not self.current_model:
                return []

            result = self.model_manager.infer(
                self.current_model, image_path, conf_threshold, iou_threshold
            )
            if not result:
                return []

            annotations = []
            for box in result.boxes:
                raw_class_id = int(box.cls[0])
                conf = float(box.conf[0])
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                mapped_class_id = class_mapping.get(raw_class_id, raw_class_id)
                raw_class_name = self._get_model_class_name(result, raw_class_id)

                annotations.append({
                    'type': 'bbox',
                    'class_id': mapped_class_id,
                    'model_class_name': raw_class_name,
                    'confidence': conf,
                    'data': {
                        'x': x1,
                        'y': y1,
                        'width': x2 - x1,
                        'height': y2 - y1,
                    }
                })

            return annotations
        except Exception as e:
            print(f"Error processing image: {e}")
            return []

    def unload_model(self):
        """卸载模型"""
        model_manager.unload_all_models()
        self.current_model = None
        self.model_info = {}
        self._project_classes_cache.clear()


class BatchLabelingThread(QThread):
    """批量标注线程"""

    progress_updated = pyqtSignal(int, int, int)  # 已处理数量, 总数量, 已标注数量
    image_processed = pyqtSignal(str, int)  # 图像路径, 生成的标注数量
    batch_completed = pyqtSignal(bool, str)  # 是否成功, 消息

    def __init__(self, auto_labeler: AutoLabeler, images: List[Dict], config: Dict):
        super().__init__()
        self.auto_labeler = auto_labeler
        self.images = images
        self.config = config
        self._is_running = True
        self._is_paused = False

    def run(self):
        """运行批量标注"""
        total = len(self.images)
        processed = 0
        labeled = 0
        skipped_locked = 0

        try:
            for image in self.images:
                if not self._is_running:
                    break

                while self._is_paused:
                    time.sleep(0.1)
                    if not self._is_running:
                        break

                if not self._is_running:
                    break

                image_path = image.get('storage_path', '')
                image_id = image.get('id', 0)

                # 跳过已人工锁定的图片，避免覆盖人工精修成果
                latest_img = db.get_image(image_id) if image_id else image
                if (latest_img and latest_img.get('locked')) or image.get('locked'):
                    skipped_locked += 1
                    processed += 1
                    self.progress_updated.emit(processed, total, labeled)
                    self.image_processed.emit(image_path, 0)
                    continue

                if image_path and os.path.exists(image_path):
                    annotations = self.auto_labeler.process_single_image(
                        image_path, image_id, self.config
                    )

                    saved = 0
                    if annotations:
                        overwrite = self.config.get('overwrite_labels', False)
                        saved = self.auto_labeler.save_annotations(
                            annotations, image_id, overwrite
                        )
                        labeled += saved

                    processed += 1
                    self.progress_updated.emit(processed, total, labeled)
                    self.image_processed.emit(image_path, saved)

                time.sleep(0.01)

            lock_note = f"（已自动跳过人工锁定图片 {skipped_locked} 张）" if skipped_locked else ""
            if self._is_running:
                self.batch_completed.emit(
                    True,
                    f"批量标注完成，处理了 {processed}/{total} 张图像，生成了 {labeled} 个标注{lock_note}"
                )
            else:
                self.batch_completed.emit(
                    False,
                    f"批量标注被停止，已处理 {processed}/{total} 张图像{lock_note}"
                )
        except Exception as e:
            self.batch_completed.emit(False, f"批量标注出错: {str(e)}")

    def pause(self):
        self._is_paused = True

    def resume(self):
        self._is_paused = False

    def stop(self):
        self._is_running = False
        self._is_paused = False


class BatchLabelingManager(QObject):
    """批量标注管理器"""

    progress_updated = pyqtSignal(int, int, int, str)  # 进度, 当前, 总数, 图像名称
    batch_completed = pyqtSignal(bool, str, int)  # 成功, 消息, 处理数量

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_thread = None
        self.auto_labeler = None
        self.model_manager = None
        self.images = []

    def start_batch_processing(
        self,
        model_path: str,
        images: list,
        conf_threshold: float,
        iou_threshold: float,
        class_mapping: dict,
        model_manager,
        model_task: str = 'detect',
        overwrite_labels: bool = True,
    ):
        """开始批量处理"""
        try:
            self.model_manager = model_manager
            # 过滤掉已人工锁定的图片
            self.images = [img for img in images if not img.get('locked')]

            self.auto_labeler = AutoLabeler(model_path, model_manager)

            config = {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'class_mappings': class_mapping,
                'overwrite_labels': overwrite_labels,
            }

            match = re.match(r'(yolov?)(\d+)([a-z]+)', model_path, re.I)
            if match:
                version_num = match.group(2)
                size = match.group(3)
                version = f"YOLOv{version_num}"
                load_config = {
                    'model_version': version,
                    'model_size': size,
                    'model_source': 'official',
                    'model_task': model_task,
                    'class_mappings': class_mapping,
                }
                self.auto_labeler.load_model(load_config)
            elif os.path.exists(model_path):
                load_config = {
                    'model_version': '',
                    'model_size': '',
                    'model_source': 'custom',
                    'model_task': model_task,
                    'custom_model_path': model_path,
                    'class_mappings': class_mapping,
                }
                self.auto_labeler.load_model(load_config)

            self.current_thread = BatchLabelingThread(
                self.auto_labeler, self.images, config
            )
            self.current_thread.progress_updated.connect(self.on_progress_updated)
            self.current_thread.batch_completed.connect(self.on_batch_completed)
            self.current_thread.start()
        except Exception as e:
            print(f"Error starting batch processing: {e}")

    def on_progress_updated(self, processed: int, total: int, labeled: int):
        if hasattr(self, 'progress_updated') and total > 0:
            current_image = self.images[processed - 1] if 0 < processed <= len(self.images) else {}
            image_name = os.path.basename(current_image.get('storage_path', ''))
            progress = int((processed / total) * 100)
            self.progress_updated.emit(progress, processed, total, image_name)

    def on_batch_completed(self, success: bool, message: str):
        if hasattr(self, 'batch_completed'):
            processed_count = len(self.images)
            self.batch_completed.emit(success, message, processed_count)

    def pause(self):
        if self.current_thread:
            self.current_thread.pause()

    def resume(self):
        if self.current_thread:
            self.current_thread.resume()

    def stop(self):
        if self.current_thread:
            self.current_thread.stop()
            self.current_thread.wait()

    def is_running(self) -> bool:
        return bool(self.current_thread and self.current_thread.isRunning())

    def cleanup(self):
        self.stop()
        if self.auto_labeler:
            self.auto_labeler.unload_model()
