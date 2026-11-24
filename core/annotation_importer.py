# -*- coding: utf-8 -*-
"""
标注导入器
支持YOLO、COCO、VOC格式的标注导入（带内存索引与缺失图片自动补导）
"""

from collections import defaultdict
import json
import random
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from core.import_manager import ImportManager
from models.database import db


def _default_class_color(class_id: int) -> str:
    palette = [
        "#EF4444", "#3B82F6", "#10B981", "#F59E0B",
        "#8B5CF6", "#EC4899", "#06B6D4", "#84CC16",
        "#F97316", "#6366F1",
    ]
    if 0 <= class_id < len(palette):
        return palette[class_id]
    rng = random.Random(class_id)
    return f"#{rng.randint(40, 220):02x}{rng.randint(40, 220):02x}{rng.randint(40, 220):02x}"


class AnnotationImporter:
    """标注导入器"""

    IMAGE_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp')

    def __init__(self, project_id: int):
        """
        初始化标注导入器

        Args:
            project_id: 项目ID
        """
        self.project_id = project_id
        self.project = db.get_project(project_id)
        if not self.project:
            raise ValueError(f"项目 {project_id} 不存在")

        # 获取项目类别信息
        raw_classes = self.project.get('classes')
        if isinstance(raw_classes, str):
            try:
                self.classes = json.loads(raw_classes) if raw_classes else []
            except Exception:
                self.classes = []
        elif isinstance(raw_classes, list):
            self.classes = list(raw_classes)
        else:
            self.classes = []

        self.class_map = {cls['name']: cls['id'] for cls in self.classes if 'name' in cls and 'id' in cls}
        self._id_to_class = {cls['id']: cls for cls in self.classes if 'id' in cls}

        # 惰性初始化图像导入管理器
        self._import_manager: Optional[ImportManager] = None

        # 构建内存图像索引，消除每次查找时的全表扫描 O(N²)
        self._images_by_filename: Dict[str, Dict] = {}
        self._images_by_original_path: Dict[str, Dict] = {}
        self._rebuild_image_index()

    def _rebuild_image_index(self):
        """一次性加载项目图像记录并建立内存字典索引"""
        self._images_by_filename.clear()
        self._images_by_original_path.clear()
        for image in db.get_project_images(self.project_id):
            self._index_image_record(image)

    def _index_image_record(self, image: Optional[Dict]):
        """将单条图像记录加入内存索引"""
        if not image:
            return
        filename = image.get('filename')
        if filename:
            self._images_by_filename[filename] = image
        orig = image.get('original_path')
        if orig:
            self._images_by_original_path[orig] = image
            try:
                self._images_by_original_path[str(Path(orig).resolve())] = image
            except Exception:
                pass

    def _ensure_class(self, class_name: str, preferred_id: Optional[int] = None) -> int:
        """确保类别存在于项目 classes 中，返回对应的 class_id"""
        if class_name in self.class_map:
            return self.class_map[class_name]

        if preferred_id is not None and preferred_id not in self._id_to_class:
            cid = preferred_id
        else:
            cid = max(self._id_to_class.keys(), default=-1) + 1

        new_cls = {
            'id': cid,
            'name': class_name,
            'color': _default_class_color(cid),
        }
        self.classes.append(new_cls)
        self.class_map[class_name] = cid
        self._id_to_class[cid] = new_cls
        db.update_project(self.project_id, classes=self.classes)
        return cid

    def _ensure_class_id(self, class_id: int, fallback_name: Optional[str] = None) -> str:
        """确保指定 class_id 已注册，返回类别名称"""
        if class_id in self._id_to_class:
            return self._id_to_class[class_id]['name']

        name = fallback_name or f"class_{class_id}"
        # 避免名称与其它已有类别冲突
        if name in self.class_map and self.class_map[name] != class_id:
            name = f"{name}_{class_id}"

        new_cls = {
            'id': class_id,
            'name': name,
            'color': _default_class_color(class_id),
        }
        self.classes.append(new_cls)
        self.class_map[name] = class_id
        self._id_to_class[class_id] = new_cls
        db.update_project(self.project_id, classes=self.classes)
        return name

    def _load_yolo_classes_txt(self, labels_dir: Path, images_dir: Optional[Path] = None):
        """如果存在 classes.txt，自动读取并注册类别"""
        candidates = [
            labels_dir / "classes.txt",
            labels_dir.parent / "classes.txt",
        ]
        if images_dir:
            candidates.append(images_dir / "classes.txt")
            candidates.append(images_dir.parent / "classes.txt")

        for cls_file in candidates:
            if cls_file.is_file():
                try:
                    names = [
                        line.strip()
                        for line in cls_file.read_text(encoding='utf-8').splitlines()
                        if line.strip()
                    ]
                    for idx, name in enumerate(names):
                        if idx not in self._id_to_class:
                            self._ensure_class_id(idx, fallback_name=name)
                    break
                except Exception as e:
                    print(f"读取 classes.txt 失败 {cls_file}: {e}")

    def import_yolo_annotations(self, labels_dir: str, images_dir: str = None, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入YOLO格式的标注

        Args:
            labels_dir: YOLO标签文件夹路径
            images_dir: 对应的图像文件夹路径（可选）
            overwrite: 是否覆盖已有的标注

        Returns:
            (成功导入数量, 跳过数量)
        """
        labels_dir = Path(labels_dir)
        if not labels_dir.exists():
            raise ValueError(f"标签文件夹不存在: {labels_dir}")

        images_dir_path = Path(images_dir) if images_dir else None
        self._load_yolo_classes_txt(labels_dir, images_dir_path)

        imported = 0
        skipped = 0

        txt_files = sorted(labels_dir.glob("*.txt"))

        for txt_file in txt_files:
            # 跳过类别定义文件本身
            if txt_file.name.lower() == "classes.txt":
                continue

            try:
                # 优先检查项目中是否已导入同名图像，否则在磁盘目录查找对应原图并自动补导
                image_file = self._find_corresponding_image(txt_file, images_dir_path)
                image_record = None

                if image_file:
                    image_record = self._find_image_record(str(image_file))
                    if not image_record:
                        image_record = self._import_image_if_needed(str(image_file))
                else:
                    # 原图不在标签旁时，尝试按 stem 匹配项目内已导入图片
                    image_record = self._find_image_record_by_stem(txt_file.stem)

                if not image_record:
                    print(f"未找到对应的图像文件: {txt_file}")
                    skipped += 1
                    continue

                image_info = self._resolve_image_dimensions(image_record, image_file)
                if not image_info:
                    skipped += 1
                    continue

                existing_annotations = db.get_image_annotations(image_record['id'])
                if existing_annotations and not overwrite:
                    skipped += 1
                    continue

                if existing_annotations and overwrite:
                    db.delete_image_annotations(image_record['id'])

                raw_content = txt_file.read_text(encoding='utf-8').strip()
                is_empty_label_file = (raw_content == "")
                annotations = self._parse_yolo_file(txt_file, image_info) if not is_empty_label_file else []

                project_task = self.project.get('type', 'detect')
                for ann in annotations:
                    if project_task == 'segment' and 'points' in ann['data']:
                        annotation_type = 'polygon'
                    elif project_task == 'pose' and 'keypoints' in ann['data']:
                        annotation_type = 'keypoint'
                    elif project_task == 'obb' and 'angle' in ann['data']:
                        annotation_type = 'obb'
                    else:
                        annotation_type = 'bbox'

                    db.add_annotation(
                        image_id=image_record['id'],
                        project_id=self.project_id,
                        class_id=ann['class_id'],
                        class_name=ann['class_name'],
                        annotation_type=annotation_type,
                        data=ann['data']
                    )

                # 空标签文件代表负样本，视为“已标注”
                if not annotations and is_empty_label_file:
                    db.update_image_status(image_record['id'], 'annotated')

                imported += len(annotations)

            except Exception as e:
                print(f"导入YOLO标注失败 {txt_file}: {e}")
                skipped += 1

        return imported, skipped

    def import_coco_annotations(self, coco_file: str, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入COCO格式的标注（按图片聚合处理，支持自动补导同目录/images子目录图片）

        Args:
            coco_file: COCO JSON文件路径
            overwrite: 是否覆盖已有的标注

        Returns:
            (成功导入数量, 跳过数量)
        """
        coco_path = Path(coco_file)
        if not coco_path.exists():
            raise ValueError(f"COCO文件不存在: {coco_path}")

        try:
            with open(coco_path, 'r', encoding='utf-8') as f:
                coco_data = json.load(f)
        except Exception as e:
            raise ValueError(f"读取COCO文件失败: {e}")

        imported = 0
        skipped = 0

        images = {img['id']: img for img in coco_data.get('images', []) if 'id' in img}
        categories = {cat['id']: cat for cat in coco_data.get('categories', []) if 'id' in cat}

        # 将COCO类别同步映射到项目类别（保证0-based连续ID，同时保留已有同名类别ID）
        coco_cat_to_project: Dict[int, Tuple[int, str]] = {}
        for cat_id, cat in sorted(categories.items(), key=lambda x: x[0]):
            cat_name = str(cat.get('name', f"class_{cat_id}"))
            proj_cid = self._ensure_class(cat_name)
            coco_cat_to_project[cat_id] = (proj_cid, cat_name)

        # 按 image_id 聚合标注，修复单图多目标时后续目标被跳过或相互覆盖的问题
        anns_by_image = defaultdict(list)
        orphan_count = 0
        for ann in coco_data.get('annotations', []):
            img_id = ann.get('image_id')
            if img_id in images:
                anns_by_image[img_id].append(ann)
            else:
                orphan_count += 1
        skipped += orphan_count

        search_dirs = [
            coco_path.parent,
            coco_path.parent / "images",
            coco_path.parent.parent / "images",
            coco_path.parent.parent,
        ]
        project_task = self.project.get('type', 'detect')

        for coco_img_id, image_info in images.items():
            img_anns = anns_by_image.get(coco_img_id, [])
            file_name = image_info.get('file_name', '')
            if not file_name:
                skipped += max(1, len(img_anns))
                continue

            base_name = Path(file_name).name
            image_record = (
                self._find_image_record_by_filename(base_name)
                or self._import_image_if_needed(file_name, search_dirs=search_dirs)
            )
            if not image_record:
                skipped += max(1, len(img_anns))
                continue

            existing_annotations = db.get_image_annotations(image_record['id'])
            if existing_annotations and not overwrite:
                skipped += len(img_anns)
                continue

            if existing_annotations and overwrite:
                db.delete_image_annotations(image_record['id'])

            if not img_anns:
                # COCO images列表中有记录但无annotations，视为负样本
                db.update_image_status(image_record['id'], 'annotated')
                continue

            for ann in img_anns:
                try:
                    category_id = ann.get('category_id')
                    if category_id not in coco_cat_to_project:
                        skipped += 1
                        continue

                    proj_class_id, proj_class_name = coco_cat_to_project[category_id]

                    # 解析有效多边形轮廓
                    polygon_points_list = []
                    segmentation = ann.get('segmentation')
                    if isinstance(segmentation, list):
                        for seg in segmentation:
                            if isinstance(seg, list) and len(seg) >= 6:
                                pts = [
                                    {'x': float(seg[j]), 'y': float(seg[j + 1])}
                                    for j in range(0, len(seg) - 1, 2)
                                ]
                                if len(pts) >= 3:
                                    polygon_points_list.append(pts)

                    has_bbox = 'bbox' in ann and isinstance(ann['bbox'], (list, tuple)) and len(ann['bbox']) >= 4

                    # 根据项目任务类型优先选择多边形或边界框，避免同一目标重复生成bbox+polygon两条记录
                    if polygon_points_list and (project_task == 'segment' or not has_bbox):
                        for pts in polygon_points_list:
                            db.add_annotation(
                                image_id=image_record['id'],
                                project_id=self.project_id,
                                class_id=proj_class_id,
                                class_name=proj_class_name,
                                annotation_type='polygon',
                                data={'points': pts},
                            )
                            imported += 1
                    elif has_bbox:
                        bbox = ann['bbox']
                        data = {
                            'x': float(bbox[0]),
                            'y': float(bbox[1]),
                            'width': float(bbox[2]),
                            'height': float(bbox[3]),
                        }
                        db.add_annotation(
                            image_id=image_record['id'],
                            project_id=self.project_id,
                            class_id=proj_class_id,
                            class_name=proj_class_name,
                            annotation_type='bbox',
                            data=data,
                        )
                        imported += 1
                    else:
                        skipped += 1

                except Exception as e:
                    print(f"导入COCO标注失败: {e}")
                    skipped += 1

        return imported, skipped

    def import_voc_annotations(self, voc_dir: str, overwrite: bool = False) -> Tuple[int, int]:
        """
        导入Pascal VOC格式的标注（支持从同级JPEGImages/images自动补导图片并自动注册类别）

        Args:
            voc_dir: VOC标注文件夹路径
            overwrite: 是否覆盖已有的标注

        Returns:
            (成功导入数量, 跳过数量)
        """
        voc_path = Path(voc_dir)
        if not voc_path.exists():
            raise ValueError(f"VOC文件夹不存在: {voc_path}")

        imported = 0
        skipped = 0

        search_dirs = [
            voc_path,
            voc_path.parent / "JPEGImages",
            voc_path.parent / "images",
            voc_path.parent,
        ]

        xml_files = sorted(voc_path.glob("*.xml"))

        for xml_file in xml_files:
            try:
                tree = ET.parse(xml_file)
                root = tree.getroot()

                filename_elem = root.find('filename')
                filename = filename_elem.text.strip() if (filename_elem is not None and filename_elem.text) else ""
                path_elem = root.find('path')
                xml_img_path = path_elem.text.strip() if (path_elem is not None and path_elem.text) else ""

                image_record = None
                if filename:
                    image_record = self._find_image_record_by_filename(Path(filename).name)
                if not image_record:
                    image_record = self._find_image_record_by_stem(xml_file.stem)
                if not image_record:
                    image_record = self._import_image_if_needed(
                        filename or xml_img_path or xml_file.stem,
                        search_dirs=search_dirs,
                        stem_hint=xml_file.stem,
                    )

                if not image_record:
                    skipped += 1
                    continue

                existing_annotations = db.get_image_annotations(image_record['id'])
                if existing_annotations and not overwrite:
                    skipped += 1
                    continue

                if existing_annotations and overwrite:
                    db.delete_image_annotations(image_record['id'])

                objects = root.findall('object')
                if not objects:
                    db.update_image_status(image_record['id'], 'annotated')
                    continue

                for obj in objects:
                    try:
                        name_elem = obj.find('name')
                        bbox_elem = obj.find('bndbox')
                        if name_elem is None or not name_elem.text or bbox_elem is None:
                            continue

                        class_name = name_elem.text.strip()
                        class_id = self._ensure_class(class_name)

                        xmin = float(bbox_elem.find('xmin').text)
                        ymin = float(bbox_elem.find('ymin').text)
                        xmax = float(bbox_elem.find('xmax').text)
                        ymax = float(bbox_elem.find('ymax').text)

                        data = {
                            'x': xmin,
                            'y': ymin,
                            'width': xmax - xmin,
                            'height': ymax - ymin,
                        }

                        db.add_annotation(
                            image_id=image_record['id'],
                            project_id=self.project_id,
                            class_id=class_id,
                            class_name=class_name,
                            annotation_type='bbox',
                            data=data,
                        )
                        imported += 1

                    except Exception as e:
                        print(f"解析VOC对象失败: {e}")
                        skipped += 1

            except Exception as e:
                print(f"导入VOC标注失败 {xml_file}: {e}")
                skipped += 1

        return imported, skipped

    def _parse_yolo_file(self, txt_file: Path, image_info: Dict) -> List[Dict]:
        """解析YOLO标注文件"""
        annotations = []

        try:
            with open(txt_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            img_width = image_info['width']
            img_height = image_info['height']
            project_task = self.project.get('type', 'detect')

            for line in lines:
                line = line.strip()
                if not line:
                    continue

                parts = line.split()
                if len(parts) < 1:
                    continue

                class_id = int(parts[0])
                class_name = self._ensure_class_id(class_id)

                if project_task == 'classify':
                    annotations.append({
                        'class_id': class_id,
                        'class_name': class_name,
                        'data': {'class_id': class_id}
                    })
                elif len(parts) >= 5:
                    x_center = float(parts[1])
                    y_center = float(parts[2])
                    width = float(parts[3])
                    height = float(parts[4])

                    if project_task == 'segment' and len(parts) > 5:
                        # YOLO分割格式: class_id x1 y1 x2 y2 ... xn yn (从第1项开始全部为顶点对)
                        points = []
                        for i in range(1, len(parts) - 1, 2):
                            px = float(parts[i]) * img_width
                            py = float(parts[i + 1]) * img_height
                            points.append({'x': px, 'y': py})

                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {'points': points}
                        })
                    elif project_task == 'pose' and len(parts) > 5:
                        keypoints = []
                        for i in range(5, len(parts), 3):
                            if i + 2 < len(parts):
                                kp_x = float(parts[i]) * img_width
                                kp_y = float(parts[i + 1]) * img_height
                                kp_v = float(parts[i + 2])
                                keypoints.append({
                                    'x': kp_x,
                                    'y': kp_y,
                                    'v': kp_v
                                })

                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': (x_center - width / 2) * img_width,
                                'y': (y_center - height / 2) * img_height,
                                'width': width * img_width,
                                'height': height * img_height,
                                'keypoints': keypoints
                            }
                        })
                    elif project_task == 'obb' and len(parts) > 5:
                        angle = float(parts[5]) if len(parts) > 5 else 0.0
                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': (x_center - width / 2) * img_width,
                                'y': (y_center - height / 2) * img_height,
                                'width': width * img_width,
                                'height': height * img_height,
                                'angle': angle
                            }
                        })
                    else:
                        x = (x_center - width / 2) * img_width
                        y = (y_center - height / 2) * img_height
                        w = width * img_width
                        h = height * img_height

                        annotations.append({
                            'class_id': class_id,
                            'class_name': class_name,
                            'data': {
                                'x': x,
                                'y': y,
                                'width': w,
                                'height': h
                            }
                        })

        except Exception as e:
            print(f"解析YOLO文件失败 {txt_file}: {e}")

        return annotations

    def _find_corresponding_image(self, txt_file: Path, images_dir: Path = None) -> Optional[Path]:
        """查找YOLO标签对应的磁盘图像文件（兼容大小写扩展名与标准 ../images 目录）"""
        base_name = txt_file.stem
        candidate_dirs = [txt_file.parent]
        if images_dir:
            candidate_dirs.insert(0, images_dir)
        else:
            # 兼容标准 YOLO 目录结构: dataset/labels/xxx.txt -> dataset/images/xxx.jpg
            sibling_images = txt_file.parent.parent / "images"
            if sibling_images.is_dir():
                candidate_dirs.append(sibling_images)
            split_images = txt_file.parent.parent.parent / "images" / txt_file.parent.name
            if split_images.is_dir():
                candidate_dirs.append(split_images)

        for directory in candidate_dirs:
            for ext in self.IMAGE_EXTENSIONS:
                for candidate_ext in (ext, ext.upper()):
                    image_file = directory / f"{base_name}{candidate_ext}"
                    if image_file.is_file():
                        return image_file
        return None

    def _resolve_image_dimensions(self, image_record: Dict, image_file: Optional[Path] = None) -> Optional[Dict]:
        """优先从数据库记录读取图像宽高，缺失时回退读取磁盘图像"""
        w = image_record.get('width')
        h = image_record.get('height')
        if w and h:
            return {
                'width': int(w),
                'height': int(h),
                'format': image_record.get('format') or 'unknown',
            }
        target_path = str(image_file) if image_file else image_record.get('storage_path')
        if target_path:
            return self._get_image_info(target_path)
        return None

    def _get_image_info(self, image_path: str) -> Optional[Dict]:
        """获取图像尺寸信息"""
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                width, height = img.size
                return {
                    'width': width,
                    'height': height,
                    'format': img.format.lower() if img.format else 'unknown'
                }
        except Exception as e:
            print(f"获取图像信息失败 {image_path}: {e}")
            return None

    def _find_image_record(self, image_path: str) -> Optional[Dict]:
        """通过内存索引 O(1) 查找数据库中的图像记录"""
        if image_path in self._images_by_original_path:
            return self._images_by_original_path[image_path]
        try:
            resolved = str(Path(image_path).resolve())
            if resolved in self._images_by_original_path:
                return self._images_by_original_path[resolved]
        except Exception:
            pass
        return self._images_by_filename.get(Path(image_path).name)

    def _find_image_record_by_filename(self, filename: str) -> Optional[Dict]:
        """根据文件名 O(1) 查找图像记录"""
        return self._images_by_filename.get(filename)

    def _find_image_record_by_stem(self, stem: str) -> Optional[Dict]:
        """根据无后缀文件名查找项目内已有图像记录"""
        for ext in self.IMAGE_EXTENSIONS:
            for candidate_ext in (ext, ext.upper()):
                rec = self._images_by_filename.get(f"{stem}{candidate_ext}")
                if rec:
                    return rec
        return None

    def _resolve_missing_image_path(
        self,
        image_ref: str,
        search_dirs: Optional[List[Path]] = None,
        stem_hint: Optional[str] = None,
    ) -> Optional[Path]:
        """在候选路径中定位尚未导入的图像原文件"""
        if image_ref:
            direct = Path(image_ref)
            if direct.is_file():
                return direct

        dirs = [d for d in (search_dirs or []) if d and d.is_dir()]
        base_name = Path(image_ref).name if image_ref else ""

        for d in dirs:
            if image_ref:
                rel_candidate = d / image_ref
                if rel_candidate.is_file():
                    return rel_candidate
            if base_name:
                name_candidate = d / base_name
                if name_candidate.is_file():
                    return name_candidate

        target_stem = stem_hint or (Path(base_name).stem if base_name else "")
        if target_stem:
            for d in dirs:
                for ext in self.IMAGE_EXTENSIONS:
                    for candidate_ext in (ext, ext.upper()):
                        p = d / f"{target_stem}{candidate_ext}"
                        if p.is_file():
                            return p

        return None

    def _import_image_if_needed(
        self,
        image_path: str,
        search_dirs: Optional[List[Path]] = None,
        stem_hint: Optional[str] = None,
    ) -> Optional[Dict]:
        """当标注对应的图像尚未导入项目时，自动定位并导入该图像"""
        resolved_path = self._resolve_missing_image_path(
            image_path, search_dirs=search_dirs, stem_hint=stem_hint
        )
        if not resolved_path:
            return None

        if self._import_manager is None:
            self._import_manager = ImportManager(self.project_id)

        record = self._import_manager.import_and_get_image(str(resolved_path))
        if record:
            self._index_image_record(record)
        return record

    def _get_class_name_by_id(self, class_id: int) -> str:
        """根据类别ID获取类别名称"""
        if class_id in self._id_to_class:
            return self._id_to_class[class_id]['name']
        return f"class_{class_id}"
