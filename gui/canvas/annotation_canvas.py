# -*- coding: utf-8 -*-
"""标注画布组件：负责图像缩放平移、矩形/多边形/关键点/OBB 绘制与 SAM 交互渲染。"""

import math
import os
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PyQt6.QtCore import QPoint, QPointF, QRect, Qt, pyqtSignal
from PyQt6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QImage,
    QKeyEvent,
    QMouseEvent,
    QPainter,
    QPen,
    QPixmap,
    QWheelEvent,
)
from PyQt6.QtWidgets import QFrame

from core.sam_manager import SAMInferenceWorker
from gui.styles import COLORS
from gui.ui_scale import ui_font, ui_pixels


class AnnotationCanvas(QFrame):
    """标注画布组件"""
    
    annotation_created = pyqtSignal(dict)  # 标注创建信号
    annotation_selected = pyqtSignal(int)  # 标注选中信号
    annotation_modified = pyqtSignal(int, dict)  # 标注修改信号
    annotation_deleted = pyqtSignal(int)  # 标注删除信号
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setFrameStyle(QFrame.Shape.StyledPanel)
        
        # 图像数据
        self.current_image = None
        self.current_image_path = None
        self.image_scale = 1.0
        self.image_offset = QPoint(0, 0)
        
        # 标注数据
        self.annotations = []  # 当前图像的所有标注
        self.image_locked = False  # 整图锁定(人工修改过,AI不再处理)
        self.selected_annotation_id = None
        self.current_tool = 'rectangle'  # rectangle, polygon, move
        self.drawing = False
        self.start_point = None
        self.current_point = None
        
        # 多边形绘制
        self.polygon_points = []
        
        # 关键点绘制
        self.keypoints = []
        self.drawing_keypoint = False
        
        # OBB绘制状态
        self.obb_state = 0  # 0: 未开始, 1: 已确定第一个点, 2: 已确定第一条边
        self.obb_points = []
        
        # 编辑状态
        self.editing = False
        self.dragging = False
        self.resizing = False
        # 多边形顶点拖动（move工具下）
        self.dragging_vertex = False
        self.drag_vertex_index = None
        self.drag_start = None
        self.drag_start_annotation = None
        self.resize_handle = None
        self.resize_start_rect = None
        
        # 图片平移
        self.panning = False
        self.pan_start = None
        self.pan_start_offset = None
        
        # 手柄大小
        self.handle_size = 8
        # 多边形顶点命中阈值（控件坐标，像素）
        self.vertex_hit_radius = 8
        # 多边形顶点“磁吸/辅助命中”半径（比命中半径略大一点）
        self.vertex_snap_radius = 14
        # 当前悬停的多边形顶点 (annotation_id, vertex_index) / None
        self.hover_vertex = None
        
        # 类别颜色与名称映射（动态获取，仅用于画布显示）
        self.class_colors = {}
        self.class_names = {}
        
        # 当前选中的类别ID
        self.current_class_id = 0
        
        # 批量处理模式
        self.batch_process_mode = False
        self.batch_process_points = []
        self.batch_process_dialog = None
        
        # SAM标注模式
        self.sam_mode_active = False
        self.sam_config = None
        self.sam_image_path = None
        self.sam_points = []  # [(x, y), ...]
        self.sam_bboxes = []  # [(x1, y1, x2, y2), ...]
        self.sam_mode = 'point'  # 'point' 或 'bbox'
        self.sam_drawing_bbox = False
        self.sam_start_point = None
        self.sam_current_point = None
        # SAM运行模式：normal=普通交互(允许自动推理), memory_collect=记忆采集(禁止自动推理)
        self.sam_operation_mode = "normal"
        # SAM记忆模式显示列表
        self.memory_display_points = []  # [{"x": x, "y": y, "obj_id": id}, ...]
        self.memory_display_bboxes = []  # [{"x1": x1, "y1": y1, "x2": x2, "y2": y2, "obj_id": id}, ...]
        
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        # 初始样式会在主题变化时被更新
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['sidebar']};
                border: 2px solid {COLORS['border']};
            }}
        """)
        
        # 设置鼠标追踪
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
    
    def load_image(self, image_path: str):
        """加载图像"""
        if not image_path or not os.path.exists(image_path):
            self.current_image = None
            self.current_image_path = None
            self.update()
            return
        
        # 使用OpenCV加载图像
        img = cv2.imread(image_path)
        if img is not None:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            h, w, ch = img.shape
            bytes_per_line = ch * w
            qt_image = QImage(img.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            self.current_image = QPixmap.fromImage(qt_image)
            self.current_image_path = image_path
            
            # 重置视图
            self.reset_view()
            self.update()
    
    def reset_view(self):
        """重置视图"""
        if self.current_image is None:
            return
        
        # 计算适应窗口的缩放比例
        widget_rect = self.rect()
        img_width = self.current_image.width()
        img_height = self.current_image.height()
        
        scale_x = (widget_rect.width() - 40) / img_width
        scale_y = (widget_rect.height() - 40) / img_height
        self.image_scale = min(scale_x, scale_y, 1.0)
        
        # 居中显示
        scaled_width = img_width * self.image_scale
        scaled_height = img_height * self.image_scale
        self.image_offset = QPoint(
            int((widget_rect.width() - scaled_width) / 2),
            int((widget_rect.height() - scaled_height) / 2)
        )
    
    def set_annotations(self, annotations: List[Dict]):
        """设置标注数据"""
        self.annotations = annotations
        self.selected_annotation_id = None
        self.update()
    
    def set_tool(self, tool: str):
        """设置当前工具"""
        self.current_tool = tool
        self.drawing = False
        self.polygon_points = []
        # 清除关键点绘制状态
        if hasattr(self, 'keypoints'):
            self.keypoints = []
        
        # 处理SAM工具
        if tool == 'sam':
            self.sam_mode_active = True
            self.sam_points = []
            self.sam_bboxes = []
            self.sam_mode = 'point'
        else:
            self.sam_mode_active = False
            self.sam_operation_mode = "normal"
        
        self.update()

    def set_sam_operation_mode(self, mode: str):
        """设置SAM交互模式。"""
        self.sam_operation_mode = mode if mode in ("normal", "memory_collect") else "normal"

    def _sam_auto_infer_enabled(self) -> bool:
        """当前状态下是否允许SAM自动推理。"""
        return self.sam_mode_active and self.sam_operation_mode == "normal"
    
    def image_to_widget(self, x: float, y: float) -> QPoint:
        """图像坐标转换为控件坐标"""
        widget_x = int(x * self.image_scale + self.image_offset.x())
        widget_y = int(y * self.image_scale + self.image_offset.y())
        return QPoint(widget_x, widget_y)
    
    def widget_to_image(self, x: int, y: int) -> Tuple[float, float]:
        """控件坐标转换为图像坐标"""
        img_x = (x - self.image_offset.x()) / self.image_scale
        img_y = (y - self.image_offset.y()) / self.image_scale
        return (img_x, img_y)
    
    def paintEvent(self, event):
        """绘制事件"""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # 绘制背景
        painter.fillRect(self.rect(), QColor(COLORS['sidebar']))
        
        if self.current_image is None:
            # 显示提示文字
            painter.setPen(QColor(COLORS['text_secondary']))
            painter.setFont(ui_font("Microsoft YaHei", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "请选择一张图片开始标注")
            return
        
        # 绘制图像
        scaled_pixmap = self.current_image.scaled(
            int(self.current_image.width() * self.image_scale),
            int(self.current_image.height() * self.image_scale),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawPixmap(self.image_offset, scaled_pixmap)
        
        # 绘制标注
        self.draw_annotations(painter)

        # 整图锁定横幅(人工修改过,AI不再处理)
        if self.image_locked:
            self.draw_lock_banner(painter)
        
        # 绘制正在绘制的矩形
        if self.drawing and self.current_tool == 'rectangle' and self.start_point and self.current_point:
            self.draw_drawing_rectangle(painter)
        
        # 绘制正在绘制的多边形
        if self.current_tool == 'polygon' and len(self.polygon_points) > 0:
            self.draw_drawing_polygon(painter)
        
        # 绘制正在绘制的旋转矩形
        if self.current_tool == 'obb' and len(self.obb_points) > 0:
            self.draw_drawing_obb(painter)
        
        # 绘制鼠标辅助线
        if self.current_image and self.current_point:
            self.draw_guide_lines(painter)
        
        # 批量处理模式：绘制已选择的像素点
        if self.batch_process_mode and self.batch_process_points:
            self.draw_batch_process_points(painter)
        
        # SAM模式：绘制点和框
        if self.sam_mode_active:
            self.draw_sam_elements(painter)
    
    def draw_sam_elements(self, painter: QPainter):
        """绘制SAM模式的点和框"""
        # 绘制记忆对象的所有标记（不同ID用不同颜色）
        memory_points = getattr(self, 'memory_display_points', [])
        memory_bboxes = getattr(self, 'memory_display_bboxes', [])
        
        if memory_points:
            # 为不同ID分配不同颜色
            id_colors = {}
            color_list = [
                QColor(255, 0, 0),    # 红
                QColor(0, 255, 0),    # 绿
                QColor(0, 0, 255),    # 蓝
                QColor(255, 255, 0),  # 黄
                QColor(255, 0, 255),  # 紫
                QColor(0, 255, 255),  # 青
                QColor(255, 128, 0),  # 橙
                QColor(128, 0, 255),  # 紫红
            ]
            
            for point_data in memory_points:
                obj_id = point_data.get("obj_id", 0)
                if obj_id not in id_colors:
                    id_colors[obj_id] = color_list[obj_id % len(color_list)]
                color = id_colors[obj_id]
                
                img_x = point_data["x"]
                img_y = point_data["y"]
                widget_pos = self.image_to_widget(img_x, img_y)
                
                # 绘制点
                painter.setPen(QPen(color, 2))
                painter.setBrush(QBrush(color))
                radius = 8
                painter.drawEllipse(widget_pos, radius, radius)
                
                # 绘制ID
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(ui_font("Microsoft YaHei", 10, QFont.Weight.Bold))
                painter.drawText(widget_pos.x() + radius + 2, widget_pos.y() - radius, f"ID:{obj_id}")
        
        # 绘制记忆对象的所有框
        if memory_bboxes:
            id_colors = {}
            color_list = [
                QColor(255, 0, 0), QColor(0, 255, 0), QColor(0, 0, 255),
                QColor(255, 255, 0), QColor(255, 0, 255), QColor(0, 255, 255),
                QColor(255, 128, 0), QColor(128, 0, 255),
            ]
            
            for bbox_data in memory_bboxes:
                obj_id = bbox_data.get("obj_id", 0)
                if obj_id not in id_colors:
                    id_colors[obj_id] = color_list[obj_id % len(color_list)]
                color = id_colors[obj_id]
                
                x1 = bbox_data["x1"]
                y1 = bbox_data["y1"]
                x2 = bbox_data["x2"]
                y2 = bbox_data["y2"]
                
                p1 = self.image_to_widget(x1, y1)
                p2 = self.image_to_widget(x2, y2)
                
                # 绘制框
                pen = QPen(color, 2)
                pen.setStyle(Qt.PenStyle.SolidLine)
                painter.setPen(pen)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                rect = QRect(p1, p2)
                painter.drawRect(rect)
                
                # 绘制ID标签
                painter.setBrush(QBrush(color))
                painter.setPen(QPen(color, 1))
                label_rect = QRect(p1.x(), p1.y() - 20, 40, 20)
                painter.drawRect(label_rect)
                painter.setPen(QColor(255, 255, 255))
                painter.setFont(ui_font("Microsoft YaHei", 9))
                painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, f"ID:{obj_id}")
        
        # 绘制当前正在添加的点（临时标记）
        painter.setPen(QPen(QColor(0, 255, 0), 2))
        painter.setBrush(QBrush(QColor(0, 255, 0)))
        for i, (img_x, img_y) in enumerate(self.sam_points):
            widget_pos = self.image_to_widget(img_x, img_y)
            radius = 8
            painter.drawEllipse(widget_pos, radius, radius)
            # 绘制点编号
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(ui_font("Microsoft YaHei", 10, QFont.Weight.Bold))
            painter.drawText(widget_pos.x() + radius + 2, widget_pos.y() - radius, str(i + 1))
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0)))
        
        # 绘制正在绘制的框（虚线）
        if self.sam_drawing_bbox and self.sam_start_point and self.sam_current_point:
            pen = QPen(QColor(255, 0, 0), 2)
            pen.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            rect = QRect(self.sam_start_point, self.sam_current_point)
            painter.drawRect(rect)
    
    def draw_batch_process_points(self, painter: QPainter):
        """绘制批量处理模式下选择的像素点"""
        # 设置绘制样式
        painter.setPen(QPen(QColor(255, 0, 0), 2))
        painter.setBrush(QBrush(QColor(255, 0, 0)))
        
        # 绘制每个像素点
        for i, (img_x, img_y) in enumerate(self.batch_process_points):
            # 将图像坐标转换为控件坐标
            widget_pos = self.image_to_widget(img_x, img_y)
            
            # 绘制圆点
            radius = 6
            painter.drawEllipse(widget_pos, radius, radius)
            
            # 绘制点编号
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(ui_font("Microsoft YaHei", 10, QFont.Weight.Bold))
            painter.drawText(widget_pos.x() + radius + 2, widget_pos.y() - radius, str(i + 1))
            
            # 恢复绘制样式
            painter.setPen(QPen(QColor(255, 0, 0), 2))
            painter.setBrush(QBrush(QColor(255, 0, 0)))
    
    def draw_annotations(self, painter: QPainter):
        """绘制所有标注"""
        for annotation in self.annotations:
            ann_id = annotation['id']
            ann_type = annotation.get('type', 'bbox')
            data = annotation.get('data', {})
            class_id = annotation.get('class_id', 0)
            
            # 获取颜色（优先从class_colors字典，否则使用默认灰色）
            color = self.class_colors.get(class_id, QColor(128, 128, 128))
            if isinstance(color, str):
                color = QColor(color)
            
            # 如果是选中的标注，使用高亮颜色
            is_selected = (ann_id == self.selected_annotation_id)
            pen_width = 3 if is_selected else 2
            
            pen = QPen(color)
            pen.setWidth(pen_width)
            painter.setPen(pen)
            
            brush = QBrush(color)
            brush.setStyle(Qt.BrushStyle.NoBrush)
            painter.setBrush(brush)
            
            if ann_type == 'bbox':
                self.draw_bbox(painter, data, is_selected)
            elif ann_type == 'polygon':
                self.draw_polygon(painter, data, is_selected)
            elif ann_type == 'keypoint':
                self.draw_keypoints(painter, data, is_selected)
            elif ann_type == 'obb':
                self.draw_obb(painter, data, is_selected)

            # 锁定的标注绘制锁形标记(人工修改过或手动锁定)
            has_lock = bool((annotation.get('attributes') or {}).get('locked'))
            if has_lock:
                self.draw_lock_marker(painter, ann_type, data)

            # 在画布框体上显示类别名徽标（纯UI渲染，不写入任何导出的txt文件）
            label_text = self._get_class_label_text(class_id, annotation.get('class_name', ''))
            if label_text:
                anchor = self._get_annotation_label_anchor(ann_type, data)
                if anchor is not None:
                    self.draw_class_label(
                        painter, anchor, label_text, color,
                        is_selected=is_selected, has_lock=has_lock
                    )

    def _get_class_label_text(self, class_id: int, fallback_name: str = "") -> str:
        """获取用于画布显示的类别名称"""
        if class_id in self.class_names and self.class_names[class_id]:
            return str(self.class_names[class_id]).strip()
        return str(fallback_name or "").strip()

    def _get_annotation_label_anchor(self, ann_type: str, data: Dict) -> Optional[QPoint]:
        """计算标注类别名标签在控件坐标系下的锚点（左上角/最高顶点）"""
        if ann_type == 'bbox':
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            p1 = self.image_to_widget(x, y)
            p2 = self.image_to_widget(x + width, y + height)
            return QPoint(min(p1.x(), p2.x()), min(p1.y(), p2.y()))
        elif ann_type == 'polygon':
            pts = data.get('points', [])
            if not pts:
                return None
            widget_pts = [self.image_to_widget(pt['x'], pt['y']) for pt in pts if 'x' in pt and 'y' in pt]
            return min(widget_pts, key=lambda p: (p.y(), p.x())) if widget_pts else None
        elif ann_type == 'keypoint':
            kps = [kp for kp in data.get('keypoints', []) if kp.get('visible', 1)]
            if not kps:
                return None
            widget_pts = [self.image_to_widget(kp.get('x', 0), kp.get('y', 0)) for kp in kps]
            return min(widget_pts, key=lambda p: (p.y(), p.x())) if widget_pts else None
        elif ann_type == 'obb':
            pts = data.get('points', [])
            if pts and isinstance(pts[0], dict) and 'x' in pts[0] and 'y' in pts[0]:
                widget_pts = [self.image_to_widget(pt['x'], pt['y']) for pt in pts]
                return min(widget_pts, key=lambda p: (p.y(), p.x()))
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            angle = data.get('angle', 0)
            center = self.image_to_widget(x, y)
            half_w = width * self.image_scale / 2
            half_h = height * self.image_scale / 2
            corners = []
            for dx, dy in ((half_w, -half_h), (-half_w, -half_h), (-half_w, half_h), (half_w, half_h)):
                rx = dx * math.cos(angle) - dy * math.sin(angle)
                ry = dx * math.sin(angle) + dy * math.cos(angle)
                corners.append(QPoint(int(center.x() + rx), int(center.y() + ry)))
            return min(corners, key=lambda p: (p.y(), p.x()))
        return None

    def draw_class_label(
        self,
        painter: QPainter,
        anchor: QPoint,
        label_text: str,
        color: QColor,
        is_selected: bool = False,
        has_lock: bool = False,
    ):
        """在目标框左上角绘制类别名胶囊标签（贴顶自动内移防裁切，自适应文字深浅对比度）"""
        if not label_text or anchor is None:
            return

        painter.save()
        try:
            font = ui_font("Microsoft YaHei", 9, QFont.Weight.Bold)
            painter.setFont(font)
            fm = painter.fontMetrics()

            pad_x = max(4, ui_pixels(6))
            pad_y = max(2, ui_pixels(2))
            tag_w = fm.horizontalAdvance(label_text) + pad_x * 2
            tag_h = fm.height() + pad_y * 2

            tx = anchor.x()
            ty = anchor.y() - tag_h - 1

            # 靠近图片或画布上边缘时自动翻入框内左上角，并避让锁定小图标
            top_limit = max(2, self.image_offset.y())
            if ty < top_limit:
                ty = anchor.y() + 2
                if has_lock:
                    tx = anchor.x() + max(15, ui_pixels(16))

            # 防止右侧超出画布被截断
            max_x = max(2, self.width() - tag_w - 2)
            tx = max(2, min(tx, max_x))

            tag_rect = QRect(tx, ty, tag_w, tag_h)
            bg_color = QColor(color.red(), color.green(), color.blue(), 235 if is_selected else 205)
            border_color = color.darker(115)

            painter.setPen(QPen(border_color, 1))
            painter.setBrush(QBrush(bg_color))
            painter.drawRoundedRect(tag_rect, 3, 3)

            # 根据背景色感知亮度自动选择深色或白色文字，保证任意类别颜色下清晰可读
            luminance = 0.299 * color.red() + 0.587 * color.green() + 0.114 * color.blue()
            text_color = QColor(20, 20, 20) if luminance > 165 else QColor(255, 255, 255)
            painter.setPen(text_color)
            painter.drawText(tag_rect, Qt.AlignmentFlag.AlignCenter, label_text)
        finally:
            painter.restore()

    def draw_lock_marker(self, painter: QPainter, ann_type: str, data: Dict):
        """在锁定的标注角落绘制小锁图标"""
        # 锚点: bbox左上角 / 其他取第一个点
        if ann_type == 'bbox':
            ax, ay = data.get('x', 0), data.get('y', 0)
        elif ann_type == 'polygon':
            pts = data.get('points', [])
            if not pts:
                return
            ax, ay = pts[0]['x'], pts[0]['y']
        elif ann_type == 'obb':
            pts = data.get('points', [])
            if not pts:
                return
            ax, ay = pts[0]['x'], pts[0]['y']
        elif ann_type == 'keypoint':
            kps = data.get('keypoints', [])
            if not kps:
                return
            ax, ay = kps[0].get('x', 0), kps[0].get('y', 0)
        else:
            return

        pos = self.image_to_widget(ax, ay)
        s = 9  # 锁体尺寸
        bx, by = pos.x() + 3, pos.y() + 3

        # 锁体
        body = QColor(245, 166, 35)
        painter.setPen(QPen(QColor(120, 70, 0), 1))
        painter.setBrush(QBrush(body))
        painter.drawRect(bx, by + s // 2, s, s - s // 3)
        # 锁环
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawArc(bx + s // 4, by - s // 3 + s // 2,
                        s // 2, s // 2, 0, 180 * 16)
        # 锁孔
        painter.setBrush(QBrush(QColor(120, 70, 0)))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(bx + s / 2, by + 2 * s / 3), 1.5, 1.5)

    def draw_lock_banner(self, painter: QPainter):
        """整图锁定醒目横幅: 橙色描边 + 顶部徽标"""
        rect = self.rect()

        # 四周加粗橙色描边
        painter.setPen(QPen(QColor(255, 140, 0), 4))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(rect.adjusted(2, 2, -2, -2))

        # 顶部中央徽标
        font = ui_font("Microsoft YaHei", 11, QFont.Weight.Bold)
        painter.setFont(font)
        if getattr(self, 'image_modified', False):
            text = "✏️🔒 已复标修正并锁定 · AI不再覆盖此图"
        else:
            text = "🔒 已确认锁定 · AI不再覆盖此图"
        fm = painter.fontMetrics()
        w = fm.horizontalAdvance(text) + 28
        h = fm.height() + 14
        badge = QRect(
            rect.center().x() - w // 2, rect.top() + 12, w, h
        )
        painter.setPen(QPen(QColor(120, 70, 0), 2))
        painter.setBrush(QBrush(QColor(255, 140, 0, 230)))
        painter.drawRoundedRect(badge, 8, 8)
        painter.setPen(QColor(40, 22, 0))
        painter.drawText(badge, Qt.AlignmentFlag.AlignCenter, text)
    
    def draw_bbox(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制矩形框"""
        x = data.get('x', 0)
        y = data.get('y', 0)
        width = data.get('width', 0)
        height = data.get('height', 0)
        
        top_left = self.image_to_widget(x, y)
        bottom_right = self.image_to_widget(x + width, y + height)
        
        rect = QRect(top_left, bottom_right)
        painter.drawRect(rect)
        
        # 如果是选中状态，绘制调整手柄
        if is_selected:
            self.draw_resize_handles(painter, rect)
    
    def draw_polygon(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制多边形"""
        points = data.get('points', [])
        if len(points) < 3:
            return
        
        widget_points = []
        for point in points:
            widget_point = self.image_to_widget(point['x'], point['y'])
            widget_points.append(widget_point)
        
        # 绘制多边形
        for i in range(len(widget_points)):
            p1 = widget_points[i]
            p2 = widget_points[(i + 1) % len(widget_points)]
            painter.drawLine(p1, p2)
        
        # 绘制顶点
        for point in widget_points:
            painter.drawEllipse(point, 4, 4)
    
    def draw_keypoints(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制关键点"""
        keypoints = data.get('keypoints', [])
        if not keypoints:
            return
        
        # 绘制关键点之间的连接线
        if len(keypoints) > 1:
            for i in range(len(keypoints) - 1):
                kp1 = keypoints[i]
                kp2 = keypoints[i + 1]
                if kp1.get('visible', 1) and kp2.get('visible', 1):
                    p1 = self.image_to_widget(kp1['x'], kp1['y'])
                    p2 = self.image_to_widget(kp2['x'], kp2['y'])
                    painter.drawLine(p1, p2)
        
        # 绘制关键点
        for kp in keypoints:
            if kp.get('visible', 1):
                point = self.image_to_widget(kp['x'], kp['y'])
                # 绘制关键点圆圈
                radius = 6 if is_selected else 4
                painter.drawEllipse(point, radius, radius)
                # 绘制关键点索引（如果有）
                if 'id' in kp:
                    painter.drawText(point.x() + radius + 2, point.y() - radius, str(kp['id']))
    
    def draw_obb(self, painter: QPainter, data: Dict, is_selected: bool):
        """绘制旋转矩形"""
        x = data.get('x', 0)
        y = data.get('y', 0)
        width = data.get('width', 0)
        height = data.get('height', 0)
        angle = data.get('angle', 0)  # 旋转角度（弧度）
        
        # 计算矩形的四个顶点
        center = self.image_to_widget(x, y)
        half_width = width * self.image_scale / 2
        half_height = height * self.image_scale / 2
        
        # 计算四个顶点
        import math
        points = []
        # 定义四个顶点相对于中心点的基础偏移（未旋转时）
        # 顺序：p1, p2, p3, p4 对应 create_obb_annotation_with_points 中的四个点
        base_offsets = [
            (half_width, -half_height),  # p1: 右上
            (-half_width, -half_height),  # p2: 左上
            (-half_width, half_height),   # p3: 左下
            (half_width, half_height)     # p4: 右下
        ]
        
        for dx, dy in base_offsets:
            # 应用旋转
            rotated_dx = dx * math.cos(angle) - dy * math.sin(angle)
            rotated_dy = dx * math.sin(angle) + dy * math.cos(angle)
            # 计算最终顶点坐标
            vertex_x = center.x() + rotated_dx
            vertex_y = center.y() + rotated_dy
            points.append(QPoint(int(vertex_x), int(vertex_y)))
        
        # 绘制旋转矩形
        for i in range(4):
            p1 = points[i]
            p2 = points[(i + 1) % 4]
            painter.drawLine(p1, p2)
        
        # 如果是选中状态，绘制调整手柄
        if is_selected:
            for point in points:
                handle_rect = QRect(
                    point.x() - self.handle_size // 2,
                    point.y() - self.handle_size // 2,
                    self.handle_size,
                    self.handle_size
                )
                painter.drawRect(handle_rect)
    
    def draw_resize_handles(self, painter: QPainter, rect: QRect):
        """绘制调整大小的手柄"""
        handle_size = 8
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        
        # 四个角
        corners = [
            rect.topLeft(),
            rect.topRight(),
            rect.bottomLeft(),
            rect.bottomRight()
        ]
        
        for corner in corners:
            handle_rect = QRect(
                corner.x() - handle_size // 2,
                corner.y() - handle_size // 2,
                handle_size,
                handle_size
            )
            painter.drawRect(handle_rect)
    
    def _get_active_draw_color(self) -> QColor:
        """获取当前选中类别对应的绘制颜色（无类别时回退主色）"""
        raw = self.class_colors.get(self.current_class_id)
        if raw:
            c = QColor(raw) if isinstance(raw, str) else QColor(raw)
            if c.isValid():
                return c
        return QColor(COLORS['primary'])

    def draw_drawing_rectangle(self, painter: QPainter):
        """绘制正在绘制的矩形（同步实时预览当前类别名）"""
        draw_color = self._get_active_draw_color()
        pen = QPen(draw_color)
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        
        fill_color = QColor(draw_color)
        fill_color.setAlpha(50)
        painter.setBrush(QBrush(fill_color))
        
        rect = QRect(self.start_point, self.current_point)
        painter.drawRect(rect)

        label_text = self._get_class_label_text(self.current_class_id)
        if label_text:
            norm_rect = rect.normalized()
            self.draw_class_label(painter, norm_rect.topLeft(), label_text, draw_color, is_selected=True)
    
    def draw_drawing_polygon(self, painter: QPainter):
        """绘制正在绘制的多边形（同步实时预览当前类别名）"""
        draw_color = self._get_active_draw_color()
        pen = QPen(draw_color)
        pen.setWidth(2)
        painter.setPen(pen)
        
        # 绘制已有点
        for point in self.polygon_points:
            painter.drawEllipse(point, 4, 4)
        
        # 绘制连线
        if len(self.polygon_points) > 1:
            for i in range(len(self.polygon_points) - 1):
                painter.drawLine(self.polygon_points[i], self.polygon_points[i + 1])
        
        # 绘制从最后一点到当前鼠标的线
        if len(self.polygon_points) > 0 and self.current_point:
            painter.drawLine(self.polygon_points[-1], self.current_point)

        label_text = self._get_class_label_text(self.current_class_id)
        if label_text and self.polygon_points:
            anchor = min(self.polygon_points, key=lambda p: (p.y(), p.x()))
            self.draw_class_label(painter, anchor, label_text, draw_color, is_selected=True)
    
    def draw_drawing_obb(self, painter: QPainter):
        """绘制正在绘制的旋转矩形"""
        if len(self.obb_points) == 0:
            return
        
        # 设置绘制样式
        pen = QPen(QColor(COLORS['primary']))
        pen.setWidth(2)
        pen.setStyle(Qt.PenStyle.DashLine)
        painter.setPen(pen)
        
        # 使用主色调的半透明版本
        primary_color = QColor(COLORS['primary'])
        primary_color.setAlpha(50)
        painter.setBrush(QBrush(primary_color))
        
        if len(self.obb_points) == 1:
            # 只绘制第一个点
            point = self.obb_points[0]
            painter.drawEllipse(point, 4, 4)
        elif len(self.obb_points) == 2:
            # 绘制第一条边
            p1 = self.obb_points[0]
            p2 = self.obb_points[1]
            painter.drawLine(p1, p2)
            painter.drawEllipse(p1, 4, 4)
            painter.drawEllipse(p2, 4, 4)
            # 如果有鼠标位置，绘制辅助线、垂线和半透明OBB
            if self.current_point:
                # 计算垂直线
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                # 计算垂线方向向量
                perp_dx = -dy
                perp_dy = dx
                # 归一化
                length = math.sqrt(perp_dx ** 2 + perp_dy ** 2)
                if length > 0:
                    perp_dx /= length
                    perp_dy /= length
                # 绘制垂线
                perp_p1 = QPoint(int(p2.x() + perp_dx * 100), int(p2.y() + perp_dy * 100))
                perp_p2 = QPoint(int(p2.x() - perp_dx * 100), int(p2.y() - perp_dy * 100))
                painter.setPen(QPen(QColor(COLORS['primary']), 1, Qt.PenStyle.DotLine))
                painter.drawLine(perp_p1, perp_p2)
                # 计算垂线上的点（鼠标位置在垂线上的投影）
                mouse_dx = self.current_point.x() - p2.x()
                mouse_dy = self.current_point.y() - p2.y()
                # 计算投影长度
                proj_length = mouse_dx * perp_dx + mouse_dy * perp_dy
                # 计算垂线上的点
                perp_point = QPoint(int(p2.x() + perp_dx * proj_length), int(p2.y() + perp_dy * proj_length))
                # 计算第四个点
                p4 = QPoint(int(p1.x() + (perp_point.x() - p2.x())), int(p1.y() + (perp_point.y() - p2.y())))
                # 绘制OBB
                painter.setPen(QPen(QColor(COLORS['primary']), 2, Qt.PenStyle.DashLine))
                painter.drawLine(p1, p2)
                painter.drawLine(p2, perp_point)
                painter.drawLine(perp_point, p4)
                painter.drawLine(p4, p1)
                # 绘制所有点
                painter.drawEllipse(p1, 4, 4)
                painter.drawEllipse(p2, 4, 4)
                painter.drawEllipse(perp_point, 4, 4)
                painter.drawEllipse(p4, 4, 4)
                # 绘制鼠标到垂点的辅助线
                painter.setPen(QPen(QColor(COLORS['primary']), 1, Qt.PenStyle.DotLine))
                painter.drawLine(self.current_point, perp_point)
                painter.setPen(QPen(QColor(COLORS['primary']), 2, Qt.PenStyle.DashLine))
        elif len(self.obb_points) == 3:
            # 绘制完整的OBB
            p1 = self.obb_points[0]
            p2 = self.obb_points[1]
            p3 = self.obb_points[2]
            
            # 计算第四个点
            dx = p3.x() - p2.x()
            dy = p3.y() - p2.y()
            p4 = QPoint(p1.x() + dx, p1.y() + dy)
            
            # 绘制OBB
            painter.drawLine(p1, p2)
            painter.drawLine(p2, p3)
            painter.drawLine(p3, p4)
            painter.drawLine(p4, p1)
            
            # 绘制所有点
            painter.drawEllipse(p1, 4, 4)
            painter.drawEllipse(p2, 4, 4)
            painter.drawEllipse(p3, 4, 4)
            painter.drawEllipse(p4, 4, 4)

        label_text = self._get_class_label_text(self.current_class_id)
        if label_text and self.obb_points:
            anchor = min(self.obb_points, key=lambda p: (p.y(), p.x()))
            self.draw_class_label(
                painter, anchor, label_text, self._get_active_draw_color(), is_selected=True
            )
    
    def draw_guide_lines(self, painter: QPainter):
        """绘制鼠标辅助线"""
        if not self.current_image or not self.current_point:
            return
        
        # 获取图像区域
        img_rect = self.rect()
        scaled_width = int(self.current_image.width() * self.image_scale)
        scaled_height = int(self.current_image.height() * self.image_scale)
        
        # 计算图像显示区域的边界
        img_left = self.image_offset.x()
        img_top = self.image_offset.y()
        img_right = img_left + scaled_width
        img_bottom = img_top + scaled_height
        
        # 获取鼠标位置
        mouse_x = self.current_point.x()
        mouse_y = self.current_point.y()
        
        # 检查鼠标是否在图像区域内
        if not (img_left <= mouse_x <= img_right and img_top <= mouse_y <= img_bottom):
            return
        
        # 设置辅助线样式
        pen = QPen(QColor(255, 255, 255, 150))  # 半透明白色
        pen.setStyle(Qt.PenStyle.DashLine)
        pen.setWidth(2)  # 加粗
        painter.setPen(pen)
        
        # 绘制水平线（穿过鼠标）
        painter.drawLine(img_left, mouse_y, img_right, mouse_y)
        
        # 绘制垂直线（穿过鼠标）
        painter.drawLine(mouse_x, img_top, mouse_x, img_bottom)
    
    def mousePressEvent(self, event: QMouseEvent):
        """鼠标按下事件"""
        if self.current_image is None:
            return
        
        # 批量处理模式：点击选择像素点
        if self.batch_process_mode and event.button() == Qt.MouseButton.LeftButton:
            # 将鼠标位置转换为图像坐标
            img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
            
            # 检查是否在图像范围内
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            
            if 0 <= img_x <= img_width and 0 <= img_y <= img_height:
                # 添加像素点
                self.batch_process_points.append((int(img_x), int(img_y)))
                
                # 更新对话框中的显示
                if self.batch_process_dialog:
                    self.batch_process_dialog.add_point(int(img_x), int(img_y))
                
                self.update()
            return
        
        # SAM模式处理
        if self.sam_mode_active:
            if event.button() == Qt.MouseButton.LeftButton:
                # 左键添加点提示
                img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
                if self.current_image:
                    img_w = self.current_image.width()
                    img_h = self.current_image.height()
                    img_x = max(0, min(img_x, img_w - 1))
                    img_y = max(0, min(img_y, img_h - 1))
                self.sam_points.append((img_x, img_y))
                self.update()
                
                # 检查是否按住Ctrl
                is_ctrl_pressed = event.modifiers() & Qt.KeyboardModifier.ControlModifier
                
                # 如果没有按住Ctrl，立即进行推理
                if not is_ctrl_pressed:
                    if self._sam_auto_infer_enabled():
                        project_type = getattr(self, 'sam_project_type', 'segment')
                        self.run_sam_inference(project_type)
                return
            elif event.button() == Qt.MouseButton.RightButton:
                # 右键开始画框
                self.sam_mode = 'bbox'
                self.sam_drawing_bbox = True
                self.sam_start_point = event.pos()
                self.sam_current_point = event.pos()
                return
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.current_tool == 'rectangle':
                self.drawing = True
                self.start_point = event.pos()
                self.current_point = event.pos()
            elif self.current_tool == 'polygon':
                # 检查是否是吸附到初始点的情况
                if len(self.polygon_points) > 0:
                    initial_point = self.polygon_points[0]
                    distance = (event.pos() - initial_point).manhattanLength()
                    # 如果距离小于吸附阈值，完成多边形标注
                    if distance < 10:
                        self.create_polygon_annotation()
                        return
                # 否则添加新点
                self.polygon_points.append(event.pos())
                self.update()
            elif self.current_tool == 'keypoint':
                # 转换为图像坐标
                x, y = self.widget_to_image(event.pos().x(), event.pos().y())
                
                # 限制坐标在图像范围内
                if self.current_image:
                    img_width = self.current_image.width()
                    img_height = self.current_image.height()
                    x = max(0, min(x, img_width))
                    y = max(0, min(y, img_height))
                
                # 创建关键点标注
                keypoint = {
                    'x': x,
                    'y': y,
                    'visible': 1
                }
                
                # 添加到关键点列表
                self.keypoints.append(keypoint)
                
                # 如果已经有多个关键点，创建标注
                if len(self.keypoints) >= 1:
                    annotation = {
                        'type': 'keypoint',
                        'class_id': self.current_class_id,
                        'data': {
                            'keypoints': self.keypoints.copy()
                        }
                    }
                    self.annotation_created.emit(annotation)
                    # 重置关键点列表，准备下一个标注
                    self.keypoints = []
            elif self.current_tool == 'obb':
                if self.obb_state == 0:
                    # 第一步：确定第一个点（固定角度的起始点）
                    self.obb_state = 1
                    self.obb_points = [event.pos()]
                    self.update()
                elif self.obb_state == 1:
                    # 第二步：确定第一条边的另一个端点
                    self.obb_state = 2
                    self.obb_points.append(event.pos())
                    self.update()
                elif self.obb_state == 2:
                    # 第三步：确定邻边的另一个端点，完成OBB创建
                    # 计算垂线上的点（与draw_drawing_obb方法相同的逻辑）
                    p1 = self.obb_points[0]
                    p2 = self.obb_points[1]
                    # 计算垂直线
                    dx = p2.x() - p1.x()
                    dy = p2.y() - p1.y()
                    # 计算垂线方向向量
                    perp_dx = -dy
                    perp_dy = dx
                    # 归一化
                    length = math.sqrt(perp_dx ** 2 + perp_dy ** 2)
                    if length > 0:
                        perp_dx /= length
                        perp_dy /= length
                    # 计算从p2到当前鼠标的向量
                    mouse_dx = event.pos().x() - p2.x()
                    mouse_dy = event.pos().y() - p2.y()
                    # 计算投影长度
                    proj_length = mouse_dx * perp_dx + mouse_dy * perp_dy
                    # 计算垂线上的点
                    perp_point = QPoint(int(p2.x() + perp_dx * proj_length), int(p2.y() + perp_dy * proj_length))
                    # 将垂线上的点添加到obb_points列表中
                    self.obb_points.append(perp_point)
                    # 创建OBB标注
                    self.create_obb_annotation_with_points()
                    # 重置状态
                    self.obb_state = 0
                    self.obb_points = []
                    self.update()
            elif self.current_tool == 'move':
                # 检查是否点击了调整手柄
                handle_info = self.get_resize_handle_at(event.pos())
                if handle_info:
                    self.resizing = True
                    self.resize_handle = handle_info['handle']
                    self.selected_annotation_id = handle_info['annotation_id']
                    # 获取选中的标注数据
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation:
                        self.drag_start = event.pos()
                        self.resize_start_rect = annotation['data'].copy()
                        self.annotation_selected.emit(self.selected_annotation_id)
                else:
                    # 检查是否点击了某个标注
                    clicked_annotation = self.get_annotation_at(event.pos())
                    if clicked_annotation:
                        self.selected_annotation_id = clicked_annotation['id']
                        ann_type = clicked_annotation.get('type', 'bbox')
                        ann_data = clicked_annotation.get('data', {})

                        # 点到多边形顶点：拖动该点，而不是拖动整个多边形
                        if ann_type == 'polygon':
                            # 先用更宽松的“最近点磁吸”选择顶点，提升易用性
                            vertex_idx = self.get_nearest_polygon_vertex_at(event.pos(), clicked_annotation, self.vertex_snap_radius)
                            if vertex_idx is None:
                                vertex_idx = self.get_polygon_vertex_at(event.pos(), clicked_annotation)
                            if vertex_idx is not None:
                                self.dragging_vertex = True
                                self.drag_vertex_index = vertex_idx
                                self.drag_start = event.pos()
                                self.drag_start_annotation = {
                                    'points': [p.copy() for p in ann_data.get('points', [])]
                                }
                                self.annotation_selected.emit(self.selected_annotation_id)
                                self.update()
                                return

                        # 否则：拖动整个标注
                        self.dragging = True
                        self.drag_start = event.pos()
                        # 深度复制标注数据，特别是多边形的点
                        if ann_type == 'polygon' and 'points' in ann_data:
                            # 对多边形点进行深度复制
                            self.drag_start_annotation = {'points': [p.copy() for p in ann_data.get('points', [])]}
                        elif ann_type == 'keypoint' and 'keypoints' in ann_data:
                            # 对关键点进行深度复制
                            self.drag_start_annotation = {'keypoints': [kp.copy() for kp in ann_data.get('keypoints', [])]}
                        else:
                            self.drag_start_annotation = ann_data.copy() if isinstance(ann_data, dict) else {}
                        self.annotation_selected.emit(self.selected_annotation_id)
                    else:
                        # 没有点击标注，开始平移图片
                        self.panning = True
                        self.pan_start = event.pos()
                        self.pan_start_offset = QPoint(self.image_offset)
                        self.setCursor(Qt.CursorShape.ClosedHandCursor)
        
        self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """鼠标移动事件"""
        # SAM模式：更新框绘制
        if self.sam_mode_active and self.sam_drawing_bbox:
            self.sam_current_point = event.pos()
            self.update()
            return
        
        # 多边形标注：添加初始点吸附效果
        if self.current_tool == 'polygon' and len(self.polygon_points) > 0:
            initial_point = self.polygon_points[0]
            distance = (event.pos() - initial_point).manhattanLength()
            # 吸附阈值：10像素
            if distance < 10:
                self.current_point = initial_point
            else:
                self.current_point = event.pos()
        else:
            self.current_point = event.pos()
        
        if self.drawing and self.current_tool == 'rectangle':
            self.update()
        elif self.current_tool == 'polygon':
            self.update()
        elif self.current_tool == 'move':
            if self.resizing and self.resize_handle and self.resize_start_rect:
                # 调整大小
                self.resize_annotation(event.pos())
                self.update()
            elif self.dragging_vertex and self.selected_annotation_id is not None:
                # 拖动多边形的某个顶点
                annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                if annotation and annotation.get('type') == 'polygon':
                    points = annotation.get('data', {}).get('points', [])
                    idx = self.drag_vertex_index
                    if idx is not None and 0 <= idx < len(points):
                        img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
                        if self.current_image:
                            img_w = self.current_image.width()
                            img_h = self.current_image.height()
                            img_x = max(0, min(img_x, img_w))
                            img_y = max(0, min(img_y, img_h))
                        points[idx]['x'] = img_x
                        points[idx]['y'] = img_y
                self.update()
            elif self.dragging and self.drag_start and self.drag_start_annotation:
                # 拖动标注
                self.drag_annotation(event.pos())
                self.update()
            elif self.panning and self.pan_start and self.pan_start_offset:
                # 平移图片
                delta = event.pos() - self.pan_start
                self.image_offset = QPoint(
                    self.pan_start_offset.x() + delta.x(),
                    self.pan_start_offset.y() + delta.y()
                )
                self.update()
            else:
                # 检查鼠标是否在手柄上，改变光标
                handle_info = self.get_resize_handle_at(event.pos())
                if handle_info:
                    self.setCursor(Qt.CursorShape.SizeAllCursor)
                else:
                    annotation = self.get_annotation_at(event.pos())
                    if annotation:
                        # 轻微磁吸：靠近多边形顶点时，更容易“抓住点”
                        if annotation.get('type') == 'polygon':
                            vidx = self.get_nearest_polygon_vertex_at(event.pos(), annotation, self.vertex_snap_radius)
                            if vidx is not None:
                                self.hover_vertex = (annotation.get('id'), vidx)
                                self.setCursor(Qt.CursorShape.PointingHandCursor)
                            else:
                                self.hover_vertex = None
                                self.setCursor(Qt.CursorShape.OpenHandCursor)
                        else:
                            self.hover_vertex = None
                        self.setCursor(Qt.CursorShape.OpenHandCursor)
                    else:
                        self.hover_vertex = None
                        # 图片放大时可以平移
                        if self.image_scale > 1.0:
                            self.setCursor(Qt.CursorShape.OpenHandCursor)
                        else:
                            self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # 非移动工具时，设置为箭头光标
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        # 更新鼠标位置信息
        if self.current_image:
            img_x, img_y = self.widget_to_image(event.pos().x(), event.pos().y())
        
        # 触发重绘以显示辅助线
        self.update()
    
    def mouseReleaseEvent(self, event: QMouseEvent):
        """鼠标释放事件"""
        # SAM模式：右键释放完成框绘制
        if self.sam_mode_active and event.button() == Qt.MouseButton.RightButton:
            if self.sam_drawing_bbox:
                self.sam_drawing_bbox = False
                # 转换框坐标到图像坐标
                x1, y1 = self.widget_to_image(self.sam_start_point.x(), self.sam_start_point.y())
                x2, y2 = self.widget_to_image(self.sam_current_point.x(), self.sam_current_point.y())
                # 确保坐标顺序正确
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                self.sam_bboxes.append((x1, y1, x2, y2))
                self.update()
                # 运行推理
                if self._sam_auto_infer_enabled():
                    project_type = getattr(self, 'sam_project_type', 'segment')
                    self.run_sam_inference(project_type)
                return
        
        if event.button() == Qt.MouseButton.LeftButton:
            if self.drawing and self.current_tool == 'rectangle':
                self.drawing = False
                self.create_rectangle_annotation()
            elif self.drawing and self.current_tool == 'obb':
                self.drawing = False
                self.create_obb_annotation()
            elif self.resizing:
                start_rect = self.resize_start_rect
                self.resizing = False
                self.resize_handle = None
                self.resize_start_rect = None
                # 仅在尺寸实际发生变化时发送修改信号
                if self.selected_annotation_id is not None:
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation and annotation.get('data') != start_rect:
                        self.annotation_modified.emit(self.selected_annotation_id, annotation['data'])
            elif self.dragging_vertex:
                start_ann = self.drag_start_annotation
                self.dragging_vertex = False
                self.drag_vertex_index = None
                self.drag_start = None
                self.drag_start_annotation = None
                # 仅在顶点坐标实际发生变化时发送修改信号
                if self.selected_annotation_id is not None:
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation and annotation.get('data') != start_ann:
                        self.annotation_modified.emit(self.selected_annotation_id, annotation['data'])
            elif self.dragging:
                start_ann = self.drag_start_annotation
                self.dragging = False
                self.drag_start = None
                self.drag_start_annotation = None
                # 仅在框位置实际拖动变化时发送修改信号（避免单击选中即误触发锁定）
                if self.selected_annotation_id is not None:
                    annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
                    if annotation and annotation.get('data') != start_ann:
                        self.annotation_modified.emit(self.selected_annotation_id, annotation['data'])
            elif self.panning:
                self.panning = False
                self.pan_start = None
                self.pan_start_offset = None
        
        # 根据当前状态设置光标
        if self.current_tool == 'move':
            if self.image_scale > 1.0:
                self.setCursor(Qt.CursorShape.OpenHandCursor)
            else:
                self.setCursor(Qt.CursorShape.ArrowCursor)
        else:
            # 非移动工具时，始终设置为箭头光标
            self.setCursor(Qt.CursorShape.ArrowCursor)
        
        self.update()
    
    def get_annotation_at(self, pos: QPoint) -> Optional[Dict]:
        """获取指定位置的标注"""
        for annotation in reversed(self.annotations):
            if self.is_point_in_annotation(pos, annotation):
                return annotation
        return None

    def get_polygon_vertex_at(self, pos: QPoint, annotation: Dict) -> Optional[int]:
        """如果pos命中多边形顶点，返回顶点索引，否则None（控件坐标判定）"""
        if not annotation or annotation.get('type') != 'polygon':
            return None
        data = annotation.get('data', {})
        points = data.get('points', [])
        if not points:
            return None

        for idx, pt in enumerate(points):
            if not isinstance(pt, dict) or 'x' not in pt or 'y' not in pt:
                continue
            wp = self.image_to_widget(pt['x'], pt['y'])
            if (pos - wp).manhattanLength() <= self.vertex_hit_radius:
                return idx
        return None

    def get_nearest_polygon_vertex_at(self, pos: QPoint, annotation: Dict, radius: int) -> Optional[int]:
        """返回radius范围内最近的多边形顶点索引（控件坐标），否则None。

        用于提供轻微“磁吸/辅助命中”，让用户更容易选中顶点。
        """
        if not annotation or annotation.get('type') != 'polygon':
            return None
        data = annotation.get('data', {})
        points = data.get('points', [])
        if not points:
            return None

        best_idx = None
        best_dist = None
        for idx, pt in enumerate(points):
            if not isinstance(pt, dict) or 'x' not in pt or 'y' not in pt:
                continue
            wp = self.image_to_widget(pt['x'], pt['y'])
            d = (pos - wp).manhattanLength()
            if d <= radius and (best_dist is None or d < best_dist):
                best_dist = d
                best_idx = idx
        return best_idx
    
    def get_resize_handle_at(self, pos: QPoint) -> Optional[Dict]:
        """获取指定位置的调整手柄信息"""
        if self.selected_annotation_id is None:
            return None
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation or annotation.get('type') != 'bbox':
            return None
        
        data = annotation['data']
        x, y, width, height = data['x'], data['y'], data['width'], data['height']
        
        # 转换为控件坐标
        top_left = self.image_to_widget(x, y)
        bottom_right = self.image_to_widget(x + width, y + height)
        
        # 四个角的手柄
        handles = {
            'top_left': QRect(top_left.x() - self.handle_size, top_left.y() - self.handle_size, 
                             self.handle_size * 2, self.handle_size * 2),
            'top_right': QRect(bottom_right.x() - self.handle_size, top_left.y() - self.handle_size,
                              self.handle_size * 2, self.handle_size * 2),
            'bottom_left': QRect(top_left.x() - self.handle_size, bottom_right.y() - self.handle_size,
                                self.handle_size * 2, self.handle_size * 2),
            'bottom_right': QRect(bottom_right.x() - self.handle_size, bottom_right.y() - self.handle_size,
                                 self.handle_size * 2, self.handle_size * 2),
        }
        
        for handle_name, handle_rect in handles.items():
            if handle_rect.contains(pos):
                return {'handle': handle_name, 'annotation_id': annotation['id']}
        
        return None
    
    def drag_annotation(self, pos: QPoint):
        """拖动标注"""
        if self.drag_start is None or self.drag_start_annotation is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation:
            return
        
        # 计算拖动偏移（控件坐标）
        delta_x = pos.x() - self.drag_start.x()
        delta_y = pos.y() - self.drag_start.y()
        
        # 转换为图像坐标偏移
        img_delta_x = delta_x / self.image_scale
        img_delta_y = delta_y / self.image_scale
        
        ann_type = annotation.get('type', 'bbox')
        data = annotation['data']
        
        # 获取图像尺寸
        img_width = self.current_image.width() if self.current_image else 0
        img_height = self.current_image.height() if self.current_image else 0
        
        if ann_type == 'bbox':
            new_x = self.drag_start_annotation['x'] + img_delta_x
            new_y = self.drag_start_annotation['y'] + img_delta_y
            width = data.get('width', 0)
            height = data.get('height', 0)
            
            # 限制在图像范围内
            data['x'] = max(0, min(new_x, img_width - width))
            data['y'] = max(0, min(new_y, img_height - height))
        elif ann_type == 'polygon':
            # 确保drag_start_annotation包含正确的点数据
            if 'points' in self.drag_start_annotation and 'points' in data:
                start_points = self.drag_start_annotation['points']
                # 确保点的数量匹配
                if len(start_points) == len(data['points']):
                    for i, point in enumerate(data['points']):
                        if i < len(start_points):
                            # 直接使用原始点加上偏移量，避免累积误差
                            new_x = start_points[i]['x'] + img_delta_x
                            new_y = start_points[i]['y'] + img_delta_y
                            # 限制在图像范围内
                            point['x'] = max(0, min(new_x, img_width))
                            point['y'] = max(0, min(new_y, img_height))
        elif ann_type == 'obb':
            # 确保drag_start_annotation包含正确的OBB数据
            if 'x' in self.drag_start_annotation and 'y' in self.drag_start_annotation:
                new_x = self.drag_start_annotation['x'] + img_delta_x
                new_y = self.drag_start_annotation['y'] + img_delta_y
                # 限制在图像范围内
                data['x'] = max(0, min(new_x, img_width))
                data['y'] = max(0, min(new_y, img_height))
        elif ann_type == 'keypoint':
            # 确保drag_start_annotation包含正确的关键点数据
            if 'keypoints' in self.drag_start_annotation and 'keypoints' in data:
                start_keypoints = self.drag_start_annotation['keypoints']
                # 确保关键点数量匹配
                if len(start_keypoints) == len(data['keypoints']):
                    for i, kp in enumerate(data['keypoints']):
                        if i < len(start_keypoints):
                            new_x = start_keypoints[i]['x'] + img_delta_x
                            new_y = start_keypoints[i]['y'] + img_delta_y
                            # 限制在图像范围内
                            kp['x'] = max(0, min(new_x, img_width))
                            kp['y'] = max(0, min(new_y, img_height))
    
    def resize_annotation(self, pos: QPoint):
        """调整标注大小"""
        if self.resize_handle is None or self.resize_start_rect is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.selected_annotation_id), None)
        if not annotation:
            return
        
        data = annotation['data']
        start = self.resize_start_rect
        
        # 将鼠标位置转换为图像坐标
        img_x, img_y = self.widget_to_image(pos.x(), pos.y())
        
        # 限制鼠标位置在图像范围内
        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            img_x = max(0, min(img_x, img_width))
            img_y = max(0, min(img_y, img_height))
        
        if self.resize_handle == 'top_left':
            new_x = min(img_x, start['x'] + start['width'])
            new_y = min(img_y, start['y'] + start['height'])
            data['x'] = max(0, new_x)
            data['y'] = max(0, new_y)
            data['width'] = start['x'] + start['width'] - data['x']
            data['height'] = start['y'] + start['height'] - data['y']
        elif self.resize_handle == 'top_right':
            new_y = min(img_y, start['y'] + start['height'])
            data['x'] = start['x']
            data['y'] = max(0, new_y)
            data['width'] = min(img_x, img_width) - start['x'] if self.current_image else img_x - start['x']
            data['height'] = start['y'] + start['height'] - data['y']
        elif self.resize_handle == 'bottom_left':
            new_x = min(img_x, start['x'] + start['width'])
            data['x'] = max(0, new_x)
            data['y'] = start['y']
            data['width'] = start['x'] + start['width'] - data['x']
            data['height'] = min(img_y, img_height) - start['y'] if self.current_image else img_y - start['y']
        elif self.resize_handle == 'bottom_right':
            data['x'] = start['x']
            data['y'] = start['y']
            data['width'] = min(img_x, img_width) - start['x'] if self.current_image else img_x - start['x']
            data['height'] = min(img_y, img_height) - start['y'] if self.current_image else img_y - start['y']
        
        # 确保宽度和高度为正且不超过图像范围
        if data['width'] < 0:
            data['x'] += data['width']
            data['width'] = abs(data['width'])
        if data['height'] < 0:
            data['y'] += data['height']
            data['height'] = abs(data['height'])
        
        # 最终限制在图像范围内
        if self.current_image:
            data['x'] = max(0, min(data['x'], img_width))
            data['y'] = max(0, min(data['y'], img_height))
            data['width'] = min(data['width'], img_width - data['x'])
            data['height'] = min(data['height'], img_height - data['y'])
    
    def mouseDoubleClickEvent(self, event: QMouseEvent):
        """鼠标双击事件 - 完成多边形绘制"""
        if self.current_tool == 'polygon' and len(self.polygon_points) >= 3:
            self.create_polygon_annotation()
    
    def wheelEvent(self, event: QWheelEvent):
        """鼠标滚轮事件 - 缩放"""
        if self.current_image is None:
            return
        
        # 获取鼠标位置
        mouse_pos = event.position().toPoint()
        
        # 计算缩放前鼠标对应的图像坐标
        img_x_before = (mouse_pos.x() - self.image_offset.x()) / self.image_scale
        img_y_before = (mouse_pos.y() - self.image_offset.y()) / self.image_scale
        
        # 计算缩放因子
        delta = event.angleDelta().y()
        zoom_factor = 1.1 if delta > 0 else 0.9
        
        # 应用缩放
        new_scale = self.image_scale * zoom_factor
        new_scale = max(0.1, min(5.0, new_scale))  # 限制缩放范围
        
        # 调整偏移量，使鼠标位置对应的图像点保持不变
        self.image_offset = QPoint(
            int(mouse_pos.x() - img_x_before * new_scale),
            int(mouse_pos.y() - img_y_before * new_scale)
        )
        self.image_scale = new_scale
        
        self.update()
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件"""
        from PyQt6.QtCore import QSettings
        
        # 获取快捷键设置
        settings = QSettings("Better-YOLO", "Settings")
        reset_view_key = settings.value("reset_view_shortcut", "R").upper()
        
        # 重置视图快捷键
        if event.text().upper() == reset_view_key:
            self.reset_view()
            self.update()
            return
        
        if event.key() == Qt.Key.Key_Escape:
            # SAM模式：退出SAM模式
            if self.sam_mode_active:
                self.sam_mode_active = False
                self.sam_points = []
                self.sam_bboxes = []
                self.sam_drawing_bbox = False
                self.set_tool('rectangle')  # 切换回默认工具
                self.update()
                return
            # 取消当前操作
            if self.current_tool == 'polygon' and len(self.polygon_points) > 0:
                self.polygon_points = []
                self.update()
            elif self.drawing:
                self.drawing = False
                self.update()
        elif event.key() == Qt.Key.Key_Return or event.key() == Qt.Key.Key_Enter:
            # 完成多边形绘制
            if self.current_tool == 'polygon' and len(self.polygon_points) >= 3:
                self.create_polygon_annotation()
        elif self._matches_delete_shortcut(event, settings):
            # 删除选中的标注
            if self.selected_annotation_id is not None:
                self.annotation_deleted.emit(self.selected_annotation_id)
        else:
            # 将未处理的事件传递给父组件
            self.parent().keyPressEvent(event)

    @staticmethod
    def _matches_delete_shortcut(event: QKeyEvent, settings) -> bool:
        delete_key = str(settings.value("delete_shortcut", "DELETE")).strip().upper()
        special_map = {
            "DELETE": Qt.Key.Key_Delete,
            "BACKSPACE": Qt.Key.Key_Backspace,
            "SPACE": Qt.Key.Key_Space,
            "TAB": Qt.Key.Key_Tab,
        }
        if delete_key in special_map:
            return event.key() == special_map[delete_key]
        return bool(delete_key) and event.text().upper() == delete_key
    
    def check_annotation_selection(self, pos: QPoint):
        """检查是否选中了某个标注"""
        for annotation in reversed(self.annotations):  # 从后往前检查，先检查上面的
            if self.is_point_in_annotation(pos, annotation):
                self.selected_annotation_id = annotation['id']
                self.annotation_selected.emit(annotation['id'])
                self.update()
                return
        
        # 没有选中任何标注
        self.selected_annotation_id = None
        self.update()
    
    def is_point_in_annotation(self, pos: QPoint, annotation: Dict) -> bool:
        """检查点是否在标注内"""
        ann_type = annotation.get('type', 'bbox')
        data = annotation.get('data', {})
        
        if ann_type == 'bbox':
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            
            top_left = self.image_to_widget(x, y)
            bottom_right = self.image_to_widget(x + width, y + height)
            
            return (top_left.x() <= pos.x() <= bottom_right.x() and
                    top_left.y() <= pos.y() <= bottom_right.y())
        
        elif ann_type == 'polygon':
            # 简化的多边形检测
            points = data.get('points', [])
            if len(points) < 3:
                return False
            
            # 转换为控件坐标
            widget_points = []
            for point in points:
                widget_point = self.image_to_widget(point['x'], point['y'])
                widget_points.append((widget_point.x(), widget_point.y()))
            
            # 使用射线法检测点是否在多边形内
            return self.point_in_polygon(pos.x(), pos.y(), widget_points)
        elif ann_type == 'obb':
            # 计算OBB的四个顶点
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            angle = data.get('angle', 0)
            
            # 转换为控件坐标的四个顶点
            widget_points = []
            for i in range(4):
                vertex_angle = angle + i * math.pi / 2
                vertex_x = x + width * math.cos(vertex_angle) - height * math.sin(vertex_angle)
                vertex_y = y + width * math.sin(vertex_angle) + height * math.cos(vertex_angle)
                widget_point = self.image_to_widget(vertex_x, vertex_y)
                widget_points.append((widget_point.x(), widget_point.y()))
            
            # 使用射线法检测点是否在OBB内
            return self.point_in_polygon(pos.x(), pos.y(), widget_points)
        elif ann_type == 'keypoint':
            # 检查是否点击了任何一个关键点
            keypoints = data.get('keypoints', [])
            for kp in keypoints:
                kp_x = kp.get('x', 0)
                kp_y = kp.get('y', 0)
                # 转换为控件坐标
                widget_kp = self.image_to_widget(kp_x, kp_y)
                # 计算鼠标位置与关键点的距离
                distance = math.sqrt((pos.x() - widget_kp.x()) ** 2 + (pos.y() - widget_kp.y()) ** 2)
                # 设置阈值，10像素范围内视为点击了关键点
                if distance <= 10:
                    return True
        
        return False
    
    def point_in_polygon(self, x: int, y: int, polygon: List[Tuple[int, int]]) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def create_rectangle_annotation(self):
        """创建矩形标注"""
        if self.start_point is None or self.current_point is None:
            return
        
        # 转换为图像坐标
        x1, y1 = self.widget_to_image(self.start_point.x(), self.start_point.y())
        x2, y2 = self.widget_to_image(self.current_point.x(), self.current_point.y())
        
        # 确保 x1 < x2, y1 < y2
        x = min(x1, x2)
        y = min(y1, y2)
        width = abs(x2 - x1)
        height = abs(y2 - y1)
        
        # 限制坐标在图像范围内
        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            
            # 限制x和y在图像范围内
            x = max(0, min(x, img_width))
            y = max(0, min(y, img_height))
            
            # 限制width和height不超出图像范围
            width = min(width, img_width - x)
            height = min(height, img_height - y)
        
        # 过滤太小的标注
        if width < 5 or height < 5:
            return
        
        annotation = {
            'type': 'bbox',
            'class_id': self.current_class_id,
            'data': {
                'x': x,
                'y': y,
                'width': width,
                'height': height
            }
        }
        
        self.annotation_created.emit(annotation)
        self.start_point = None
        self.current_point = None
    
    def create_polygon_annotation(self):
        """创建多边形标注"""
        if len(self.polygon_points) < 3:
            return
        
        # 转换为图像坐标
        points = []
        for point in self.polygon_points:
            x, y = self.widget_to_image(point.x(), point.y())
            
            # 限制坐标在图像范围内
            if self.current_image:
                img_width = self.current_image.width()
                img_height = self.current_image.height()
                x = max(0, min(x, img_width))
                y = max(0, min(y, img_height))
            
            points.append({'x': x, 'y': y})
        
        annotation = {
            'type': 'polygon',
            'class_id': self.current_class_id,
            'data': {
                'points': points
            }
        }
        
        self.annotation_created.emit(annotation)
        self.polygon_points = []
    
    def run_sam_inference(self, project_type: str = 'segment'):
        """运行SAM推理"""
        if not self._sam_auto_infer_enabled():
            # 非normal模式下禁止自动推理
            return
        if not self.sam_config or not self.sam_image_path:
            return
        
        if not self.sam_points and not self.sam_bboxes:
            return
        
        # 保存项目类型供回调使用
        self.sam_project_type = project_type
        
        # 创建推理线程
        self.sam_worker = SAMInferenceWorker(
            self.sam_config,
            self.sam_image_path,
            points=self.sam_points,
            bboxes=self.sam_bboxes
        )
        self.sam_worker.inference_finished.connect(
            lambda success, msg, masks: self.on_sam_inference_finished(success, msg, masks, self.sam_project_type)
        )
        self.sam_worker.start()
    
    def on_sam_inference_finished(self, success: bool, message: str, masks, project_type: str = 'segment'):
        """SAM推理完成回调"""
        if self.sam_operation_mode != "normal":
            # 记忆采集阶段丢弃任何异步返回，避免误落标注
            self.sam_points = []
            self.sam_bboxes = []
            self.update()
            return

        mask_array = masks
        mask_scores = None
        if isinstance(masks, dict):
            mask_array = masks.get("masks")
            mask_scores = masks.get("scores")

        if mask_array is not None and not isinstance(mask_array, np.ndarray):
            mask_array = np.asarray(mask_array)

        if success and mask_array is not None and len(mask_array) > 0:
            # SAM常返回多个候选mask，优先取最高分mask，避免并集导致过分膨胀
            if len(mask_array.shape) == 2:
                selected_mask = mask_array
            else:
                selected_index = 0
                if mask_scores is not None and len(mask_scores) == len(mask_array):
                    selected_index = int(np.argmax(mask_scores))
                selected_mask = mask_array[selected_index]

            if len(selected_mask.shape) > 2:
                selected_mask = selected_mask.squeeze()
            mask_uint8 = (selected_mask > 0).astype(np.uint8) * 255
            
            # 查找所有轮廓
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if not contours:
                self.sam_points = []
                self.sam_bboxes = []
                self.update()
                return
            
            # 获取所有轮廓的边界框（合并所有轮廓）
            all_x, all_y, all_x2, all_y2 = [], [], [], []
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                all_x.append(x)
                all_y.append(y)
                all_x2.append(x + w)
                all_y2.append(y + h)
            
            # 计算合并后的边界框
            min_x = min(all_x)
            min_y = min(all_y)
            max_x2 = max(all_x2)
            max_y2 = max(all_y2)
            total_w = max_x2 - min_x
            total_h = max_y2 - min_y
            
            if project_type == 'detect':
                # detect任务：创建合并后的边界框
                annotation = {
                    'type': 'bbox',
                    'class_id': self.current_class_id,
                    'data': {
                        'x': float(min_x),
                        'y': float(min_y),
                        'width': float(total_w),
                        'height': float(total_h)
                    }
                }
                
                self.annotation_created.emit(annotation)
            else:
                # segment或其他任务：创建多边形标注（使用最大轮廓）
                largest_contour = max(contours, key=cv2.contourArea)
                
                # 简化轮廓
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                
                # 转换为点列表
                points = []
                for point in approx_contour:
                    x, y = point[0]
                    points.append({'x': float(x), 'y': float(y)})
                
                # 创建多边形标注
                annotation = {
                    'type': 'polygon',
                    'class_id': self.current_class_id,
                    'data': {
                        'points': points
                    }
                }
                
                self.annotation_created.emit(annotation)
            
            # 清除SAM状态
            self.sam_points = []
            self.sam_bboxes = []
            self.update()
        else:
            # 推理失败，清除状态
            self.sam_points = []
            self.sam_bboxes = []
            self.update()
    
    def create_obb_annotation_with_points(self):
        """根据三个点创建旋转矩形标注"""
        if len(self.obb_points) != 3:
            return
        
        # 转换为图像坐标
        p1 = self.obb_points[0]
        p2 = self.obb_points[1]
        p3 = self.obb_points[2]
        
        # 转换控件坐标为图像坐标
        x1, y1 = self.widget_to_image(p1.x(), p1.y())
        x2, y2 = self.widget_to_image(p2.x(), p2.y())
        x3, y3 = self.widget_to_image(p3.x(), p3.y())
        
        # 计算向量
        vec1 = (x2 - x1, y2 - y1)
        
        # 计算垂线方向向量
        perp_dx = -vec1[1]
        perp_dy = vec1[0]
        
        # 归一化
        length = math.sqrt(perp_dx ** 2 + perp_dy ** 2)
        if length > 0:
            perp_dx /= length
            perp_dy /= length
        
        # 计算从p2到p3的向量
        vec3 = (x3 - x2, y3 - y2)
        
        # 计算投影长度
        proj_length = vec3[0] * perp_dx + vec3[1] * perp_dy
        
        # 计算垂线上的点
        perp_x = x2 + perp_dx * proj_length
        perp_y = y2 + perp_dy * proj_length
        
        # 计算第四个点
        x4 = x1 + (perp_x - x2)
        y4 = y1 + (perp_y - y2)
        
        # 计算宽度和高度
        width = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        height = math.sqrt((perp_x - x2) ** 2 + (perp_y - y2) ** 2)
        
        # 计算旋转角度（弧度）
        angle = math.atan2(vec1[1], vec1[0])
        
        # 计算中心点
        center_x = (x1 + x2 + perp_x + x4) / 4
        center_y = (y1 + y2 + perp_y + y4) / 4
        
        # 限制坐标在图像范围内
        if self.current_image:
            img_width = self.current_image.width()
            img_height = self.current_image.height()
            center_x = max(0, min(center_x, img_width))
            center_y = max(0, min(center_y, img_height))
            width = min(width, img_width)
            height = min(height, img_height)
        
        # 过滤太小的标注
        if width < 5 or height < 5:
            return
        
        annotation = {
            'type': 'obb',
            'class_id': self.current_class_id,
            'data': {
                'x': center_x,
                'y': center_y,
                'width': width,
                'height': height,
                'angle': angle
            }
        }
        
        self.annotation_created.emit(annotation)
    
    def resizeEvent(self, event):
        """窗口大小改变"""
        super().resizeEvent(event)
        if self.current_image:
            self.reset_view()
