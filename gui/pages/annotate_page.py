# -*- coding: utf-8 -*-
"""
标注页面
支持矩形框、多边形标注，快捷键操作，自动保存
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QScrollArea, QGridLayout, QFrame, QFileDialog, QProgressBar,
    QMenu, QMessageBox, QComboBox, QLineEdit, QSplitter,
    QListWidget, QListWidgetItem, QButtonGroup,
    QRadioButton, QSpinBox, QDoubleSpinBox, QFormLayout,
    QGroupBox, QCheckBox, QSlider, QTextEdit, QInputDialog,
    QSizePolicy, QToolButton,
    QColorDialog, QDialog, QApplication, QAbstractSpinBox
)
import math

# 导入自动标注相关模块
from gui.pages.auto_label_dialog import AutoLabelDialog
from gui.pages.batch_process_dialog import BatchProcessDialog
from core.auto_labeler import BatchLabelingManager
from core.model_manager import ModelManager, get_pretrained_dir
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSize, QPoint, QRect, QTimer, QPointF
from PyQt6.QtGui import QPixmap, QImage, QPainter, QColor, QFont, QKeyEvent, QMouseEvent, QWheelEvent, QAction, QIcon, QPen, QBrush, QShortcut, QKeySequence
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
import gc
import threading
import random

from gui.styles import COLORS
from gui.ui_scale import ui_font, ui_pixels
from gui.widgets.flow_layout import FlowLayout
from models.database import db
from gui.widgets.loading_dialog import LoadingOverlay

from core.sam_manager import (
    SAM3_DOWNLOAD_URL,
    SAMInferenceWorker,
    SAMMemoryPredictorManager,
    SAMModelManager,
)
from gui.canvas.annotation_canvas import AnnotationCanvas
from gui.pages.annotate_workers import (
    NEGATIVE_SAMPLE_CLASS_ID,
    AnnotateImageLoadWorker,
    RandomSampleDeleteWorker,
)
from gui.widgets.sam_memory_dialog import SAMMemoryObjectsDialog


class AnnotatePage(QWidget):
    """标注页面"""

    # 项目类别在标注页发生增/删/改时发出，通知AI标注页同步标签名
    classes_changed = pyqtSignal(dict)
    # 项目图片与标签名批量重命名后发出，通知导入页与AI标注页刷新
    images_renamed = pyqtSignal(int)
    
    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.current_image_id = None
        self.current_image_data = None
        self.images = []
        self.annotations = []
        self.classes = []
        self.current_class_id = 0
        self.history = []  # 撤销历史
        self.history_index = -1
        self.load_worker = None  # 加载线程
        self.random_delete_worker = None
        self.random_delete_progress = None
        self._sample_stats_project_id = None
        self._sample_class_counts_cache = {}
        self._negative_sample_count_cache = 0
        self._sample_stats_dirty = True
        self.default_draw_tool = 'rectangle'
        
        # 自动标注相关属性
        self.auto_label_dialog = None
        self.model_manager = None
        self.batch_labeling_manager = None
        self.sam_memory_objects = []
        self.sam_memory_dialog = None
        self.prev_image_shortcut = None
        self.next_image_shortcut = None
        self.lock_image_shortcut = None
        self._auto_save_timer = QTimer(self)
        self._auto_save_timer.timeout.connect(self._on_auto_save_tick)
        
        self.init_ui()
        self.apply_user_settings()
    
    def init_ui(self):
        """初始化界面"""
        self.main_layout = QVBoxLayout(self)
        self.main_layout.setContentsMargins(0, 0, 0, 0)
        self.main_layout.setSpacing(0)
        
        # 创建分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # 左侧：图片列表
        self.left_panel = self.create_left_panel()
        splitter.addWidget(self.left_panel)
        
        # 中间：标注画布
        self.center_panel = self.create_center_panel()
        splitter.addWidget(self.center_panel)
        
        # 右侧：属性面板
        self.right_panel = self.create_right_panel()
        splitter.addWidget(self.right_panel)
        
        # 设置分割器比例
        splitter.setSizes([250, 700, 250])
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setStretchFactor(2, 0)
        
        self.main_layout.addWidget(splitter)
        
        # 底部状态栏
        self.status_bar = self.create_status_bar()
        self.main_layout.addWidget(self.status_bar)

        self._init_navigation_shortcuts()

    def showEvent(self, event):
        """显示页面时刷新快捷键与自动保存配置。"""
        self.apply_user_settings()
        super().showEvent(event)

    def apply_user_settings(self):
        """同步偏好设置中的快捷键和自动保存配置。"""
        from PyQt6.QtCore import QSettings

        self._refresh_navigation_shortcuts()
        settings = QSettings("Better-YOLO", "Settings")
        enabled = settings.value("auto_save_enabled", True, type=bool)
        try:
            interval_min = max(1, int(settings.value("auto_save_interval", 5)))
        except (TypeError, ValueError):
            interval_min = 5

        if enabled:
            self._auto_save_timer.start(interval_min * 60 * 1000)
        else:
            self._auto_save_timer.stop()

    def _on_auto_save_tick(self):
        """定时自动保存当前打开图片的标注状态。"""
        if not self.current_project_id or not self.current_image_id or not self.annotations:
            return
        saved = 0
        for ann in self.annotations:
            try:
                db.update_annotation(
                    ann['id'],
                    data=ann.get('data', {}).copy(),
                    class_id=ann.get('class_id'),
                    class_name=ann.get('class_name'),
                )
                saved += 1
            except Exception:
                pass
        if saved:
            self.show_save_feedback(f"已自动保存 ({saved})")

    def _init_navigation_shortcuts(self):
        """初始化翻页与整图锁定快捷键，避免依赖控件焦点。"""
        self.prev_image_shortcut = QShortcut(QKeySequence(), self)
        self.prev_image_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.prev_image_shortcut.activated.connect(self._trigger_prev_image_shortcut)

        self.next_image_shortcut = QShortcut(QKeySequence(), self)
        self.next_image_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.next_image_shortcut.activated.connect(self._trigger_next_image_shortcut)

        self.lock_image_shortcut = QShortcut(QKeySequence(), self)
        self.lock_image_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.lock_image_shortcut.activated.connect(self._trigger_lock_image_shortcut)

        self._refresh_navigation_shortcuts()

    def _refresh_navigation_shortcuts(self):
        """读取设置中的翻页与整图锁定快捷键。"""
        from PyQt6.QtCore import QSettings

        settings = QSettings("Better-YOLO", "Settings")
        prev_image_key = str(settings.value("prev_image_shortcut", "A")).upper()
        next_image_key = str(settings.value("next_image_shortcut", "D")).upper()
        lock_image_key = str(settings.value("lock_image_shortcut", "L")).upper()
        self.prev_image_shortcut.setKey(QKeySequence(prev_image_key))
        self.next_image_shortcut.setKey(QKeySequence(next_image_key))
        if self.lock_image_shortcut is not None:
            self.lock_image_shortcut.setKey(QKeySequence(lock_image_key))
        self._update_lock_button_state_image()

    def _should_handle_navigation_shortcut(self) -> bool:
        """在当前焦点状态下是否允许触发翻页快捷键。"""
        if QApplication.activeWindow() is not self.window():
            return False
        if QApplication.activePopupWidget() is not None:
            return False

        focus_widget = QApplication.focusWidget()
        if focus_widget is None:
            return True

        if isinstance(focus_widget, (QLineEdit, QTextEdit, QAbstractSpinBox)):
            return False
        if focus_widget.inherits("QPlainTextEdit"):
            return False

        return True

    def _trigger_prev_image_shortcut(self):
        """上一张快捷键入口。"""
        if self._should_handle_navigation_shortcut():
            self.prev_image()

    def _trigger_next_image_shortcut(self):
        """下一张快捷键入口。"""
        if self._should_handle_navigation_shortcut():
            self.next_image()

    def _trigger_lock_image_shortcut(self):
        """整图锁定/解锁快捷键入口。"""
        if self._should_handle_navigation_shortcut() and self.current_image_id:
            self.on_toggle_lock()

    def create_left_panel(self) -> QWidget:
        """创建左侧面板 - 图片列表"""
        panel = QFrame()
        panel.setObjectName("annotate_create_left_panel")
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(300)
        panel.setStyleSheet(f"""
            QFrame#annotate_create_left_panel {{
                background-color: {COLORS['panel']};
                border-right: 1px solid {COLORS['border']};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 标题
        title = QLabel("图片列表")
        title.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
        # 任务类型选择器
        task_layout = QHBoxLayout()
        task_label = QLabel("任务类型:")
        task_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        task_layout.addWidget(task_label)
        
        self.task_combo = QComboBox()
        self.task_combo.addItems(["detect", "segment", "pose", "classify"])
        self.task_combo.currentTextChanged.connect(self.on_task_changed)
        task_layout.addWidget(self.task_combo)
        layout.addLayout(task_layout)
        
        # 筛选
        self.image_filter = QComboBox()
        self.image_filter.addItems(["全部", "未标注", "已标注", "待复核", "已锁定", "已复标修正"])
        self.image_filter.currentTextChanged.connect(self.filter_images)
        layout.addWidget(self.image_filter)
        
        # 图片列表
        self.image_list = QListWidget()
        self.image_list.setIconSize(QSize(80, 80))
        self.image_list.setSpacing(4)
        self.image_list.itemClicked.connect(self.on_image_selected)
        self.image_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
            QListWidget::item {{
                background-color: {COLORS['panel']};
                border-radius: 4px;
                padding: 4px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary']};
            }}
        """)
        layout.addWidget(self.image_list)
        
        # 导航按钮
        nav_layout = QHBoxLayout()
        self.btn_prev = QPushButton("◀ 上一张")
        self.btn_prev.clicked.connect(self.prev_image)
        self.btn_next = QPushButton("下一张 ▶")
        self.btn_next.clicked.connect(self.next_image)
        nav_layout.addWidget(self.btn_prev)
        nav_layout.addWidget(self.btn_next)
        layout.addLayout(nav_layout)
        
        return panel
    
    def create_center_panel(self) -> QWidget:
        """创建中间面板 - 标注画布"""
        panel = QWidget()
        panel.setMinimumWidth(620)
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 工具栏
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)
        
        # 标注画布
        self.canvas = AnnotationCanvas()
        self.canvas.annotation_created.connect(self.on_annotation_created)
        self.canvas.annotation_selected.connect(self.on_annotation_selected)
        self.canvas.annotation_modified.connect(self.on_annotation_modified)
        self.canvas.annotation_deleted.connect(self.on_annotation_deleted)
        layout.addWidget(self.canvas, stretch=1)
        
        # 初始化时根据当前任务类型调整工具按钮的可见性
        current_task = self.task_combo.currentText()
        self.adjust_tool_visibility(current_task)
        
        return panel
    
    def create_toolbar(self) -> QWidget:
        """创建工具栏"""
        toolbar = QFrame()
        toolbar.setObjectName("annotate_create_toolbar")
        toolbar.setStyleSheet(f"""
            QFrame#annotate_create_toolbar {{
                background-color: {COLORS['panel']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        toolbar_layout = FlowLayout(toolbar, spacing=10)
        toolbar_layout.setContentsMargins(8, 8, 8, 8)
        base_tool_button_style = self._toolbar_chip_style('tool')
        
        # 工具按钮组
        self.tool_group = QButtonGroup(self)
        self.tool_group.setExclusive(True)
        
        # 绘制工具（矩形/多边形合并）
        self.btn_draw_tool = QToolButton()
        self.btn_draw_tool.setCheckable(True)
        self.btn_draw_tool.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_draw_tool.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_draw_tool.setStyleSheet(base_tool_button_style)
        self.btn_draw_tool.clicked.connect(self.activate_default_draw_tool)
        self.draw_tool_menu = QMenu(self)
        self.action_draw_rectangle = self.draw_tool_menu.addAction("矩形")
        self.action_draw_polygon = self.draw_tool_menu.addAction("多边形")
        self.action_draw_rectangle.triggered.connect(lambda: self.select_draw_tool('rectangle'))
        self.action_draw_polygon.triggered.connect(lambda: self.select_draw_tool('polygon'))
        self.btn_draw_tool.setMenu(self.draw_tool_menu)
        self.btn_draw_tool.hide()  # 默认隐藏
        self.tool_group.addButton(self.btn_draw_tool)
        
        # 关键点工具
        self.btn_keypoint = QPushButton("📍 关键点")
        self.btn_keypoint.setToolTip("关键点工具")
        self.btn_keypoint.setCheckable(True)
        self.btn_keypoint.clicked.connect(lambda: self.set_tool('keypoint'))
        self.btn_keypoint.setStyleSheet(base_tool_button_style)
        self.btn_keypoint.hide()  # 默认隐藏
        self.tool_group.addButton(self.btn_keypoint)
        
        # # OBB工具
        # self.btn_obb = QPushButton("🔲 旋转矩形 (O)")
        # self.btn_obb.setCheckable(True)
        # self.btn_obb.clicked.connect(lambda: self.set_tool('obb'))
        # toolbar.addWidget(self.btn_obb)
        # self.btn_obb.hide()  # 默认隐藏
        # self.tool_group.addButton(self.btn_obb)
        
        # 移动工具
        self.btn_move = QPushButton("✋ 移动")
        self.btn_move.setToolTip("移动视图")
        self.btn_move.setCheckable(True)
        self.btn_move.clicked.connect(lambda: self.set_tool('move'))
        self.btn_move.setStyleSheet(base_tool_button_style)
        self.tool_group.addButton(self.btn_move)

        # 删除按钮（主动作删标注，下拉菜单删图片）
        self.btn_delete = QToolButton()
        self.btn_delete.setText("🗑️ 删除")
        self.btn_delete.setToolTip("删除当前选中标注\n快捷键: Delete")
        self.btn_delete.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextOnly)
        self.btn_delete.setPopupMode(QToolButton.ToolButtonPopupMode.MenuButtonPopup)
        self.btn_delete.setStyleSheet(self._toolbar_chip_style('danger'))
        self.btn_delete.clicked.connect(self.delete_selected_annotation)
        self.delete_menu = QMenu(self)
        self.action_delete_current_image = self.delete_menu.addAction("删除当前图片")
        self.action_delete_current_image.triggered.connect(self.delete_current_image)
        self.delete_menu.aboutToShow.connect(self.update_delete_menu_state)
        self.btn_delete.setMenu(self.delete_menu)
        
        # 撤销按钮
        self.btn_undo = QPushButton("↶ 撤销")
        self.btn_undo.setToolTip("撤销上一步\n快捷键: Ctrl+Z")
        self.btn_undo.clicked.connect(self.undo)
        self.btn_undo.setStyleSheet(base_tool_button_style)
        
        # 自动标注按钮
        self.btn_auto_label = QPushButton("🤖 自动标注")
        self.btn_auto_label.setMenu(self.create_auto_label_menu())
        self.btn_auto_label.setStyleSheet(self._toolbar_menu_button_style('ai_blue'))
        
        # SAM自动标注按钮
        self.btn_sam = QPushButton("🎯 SAM")
        self.btn_sam.setToolTip("使用SAM进行交互式分割标注")
        self.btn_sam.setStyleSheet(self._toolbar_menu_button_style('ai_green'))
        self.apply_sam_button_mode()
        
        # 批量处理按钮
        self.btn_batch_process = QPushButton("📋 批处理")
        self.btn_batch_process.clicked.connect(self.show_batch_process_dialog)
        self.btn_batch_process.setToolTip("批量处理工具")
        self.btn_batch_process.setStyleSheet(self._toolbar_chip_style('ai_olive'))

        self.toolbar_draw_buttons = [self.btn_draw_tool, self.btn_keypoint, self.btn_move]
        self.toolbar_edit_buttons = [self.btn_delete, self.btn_undo]
        self.toolbar_ai_buttons = [self.btn_auto_label, self.btn_sam, self.btn_batch_process]
        self.toolbar_draw_group = self._create_toolbar_button_group(self.toolbar_draw_buttons)
        self.toolbar_edit_group = self._create_toolbar_button_group(self.toolbar_edit_buttons)
        self.toolbar_ai_group = self._create_toolbar_button_group(self.toolbar_ai_buttons)
        self.toolbar_groups = [
            (self.toolbar_draw_group, self.toolbar_draw_buttons),
            (self.toolbar_edit_group, self.toolbar_edit_buttons),
            (self.toolbar_ai_group, self.toolbar_ai_buttons),
        ]

        toolbar_layout.addWidget(self.toolbar_draw_group)
        toolbar_layout.addWidget(self.toolbar_edit_group)
        toolbar_layout.addWidget(self.toolbar_ai_group)

        self.refresh_draw_tool_button()
        self.refresh_toolbar_button_layout()
        
        return toolbar

    def _create_toolbar_button_group(self, buttons) -> QWidget:
        """创建工具栏按钮组，显式控制组内间距。"""
        group_widget = QWidget()
        layout = QHBoxLayout(group_widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        for button in buttons:
            layout.addWidget(button)
        return group_widget

    def _toolbar_chip_style(self, role: str) -> str:
        """生成顶部工具栏按钮样式。"""
        palette_map = {
            'tool': {
                'bg': '#154B78',
                'hover': '#1D629B',
                'active': COLORS['primary'],
                'border': '#2871A8',
                'text': '#F7FBFF',
            },
            'danger': {
                'bg': '#154B78',
                'hover': '#1D629B',
                'active': COLORS['primary'],
                'border': '#2871A8',
                'text': '#F7FBFF',
            },
            'ai_blue': {
                'bg': '#154B78',
                'hover': '#1D629B',
                'active': COLORS['primary'],
                'border': '#2871A8',
                'text': '#F7FBFF',
            },
            'ai_green': {
                'bg': '#154B78',
                'hover': '#1D629B',
                'active': COLORS['primary'],
                'border': '#2871A8',
                'text': '#F7FBFF',
            },
            'ai_gold': {
                'bg': '#154B78',
                'hover': '#1D629B',
                'active': COLORS['primary'],
                'border': '#2871A8',
                'text': '#F7FBFF',
            },
            'ai_olive': {
                'bg': '#154B78',
                'hover': '#1D629B',
                'active': COLORS['primary'],
                'border': '#2871A8',
                'text': '#F7FBFF',
            },
        }
        palette = palette_map[role]
        return f"""
            QPushButton, QToolButton {{
                background-color: {palette['bg']};
                color: {palette['text']};
                border: 1px solid {palette['border']};
                border-radius: 6px;
                padding: 4px 10px;
                font-weight: 600;
            }}
            QPushButton:hover, QToolButton:hover {{
                background-color: {palette['hover']};
                border-color: {palette['active']};
            }}
            QPushButton:checked, QToolButton:checked {{
                background-color: {palette['active']};
                color: white;
                border-color: {palette['active']};
            }}
            QToolButton::menu-button {{
                subcontrol-origin: padding;
                subcontrol-position: right center;
                width: 16px;
                background-color: rgba(255, 255, 255, 0.08);
                border-left: 1px solid rgba(255, 255, 255, 0.14);
                border-top-right-radius: 6px;
                border-bottom-right-radius: 6px;
            }}
            QToolButton::menu-button:hover {{
                background-color: rgba(255, 255, 255, 0.14);
            }}
            QToolButton::menu-arrow {{
                width: 10px;
                height: 10px;
            }}
            QToolButton::menu-indicator {{
                width: 10px;
                height: 10px;
            }}
        """

    def _toolbar_menu_button_style(self, role: str) -> str:
        """菜单按钮样式：保留下拉菜单能力，但隐藏倒三角指示器。"""
        base_style = self._toolbar_chip_style(role)
        return base_style + """
            QPushButton::menu-indicator {
                image: none;
                width: 0px;
                subcontrol-origin: padding;
                subcontrol-position: right center;
            }
        """

    def _get_toolbar_button_width(self, button) -> int:
        """计算按钮建议宽度，菜单按钮额外预留箭头空间。"""
        text_width = button.fontMetrics().horizontalAdvance(button.text())
        extra_padding = ui_pixels(40)
        if isinstance(button, QToolButton) and button.menu() is not None:
            extra_padding += ui_pixels(18)
        return text_width + extra_padding

    def refresh_toolbar_button_layout(self):
        """按分组统一顶部按钮尺寸。"""
        button_height = ui_pixels(36)
        for group_widget, group in getattr(self, 'toolbar_groups', []):
            visible_buttons = [
                button for button in group
                if button is not None and not button.isHidden()
            ]
            group_widget.setVisible(bool(visible_buttons))
            if not visible_buttons:
                continue
            group_width = max(self._get_toolbar_button_width(button) for button in visible_buttons)
            for button in visible_buttons:
                button.setFixedHeight(button_height)
                button.setFixedWidth(group_width)

    def refresh_draw_tool_button(self):
        """刷新绘制工具按钮文本与选项状态。"""
        if not hasattr(self, 'btn_draw_tool'):
            return

        if self.default_draw_tool == 'polygon':
            self.btn_draw_tool.setText("🛑 绘制")
            self.btn_draw_tool.setToolTip("当前默认绘制模式: 多边形\n点击主按钮直接进入多边形绘制")
        else:
            self.btn_draw_tool.setText("🟦 绘制")
            self.btn_draw_tool.setToolTip("当前默认绘制模式: 矩形\n点击主按钮直接进入矩形绘制")

        if hasattr(self, 'action_draw_rectangle'):
            self.action_draw_rectangle.setCheckable(True)
            self.action_draw_rectangle.setChecked(self.default_draw_tool == 'rectangle')
        if hasattr(self, 'action_draw_polygon'):
            self.action_draw_polygon.setCheckable(True)
            self.action_draw_polygon.setChecked(self.default_draw_tool == 'polygon')

        self.refresh_toolbar_button_layout()

    def select_draw_tool(self, tool: str):
        """选择默认绘制工具，并立即切换到该工具。"""
        if tool not in ('rectangle', 'polygon'):
            return
        self.default_draw_tool = tool
        self.refresh_draw_tool_button()
        self.btn_draw_tool.setChecked(True)
        self.set_tool(tool)

    def activate_default_draw_tool(self):
        """激活当前默认绘制工具。"""
        self.btn_draw_tool.setChecked(True)
        self.set_tool(self.default_draw_tool)

    def update_delete_menu_state(self):
        """更新删除菜单状态。"""
        if hasattr(self, 'action_delete_current_image'):
            self.action_delete_current_image.setEnabled(bool(self.current_image_id))
    
    def apply_sam_button_mode(self):
        """根据SAM设置刷新按钮行为：普通模式/记忆模式。"""
        if not hasattr(self, "btn_sam"):
            return
        try:
            self.btn_sam.clicked.disconnect()
        except Exception:
            pass
        self.btn_sam.setMenu(None)

        sam_config = AutoLabelDialog.get_saved_sam_config()
        sam_type = sam_config.get("sam_type", "SAM")
        usage_mode = sam_config.get("usage_mode", "normal")

        if usage_mode == "memory" and sam_type in ("SAM2", "SAM3"):
            self.btn_sam.setText("🎯 SAM记忆")
            self.btn_sam.setToolTip("SAM记忆标注：更新记忆/清空记忆/单张推理/批量推理")
            self.btn_sam.setMenu(self.create_sam_memory_menu())
        else:
            self.btn_sam.setText("🎯 SAM")
            self.btn_sam.setToolTip("使用SAM进行交互式分割标注")
            self.btn_sam.clicked.connect(self.start_sam_annotation)
        self.refresh_toolbar_button_layout()

    def create_sam_memory_menu(self) -> QMenu:
        menu = QMenu(self)
        act_update = QAction("更新记忆", self)
        act_clear = QAction("清空记忆", self)
        act_single = QAction("单张推理", self)
        act_batch = QAction("批量推理", self)
        act_update.triggered.connect(self.start_sam_memory_update)
        act_clear.triggered.connect(self.clear_sam_memory)
        act_single.triggered.connect(self.run_sam_memory_single)
        act_batch.triggered.connect(self.run_sam_memory_batch)
        menu.addAction(act_update)
        menu.addAction(act_clear)
        menu.addSeparator()
        menu.addAction(act_single)
        menu.addAction(act_batch)
        return menu
    
    def create_right_panel(self) -> QWidget:
        """创建右侧面板 - 属性面板"""
        panel = QFrame()
        panel.setObjectName("annotate_create_right_panel")
        panel.setFrameStyle(QFrame.Shape.StyledPanel)
        panel.setMaximumWidth(300)
        panel.setStyleSheet(f"""
            QFrame#annotate_create_right_panel {{
                background-color: {COLORS['panel']};
                border-left: 1px solid {COLORS['border']};
            }}
        """)
        
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        
        # 类别列表
        class_group = QGroupBox("类别列表")
        class_group.setStyleSheet(f"""
            QGroupBox {{
                color: {COLORS['text_primary']};
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px;
            }}
        """)
        class_layout = QVBoxLayout(class_group)
        
        self.class_list = QListWidget()
        self.class_list.setStyleSheet(f"""
            QListWidget {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
            }}
            QListWidget::item {{
                padding: 6px 8px;
                min-height: 24px;
            }}
            QListWidget::item:selected {{
                background-color: {COLORS['primary']};
            }}
        """)
        self.class_list.setSpacing(2)
        self.class_list.itemClicked.connect(self.on_class_selected)
        self.class_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.class_list.customContextMenuRequested.connect(self.show_class_context_menu)
        self.class_list.setMinimumHeight(130)
        class_layout.addWidget(self.class_list, 1)
        
        compact_action_button_height = 24
        compact_action_button_style = f"""
            QPushButton {{
                min-height: {compact_action_button_height}px;
                max-height: {compact_action_button_height}px;
                padding: 0 10px;
            }}
        """

        # 添加类别按钮
        self.btn_add_class = QPushButton("+ 添加类别")
        self.btn_add_class.clicked.connect(self.add_class)
        self.btn_add_class.setFixedHeight(compact_action_button_height)
        self.btn_add_class.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_add_class.setStyleSheet(compact_action_button_style)
        self.btn_apply_attr = QPushButton("应用修改")
        self.btn_apply_attr.clicked.connect(self.apply_annotation_changes)
        self.btn_apply_attr.setFixedHeight(compact_action_button_height)
        self.btn_apply_attr.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_apply_attr.setStyleSheet(compact_action_button_style)
        self.btn_apply_attr.setEnabled(False)
        class_button_row = QHBoxLayout()
        class_button_row.setContentsMargins(0, 0, 0, 0)
        class_button_row.setSpacing(8)
        class_button_row.addWidget(self.btn_add_class, 1)
        class_button_row.addWidget(self.btn_apply_attr, 1)
        class_layout.addLayout(class_button_row)

        # 整图锁定 / 手动保存
        self.btn_lock = QPushButton("🔒 锁定本图")
        self.btn_lock.setToolTip(
            "锁定/解锁当前整张图片（默认快捷键 L）。\n"
            "• 已锁定的图片在 AI/YOLO 批量标注时自动跳过，并可作为正样本示例；\n"
            "• 手动增/删/改任何标注框或修改类别都会自动锁定本图；\n"
            "• 若本图经过 AI 标注且人工修改了结果，将额外记录为 ✏️ 复标修正；\n"
            "• 解锁本图会同时将其移出 AI 断点，便于单独重新跑 AI 标注。"
        )
        self.btn_lock.setFixedHeight(compact_action_button_height)
        self.btn_lock.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_lock.setStyleSheet(compact_action_button_style)
        self.btn_lock.setEnabled(False)
        self.btn_lock.clicked.connect(self.on_toggle_lock)

        self.btn_save_ann = QPushButton("💾 保存")
        self.btn_save_ann.setToolTip("把当前图片的所有标注保存到数据库")
        self.btn_save_ann.setFixedHeight(compact_action_button_height)
        self.btn_save_ann.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.btn_save_ann.setStyleSheet(compact_action_button_style)
        self.btn_save_ann.clicked.connect(self.on_save_annotations)

        ann_action_row = QHBoxLayout()
        ann_action_row.setContentsMargins(0, 0, 0, 0)
        ann_action_row.setSpacing(8)
        ann_action_row.addWidget(self.btn_lock, 1)
        ann_action_row.addWidget(self.btn_save_ann, 1)
        class_layout.addLayout(ann_action_row)
        
        layout.addWidget(class_group, 1)
        
        # 标注属性 / 样本调节
        attr_group = QGroupBox("标注属性")
        attr_group.setStyleSheet(class_group.styleSheet())
        attr_layout = QFormLayout(attr_group)
        attr_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        attr_layout.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        attr_layout.setVerticalSpacing(8)
        attr_layout.setHorizontalSpacing(10)

        field_style = f"""
            QComboBox, QSpinBox {{
                min-height: 32px;
                padding: 3px 8px;
                color: {COLORS['text_primary']};
            }}
        """
        # 当前标注类别
        self.attr_class = QComboBox()
        self.attr_class.currentIndexChanged.connect(self.on_attr_class_changed)
        self.attr_class.hide()
        
        # 当前目标标签的样本数
        self.sample_target_class = QComboBox()
        self.sample_target_class.currentIndexChanged.connect(self.on_sample_target_changed)
        self.sample_target_class.setStyleSheet(field_style)
        attr_layout.addRow("删样标签:", self.sample_target_class)

        self.sample_count_label = QLabel("0")
        self.sample_count_label.setMinimumHeight(32)
        self.sample_count_label.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft)
        self.sample_count_label.setStyleSheet(f"""
            QLabel {{
                color: {COLORS['text_primary']};
                padding: 6px 8px;
                border: 1px solid {COLORS['border']};
                background-color: {COLORS['sidebar']};
                border-radius: 4px;
            }}
        """)
        attr_layout.addRow("标签个数:", self.sample_count_label)

        self.sample_delete_count = QSpinBox()
        self.sample_delete_count.setRange(0, 0)
        self.sample_delete_count.setStyleSheet(field_style)
        attr_layout.addRow("随机删去:", self.sample_delete_count)

        self.sample_hint = QLabel("按整张图片随机删除；负样本指已标注但无任何框的图片。")
        self.sample_hint.setWordWrap(True)
        self.sample_hint.setStyleSheet(f"color: {COLORS['text_secondary']}; padding: 2px 0;")
        attr_layout.addRow(self.sample_hint)

        self.btn_mark_negative_sample = QPushButton("标注为负样本")
        self.btn_mark_negative_sample.clicked.connect(self.mark_current_image_as_negative_sample)
        self.btn_mark_negative_sample.setFixedHeight(compact_action_button_height)
        self.btn_mark_negative_sample.setStyleSheet(compact_action_button_style)
        self.btn_mark_negative_sample.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        self.btn_delete_random_samples = QPushButton("随机删除样本")
        self.btn_delete_random_samples.clicked.connect(self.delete_random_samples_for_target)
        self.btn_delete_random_samples.setFixedHeight(compact_action_button_height)
        self.btn_delete_random_samples.setStyleSheet(compact_action_button_style)
        self.btn_delete_random_samples.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)

        action_button_widget = QWidget()
        action_button_layout = QHBoxLayout(action_button_widget)
        action_button_layout.setContentsMargins(0, 0, 0, 0)
        action_button_layout.setSpacing(8)
        action_button_layout.addWidget(self.btn_mark_negative_sample, 1)
        action_button_layout.addWidget(self.btn_delete_random_samples, 1)
        attr_layout.addRow(action_button_widget)
        
        layout.addWidget(attr_group)
        
        # 导出功能（保持极简紧凑，不占用上方类别列表高度；格式与目录结构在点击导出后弹窗选择）
        export_group = QGroupBox("数据导出")
        export_group.setStyleSheet(class_group.styleSheet())
        export_layout = QVBoxLayout(export_group)
        export_layout.setSpacing(6)
        export_layout.setContentsMargins(8, 8, 8, 8)
        
        # 保留隐藏的默认格式控件以兼容编程调用，不在右侧窄栏中占空间
        self.export_format = QComboBox()
        self.export_format.addItems(["YOLO格式", "COCO格式"])
        self.export_format.hide()
        
        # 一键重命名与导出按钮
        self.btn_rename_dataset = QPushButton("✏️ 一键重命名图片与标签")
        self.btn_rename_dataset.setFixedHeight(compact_action_button_height + 2)
        self.btn_rename_dataset.setStyleSheet(compact_action_button_style)
        self.btn_rename_dataset.setToolTip(
            "按序号一键重命名图片与对应标签文件（默认 img_001 起，可自定义起始编号，支持当前项目与外部目录）"
        )
        self.btn_rename_dataset.clicked.connect(self.open_rename_dialog)
        export_layout.addWidget(self.btn_rename_dataset)

        self.btn_export_annotations = QPushButton("📤 导出标注文件")
        self.btn_export_annotations.setFixedHeight(compact_action_button_height + 2)
        self.btn_export_annotations.setStyleSheet(compact_action_button_style)
        self.btn_export_annotations.clicked.connect(self.on_click_export_annotations)
        export_layout.addWidget(self.btn_export_annotations)
        
        self.btn_export_dataset = QPushButton("📦 导出完整数据集")
        self.btn_export_dataset.setFixedHeight(compact_action_button_height + 2)
        self.btn_export_dataset.setStyleSheet(compact_action_button_style)
        self.btn_export_dataset.clicked.connect(self.on_click_export_dataset)
        export_layout.addWidget(self.btn_export_dataset)
        
        layout.addWidget(export_group)
        
        return panel
    
    def create_auto_label_menu(self) -> QMenu:
        """创建自动标注下拉菜单"""
        menu = QMenu()
        
        # 设置选项
        action_settings = menu.addAction("⚙️ 设置")
        action_settings.triggered.connect(self.show_auto_label_settings)
        
        # 单张推理选项
        action_single = menu.addAction("🔍 单张推理")
        action_single.triggered.connect(self.run_single_inference)
        
        # 批量推理选项
        action_batch = menu.addAction("📋 批量推理")
        action_batch.triggered.connect(self.run_batch_inference)
        
        return menu
    
    def create_status_bar(self) -> QFrame:
        """创建状态栏"""
        status_bar = QFrame()
        status_bar.setObjectName("annotate_create_status_bar")
        status_bar.setFrameStyle(QFrame.Shape.StyledPanel)
        status_bar.setMaximumHeight(40)
        status_bar.setStyleSheet(f"""
            QFrame#annotate_create_status_bar {{
                background-color: {COLORS['panel']};
                border-top: 1px solid {COLORS['border']};
            }}
            QLabel {{
                color: {COLORS['text_secondary']};
                font-size: 12px;
                padding: 4px 12px;
            }}
        """)
        
        layout = QHBoxLayout(status_bar)
        layout.setContentsMargins(8, 4, 8, 4)
        
        self.status_image = QLabel("当前: 0/0")
        layout.addWidget(self.status_image)
        
        layout.addWidget(QLabel("|"))
        
        self.status_annotation = QLabel("标注: 0")
        layout.addWidget(self.status_annotation)
        
        layout.addWidget(QLabel("|"))
        
        self.status_position = QLabel("位置: --")
        layout.addWidget(self.status_position)
        
        layout.addStretch()
        
        self.status_tool = QLabel("工具: 矩形")
        layout.addWidget(self.status_tool)

        # 保存成功临时提示
        self.status_save = QLabel("")
        self.status_save.setStyleSheet("color: #4caf50; font-weight: bold;")
        layout.addWidget(self.status_save)

        return status_bar
    
    def set_project(self, project_id: int):
        """设置当前项目"""
        # 即使项目ID相同，也重新加载数据（确保图片列表更新）
        self.current_project_id = project_id
        self._invalidate_sample_stats_cache()
        
        # 显示加载动画
        self.loading_overlay = LoadingOverlay(self, "正在加载项目数据...")
        self.loading_overlay.show_loading()
        
        # 创建后台线程来加载项目数据
        from PyQt6.QtCore import QThread, pyqtSignal
        
        class ProjectLoadThread(QThread):
            """项目数据加载线程"""
            
            data_loaded = pyqtSignal(dict)
            finished = pyqtSignal()
            
            def __init__(self, project_id):
                super().__init__()
                self.project_id = project_id
            
            def run(self):
                """运行线程"""
                try:
                    # 加载项目信息
                    project = db.get_project(self.project_id)
                    classes = []
                    
                    if project:
                        # 加载类别（默认不预设任何类别）
                        import json
                        try:
                            classes = json.loads(project.get('classes', '[]'))
                            if not isinstance(classes, list):
                                classes = []
                        except Exception:
                            classes = []
                    
                    # 加载图片列表数据
                    images = db.get_project_images(self.project_id)
                    
                    # 发送加载完成信号
                    self.data_loaded.emit({'classes': classes, 'images': images})
                finally:
                    self.finished.emit()
        
        # 创建并启动线程
        self.load_thread = ProjectLoadThread(project_id)
        self.load_thread.data_loaded.connect(self.on_project_data_loaded)
        self.load_thread.finished.connect(self.on_project_load_finished)
        self.load_thread.start()
    
    def on_project_data_loaded(self, data):
        """项目数据加载完成回调"""
        # 更新类别
        self.classes = data.get('classes', [])
        self.update_class_list()
        
        # 保存图片数据
        self.images = data.get('images', [])
        
        # 设置任务类型选择器
        if self.current_project_id:
            project = db.get_project(self.current_project_id)
            if project and project.get('type'):
                task_type = project['type']
                if task_type in ['detect', 'segment', 'pose', 'classify', 'obb']:
                    index = self.task_combo.findText(task_type)
                    if index >= 0:
                        self.task_combo.setCurrentIndex(index)
        
        # 开始加载图片列表（使用多线程加载缩略图）
        self.load_image_list()
    
    def on_project_load_finished(self):
        """项目加载完成回调"""
        # 隐藏加载动画
        if hasattr(self, 'loading_overlay'):
            self.loading_overlay.hide_loading()
            self.loading_overlay.deleteLater()
            delattr(self, 'loading_overlay')
        
        # 清理线程
        if hasattr(self, 'load_thread'):
            self.load_thread.wait()
            delattr(self, 'load_thread')
    
    @staticmethod
    def _format_image_list_text(image: dict) -> str:
        """格式化左侧图片列表项的状态徽标与文件名:
        ○ 未标注 | ✓ 待复核 | ✓ 🔒 已确认锁定 | ✓ ✏️🔒 已复标修正并锁定
        """
        status_text = "✓" if image.get('status') == 'annotated' else "○"
        locked = bool(image.get('locked'))
        modified = bool(image.get('modified'))
        if locked and modified:
            badge = " ✏️🔒"
        elif locked:
            badge = " 🔒"
        elif modified:
            badge = " ✏️"
        else:
            badge = ""
        return f"{status_text}{badge} {image['filename']}"

    def load_image_list(self):
        """加载图片列表 - 使用多线程"""
        # 停止之前的加载
        if self.load_worker and self.load_worker.isRunning():
            self.load_worker.stop()
            self.load_worker.wait()
        
        self.image_list.clear()
        self.images = []
        
        if not self.current_project_id:
            return
        
        # 从数据库获取图片列表（很快）
        self.images = db.get_project_images(self.current_project_id)
        self._invalidate_sample_stats_cache()
        
        # 先创建所有列表项（显示占位符）
        for image in self.images:
            item = QListWidgetItem()
            item.setData(Qt.ItemDataRole.UserRole, image['id'])
            item.setText(self._format_image_list_text(image))
            self.image_list.addItem(item)

        if hasattr(self, 'image_filter'):
            self.filter_images(self.image_filter.currentText())
        
        self.update_status_bar()
        self.update_sample_control_panel()
        
        # 启动后台加载线程加载缩略图
        if self.images:
            self.load_worker = AnnotateImageLoadWorker(self.images)
            self.load_worker.image_loaded.connect(self.on_image_loaded)
            self.load_worker.finished_loading.connect(self.on_load_finished)
            self.load_worker.start()
    
    def on_image_loaded(self, index: int, pixmap: QPixmap):
        """单个图片加载完成回调"""
        if index < self.image_list.count():
            item = self.image_list.item(index)
            if item:
                item.setIcon(QIcon(pixmap))
    
    def on_load_finished(self):
        """加载完成回调"""
        pass
    
    def update_image_list_display(self):
        """更新图片列表显示"""
        # 重新加载图片数据
        if self.current_project_id:
            self.images = db.get_project_images(self.current_project_id)
            img_map = {img['id']: img for img in self.images}
            
            # 更新图片列表项
            for i in range(self.image_list.count()):
                item = self.image_list.item(i)
                image_id = item.data(Qt.ItemDataRole.UserRole)
                image_data = img_map.get(image_id)
                if image_data:
                    item.setText(self._format_image_list_text(image_data))

            if hasattr(self, 'image_filter'):
                self.filter_images(self.image_filter.currentText())
    
    def refresh_classes(self):
        """从数据库重新加载类别并刷新列表(AI标注页添加标签后同步)"""
        if not self.current_project_id:
            return
        project = db.get_project(self.current_project_id)
        if not project:
            return
        classes = project.get('classes') or []
        if isinstance(classes, str):
            try:
                import json as _json
                classes = _json.loads(classes)
            except (ValueError, TypeError):
                classes = []
        self.classes = classes
        self.update_class_list()

    def update_class_list(self):
        """更新类别列表"""
        current_attr_class = self.attr_class.currentData()
        current_sample_target = self.sample_target_class.currentData()
        current_selected_class = self.current_class_id
        class_sample_counts, negative_sample_count = self._get_sample_stats()
        self.class_list.clear()
        self.attr_class.clear()
        self.sample_target_class.blockSignals(True)
        self.sample_target_class.clear()
        
        # 更新canvas的类别颜色与类别名称（画布显示标签用）
        class_colors = {}
        class_names = {}
        for cls in self.classes:
            class_colors[cls['id']] = cls.get('color', '#808080')
            class_names[cls['id']] = cls.get('name', '')
        self.canvas.class_colors = class_colors
        self.canvas.class_names = class_names
        
        for cls in self.classes:
            sample_count = class_sample_counts.get(cls['id'], 0)
            # 创建带颜色的列表项
            item = QListWidgetItem(f"■ {cls['name']} ({sample_count})")
            item.setData(Qt.ItemDataRole.UserRole, cls['id'])
            
            # 设置颜色
            color = QColor(cls.get('color', '#808080'))
            item.setForeground(color)
            item.setSizeHint(QSize(item.sizeHint().width(), 30))
            
            self.class_list.addItem(item)
            
            # 添加到属性面板的下拉框
            self.attr_class.addItem(cls['name'], cls['id'])
            self.sample_target_class.addItem(cls['name'], cls['id'])

        self.sample_target_class.addItem("负样本", NEGATIVE_SAMPLE_CLASS_ID)

        if current_attr_class is not None:
            attr_index = self.attr_class.findData(current_attr_class)
            if attr_index >= 0:
                self.attr_class.setCurrentIndex(attr_index)

        if current_sample_target is None:
            current_sample_target = self.current_class_id if self.classes else NEGATIVE_SAMPLE_CLASS_ID
        sample_index = self.sample_target_class.findData(current_sample_target)
        if sample_index >= 0:
            self.sample_target_class.setCurrentIndex(sample_index)
        elif self.sample_target_class.count() > 0:
            self.sample_target_class.setCurrentIndex(0)
        self.sample_target_class.blockSignals(False)
        self.update_sample_control_panel(class_sample_counts, negative_sample_count)
        
        # 默认选中第一个类别
        if self.class_list.count() > 0:
            selected_row = next(
                (row for row in range(self.class_list.count())
                 if self.class_list.item(row).data(Qt.ItemDataRole.UserRole) == current_selected_class),
                0
            )
            self.class_list.setCurrentRow(selected_row)
            self.on_class_selected()
        self.canvas.update()

    def get_sample_target_images(self, target_class_id):
        """获取指定标签对应的样本图像列表"""
        if not self.current_project_id or target_class_id is None:
            return []

        if target_class_id == NEGATIVE_SAMPLE_CLASS_ID:
            return db.get_negative_sample_images(self.current_project_id, annotated_only=True)

        return db.get_project_images_by_class(self.current_project_id, target_class_id)

    def _invalidate_sample_stats_cache(self):
        """标记样本统计缓存失效。"""
        self._sample_stats_dirty = True

    def _get_sample_stats(self, force_refresh=False):
        """获取当前项目样本统计缓存。"""
        if not self.current_project_id:
            self._sample_stats_project_id = None
            self._sample_class_counts_cache = {}
            self._negative_sample_count_cache = 0
            self._sample_stats_dirty = False
            return {}, 0

        should_refresh = (
            force_refresh
            or self._sample_stats_dirty
            or self._sample_stats_project_id != self.current_project_id
        )
        if should_refresh:
            self._sample_class_counts_cache = db.get_project_image_counts_by_class(self.current_project_id)
            self._negative_sample_count_cache = db.get_negative_sample_image_count(
                self.current_project_id,
                annotated_only=True
            )
            self._sample_stats_project_id = self.current_project_id
            self._sample_stats_dirty = False

        return self._sample_class_counts_cache, self._negative_sample_count_cache

    def get_project_sample_counts(self, force_refresh=False):
        """获取当前项目各类别对应的样本图数量。"""
        return self._get_sample_stats(force_refresh=force_refresh)[0]

    def get_negative_sample_count(self, force_refresh=False):
        """获取当前项目负样本图数量。"""
        return self._get_sample_stats(force_refresh=force_refresh)[1]

    def get_class_sample_count(self, class_id):
        """获取某个类别对应的样本图数量"""
        return self.get_project_sample_counts().get(class_id, 0)

    def _get_selected_class_list_id(self) -> Optional[int]:
        """获取类别列表当前选中的类别ID。"""
        current_item = self.class_list.currentItem()
        if current_item is None:
            return None
        return current_item.data(Qt.ItemDataRole.UserRole)

    def _set_class_list_selection(self, class_id: Optional[int]):
        """按类别ID同步类别列表选中状态。"""
        if class_id is None:
            self.class_list.clearSelection()
            self._refresh_annotation_class_controls()
            return

        for row in range(self.class_list.count()):
            item = self.class_list.item(row)
            if item and item.data(Qt.ItemDataRole.UserRole) == class_id:
                self.class_list.setCurrentRow(row)
                self.current_class_id = class_id
                self.canvas.current_class_id = class_id
                break

        self._sync_attr_class_combo_from_list()
        self._refresh_annotation_class_controls()

    def _sync_attr_class_combo_from_list(self):
        """将类别列表当前选择同步到属性下拉框。"""
        class_id = self._get_selected_class_list_id()
        self.attr_class.blockSignals(True)
        if class_id is None:
            self.attr_class.setCurrentIndex(-1)
        else:
            index = self.attr_class.findData(class_id)
            if index >= 0:
                self.attr_class.setCurrentIndex(index)
        self.attr_class.blockSignals(False)

    def _refresh_annotation_class_controls(self):
        """根据当前标注和类别选择刷新“应用修改”按钮状态。"""
        selected_annotation_id = self.canvas.selected_annotation_id
        if selected_annotation_id is None:
            self.btn_apply_attr.setEnabled(False)
            return

        annotation = next((ann for ann in self.annotations if ann['id'] == selected_annotation_id), None)
        if annotation is None:
            self.btn_apply_attr.setEnabled(False)
            return

        selected_class_id = self._get_selected_class_list_id()
        self.btn_apply_attr.setEnabled(
            selected_class_id is not None and selected_class_id != annotation.get('class_id')
        )

    def refresh_class_list_counts(self, class_sample_counts=None):
        """刷新类别列表中的样本数量显示"""
        if class_sample_counts is None:
            class_sample_counts = self.get_project_sample_counts()

        for row in range(self.class_list.count()):
            item = self.class_list.item(row)
            if not item:
                continue
            class_id = item.data(Qt.ItemDataRole.UserRole)
            class_info = next((cls for cls in self.classes if cls['id'] == class_id), None)
            if not class_info:
                continue
            sample_count = class_sample_counts.get(class_id, 0)
            item.setText(f"■ {class_info['name']} ({sample_count})")

    def update_sample_control_panel(self, class_sample_counts=None, negative_sample_count=None):
        """刷新样本调节面板"""
        if class_sample_counts is None or negative_sample_count is None:
            cached_class_counts, cached_negative_sample_count = self._get_sample_stats()
            if class_sample_counts is None:
                class_sample_counts = cached_class_counts
            if negative_sample_count is None:
                negative_sample_count = cached_negative_sample_count

        self.refresh_class_list_counts(class_sample_counts)
        target_class_id = self.sample_target_class.currentData()
        if target_class_id is None:
            self.sample_count_label.setText("0")
            self.sample_delete_count.blockSignals(True)
            self.sample_delete_count.setRange(0, 0)
            self.sample_delete_count.setValue(0)
            self.sample_delete_count.blockSignals(False)
            return

        if target_class_id == NEGATIVE_SAMPLE_CLASS_ID:
            sample_count = negative_sample_count
        else:
            sample_count = class_sample_counts.get(target_class_id, 0)
        self.sample_count_label.setText(str(sample_count))

        self.sample_delete_count.blockSignals(True)
        self.sample_delete_count.setRange(0, sample_count)
        if self.sample_delete_count.value() > sample_count:
            self.sample_delete_count.setValue(sample_count)
        self.sample_delete_count.blockSignals(False)

    def on_sample_target_changed(self, index):
        """删样目标切换事件"""
        self.update_sample_control_panel()
    
    def init_auto_label_components(self):
        """初始化自动标注组件"""
        if not self.model_manager:
            self.model_manager = ModelManager()
        
        if not self.auto_label_dialog:
            self.auto_label_dialog = AutoLabelDialog(self)
            self.auto_label_dialog.single_inference_requested.connect(self.on_single_inference_requested)
            self.auto_label_dialog.batch_inference_requested.connect(self.on_batch_inference_requested)
        
        if not self.batch_labeling_manager:
            self.batch_labeling_manager = BatchLabelingManager()
            self.batch_labeling_manager.progress_updated.connect(self.on_batch_inference_progress)
            self.batch_labeling_manager.batch_completed.connect(self.on_batch_inference_completed)
        
        # 初始化加载动画
        if not hasattr(self, 'loading_label'):
            self.loading_label = QLabel("加载中...")
            self.loading_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.loading_label.setStyleSheet("""
                QLabel {
                    background-color: rgba(0, 0, 0, 0.7);
                    color: white;
                    font-size: 16px;
                    font-weight: bold;
                    padding: 20px;
                    border-radius: 8px;
                }
            """)
            self.loading_label.hide()
            # 将加载动画添加到主布局
            self.main_layout.addWidget(self.loading_label)
            self.loading_label.setGeometry(
                self.width() // 2 - 100,
                self.height() // 2 - 50,
                200,
                100
            )
            self.loading_label.raise_()
    
    def show_loading_animation(self, message):
        """显示加载动画"""
        if not hasattr(self, 'loading_label'):
            self.init_auto_label_components()
        
        self.loading_label.setText(message)
        self.loading_label.setGeometry(
            self.width() // 2 - 150,
            self.height() // 2 - 50,
            300,
            100
        )
        self.loading_label.show()
        self.loading_label.raise_()
        # 强制刷新界面
        self.repaint()
    
    def hide_loading_animation(self):
        """隐藏加载动画"""
        if hasattr(self, 'loading_label'):
            self.loading_label.hide()
    
    def show_auto_label_settings(self):
        """显示自动标注设置对话框"""
        self.init_auto_label_components()
        self.auto_label_dialog.set_classes(self.classes)
        
        # 显示对话框
        if self.auto_label_dialog.exec() == QDialog.DialogCode.Accepted:
            # 获取任务类型
            model_task = self.auto_label_dialog.get_model_task()
            
            # 保存设置
            self.auto_label_settings = {
                'model_path': self.auto_label_dialog.get_model_path(),
                'model_task': model_task,  # 保存任务类型
                'conf_threshold': self.auto_label_dialog.sb_conf_threshold.value(),
                'iou_threshold': self.auto_label_dialog.sb_iou_threshold.value(),
                'class_mapping': self.auto_label_dialog.get_class_mappings(),
                'only_unlabeled': self.auto_label_dialog.chk_only_unlabeled.isChecked(),
                'overwrite_labels': self.auto_label_dialog.chk_overwrite.isChecked()
            }
            
            # 输出调试信息

            
            # 更新标注页面的类别列表
            if hasattr(self.auto_label_dialog, 'project_classes'):
                new_classes = self.auto_label_dialog.project_classes
                if new_classes != self.classes:
                    self.classes = new_classes
                    # 更新数据库
                    db.update_project(self.current_project_id, classes=self.classes)
                    # 更新界面并通知AI标注页同步标签
                    self.update_class_list()
                    self.classes_changed.emit({'action': 'sync'})
                    QMessageBox.information(self, "成功", "类别列表已更新")
            self.apply_sam_button_mode()

    def start_sam_annotation(self):
        """开始SAM交互式标注"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        if not self.current_image_id:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return

        if not self.classes:
            QMessageBox.information(self, "提示", "当前项目暂无类别，请先在右侧“类别列表”中添加类别后再使用 SAM 标注")
            return
        
        # 获取当前图片路径
        current_image = None
        for img in self.images:
            if img['id'] == self.current_image_id:
                current_image = img
                break
        
        if not current_image:
            QMessageBox.warning(self, "提示", "无法获取当前图片信息")
            return
        
        image_path = current_image.get('storage_path', '')
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "图片文件不存在")
            return
        
        # 获取已保存的SAM配置（避免临时弹窗回落到默认值）
        sam_config = AutoLabelDialog.get_saved_sam_config()
        
        if not sam_config or not sam_config.get('model_file'):
            QMessageBox.warning(self, "提示", "请先配置SAM模型\n点击: 自动标注 → 设置 → SAM自动标注")
            return
        
        # 获取项目类型
        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        
        # 进入SAM标注模式
        self.canvas.set_tool('sam')
        self.canvas.set_sam_operation_mode("normal")
        self.canvas.sam_config = sam_config
        self.canvas.sam_image_path = image_path
        self.canvas.sam_project_type = project_type
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self.canvas.sam_mode = 'point'  # 'point' 或 'bbox'
        
        # 显示提示
        task_name = "边界框" if project_type == 'detect' else "分割"
        QMessageBox.information(self, "SAM标注模式", 
            f"进入SAM交互式标注模式 ({task_name}任务):\n"
            "• 左键点击: 添加点提示并推理\n"
            "• Ctrl+左键: 添加多个点（不立即推理）\n"
            "• 右键拖拽: 绘制框提示\n"
            "• 松开鼠标: 自动推理\n"
            "• 按ESC键: 退出SAM模式")

    def _get_current_image_path(self) -> str:
        if not self.current_image_id:
            return ""
        for img in self.images:
            if img['id'] == self.current_image_id:
                return img.get('storage_path', '')
        return ""

    def _prepare_memory_mode_context(self):
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return None, None
        image_path = self._get_current_image_path()
        if not image_path or not os.path.exists(image_path):
            QMessageBox.warning(self, "提示", "请先选择有效图片")
            return None, None
        sam_config = AutoLabelDialog.get_saved_sam_config()
        if sam_config.get("usage_mode") != "memory" or sam_config.get("sam_type") not in ("SAM2", "SAM3"):
            QMessageBox.warning(self, "提示", "当前SAM设置不是记忆标注模式（仅SAM2/SAM3支持）")
            return None, None
        return sam_config, image_path

    def start_sam_memory_update(self):
        """打开记忆对象管理并允许在画布上输入对象提示。"""
        sam_config, image_path = self._prepare_memory_mode_context()
        if not sam_config:
            return

        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'

        self.canvas.set_tool('sam')
        self.canvas.set_sam_operation_mode("memory_collect")
        # 清理进入记忆模式前残留的提示与异步回调影响
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self.canvas.sam_config = sam_config
        self.canvas.sam_image_path = image_path
        self.canvas.sam_project_type = project_type

        if self.sam_memory_dialog is None:
            self.sam_memory_dialog = SAMMemoryObjectsDialog(self)
            self.sam_memory_dialog.add_requested.connect(self.add_sam_memory_object_from_canvas)
            self.sam_memory_dialog.delete_requested.connect(self.delete_sam_memory_object)
            self.sam_memory_dialog.save_requested.connect(self.save_sam_memory_and_infer_current)
            self.sam_memory_dialog.closed.connect(self.on_sam_memory_dialog_closed)
        self.sam_memory_dialog.update_objects(self.sam_memory_objects)
        self.sam_memory_dialog.show()
        self.sam_memory_dialog.raise_()
        # 清空画布上的临时标记，并显示所有已有对象的标记
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self._draw_all_memory_objects_on_canvas()

        QMessageBox.information(self, "记忆标注", "请在图片上添加点/框提示后点击“添加对象“，\n可以指定ID来更新已有对象。")

    def add_sam_memory_object_from_canvas(self):
        print(f"[SAM Memory] 添加对象被调用")
        points = list(self.canvas.sam_points)
        bboxes = list(self.canvas.sam_bboxes)
        print(f"[SAM Memory] 当前画布标记: points={len(points)}, bboxes={len(bboxes)}")
        
        if not points and not bboxes:
            QMessageBox.warning(self, "提示", "请先在图片上添加点或框提示")
            return
        
        # 弹窗输入对象ID
        from PyQt6.QtWidgets import QInputDialog
        
        # 获取建议的ID（已有ID的最大值+1）
        suggested_id = 0
        if self.sam_memory_objects:
            suggested_id = max(x["obj_id"] for x in self.sam_memory_objects) + 1
        
        print(f"[SAM Memory] 建议ID: {suggested_id}")
        
        try:
            obj_id, ok = QInputDialog.getInt(
                self,
                "添加记忆对象",
                "请输入对象ID:",
                value=suggested_id,
                min=0,
                max=999
            )
        except Exception as e:
            print(f"[SAM Memory] 输入对话框出错: {e}")
            return
        
        if not ok:
            print(f"[SAM Memory] 用户取消输入")
            return
        
        print(f"[SAM Memory] 用户输入ID: {obj_id}")
        
        # 检查是否已存在该ID，如果存在则更新
        existing_idx = None
        for i, obj in enumerate(self.sam_memory_objects):
            if obj["obj_id"] == obj_id:
                existing_idx = i
                break
        
        if existing_idx is not None:
            # 更新已有对象
            reply = QMessageBox.question(
                self,
                "ID已存在",
                f"ID {obj_id} 已存在，是否覆盖？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
            )
            if reply == QMessageBox.StandardButton.Yes:
                self.sam_memory_objects[existing_idx] = {
                    "obj_id": obj_id,
                    "points": points,
                    "bboxes": bboxes,
                }
                print(f"[SAM Memory] 更新已有对象 ID={obj_id}")
        else:
            # 添加新对象
            self.sam_memory_objects.append({
                "obj_id": obj_id,
                "points": points,
                "bboxes": bboxes,
            })
            print(f"[SAM Memory] 添加新对象 ID={obj_id}")
        
        # 清空画布上的临时标记
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self.canvas.update()
        
        # 更新对话框列表
        if self.sam_memory_dialog:
            self.sam_memory_dialog.update_objects(self.sam_memory_objects)
        
        # 在图上显示所有对象的标记
        print(f"[SAM Memory] 开始绘制所有对象标记")
        self._draw_all_memory_objects_on_canvas()
        print(f"[SAM Memory] 绘制完成")

    def delete_sam_memory_object(self, index: int):
        if 0 <= index < len(self.sam_memory_objects):
            del self.sam_memory_objects[index]
            if self.sam_memory_dialog:
                self.sam_memory_dialog.update_objects(self.sam_memory_objects)
            # 重新绘制所有对象标记
            self._draw_all_memory_objects_on_canvas()
    
    def _draw_all_memory_objects_on_canvas(self):
        """在画布上绘制所有记忆对象的标记"""
        print(f"[SAM Memory] _draw_all_memory_objects_on_canvas 被调用")
        
        # 清空当前临时标记
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        
        # 收集所有对象的标记用于显示
        all_points = []
        all_bboxes = []
        
        print(f"[SAM Memory] 当前记忆对象数: {len(self.sam_memory_objects)}")
        
        for obj in self.sam_memory_objects:
            obj_id = obj.get("obj_id", 0)
            points = obj.get("points", [])
            bboxes = obj.get("bboxes", [])
            
            print(f"[SAM Memory] 对象 ID={obj_id}: points={len(points)}, bboxes={len(bboxes)}")
            
            # 为每个点添加ID信息
            for p in points:
                all_points.append({
                    "x": p[0],
                    "y": p[1],
                    "obj_id": obj_id
                })
            
            # 为每个框添加ID信息
            for bbox in bboxes:
                all_bboxes.append({
                    "x1": bbox[0],
                    "y1": bbox[1],
                    "x2": bbox[2],
                    "y2": bbox[3],
                    "obj_id": obj_id
                })
        
        print(f"[SAM Memory] 总显示点数: {len(all_points)}, 总显示框数: {len(all_bboxes)}")
        
        # 保存到画布的显示列表
        self.canvas.memory_display_points = all_points
        self.canvas.memory_display_bboxes = all_bboxes
        self.canvas.update()
        print(f"[SAM Memory] 画布更新完成")

    def on_sam_memory_dialog_closed(self):
        """记忆对象管理对话框关闭时的处理"""
        print(f"[SAM Memory] 对话框关闭，清除画布上的记忆标记")
        # 清除画布上的记忆标记显示
        self.canvas.memory_display_points = []
        self.canvas.memory_display_bboxes = []
        # 清空临时标记
        self.canvas.sam_points = []
        self.canvas.sam_bboxes = []
        self.canvas.update()
        # 退出SAM记忆采集模式
        self.canvas.set_sam_operation_mode("normal")
        self.canvas.sam_mode_active = False
        self.canvas.set_tool('rectangle')
        print(f"[SAM Memory] 已清除标记并退出SAM模式")

    def clear_sam_memory(self):
        SAMMemoryPredictorManager.instance().clear()
        self.sam_memory_objects = []
        if self.sam_memory_dialog:
            self.sam_memory_dialog.update_objects(self.sam_memory_objects)
        QMessageBox.information(self, "提示", "SAM记忆已清空")

    def _run_memory_predictor(self, sam_config: dict, image_path: str, update_objects: list = None):
        predictor = SAMMemoryPredictorManager.instance().get_predictor(sam_config)
        last_update_results = None
        if update_objects:
            for obj in update_objects:
                kwargs = {
                    "source": image_path,
                    "obj_ids": [obj["obj_id"]],
                    "update_memory": True,
                }
                if obj.get("bboxes"):
                    kwargs["bboxes"] = [obj["bboxes"][-1]]
                if obj.get("points"):
                    kwargs["points"] = [[p[0], p[1]] for p in obj["points"]]
                    kwargs["labels"] = [1] * len(obj["points"])
                update_results = predictor(**kwargs)
                if update_results:
                    last_update_results = update_results
        infer_results = predictor(source=image_path)
        # 某些场景下“同图更新记忆后立刻推理同图”会返回空，回退到最近一次更新结果
        if self._memory_results_empty(infer_results) and not self._memory_results_empty(last_update_results):
            return last_update_results
        return infer_results

    @staticmethod
    def _memory_results_empty(results) -> bool:
        if not results or len(results) == 0:
            return True
        result = results[0]
        if not hasattr(result, "masks") or result.masks is None or result.masks.data is None:
            return True
        try:
            return len(result.masks.data) == 0
        except Exception:
            return False

    def save_sam_memory_and_infer_current(self):
        sam_config, image_path = self._prepare_memory_mode_context()
        if not sam_config:
            return
        if not self.sam_memory_objects:
            QMessageBox.warning(self, "提示", "请先添加至少一个记忆对象")
            return
        try:
            self.canvas.set_sam_operation_mode("normal")
            results = self._run_memory_predictor(sam_config, image_path, update_objects=self.sam_memory_objects)
            ok, msg, ann_count = self._apply_memory_results_to_current_image(results)
            if ok:
                QMessageBox.information(
                    self,
                    "记忆推理完成",
                    f"记忆对象数: {len(self.sam_memory_objects)}\n当前图新增标注: {ann_count}"
                )
            else:
                QMessageBox.warning(self, "记忆推理无结果", msg)
        except Exception as e:
            QMessageBox.critical(self, "记忆推理失败", str(e))

    def _apply_memory_results_to_current_image(self, results):
        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        anns = self._build_annotations_from_memory_results(results, project_type)
        if not anns:
            summary = self._summarize_memory_results(results)
            return False, f"模型返回了空结果（没有可用mask/轮廓）。\n结果摘要: {summary}\n请尝试补充提示点或框后重试。", 0
        for ann in anns:
            self.on_annotation_created(ann)
        self.load_annotations()
        return True, "ok", len(anns)

    @staticmethod
    def _summarize_memory_results(results) -> str:
        if not results or len(results) == 0:
            return "results=empty"
        result = results[0]
        if not hasattr(result, "masks") or result.masks is None or result.masks.data is None:
            return "masks=None"
        data = result.masks.data
        try:
            shape = tuple(data.shape)
        except Exception:
            shape = "unknown"
        try:
            count = len(data)
        except Exception:
            count = "unknown"
        return f"masks.shape={shape}, count={count}"

    def run_sam_memory_single(self):
        sam_config, image_path = self._prepare_memory_mode_context()
        if not sam_config:
            return
        project = db.get_project(self.current_project_id)
        _ = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        try:
            self.canvas.set_sam_operation_mode("normal")
            results = self._run_memory_predictor(sam_config, image_path)
            ok, msg, ann_count = self._apply_memory_results_to_current_image(results)
            if ok:
                QMessageBox.information(self, "单张记忆推理完成", f"当前图新增标注: {ann_count}")
            else:
                QMessageBox.warning(self, "单张记忆推理无结果", msg)
        except Exception as e:
            QMessageBox.critical(self, "推理失败", str(e))

    def run_sam_memory_batch(self):
        sam_config, _ = self._prepare_memory_mode_context()
        if not sam_config:
            return
        if not self.images:
            QMessageBox.warning(self, "提示", "当前项目没有图片")
            return
        if not self.current_image_id:
            QMessageBox.warning(self, "提示", "请先选择当前图片，批量推理将从该图片开始")
            return
        if not self.sam_memory_objects:
            QMessageBox.warning(self, "提示", "请先执行“更新记忆”并保存至少一个对象")
            return

        start_idx = next((i for i, img in enumerate(self.images) if img.get("id") == self.current_image_id), -1)
        if start_idx < 0:
            QMessageBox.warning(self, "提示", "未找到当前图片在列表中的位置")
            return
        images_to_process = self.images[start_idx:]
        if not images_to_process:
            QMessageBox.warning(self, "提示", "当前图片之后没有可处理图片")
            return

        project = db.get_project(self.current_project_id)
        project_type = project.get('type', 'detect') if isinstance(project, dict) else 'detect'
        processed = 0
        failed = 0
        for img in images_to_process:
            image_path = img.get("storage_path", "")
            if not image_path or not os.path.exists(image_path):
                failed += 1
                continue
            try:
                results = self._run_memory_predictor(sam_config, image_path)
                anns = self._build_annotations_from_memory_results(results, project_type)
                for ann in anns:
                    class_id = ann.get('class_id', self.current_class_id)
                    class_name = self.classes[class_id]['name'] if class_id < len(self.classes) else 'unknown'
                    db.add_annotation(
                        image_id=img['id'],
                        project_id=self.current_project_id,
                        class_id=class_id,
                        class_name=class_name,
                        annotation_type=ann['type'],
                        data=ann['data']
                    )
                if anns:
                    db.update_image_status(img['id'], 'annotated')
                processed += 1
            except Exception:
                failed += 1
        self.load_annotations()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
        QMessageBox.information(
            self,
            "批量记忆推理",
            f"处理范围: 从当前图片开始，共 {len(images_to_process)} 张\n完成: {processed} 张，失败: {failed} 张"
        )

    def _build_annotations_from_memory_results(self, results, project_type: str) -> list:
        if not results or len(results) == 0:
            return []
        result = results[0]
        if not hasattr(result, 'masks') or result.masks is None:
            return []
        masks = result.masks.data.cpu().numpy() if hasattr(result.masks.data, 'cpu') else result.masks.data
        masks = np.asarray(masks)
        if masks.ndim == 2:
            mask_list = [masks]
        elif masks.ndim == 3:
            mask_list = [m for m in masks]
        elif masks.ndim >= 4:
            # 兼容 [N, K, H, W]，统一拉平为多个2D mask
            h, w = masks.shape[-2], masks.shape[-1]
            mask_list = [m for m in masks.reshape(-1, h, w)]
        else:
            return []
        annotations = []
        for mask in mask_list:
            if mask.ndim > 2:
                mask = mask.squeeze()
            mask_uint8 = (mask > 0).astype(np.uint8) * 255
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                continue
            if project_type == 'detect':
                all_x, all_y, all_x2, all_y2 = [], [], [], []
                for contour in contours:
                    x, y, w, h = cv2.boundingRect(contour)
                    all_x.append(x)
                    all_y.append(y)
                    all_x2.append(x + w)
                    all_y2.append(y + h)
                annotations.append({
                    'type': 'bbox',
                    'class_id': self.current_class_id,
                    'data': {
                        'x': float(min(all_x)),
                        'y': float(min(all_y)),
                        'width': float(max(all_x2) - min(all_x)),
                        'height': float(max(all_y2) - min(all_y)),
                    }
                })
            else:
                largest_contour = max(contours, key=cv2.contourArea)
                epsilon = 0.005 * cv2.arcLength(largest_contour, True)
                approx_contour = cv2.approxPolyDP(largest_contour, epsilon, True)
                points = [{'x': float(p[0][0]), 'y': float(p[0][1])} for p in approx_contour]
                if points:
                    annotations.append({
                        'type': 'polygon',
                        'class_id': self.current_class_id,
                        'data': {'points': points}
                    })
        return annotations
    
    def show_batch_process_dialog(self):
        """显示批量处理标注对话框"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        
        if not self.images:
            QMessageBox.warning(self, "提示", "项目中没有图片")
            return
        
        # 创建对话框
        dialog = BatchProcessDialog(self, self.classes, len(self.images))
        dialog.process_requested.connect(self.on_batch_process_requested)
        
        # 进入像素点选择模式
        self.batch_process_dialog = dialog
        self.canvas.batch_process_mode = True
        self.canvas.batch_process_points = []
        self.canvas.batch_process_dialog = dialog
        
        # 显示对话框（非模态，允许在图片上点击）
        dialog.show()
    
    def on_batch_process_requested(self, config):
        """处理批量处理请求"""
        # 退出像素点选择模式
        self.canvas.batch_process_mode = False
        self.canvas.batch_process_points = []
        self.canvas.batch_process_dialog = None
        
        # 执行批量处理
        self.execute_batch_process(config)
    
    def execute_batch_process(self, config):
        """执行批量处理"""
        points = config['points']
        start_idx = config['start_idx']
        end_idx = config['end_idx']
        operation = config['operation']
        
        # 获取处理范围内的图片
        images_to_process = self.images[start_idx:end_idx+1]
        
        if not images_to_process:
            QMessageBox.warning(self, "提示", "没有需要处理的图片")
            return
        
        # 显示进度对话框
        from PyQt6.QtWidgets import QProgressDialog
        progress = QProgressDialog("正在批量处理标注...", "取消", 0, len(images_to_process), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.show()
        
        processed_count = 0
        modified_count = 0
        
        try:
            for i, image_data in enumerate(images_to_process):
                if progress.wasCanceled():
                    break
                
                progress.setValue(i)
                progress.setLabelText(f"正在处理第 {i+1}/{len(images_to_process)} 张图片...")
                
                image_id = image_data['id']
                annotations = db.get_image_annotations(image_id)
                img_modified = False

                if not annotations:
                    continue
                
                # 检查每个标注是否覆盖选择的像素点
                for annotation in annotations:
                    annotation_data = annotation.get('data', {})
                    annotation_type = annotation.get('type', 'bbox')
                    
                    # 检查标注是否覆盖任何选择的像素点
                    covers_point = False
                    for point in points:
                        px, py = point
                        if self.is_point_in_annotation_data(px, py, annotation_data, annotation_type):
                            covers_point = True
                            break
                    
                    if covers_point:
                        if operation == 'delete':
                            # 批量删除：检查类别是否在目标类别列表中
                            target_classes = config.get('target_classes', [])
                            if annotation.get('class_id') in target_classes:
                                db.delete_annotation(annotation['id'])
                                modified_count += 1
                                img_modified = True
                        else:
                            # 批量修改：检查类别是否在源类别列表中
                            source_classes = config.get('source_classes', [])
                            target_class = config.get('target_class')
                            if annotation.get('class_id') in source_classes:
                                target_class_info = next(
                                    (cls for cls in self.classes if cls['id'] == target_class),
                                    None
                                )
                                db.update_annotation(
                                    annotation['id'],
                                    class_id=target_class,
                                    class_name=target_class_info['name'] if target_class_info else None
                                )
                                modified_count += 1
                                img_modified = True

                # 批量修改过的图按AI快照差异同步整图锁定与复标状态
                if img_modified:
                    remaining = db.get_image_annotations(image_id)
                    if not remaining and db.get_image_ai_original(image_id) is None:
                        db.update_image_status(image_id, 'pending')
                        self._sync_image_review_state(image_id, lock=False)
                    else:
                        self._sync_image_review_state(image_id, lock=True)

                processed_count += 1
            
            progress.setValue(len(images_to_process))
            
            # 显示结果
            QMessageBox.information(
                self, 
                "批量处理完成", 
                f"处理完成！\n处理了 {processed_count} 张图片\n修改了 {modified_count} 个标注"
            )
            
            # 刷新当前图片的标注显示
            self._invalidate_sample_stats_cache()
            if self.current_image_id:
                self.load_annotations()
            self.update_sample_control_panel()
            
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量处理出错: {str(e)}")
    
    def is_point_in_annotation_data(self, px: int, py: int, data: dict, ann_type: str) -> bool:
        """检查点是否在标注数据内"""
        if ann_type == 'bbox':
            x = data.get('x', 0)
            y = data.get('y', 0)
            width = data.get('width', 0)
            height = data.get('height', 0)
            return x <= px <= x + width and y <= py <= y + height
        elif ann_type == 'polygon':
            points = data.get('points', [])
            if len(points) < 3:
                return False
            # 使用射线法判断点是否在多边形内
            return self.point_in_polygon(px, py, points)
        return False
    
    def point_in_polygon(self, x: int, y: int, polygon: list) -> bool:
        """射线法判断点是否在多边形内"""
        n = len(polygon)
        inside = False
        p1x, p1y = polygon[0]['x'], polygon[0]['y']
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]['x'], polygon[i % n]['y']
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        return inside
    
    def _confirm_locked_single_inference(self) -> bool:
        """若当前图片已人工锁定，单张推理前提示确认，防止误覆盖人工成果"""
        img = next(
            (i for i in self.images if i['id'] == self.current_image_id),
            self.current_image_data,
        )
        if img and img.get('locked'):
            reply = QMessageBox.question(
                self,
                "图片已锁定",
                "当前图片已处于人工锁定状态（🔒）。\n确定仍要对该图片运行 YOLO 自动标注吗？",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            return reply == QMessageBox.StandardButton.Yes
        return True

    def run_single_inference(self):
        """运行单张图像推理"""
        if not self.current_image_data:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return
        if not self._confirm_locked_single_inference():
            return
        
        # 显示加载动画
        self.show_loading_animation("正在进行单张推理...")
        
        try:
            from core.auto_labeler import AutoLabeler
            
            if hasattr(self, 'auto_label_settings'):
                settings = self.auto_label_settings
                model_path = settings.get('model_path', "yolov8n")
                model_task = settings.get('model_task', 'detect')
                conf_threshold = settings.get('conf_threshold', 0.5)
                iou_threshold = settings.get('iou_threshold', 0.45)
                class_mapping = settings.get('class_mapping', {})
                overwrite_labels = settings.get('overwrite_labels', False)
            else:
                model_path = "yolov8n"
                model_task = 'detect'
                conf_threshold = 0.5
                iou_threshold = 0.45
                class_mapping = {}
                overwrite_labels = False
            
            labeler = AutoLabeler(model_path, self.model_manager)
            
            import re
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
                    'class_mappings': class_mapping
                }
                labeler.load_model(load_config)
            elif os.path.exists(model_path):
                load_config = {
                    'model_version': '',
                    'model_size': '',
                    'model_source': 'custom',
                    'model_task': model_task,
                    'custom_model_path': model_path,
                    'class_mappings': class_mapping
                }
                labeler.load_model(load_config)
            
            config = {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'class_mappings': class_mapping
            }
            annotations = labeler.process_single_image(
                self.current_image_data['storage_path'], 
                self.current_image_id,
                config
            )
            
            if annotations and self.current_image_id:
                labeler.save_annotations(
                    annotations,
                    self.current_image_id,
                    overwrite=overwrite_labels,
                    ignore_lock=True,
                )
                self.refresh_classes()
                self.classes_changed.emit({'action': 'sync'})
                self.load_current_image_annotations()
                self.update_image_list_display()
                QMessageBox.information(self, "成功", "自动标注完成！")
            else:
                if overwrite_labels and self.current_image_id:
                    for ann in db.get_image_annotations(self.current_image_id):
                        if not (ann.get('attributes') or {}).get('locked'):
                            db.delete_annotation(ann['id'])
                    self.load_current_image_annotations()
                QMessageBox.information(self, "提示", "未检测到目标")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标注失败: {str(e)}")
        finally:
            self.hide_loading_animation()
    
    def run_batch_inference(self):
        """运行批量推理"""
        if not self.current_project_id or len(self.images) == 0:
            QMessageBox.warning(self, "提示", "项目中没有图片")
            return
        
        self.show_loading_animation("正在进行批量推理...")
        
        try:
            if hasattr(self, 'auto_label_settings'):
                settings = self.auto_label_settings
                model_path = settings.get('model_path', "yolov8n")
                model_task = settings.get('model_task', 'detect')
                conf_threshold = settings.get('conf_threshold', 0.5)
                iou_threshold = settings.get('iou_threshold', 0.45)
                class_mapping = settings.get('class_mapping', {})
                only_unlabeled = settings.get('only_unlabeled', True)
                overwrite_labels = settings.get('overwrite_labels', False)
            else:
                model_path = "yolov8n"
                model_task = 'detect'
                conf_threshold = 0.5
                iou_threshold = 0.45
                class_mapping = {}
                only_unlabeled = True
                overwrite_labels = False
            
            if only_unlabeled:
                candidate_images = [img for img in self.images if img.get('status') not in ['annotated', 'completed']]
            else:
                candidate_images = list(self.images)

            if not candidate_images:
                QMessageBox.warning(
                    self,
                    "提示",
                    f"没有符合条件的图片\n总图片数: {len(self.images)}\n"
                    f"未标注图片数: {len([img for img in self.images if img.get('status') not in ['annotated', 'completed']])}"
                )
                return

            filtered_images = [img for img in candidate_images if not img.get('locked')]
            if not filtered_images:
                QMessageBox.information(
                    self,
                    "提示",
                    f"符合条件的 {len(candidate_images)} 张图片均已人工锁定（🔒），已自动保护跳过。\n如需重新自动标注，请先解锁对应图片。"
                )
                return
            
            self.batch_labeling_manager.start_batch_processing(
                model_path,
                filtered_images,
                conf_threshold,
                iou_threshold,
                class_mapping,
                self.model_manager,
                model_task,
                overwrite_labels=overwrite_labels,
            )
        except Exception as e:
            QMessageBox.critical(self, "错误", f"批量推理失败: {str(e)}")
        finally:
            self.hide_loading_animation()
    
    def on_single_inference_requested(self, model_path, conf_threshold, iou_threshold, class_mapping, image_path, model_task='detect'):
        """单张推理请求回调"""
        from core.auto_labeler import AutoLabeler

        if not self._confirm_locked_single_inference():
            return
        
        try:
            labeler = AutoLabeler(model_path, self.model_manager)
            
            import re
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
                    'class_mappings': class_mapping
                }
                labeler.load_model(load_config)
            elif os.path.exists(model_path):
                load_config = {
                    'model_version': '',
                    'model_size': '',
                    'model_source': 'custom',
                    'model_task': model_task,
                    'custom_model_path': model_path,
                    'class_mappings': class_mapping
                }
                labeler.load_model(load_config)
            
            config = {
                'conf_threshold': conf_threshold,
                'iou_threshold': iou_threshold,
                'class_mappings': class_mapping
            }
            annotations = labeler.process_single_image(image_path, self.current_image_id, config)
            
            overwrite_labels = False
            if hasattr(self, 'auto_label_settings'):
                overwrite_labels = self.auto_label_settings.get('overwrite_labels', False)
            
            if annotations and self.current_image_id:
                labeler.save_annotations(
                    annotations,
                    self.current_image_id,
                    overwrite=overwrite_labels,
                    ignore_lock=True,
                )
                self.refresh_classes()
                self.classes_changed.emit({'action': 'sync'})
                self.load_current_image_annotations()
                self.update_image_list_display()
                QMessageBox.information(self, "成功", "自动标注完成！")
            else:
                if overwrite_labels and self.current_image_id:
                    for ann in db.get_image_annotations(self.current_image_id):
                        if not (ann.get('attributes') or {}).get('locked'):
                            db.delete_annotation(ann['id'])
                    self.load_current_image_annotations()
                QMessageBox.information(self, "提示", "未检测到目标")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"自动标注失败: {str(e)}")
    
    def on_batch_inference_requested(self, model_path, conf_threshold, iou_threshold, class_mapping, images, only_unlabeled, model_task='detect'):
        """批量推理请求回调"""
        if only_unlabeled:
            candidate_images = [img for img in images if img.get('status') != 'annotated']
        else:
            candidate_images = list(images)
        
        if not candidate_images:
            QMessageBox.warning(self, "提示", "没有符合条件的图片")
            return

        filtered_images = [img for img in candidate_images if not img.get('locked')]
        if not filtered_images:
            QMessageBox.information(self, "提示", "符合条件的图片均已人工锁定（🔒），已自动保护跳过。")
            return

        overwrite_labels = False
        if hasattr(self, 'auto_label_settings'):
            overwrite_labels = self.auto_label_settings.get('overwrite_labels', False)
        elif self.auto_label_dialog is not None:
            overwrite_labels = self.auto_label_dialog.chk_overwrite.isChecked()
        
        self.batch_labeling_manager.start_batch_processing(
            model_path,
            filtered_images,
            conf_threshold,
            iou_threshold,
            class_mapping,
            self.model_manager,
            model_task,
            overwrite_labels=overwrite_labels,
        )
    
    def on_batch_inference_progress(self, progress, current, total, image_name):
        """批量推理进度回调"""
        # 更新状态栏
        self.status_annotation.setText(f"自动标注: {current}/{total}")
        self.status_position.setText(f"当前: {image_name}")
        self.repaint()
    
    def on_batch_inference_completed(self, success, message, processed_count):
        """批量推理完成回调"""
        self.refresh_classes()
        self.classes_changed.emit({'action': 'sync'})
        self.load_image_list()
        self.load_current_image_annotations()
        self.update_status_bar()

        if success:
            QMessageBox.information(self, "成功", message or f"批量自动标注完成！\n处理了 {processed_count} 张图片")
        else:
            QMessageBox.critical(self, "错误", f"批量自动标注失败: {message}")
    
    def load_current_image_annotations(self):
        """加载当前图像的标注"""
        if self.current_image_id:
            self.annotations = db.get_image_annotations(self.current_image_id)
            self.canvas.set_annotations(self.annotations)
            self.update_status_bar()
            self._invalidate_sample_stats_cache()
            self.update_sample_control_panel()
    
    def update_status_bar(self):
        """更新状态栏"""
        if self.images and self.current_image_id:
            # 找到当前图像的索引
            current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), -1)
            if current_index >= 0:
                self.status_image.setText(f"当前: {current_index + 1}/{len(self.images)}")
        else:
            self.status_image.setText("当前: 0/0")
        
        # 更新标注数量
        self.status_annotation.setText(f"标注: {len(self.annotations)}")
        
        # 更新工具状态
        tool_names = {'rectangle': '矩形', 'polygon': '多边形', 'move': '移动'}
        tool_name = tool_names.get(self.canvas.current_tool, '矩形')
        self.status_tool.setText(f"工具: {tool_name}")
    
    def on_class_selected(self):
        """类别选中事件"""
        current_item = self.class_list.currentItem()
        if current_item:
            self.current_class_id = current_item.data(Qt.ItemDataRole.UserRole)
            # 更新画布当前类别
            self.canvas.current_class_id = self.current_class_id
            self._sync_attr_class_combo_from_list()
        self._refresh_annotation_class_controls()
    
    def filter_images(self, filter_text: str):
        """筛选图片"""
        img_map = {img['id']: img for img in self.images}
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            image_id = item.data(Qt.ItemDataRole.UserRole)
            image_data = img_map.get(image_id)
            if not image_data:
                continue
            
            status = image_data.get('status', 'pending')
            locked = bool(image_data.get('locked'))
            modified = bool(image_data.get('modified'))
            
            if filter_text == "全部":
                item.setHidden(False)
            elif filter_text == "未标注":
                item.setHidden(status != 'pending')
            elif filter_text == "已标注":
                item.setHidden(status == 'pending')
            elif filter_text == "待复核":
                item.setHidden(not (status == 'annotated' and not locked))
            elif filter_text == "已锁定":
                item.setHidden(not locked)
            elif filter_text == "已复标修正":
                item.setHidden(not modified)
    
    def on_image_selected(self, item: QListWidgetItem):
        """图片选中事件"""
        image_id = item.data(Qt.ItemDataRole.UserRole)
        self.load_image(image_id)
    
    def on_task_changed(self, task_type):
        """任务类型切换事件"""
        if self.current_project_id:
            # 更新项目的任务类型
            db.update_project(self.current_project_id, type=task_type)
            # 重新加载当前图片的标注
            if self.current_image_id:
                self.load_annotations()
        
        # 根据任务类型调整标注工具显示
        self.adjust_tool_visibility(task_type)
    
    def adjust_tool_visibility(self, task_type):
        """根据任务类型调整标注工具的显示"""
        # 隐藏所有标注工具
        self.btn_draw_tool.hide()
        self.btn_keypoint.hide()
        
        # 根据任务类型显示对应的工具
        if task_type == 'detect':
            self.btn_draw_tool.show()
            self.select_draw_tool('rectangle')
        elif task_type == 'segment':
            self.btn_draw_tool.show()
            self.select_draw_tool('polygon')
        elif task_type == 'pose':
            self.btn_keypoint.show()
            # 默认选中关键点工具
            self.btn_keypoint.setChecked(True)
            self.set_tool('keypoint')
        elif task_type == 'classify':
            # 分类任务不需要标注工具
            pass
        
        # 移动工具始终显示
        self.btn_move.show()
        self.refresh_toolbar_button_layout()
    
    def load_image(self, image_id: int):
        """加载图片"""
        self.current_image_id = image_id
        self.update_delete_menu_state()
        
        # 找到图片数据
        image_data = next((img for img in self.images if img['id'] == image_id), None)
        if not image_data:
            return
        
        self.current_image_data = image_data
        
        # 加载到画布
        if image_data.get('storage_path'):
            self.canvas.load_image(image_data['storage_path'])
            # SAM模式下切图时，同步推理上下文到当前图片，避免继续使用旧图推理
            if self.canvas.sam_mode_active:
                self.canvas.sam_image_path = image_data['storage_path']
                self.canvas.sam_config = AutoLabelDialog.get_saved_sam_config()
                self.canvas.sam_points = []
                self.canvas.sam_bboxes = []
                self.canvas.sam_drawing_bbox = False
        
        # 加载标注
        self.load_annotations()
        
        # 更新状态栏
        self.update_status_bar()
        
        # 高亮当前项并滚动到该项
        for i in range(self.image_list.count()):
            item = self.image_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == image_id:
                item.setSelected(True)
                # 滚动到当前项
                self.image_list.scrollToItem(item)
            else:
                item.setSelected(False)
    
    def load_annotations(self):
        """加载标注"""
        if not self.current_image_id:
            return

        self.annotations = db.get_image_annotations(self.current_image_id)

        # 同步整图锁定与复标修正状态(画布横幅+按钮)
        img = next(
            (i for i in self.images if i['id'] == self.current_image_id),
            None
        )
        locked = bool(img and img.get('locked'))
        modified = bool(img and img.get('modified'))
        self.canvas.image_locked = locked
        self.canvas.image_modified = modified
        self._update_lock_button_state_image()

        self.canvas.set_annotations(self.annotations)
        self.update_status_bar()
        self._refresh_annotation_class_controls()

    def _update_lock_button_state_image(self):
        """按当前图片的整图锁定状态更新按钮（始终只代表整张图片，不被单框选中状态覆盖）"""
        if not hasattr(self, 'btn_lock'):
            return
        img = next(
            (i for i in self.images if i['id'] == self.current_image_id),
            None
        ) if self.current_image_id else None
        locked = bool(img and img.get('locked'))
        self.btn_lock.setText("🔓 解锁本图" if locked else "🔒 锁定本图")
        self.btn_lock.setEnabled(bool(self.current_image_id))

    def _sync_image_review_state(self, image_id: int = None, lock: Optional[bool] = True):
        """统一同步图片的整图锁定(locked)与人工复标修正(modified)状态。

        Args:
            image_id: 目标图片ID，默认为当前打开图片
            lock:
                - True: 人工编辑后自动锁定整张图(方案 A)
                - False: 显式解锁整张图，同时清空 modified
                - None: 保持现有 locked 状态不变，仅按 AI 快照重算 modified（例如撤销操作）
        """
        from core.qwen_annotator import compute_image_ai_modified

        image_id = image_id or self.current_image_id
        if not image_id:
            return

        try:
            if lock is True:
                db.set_image_locked(image_id, True)
                is_modified = compute_image_ai_modified(image_id)
                db.set_image_modified(image_id, is_modified)
                target_locked = True
            elif lock is False:
                db.set_image_locked(image_id, False)
                db.set_image_modified(image_id, False)
                target_locked = False
                is_modified = False
            else:
                img_row = db.get_image(image_id)
                target_locked = bool(img_row and img_row.get('locked'))
                is_modified = compute_image_ai_modified(image_id) if target_locked else False
                db.set_image_modified(image_id, is_modified)
        except Exception:
            return

        fresh_img = db.get_image(image_id)
        for img in self.images:
            if img['id'] == image_id:
                img['locked'] = 1 if target_locked else 0
                img['modified'] = 1 if is_modified else 0
                if fresh_img:
                    img['status'] = fresh_img.get('status', img.get('status'))
                break

        self.update_image_list_display()
        if image_id == self.current_image_id:
            if fresh_img:
                self.current_image_data = fresh_img
            self.canvas.image_locked = target_locked
            self.canvas.image_modified = is_modified
            self._update_lock_button_state_image()
            self.canvas.update()

    def _mark_image_modified(self, image_id: int = None):
        """兼容入口: 人工修改标注后按方案 A 自动锁定本图，并根据 AI 快照计算是否标记为 ✏️ 复标修正"""
        self._sync_image_review_state(image_id=image_id, lock=True)
    
    def set_tool(self, tool: str):
        """设置工具"""
        self.canvas.set_tool(tool)

        if tool in ('rectangle', 'polygon'):
            self.default_draw_tool = tool
            if hasattr(self, 'btn_draw_tool'):
                self.btn_draw_tool.setChecked(True)
                self.refresh_draw_tool_button()
        elif tool == 'keypoint' and hasattr(self, 'btn_keypoint'):
            self.btn_keypoint.setChecked(True)
        elif tool == 'move' and hasattr(self, 'btn_move'):
            self.btn_move.setChecked(True)
        
        # 确保status_tool存在时才更新
        if hasattr(self, 'status_tool'):
            tool_names = {
                'rectangle': '矩形',
                'polygon': '多边形',
                'move': '移动',
                'keypoint': '关键点',
                'obb': '旋转矩形'
            }
            self.status_tool.setText(f"工具: {tool_names.get(tool, tool)}")

    def clear_current_image_view(self):
        """清空当前图片显示与相关状态。"""
        self.current_image_id = None
        self.current_image_data = None
        self.annotations = []
        self.history = []
        self.history_index = -1
        self.canvas.selected_annotation_id = None
        self.canvas.image_locked = False
        self.canvas.image_modified = False
        self.canvas.set_annotations([])
        self.canvas.load_image("")
        self.clear_attribute_panel(refresh_sample_panel=False)
        self._update_lock_button_state_image()
        self.update_status_bar()
        self.update_delete_menu_state()

    def delete_current_image(self):
        """删除当前图片及其全部标注和实际文件。"""
        if not self.current_project_id or not self.current_image_id:
            QMessageBox.warning(self, "提示", "当前没有可删除的图片")
            return

        current_index = next(
            (i for i, image in enumerate(self.images) if image['id'] == self.current_image_id),
            -1
        )
        current_image = next(
            (image for image in self.images if image['id'] == self.current_image_id),
            None
        )
        if current_index < 0 or current_image is None:
            QMessageBox.warning(self, "提示", "当前图片不存在或已失效，请先重新加载")
            return

        filename = current_image.get('filename', str(self.current_image_id))
        reply = QMessageBox.question(
            self,
            "确认删除当前图片",
            (
                f"将删除当前图片“{filename}”及其全部标注，并删除实际文件。\n"
                "此操作不可恢复，是否继续？"
            ),
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        fallback_index = current_index if current_index < len(self.images) - 1 else current_index - 1
        deleted = db.delete_image(self.current_image_id)
        if not deleted:
            QMessageBox.warning(self, "提示", "删除当前图片失败")
            return

        self.load_image_list()
        self.update_class_list()

        if self.images:
            fallback_index = max(0, min(fallback_index, len(self.images) - 1))
            next_image_id = self.images[fallback_index]['id']
            self.load_image(next_image_id)
        else:
            self.clear_current_image_view()
    
    def on_annotation_created(self, annotation: dict):
        """标注创建事件"""
        if not self.current_image_id or not self.current_project_id:
            return

        # 当前项目尚未添加任何类别时，引导用户输入首个类别名称，避免写入无效 unknown 类别
        if not self.classes:
            name, ok = QInputDialog.getText(
                self, "添加类别", "当前项目暂无标注类别，请输入首个类别名称:"
            )
            name = (name or "").strip()
            if not ok or not name:
                self.canvas.update()
                return
            self.classes.append({
                'id': 0,
                'name': name,
                'color': '#EF4444',
            })
            db.update_project(self.current_project_id, classes=self.classes)
            self.current_class_id = 0
            self.canvas.current_class_id = 0
            self.update_class_list()
            self.classes_changed.emit({'action': 'add', 'name': name})
            class_id = 0
            class_name = name
        else:
            # 从标注数据中获取类别ID（由画布传递）
            class_id = annotation.get('class_id', self.current_class_id)
            matched_cls = next((c for c in self.classes if c.get('id') == class_id), None)
            if matched_cls is None:
                matched_cls = self.classes[0]
                class_id = matched_cls['id']
            class_name = matched_cls['name']
        
        # 保存到数据库（标记为人工创建）
        attrs = dict(annotation.get('attributes') or {})
        attrs.setdefault('source', 'manual')
        ann_id = db.add_annotation(
            image_id=self.current_image_id,
            project_id=self.current_project_id,
            class_id=class_id,
            class_name=class_name,
            annotation_type=annotation['type'],
            data=annotation['data'],
            attributes=attrs,
        )
        
        # 添加到历史记录
        self.add_history('create', {'annotation_id': ann_id})
        
        # 更新图片状态为已标注
        db.update_image_status(self.current_image_id, 'annotated')

        # 方案 A: 手动画框自动锁定整张图；仅在偏离 AI 快照时标记为 ✏️
        self._sync_image_review_state(self.current_image_id, lock=True)
        
        # 重新加载标注
        self.load_annotations()
        
        # 更新图片列表显示
        self.update_image_list_display()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
    
    def on_annotation_selected(self, annotation_id: int):
        """标注选中事件"""
        # 更新属性面板
        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            self.update_attribute_panel(annotation)
    
    def on_annotation_modified(self, annotation_id: int, data: dict):
        """标注修改事件（拖动或调整大小后）"""
        import copy

        old_record = db.get_annotation(annotation_id)
        if not old_record:
            return
        old_data = copy.deepcopy(old_record.get('data') or {})
        old_attrs = copy.deepcopy(old_record.get('attributes') or {})
        if old_data == data:
            return

        # 更新数据库中的标注并锁定该框
        db.update_annotation(annotation_id, data=data)
        db.set_annotation_locked(annotation_id, True)

        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            annotation['data'] = copy.deepcopy(data)
            attrs = dict(annotation.get('attributes') or {})
            attrs['locked'] = True
            annotation['attributes'] = attrs
            self.update_attribute_panel(annotation)

        # 添加到历史记录（保存修改前的真实坐标与属性，支持 Ctrl+Z 完整回退）
        self.add_history('modify', {
            'annotation_id': annotation_id,
            'old_data': old_data,
            'old_attributes': old_attrs,
        })

        # 方案 A: 人工调整框自动锁定本图，并按 AI 快照差异更新 ✏️
        self._sync_image_review_state(self.current_image_id, lock=True)

    def on_toggle_lock(self):
        """手动锁定/解锁整张图（支持按钮点击与快捷键 L）"""
        from core.qwen_annotator import remove_image_from_breakpoint

        if not self.current_image_id:
            QMessageBox.warning(self, "提示", "当前没有打开的图片")
            return

        img = next(
            (i for i in self.images if i['id'] == self.current_image_id),
            None
        )
        cur_locked = bool(img and img.get('locked'))
        new_locked = not cur_locked

        if new_locked:
            # 锁定未标注的空图时自动确认为已标注（负样本），避免出现“未标注但已锁定”的矛盾状态
            if img and img.get('status') != 'annotated':
                db.update_image_status(self.current_image_id, 'annotated')
            self._sync_image_review_state(self.current_image_id, lock=True)
            updated_img = next((i for i in self.images if i['id'] == self.current_image_id), None)
            if updated_img and updated_img.get('modified'):
                self.show_save_feedback("已锁定本图（含复标修正）")
            else:
                self.show_save_feedback("已确认并锁定本图")
        else:
            # 解锁本图：解除整图锁与单框锁，清空 ✏️，并从 AI 断点中移除该图以便单独重标
            db.clear_image_annotation_locks(self.current_image_id)
            if self.current_project_id and img and img.get('filename'):
                project = db.get_project(self.current_project_id)
                if project and project.get('storage_path'):
                    remove_image_from_breakpoint(project['storage_path'], img['filename'])
            self._sync_image_review_state(self.current_image_id, lock=False)
            self.load_annotations()
            self.show_save_feedback("已解锁本图（可重新参与AI标注）")

        self.update_status_bar()

    def on_save_annotations(self):
        """手动保存: 把当前图片的所有标注写入数据库"""
        if not self.current_image_id:
            QMessageBox.warning(self, "提示", "当前没有打开的图片")
            return

        count = 0
        for ann in self.annotations:
            try:
                db.update_annotation(
                    ann['id'],
                    data=ann.get('data', {}).copy(),
                    class_id=ann.get('class_id'),
                    class_name=ann.get('class_name')
                )
                count += 1
            except Exception:
                pass

        self.show_save_feedback(f"已保存 {count} 条标注")

    def show_save_feedback(self, text: str):
        """在状态栏显示保存成功的临时提示"""
        if not hasattr(self, 'status_save') or self.status_save is None:
            return
        self.status_save.setText(f"✔ {text}")
        if not hasattr(self, '_save_feedback_timer'):
            self._save_feedback_timer = QTimer(self)
            self._save_feedback_timer.setSingleShot(True)
            self._save_feedback_timer.timeout.connect(self._clear_save_feedback)
        self._save_feedback_timer.start(2000)

    def _clear_save_feedback(self):
        if hasattr(self, 'status_save') and self.status_save is not None:
            try:
                self.status_save.setText("")
            except RuntimeError:
                pass
    
    def on_annotation_deleted(self, annotation_id: int):
        """标注删除事件"""
        import copy

        # 保存到历史记录
        annotation = next((ann for ann in self.annotations if ann['id'] == annotation_id), None)
        if annotation:
            self.add_history('delete', copy.deepcopy(annotation))
        
        # 从数据库删除
        db.delete_annotation(annotation_id)

        # 重新加载剩余标注
        self.load_annotations()
        self.canvas.selected_annotation_id = None
        self.clear_attribute_panel(refresh_sample_panel=False)
        
        # 检查图片是否还有标注：
        # - 若该图无 AI 快照（纯人工图）且删空了全部框，恢复为未标注并解除锁定；
        # - 若该图有 AI 快照（删空了 AI 误检框），保持已标注（负样本）并锁定 + 标记 ✏️ 复标修正！
        remaining_annotations = self.annotations
        has_ai_snapshot = db.get_image_ai_original(self.current_image_id) is not None
        if not remaining_annotations and not has_ai_snapshot:
            db.update_image_status(self.current_image_id, 'pending')
            self._sync_image_review_state(self.current_image_id, lock=False)
        else:
            if has_ai_snapshot:
                db.update_image_status(self.current_image_id, 'annotated')
            self._sync_image_review_state(self.current_image_id, lock=True)

        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
    
    def delete_selected_annotation(self):
        """删除选中的标注"""
        if self.canvas.selected_annotation_id is not None:
            self.on_annotation_deleted(self.canvas.selected_annotation_id)
    
    def update_attribute_panel(self, annotation: dict):
        """更新属性面板"""
        # 设置类别
        class_id = annotation.get('class_id', 0)
        self._set_class_list_selection(class_id)
        # 始终保持整图锁定按钮状态
        self._update_lock_button_state_image()

    def clear_attribute_panel(self, refresh_sample_panel=True):
        """清空属性面板"""
        self.attr_class.blockSignals(True)
        self.attr_class.setCurrentIndex(-1)
        self.attr_class.blockSignals(False)
        self.sample_delete_count.blockSignals(True)
        self.sample_delete_count.setValue(0)
        self.sample_delete_count.blockSignals(False)
        self._update_lock_button_state_image()
        if refresh_sample_panel:
            self.update_sample_control_panel()
        self._refresh_annotation_class_controls()
    
    def on_attr_class_changed(self, index):
        """属性类别改变事件"""
        if self.canvas.selected_annotation_id is None:
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.canvas.selected_annotation_id), None)
        if not annotation:
            return
        
        class_id = self.attr_class.currentData()
        if class_id is not None:
            annotation['class_id'] = class_id
            self._set_class_list_selection(class_id)
            self.canvas.update()
    
    def apply_annotation_changes(self):
        """应用标注修改到数据库"""
        if self.canvas.selected_annotation_id is None:
            QMessageBox.warning(self, "提示", "请先选中一个标注再修改类别")
            return
        
        annotation = next((ann for ann in self.annotations if ann['id'] == self.canvas.selected_annotation_id), None)
        if not annotation:
            return
        
        # 获取新的类别信息
        class_id = self._get_selected_class_list_id()
        if class_id is None:
            QMessageBox.warning(self, "提示", "当前没有可用类别")
            return

        class_info = next((cls for cls in self.classes if cls['id'] == class_id), None)
        class_name = class_info['name'] if class_info else 'unknown'

        if class_id == annotation.get('class_id'):
            self._refresh_annotation_class_controls()
            return

        import copy
        old_class_id = annotation.get('class_id')
        old_class_name = annotation.get('class_name')
        old_attrs = copy.deepcopy(annotation.get('attributes') or {})

        # 更新到数据库并锁定该框
        updated = db.update_annotation(annotation['id'], class_id=class_id, class_name=class_name)
        if not updated:
            QMessageBox.warning(self, "提示", "标注类别修改失败")
            return
        db.set_annotation_locked(annotation['id'], True)

        # 记录撤销历史并同步整图锁定与复标状态
        self.add_history('change_class', {
            'annotation_id': annotation['id'],
            'old_class_id': old_class_id,
            'old_class_name': old_class_name,
            'old_attributes': old_attrs,
        })
        self._sync_image_review_state(self.current_image_id, lock=True)
        
        # 刷新显示
        self.load_annotations()
        self.canvas.selected_annotation_id = annotation['id']
        updated_annotation = next((ann for ann in self.annotations if ann['id'] == annotation['id']), None)
        if updated_annotation:
            self.update_attribute_panel(updated_annotation)
        self.canvas.update()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
        
        QMessageBox.information(self, "成功", "标注修改已保存")

    def delete_random_samples_for_target(self):
        """按目标标签随机删除样本图像"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return

        if self.random_delete_worker and self.random_delete_worker.isRunning():
            QMessageBox.warning(self, "提示", "随机删样任务仍在执行，请等待完成后再试")
            return

        target_class_id = self.sample_target_class.currentData()
        target_name = self.sample_target_class.currentText()
        delete_count = self.sample_delete_count.value()

        if target_class_id is None:
            QMessageBox.warning(self, "提示", "请先选择一个删样标签")
            return

        if delete_count <= 0:
            QMessageBox.warning(self, "提示", "删样数量必须大于 0")
            return

        class_sample_counts, negative_sample_count = self._get_sample_stats()
        if target_class_id == NEGATIVE_SAMPLE_CLASS_ID:
            available_count = negative_sample_count
        else:
            available_count = class_sample_counts.get(target_class_id, 0)
        if available_count == 0:
            QMessageBox.warning(self, "提示", f"当前标签“{target_name}”没有可删除的样本")
            return

        delete_count = min(delete_count, available_count)
        reply = QMessageBox.question(
            self,
            "确认删除",
            f"将随机删除 {delete_count} 张“{target_name}”样本图。\n"
            f"这是删除整张图片及其标注，不是只删框，且不可撤销。\n"
            f"是否继续？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        from PyQt6.QtWidgets import QProgressDialog

        self.random_delete_progress = QProgressDialog(
            f"正在准备删除“{target_name}”样本图...",
            "取消",
            0,
            delete_count,
            self
        )
        self.random_delete_progress.setWindowTitle("随机删除样本")
        self.random_delete_progress.setWindowModality(Qt.WindowModality.WindowModal)
        self.random_delete_progress.setAutoClose(False)
        self.random_delete_progress.setAutoReset(False)
        self.random_delete_progress.setMinimumDuration(0)
        self.random_delete_progress.setValue(0)

        self.btn_delete_random_samples.setEnabled(False)
        self.random_delete_worker = RandomSampleDeleteWorker(
            self.current_project_id,
            target_class_id,
            delete_count,
            self.current_image_id
        )
        self.random_delete_worker.progress_updated.connect(self.on_random_delete_progress)
        self.random_delete_worker.delete_finished.connect(
            lambda result, current_target_name=target_name: self.on_random_delete_finished(current_target_name, result)
        )
        self.random_delete_worker.delete_failed.connect(self.on_random_delete_failed)
        self.random_delete_progress.canceled.connect(self.random_delete_worker.stop)
        self.random_delete_worker.start()
        self.random_delete_progress.show()

    def on_random_delete_progress(self, current: int, total: int, filename: str):
        """随机删样本进度回调"""
        if not self.random_delete_progress:
            return
        self.random_delete_progress.setMaximum(max(total, 1))
        self.random_delete_progress.setValue(current)
        self.random_delete_progress.setLabelText(
            f"正在删除样本图 ({current}/{total})：{filename}"
        )

    def _cleanup_random_delete_task(self):
        """清理随机删样本任务状态"""
        if self.random_delete_progress:
            self.random_delete_progress.close()
            self.random_delete_progress.deleteLater()
            self.random_delete_progress = None
        if self.random_delete_worker:
            self.random_delete_worker.deleteLater()
            self.random_delete_worker = None
        self.btn_delete_random_samples.setEnabled(True)

    def on_random_delete_finished(self, target_name: str, result: dict):
        """随机删样本完成回调"""
        self._cleanup_random_delete_task()
        self._invalidate_sample_stats_cache()

        deleted_count = result.get('deleted_count', 0)
        failed_files = result.get('failed_files', [])
        current_image_deleted = result.get('current_image_deleted', False)
        canceled = result.get('canceled', False)
        total_count = result.get('total_count', 0)

        self.load_image_list()

        remaining_images = self.images
        if remaining_images:
            remaining_ids = {image['id'] for image in remaining_images}
            if current_image_deleted or self.current_image_id not in remaining_ids:
                self.load_image(remaining_images[0]['id'])
            elif self.current_image_id:
                self.load_image(self.current_image_id)
        else:
            self.clear_current_image_view()

        if canceled:
            QMessageBox.information(
                self,
                "删除已取消",
                f"随机删样已取消。\n已删除 {deleted_count}/{total_count} 张“{target_name}”样本图"
            )
            return

        if failed_files:
            QMessageBox.warning(
                self,
                "部分删除失败",
                f"成功删除 {deleted_count} 张图片，失败 {len(failed_files)} 张。\n"
                f"失败文件：{', '.join(failed_files[:5])}"
            )
        else:
            QMessageBox.information(
                self,
                "删除完成",
                f"已随机删除 {deleted_count} 张“{target_name}”样本图"
            )

    def on_random_delete_failed(self, error_message: str):
        """随机删样本失败回调"""
        self._cleanup_random_delete_task()
        QMessageBox.critical(self, "删除失败", f"随机删除样本出错: {error_message}")

    def _get_next_image_shortcut_key(self) -> str:
        """获取当前“下一张”快捷键。"""
        from PyQt6.QtCore import QSettings

        settings = QSettings("Better-YOLO", "Settings")
        return str(settings.value("next_image_shortcut", "D")).upper()

    def _exec_message_box_with_shortcut(
        self,
        *,
        icon,
        title: str,
        text: str,
        buttons,
        default_button=None,
        shortcut_button=None,
        shortcut_key: Optional[str] = None,
    ) -> Tuple[QMessageBox.StandardButton, bool]:
        """执行消息框，并允许指定快捷键触发某个按钮。"""
        message_box = QMessageBox(self)
        message_box.setIcon(icon)
        message_box.setWindowTitle(title)
        message_box.setText(text)
        message_box.setStandardButtons(buttons)
        if default_button is not None:
            message_box.setDefaultButton(default_button)

        shortcut_triggered = False
        if shortcut_button is not None and shortcut_key:
            target_button = message_box.button(shortcut_button)
            if target_button is not None:
                shortcut = QShortcut(QKeySequence(shortcut_key), message_box)
                message_box._shortcut = shortcut
                shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)

                def on_shortcut_activated():
                    nonlocal shortcut_triggered
                    shortcut_triggered = True
                    target_button.click()

                shortcut.activated.connect(on_shortcut_activated)

        result = QMessageBox.StandardButton(message_box.exec())
        return result, shortcut_triggered

    def mark_current_image_as_negative_sample(self):
        """将当前图片标注为负样本"""
        if not self.current_project_id or not self.current_image_id:
            QMessageBox.warning(self, "提示", "请先选择一张图片")
            return

        existing_annotations = db.get_image_annotations(self.current_image_id)
        annotation_count = len(existing_annotations)
        next_image_key = self._get_next_image_shortcut_key()

        if annotation_count > 0:
            reply, _ = self._exec_message_box_with_shortcut(
                icon=QMessageBox.Icon.Question,
                title="确认标注为负样本",
                text=(
                    f"当前图片已有 {annotation_count} 个标注。\n"
                    f"继续后会清空这些标注，并将当前图片标记为负样本。\n"
                    f"是否继续？\n\n"
                    f"按 {next_image_key} 可直接确认"
                ),
                buttons=QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                shortcut_button=QMessageBox.StandardButton.Yes,
                shortcut_key=next_image_key,
            )
            if reply != QMessageBox.StandardButton.Yes:
                return

        db.delete_image_annotations(self.current_image_id)
        db.update_image_status(self.current_image_id, 'annotated')
        # 人工标记为负样本时自动锁定本图；若清空了 AI 误检框则同步标记 ✏️ 复标修正
        self._sync_image_review_state(self.current_image_id, lock=True)
        self.current_image_data = db.get_image(self.current_image_id)

        self.load_annotations()
        self.canvas.selected_annotation_id = None
        self.clear_attribute_panel(refresh_sample_panel=False)
        self.update_image_list_display()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()

        _, shortcut_triggered = self._exec_message_box_with_shortcut(
            icon=QMessageBox.Icon.Information,
            title="完成",
            text=f"当前图片已标注为负样本。\n按 {next_image_key} 可确认并进入下一张。",
            buttons=QMessageBox.StandardButton.Ok,
            default_button=QMessageBox.StandardButton.Ok,
            shortcut_button=QMessageBox.StandardButton.Ok,
            shortcut_key=next_image_key,
        )
        if shortcut_triggered:
            self.next_image()
    
    def show_class_context_menu(self, position):
        """显示类别右键菜单"""
        item = self.class_list.itemAt(position)
        if not item:
            return
        
        menu = QMenu(self)
        
        edit_action = menu.addAction("编辑")
        delete_action = menu.addAction("删除")
        
        action = menu.exec(self.class_list.mapToGlobal(position))
        
        if action == edit_action:
            self.edit_class(item)
        elif action == delete_action:
            self.delete_class(item)
    
    def edit_class(self, item: QListWidgetItem):
        """编辑类别"""
        class_id = item.data(Qt.ItemDataRole.UserRole)
        class_info = next((c for c in self.classes if c['id'] == class_id), None)
        if not class_info:
            return
        
        old_name = class_info['name']
        
        # 编辑名称
        name, ok = QInputDialog.getText(
            self, "编辑类别", 
            "请输入类别名称:",
            text=old_name
        )
        name = (name or "").strip()
        if not ok or not name:
            return
        if any(c.get('name') == name and c.get('id') != class_id for c in self.classes):
            QMessageBox.warning(self, "提示", f"类别名称 '{name}' 已存在")
            return
        
        # 编辑颜色
        color = QColorDialog.getColor(
            QColor(class_info.get('color', '#FF0000')), 
            self, "选择类别颜色"
        )
        if not color.isValid():
            color = QColor(class_info.get('color', '#FF0000'))
        
        # 更新类别信息
        class_info['name'] = name
        class_info['color'] = color.name()
        
        # 保存到数据库并同步已有标注名称
        db.update_project(self.current_project_id, classes=self.classes)
        if old_name != name and self.current_project_id:
            db.rename_project_class_annotations(self.current_project_id, class_id, name)
        
        # 更新显示并通知AI标注页同步标签名
        self.update_class_list()
        self.load_annotations()
        if old_name != name:
            self.classes_changed.emit({
                'action': 'rename',
                'old_name': old_name,
                'new_name': name,
            })
    
    def delete_class(self, item: QListWidgetItem):
        """删除类别"""
        class_id = item.data(Qt.ItemDataRole.UserRole)
        class_info = next((c for c in self.classes if c['id'] == class_id), None)
        if not class_info:
            return
        
        deleted_name = class_info['name']
        
        # 确认删除
        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除类别 '{deleted_name}' 吗？\n该类别下的所有标注将被删除！",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 删除项目内该类别下的全部标注并重排后续 class_id，同时同步受影响图片的锁定与复标状态
        affected_img_ids = set()
        if self.current_project_id:
            for img in self.images:
                anns = db.get_image_annotations(img['id'])
                if any(ann.get('class_id') == class_id for ann in anns):
                    affected_img_ids.add(img['id'])
            db.delete_project_class_and_reindex(self.current_project_id, class_id)

        for img_id in affected_img_ids:
            remaining = db.get_image_annotations(img_id)
            if not remaining and db.get_image_ai_original(img_id) is None:
                db.update_image_status(img_id, 'pending')
                self._sync_image_review_state(img_id, lock=False)
            else:
                self._sync_image_review_state(img_id, lock=True)

        # 从列表中删除类别并重新编号
        self.classes = [c for c in self.classes if c['id'] != class_id]
        for i, cls in enumerate(self.classes):
            cls['id'] = i
        
        # 保存到数据库
        db.update_project(self.current_project_id, classes=self.classes)
        
        # 更新显示并通知AI标注页同步删除
        self._invalidate_sample_stats_cache()
        self.update_class_list()
        self.load_annotations()
        self.classes_changed.emit({'action': 'delete', 'name': deleted_name})
    
    def add_class(self):
        """添加类别"""
        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return
        name, ok = QInputDialog.getText(self, "添加类别", "请输入类别名称:")
        name = (name or "").strip()
        if ok and name:
            if any(c.get('name') == name for c in self.classes):
                QMessageBox.warning(self, "提示", f"类别名称 '{name}' 已存在")
                return

            # 选择颜色
            color = QColorDialog.getColor(QColor(255, 0, 0), self, "选择类别颜色")
            if not color.isValid():
                color = QColor(255, 0, 0)
            
            class_id = len(self.classes)
            self.classes.append({
                'id': class_id,
                'name': name,
                'color': color.name()
            })
            
            # 更新项目类别
            db.update_project(self.current_project_id, classes=self.classes)
            
            self.update_class_list()
            # 选中新添加的类别
            self.class_list.setCurrentRow(self.class_list.count() - 1)
            self.on_class_selected()
            # 通知AI标注页同步新增标签（描述留空）
            self.classes_changed.emit({'action': 'add', 'name': name})
    
    def prev_image(self):
        """上一张图片（优先在当前筛选可见项中跳转）"""
        if not self.images or not self.current_image_id:
            return
        
        current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), 0)
        for idx in range(current_index - 1, -1, -1):
            item = self.image_list.item(idx) if idx < self.image_list.count() else None
            if item is None or not item.isHidden():
                self.load_image(self.images[idx]['id'])
                return
    
    def next_image(self):
        """下一张图片（优先在当前筛选可见项中跳转）"""
        if not self.images or not self.current_image_id:
            return
        
        current_index = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), -1)
        for idx in range(current_index + 1, len(self.images)):
            item = self.image_list.item(idx) if idx < self.image_list.count() else None
            if item is None or not item.isHidden():
                self.load_image(self.images[idx]['id'])
                return
    
    def add_history(self, action: str, data: dict):
        """添加历史记录"""
        # 删除当前位置之后的历史
        self.history = self.history[:self.history_index + 1]
        
        # 添加新记录
        self.history.append({
            'action': action,
            'data': data
        })
        self.history_index = len(self.history) - 1
        
        # 限制历史记录数量
        if len(self.history) > 50:
            self.history.pop(0)
            self.history_index -= 1
    
    def undo(self):
        """撤销（支持创建、删除、移动/缩放、修改类别，并自动重算 AI 复标修正状态）"""
        if self.history_index < 0:
            return
        
        record = self.history[self.history_index]
        action = record['action']
        data = record['data']
        
        if action == 'create':
            # 撤销创建 = 删除
            ann_id = data.get('annotation_id')
            if ann_id:
                db.delete_annotation(ann_id)
        elif action == 'delete':
            # 撤销删除 = 恢复创建（保留原 attributes）
            db.add_annotation(
                image_id=data['image_id'],
                project_id=data['project_id'],
                class_id=data['class_id'],
                class_name=data['class_name'],
                annotation_type=data['type'],
                data=data['data'],
                attributes=data.get('attributes') or {},
            )
        elif action == 'modify':
            ann_id = data.get('annotation_id')
            if ann_id and 'old_data' in data:
                db.update_annotation(
                    ann_id,
                    data=data['old_data'],
                    attributes=data.get('old_attributes'),
                )
        elif action == 'change_class':
            ann_id = data.get('annotation_id')
            if ann_id:
                db.update_annotation(
                    ann_id,
                    class_id=data.get('old_class_id'),
                    class_name=data.get('old_class_name'),
                    attributes=data.get('old_attributes'),
                )
        
        self.history_index -= 1
        if self.current_image_id:
            remaining = db.get_image_annotations(self.current_image_id)
            has_ai_snapshot = db.get_image_ai_original(self.current_image_id) is not None
            if remaining:
                db.update_image_status(self.current_image_id, 'annotated')
                self._sync_image_review_state(self.current_image_id, lock=None)
            elif not has_ai_snapshot:
                db.update_image_status(self.current_image_id, 'pending')
                self._sync_image_review_state(self.current_image_id, lock=False)
            else:
                self._sync_image_review_state(self.current_image_id, lock=None)

        self.load_annotations()
        self._invalidate_sample_stats_cache()
        self.update_sample_control_panel()
    
    def update_status_bar(self):
        """更新状态栏"""
        total = len(self.images)
        current = 0
        if self.current_image_id:
            current = next((i for i, img in enumerate(self.images) if img['id'] == self.current_image_id), 0) + 1
        
        self.status_image.setText(f"当前: {current}/{total}")
        self.status_annotation.setText(f"标注: {len(self.annotations)}")
        tool_names = {
            'rectangle': '矩形',
            'polygon': '多边形',
            'move': '移动',
            'keypoint': '关键点',
            'obb': '旋转矩形'
        }
        self.status_tool.setText(f"工具: {tool_names.get(self.canvas.current_tool, self.canvas.current_tool)}")
    
    def keyPressEvent(self, event: QKeyEvent):
        """键盘事件"""
        from PyQt6.QtCore import QSettings
        
        # 获取快捷键设置
        settings = QSettings("Better-YOLO", "Settings")
        rect_tool_key = settings.value("rect_tool_shortcut", "Q").upper()
        poly_tool_key = settings.value("poly_tool_shortcut", "P").upper()
        move_tool_key = settings.value("move_tool_shortcut", "E").upper()
        lock_image_key = settings.value("lock_image_shortcut", "L").upper()
        delete_key = settings.value("delete_shortcut", "DELETE").upper()
        
        # 处理工具快捷键
        key_text = event.text().upper()
        if key_text == rect_tool_key:
            self.select_draw_tool('rectangle')
            return
        elif key_text == poly_tool_key:
            self.select_draw_tool('polygon')
            return
        elif key_text == move_tool_key:
            self.btn_move.setChecked(True)
            self.set_tool('move')
            return
        elif key_text == lock_image_key and self.current_image_id:
            self.on_toggle_lock()
            return
        elif AnnotationCanvas._matches_delete_shortcut(event, settings):
            self.delete_selected_annotation()
            return
        elif event.modifiers() == Qt.KeyboardModifier.ControlModifier and event.key() == Qt.Key.Key_Z:
            self.undo()
        else:
            # 处理数字键1-9切换标签
            if key_text in ['1', '2', '3', '4', '5', '6', '7', '8', '9']:
                class_index = int(key_text) - 1
                if class_index < self.class_list.count():
                    self.class_list.setCurrentRow(class_index)
                    self.on_class_selected()
                    return
            super().keyPressEvent(event)
    
    @staticmethod
    def _extract_xy_points(raw_points) -> List[Tuple[float, float]]:
        """兼容 [{'x':..,'y':..}] 与 [(x, y)] 两种多边形顶点格式"""
        extracted = []
        for point in raw_points or []:
            if isinstance(point, dict) and 'x' in point and 'y' in point:
                extracted.append((float(point['x']), float(point['y'])))
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                extracted.append((float(point[0]), float(point[1])))
        return extracted

    @classmethod
    def _format_yolo_annotation_lines(
        cls,
        annotations: List[Dict],
        img_width: int,
        img_height: int,
    ) -> List[str]:
        """将单张图的标注列表序列化为标准 YOLO 文本行（兼容 bbox / polygon / keypoint / obb）"""
        w_img = max(1, int(img_width or 1920))
        h_img = max(1, int(img_height or 1080))
        lines = []

        for ann in annotations:
            class_id = ann.get('class_id', 0)
            ann_type = ann.get('type', 'bbox')
            data = ann.get('data', {}) or {}

            if ann_type == 'bbox':
                x = float(data.get('x', 0))
                y = float(data.get('y', 0))
                width = float(data.get('width', 0))
                height = float(data.get('height', 0))

                x_center = (x + width / 2.0) / w_img
                y_center = (y + height / 2.0) / h_img
                norm_width = width / w_img
                norm_height = height / h_img
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}")

            elif ann_type in ('polygon', 'mask'):
                raw_pts = data.get('points') if 'points' in data else data.get('mask', [])
                pts = cls._extract_xy_points(raw_pts)
                if len(pts) >= 3:
                    coords = [
                        f"{px / w_img:.6f} {py / h_img:.6f}"
                        for px, py in pts
                    ]
                    lines.append(f"{class_id} {' '.join(coords)}")

            elif ann_type == 'keypoint':
                x = float(data.get('x', 0))
                y = float(data.get('y', 0))
                width = float(data.get('width', 0))
                height = float(data.get('height', 0))
                keypoints = data.get('keypoints', []) or []

                x_center = (x + width / 2.0) / w_img
                y_center = (y + height / 2.0) / h_img
                norm_width = width / w_img
                norm_height = height / h_img
                kp_tokens = []
                for kp in keypoints:
                    kp_x = float(kp.get('x', 0)) / w_img
                    kp_y = float(kp.get('y', 0)) / h_img
                    kp_v = kp.get('v', kp.get('visible', 1))
                    kp_tokens.append(f"{kp_x:.6f} {kp_y:.6f} {kp_v}")
                suffix = (" " + " ".join(kp_tokens)) if kp_tokens else ""
                lines.append(f"{class_id} {x_center:.6f} {y_center:.6f} {norm_width:.6f} {norm_height:.6f}{suffix}")

            elif ann_type == 'obb':
                pts = cls._extract_xy_points(data.get('points', []))
                if len(pts) == 4:
                    coords = [f"{px / w_img:.6f} {py / h_img:.6f}" for px, py in pts]
                    lines.append(f"{class_id} {' '.join(coords)}")
                else:
                    x = float(data.get('x', 0))
                    y = float(data.get('y', 0))
                    width = float(data.get('width', 0))
                    height = float(data.get('height', 0))
                    angle = float(data.get('angle', 0))
                    x_center = (x + width / 2.0) / w_img
                    y_center = (y + height / 2.0) / h_img
                    lines.append(
                        f"{class_id} {x_center:.6f} {y_center:.6f} {width / w_img:.6f} {height / h_img:.6f} {angle:.6f}"
                    )

        return lines

    @staticmethod
    def _polygon_area(pts: List[Tuple[float, float]]) -> float:
        """鞋带公式计算多边形面积"""
        if len(pts) < 3:
            return 0.0
        area = 0.0
        n = len(pts)
        for i in range(n):
            x1, y1 = pts[i]
            x2, y2 = pts[(i + 1) % n]
            area += x1 * y2 - x2 * y1
        return abs(area) * 0.5

    def _build_coco_export_dict(self, project_name: str, images: List[Dict]) -> Dict:
        """构建标准 COCO 格式导出字典（同时兼容 bbox 与 polygon 分割标注）"""
        from datetime import datetime
        coco_data = {
            "info": {
                "description": f"Annotations for {project_name}",
                "version": "1.0",
                "year": datetime.now().year,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }

        for cls in sorted(self.classes, key=lambda c: c['id']):
            coco_data["categories"].append({
                "id": cls['id'],
                "name": cls['name'],
                "supercategory": "object",
            })

        annotation_id = 1
        for image in images:
            coco_data["images"].append({
                "id": image['id'],
                "file_name": image['filename'],
                "width": image.get('width') or 1920,
                "height": image.get('height') or 1080,
                "date_captured": "",
                "license": 0,
                "coco_url": "",
                "flickr_url": "",
            })

            annotations = db.get_image_annotations(image['id'])
            for ann in annotations:
                ann_type = ann.get('type', 'bbox')
                data = ann.get('data', {}) or {}
                category_id = ann.get('class_id', 0)

                if ann_type == 'bbox':
                    x = round(float(data.get('x', 0)), 2)
                    y = round(float(data.get('y', 0)), 2)
                    width = round(float(data.get('width', 0)), 2)
                    height = round(float(data.get('height', 0)), 2)
                    coco_data["annotations"].append({
                        "id": annotation_id,
                        "image_id": image['id'],
                        "category_id": category_id,
                        "segmentation": [],
                        "area": round(width * height, 2),
                        "bbox": [x, y, width, height],
                        "iscrowd": 0,
                    })
                    annotation_id += 1

                elif ann_type in ('polygon', 'mask'):
                    raw_pts = data.get('points') if 'points' in data else data.get('mask', [])
                    pts = self._extract_xy_points(raw_pts)
                    if len(pts) >= 3:
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        bbox = [
                            round(min_x, 2),
                            round(min_y, 2),
                            round(max_x - min_x, 2),
                            round(max_y - min_y, 2),
                        ]
                        flat_seg = [round(coord, 2) for pt in pts for coord in pt]
                        coco_data["annotations"].append({
                            "id": annotation_id,
                            "image_id": image['id'],
                            "category_id": category_id,
                            "segmentation": [flat_seg],
                            "area": round(self._polygon_area(pts), 2),
                            "bbox": bbox,
                            "iscrowd": 0,
                        })
                        annotation_id += 1

        return coco_data

    def open_rename_dialog(self):
        """打开数据集图片与标签一键重命名弹窗"""
        from gui.pages.dataset_rename_dialog import DatasetRenameDialog

        dlg = DatasetRenameDialog(self, project_id=self.current_project_id)
        dlg.project_renamed.connect(self._on_project_renamed)
        dlg.exec()

    def _on_project_renamed(self, project_id: int):
        self.refresh_after_project_rename(project_id)
        self.images_renamed.emit(project_id)

    def refresh_after_project_rename(self, project_id: int = None):
        """项目内图片重命名后刷新列表并保持当前选中图片打开状态"""
        if project_id and project_id != self.current_project_id:
            return
        if not self.current_project_id:
            return
        keep_image_id = self.current_image_id
        self.load_image_list()
        if keep_image_id and any(img['id'] == keep_image_id for img in self.images):
            self.load_image(keep_image_id)
        elif self.images:
            self.load_image(self.images[0]['id'])

    def on_click_export_annotations(self):
        """点击「导出标注文件」按钮：先弹窗选择导出格式，再选择导出目录"""
        from PyQt6.QtWidgets import QDialog, QMessageBox
        from gui.pages.dataset_export_dialog import DatasetExportDialog

        if not self.current_project_id:
            QMessageBox.warning(self, "导出失败", "请先选择一个项目")
            return

        images = db.get_project_images(self.current_project_id)
        if not images:
            QMessageBox.warning(self, "导出失败", "项目中没有图片")
            return

        dlg = DatasetExportDialog(
            self,
            mode="annotations",
            image_count=len(images),
            default_format=self.export_format.currentText(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        cfg = dlg.get_config()
        self.export_format.setCurrentText(cfg["export_format"])
        self.export_annotations(export_format=cfg["export_format"])

    def on_click_export_dataset(self):
        """点击「导出完整数据集」按钮：先弹窗选择格式、目录结构(默认/服务器)与划分比例(默认 8:2)，再选择导出目录"""
        from PyQt6.QtWidgets import QDialog, QMessageBox
        from gui.pages.dataset_export_dialog import DatasetExportDialog

        if not self.current_project_id:
            QMessageBox.warning(self, "导出失败", "请先选择一个项目")
            return

        images = db.get_project_images(self.current_project_id)
        valid_images = [
            img for img in images
            if img.get('storage_path') and os.path.exists(img['storage_path'])
        ]
        if not valid_images:
            QMessageBox.warning(self, "导出失败", "项目中没有可用的图片")
            return

        dlg = DatasetExportDialog(
            self,
            mode="dataset",
            image_count=len(valid_images),
            default_format=self.export_format.currentText(),
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        cfg = dlg.get_config()
        self.export_format.setCurrentText(cfg["export_format"])
        self.export_dataset(
            export_format=cfg["export_format"],
            structure_mode=cfg["structure_mode"],
            train_ratio=cfg["train_ratio"],
            val_ratio=cfg["val_ratio"],
            shuffle=cfg["shuffle"],
        )

    def export_annotations(self, export_format: Optional[str] = None):
        """导出标注文件"""
        import json
        from PyQt6.QtWidgets import QMessageBox, QFileDialog
        
        if not self.current_project_id:
            QMessageBox.warning(self, "导出失败", "请先选择一个项目")
            return
        
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", 
            os.path.expanduser("~")
        )
        
        if not export_dir:
            return
        
        try:
            export_format = export_format or self.export_format.currentText()
            project = db.get_project(self.current_project_id)
            if not project:
                QMessageBox.warning(self, "导出失败", "无法获取项目信息")
                return
            
            project_name = project.get('name', 'untitled')
            export_path = os.path.join(export_dir, f"{project_name}_annotations")
            os.makedirs(export_path, exist_ok=True)
            
            images = db.get_project_images(self.current_project_id)
            if not images:
                QMessageBox.warning(self, "导出失败", "项目中没有图片")
                return
            
            if export_format == "YOLO格式":
                labels_dir = os.path.join(export_path, 'labels')
                os.makedirs(labels_dir, exist_ok=True)
                
                exported_count = 0
                negative_count = 0
                for image in images:
                    image_id = image['id']
                    annotations = db.get_image_annotations(image_id)
                    image_status = image.get('status', 'pending')
                    
                    if annotations or image_status == 'annotated':
                        filename = os.path.splitext(image['filename'])[0] + '.txt'
                        label_file = os.path.join(labels_dir, filename)
                        lines = self._format_yolo_annotation_lines(
                            annotations,
                            image.get('width', 1920),
                            image.get('height', 1080),
                        )
                        with open(label_file, 'w', encoding='utf-8') as f:
                            for line in lines:
                                f.write(line + '\n')
                        
                        exported_count += 1
                        if not annotations:
                            negative_count += 1
                
                classes_file = os.path.join(export_path, 'classes.txt')
                with open(classes_file, 'w', encoding='utf-8') as f:
                    for cls in sorted(self.classes, key=lambda x: x['id']):
                        f.write(f"{cls['name']}\n")
                
                QMessageBox.information(
                    self,
                    "导出成功",
                    f"已导出 {exported_count} 个标注文件到\n{export_path}\n\n"
                    f"- 负样本空标签: {negative_count} 个"
                )
                
            elif export_format == "COCO格式":
                coco_data = self._build_coco_export_dict(project_name, images)
                coco_file = os.path.join(export_path, 'annotations.json')
                with open(coco_file, 'w', encoding='utf-8') as f:
                    json.dump(coco_data, f, indent=2, ensure_ascii=False)
                
                QMessageBox.information(
                    self,
                    "导出成功",
                    f"已导出 COCO 格式标注（共 {len(coco_data['annotations'])} 条）到\n{coco_file}"
                )
                
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出过程中出错:\n{str(e)}")
    
    @staticmethod
    def _split_dataset_images(
        images: List[Dict],
        train_ratio: int = 8,
        val_ratio: int = 2,
        shuffle: bool = True,
        seed: int = 42,
    ) -> Tuple[List[Dict], List[Dict]]:
        """按给定比例（默认 8:2）将有效图片列表切分为 (train_images, val_images)"""
        import random

        pool = list(images)
        if shuffle and len(pool) > 1:
            random.Random(seed).shuffle(pool)

        n = len(pool)
        if n == 0:
            return [], []
        if val_ratio <= 0 or n == 1:
            return pool, []

        total_weight = max(1, int(train_ratio) + int(val_ratio))
        val_count = round(n * int(val_ratio) / total_weight)
        val_count = max(1, min(n - 1, val_count))
        train_count = n - val_count
        return pool[:train_count], pool[train_count:]

    def export_dataset(
        self,
        export_format: Optional[str] = None,
        structure_mode: str = "default",
        train_ratio: int = 8,
        val_ratio: int = 2,
        shuffle: bool = True,
    ):
        """导出完整数据集（支持 YOLO 默认结构 / 服务器结构与自定义训练:验证比例划分，以及 COCO 格式）"""
        import json
        import shutil
        from PyQt6.QtWidgets import QMessageBox, QFileDialog
        
        if not self.current_project_id:
            QMessageBox.warning(self, "导出失败", "请先选择一个项目")
            return
        
        export_dir = QFileDialog.getExistingDirectory(
            self, "选择导出目录", 
            os.path.expanduser("~")
        )
        
        if not export_dir:
            return
        
        try:
            export_format = export_format or self.export_format.currentText()
            project = db.get_project(self.current_project_id)
            if not project:
                QMessageBox.warning(self, "导出失败", "无法获取项目信息")
                return
            
            project_name = project.get('name', 'untitled')
            dataset_dir = os.path.join(export_dir, project_name)
            os.makedirs(dataset_dir, exist_ok=True)
            
            images = db.get_project_images(self.current_project_id)
            if not images:
                QMessageBox.warning(self, "导出失败", "项目中没有图片")
                return

            if export_format == "COCO格式":
                images_dir = os.path.join(dataset_dir, 'images')
                os.makedirs(images_dir, exist_ok=True)
                copied_count = 0
                for image in images:
                    src_image = image.get('storage_path', '')
                    if src_image and os.path.exists(src_image):
                        dst_image = os.path.join(images_dir, image['filename'])
                        shutil.copy2(src_image, dst_image)
                        copied_count += 1

                coco_data = self._build_coco_export_dict(project_name, images)
                coco_file = os.path.join(dataset_dir, 'annotations.json')
                with open(coco_file, 'w', encoding='utf-8') as f:
                    json.dump(coco_data, f, indent=2, ensure_ascii=False)

                QMessageBox.information(
                    self,
                    "导出成功",
                    f"已导出 COCO 完整数据集到\n{dataset_dir}\n\n"
                    f"- 复制图片: {copied_count} 张\n"
                    f"- 导出标注对象: {len(coco_data['annotations'])} 个"
                )
                return

            # ==================== YOLO 格式按结构与比例划分导出 ====================
            structure_mode = structure_mode or "default"
            do_shuffle = bool(shuffle)

            valid_images = [
                img for img in images
                if img.get('storage_path') and os.path.exists(img['storage_path'])
            ]
            if not valid_images:
                QMessageBox.warning(self, "导出失败", "项目中没有可用的图片文件")
                return

            # 清理目标目录下可能残留的旧结构子目录，确保两种结构切换重新导出时不混杂
            for stale_sub in ('images', 'labels', 'train', 'val'):
                stale_path = os.path.join(dataset_dir, stale_sub)
                if os.path.isdir(stale_path):
                    shutil.rmtree(stale_path)

            if structure_mode == "server":
                # 服务器结构:
                # ├── train/{images, labels}
                # └── val/{images, labels}
                train_img_dir = os.path.join(dataset_dir, 'train', 'images')
                train_lbl_dir = os.path.join(dataset_dir, 'train', 'labels')
                val_img_dir = os.path.join(dataset_dir, 'val', 'images')
                val_lbl_dir = os.path.join(dataset_dir, 'val', 'labels')
                yaml_train_rel = "train/images"
                yaml_val_rel = "val/images"
                structure_desc = "服务器结构 (train/images, val/images)"
            else:
                # 默认结构:
                # ├── images/{train, val}
                # └── labels/{train, val}
                train_img_dir = os.path.join(dataset_dir, 'images', 'train')
                val_img_dir = os.path.join(dataset_dir, 'images', 'val')
                train_lbl_dir = os.path.join(dataset_dir, 'labels', 'train')
                val_lbl_dir = os.path.join(dataset_dir, 'labels', 'val')
                yaml_train_rel = "images/train"
                yaml_val_rel = "images/val"
                structure_desc = "默认结构 (images/train, labels/train)"

            for d in (train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir):
                os.makedirs(d, exist_ok=True)

            train_images, val_images = self._split_dataset_images(
                valid_images,
                train_ratio=train_ratio,
                val_ratio=val_ratio,
                shuffle=do_shuffle,
            )

            copied_count = 0
            exported_count = 0
            negative_count = 0

            def _export_subset(subset_imgs: List[Dict], img_out_dir: str, lbl_out_dir: str):
                nonlocal copied_count, exported_count, negative_count
                for image in subset_imgs:
                    src_image = image['storage_path']
                    dst_image = os.path.join(img_out_dir, image['filename'])
                    shutil.copy2(src_image, dst_image)
                    copied_count += 1

                    image_id = image['id']
                    annotations = db.get_image_annotations(image_id)
                    image_status = image.get('status', 'pending')
                    if annotations or image_status == 'annotated':
                        txt_name = os.path.splitext(image['filename'])[0] + '.txt'
                        label_file = os.path.join(lbl_out_dir, txt_name)
                        lines = self._format_yolo_annotation_lines(
                            annotations,
                            image.get('width', 1920),
                            image.get('height', 1080),
                        )
                        with open(label_file, 'w', encoding='utf-8') as f:
                            for line in lines:
                                f.write(line + '\n')
                        exported_count += 1
                        if not annotations:
                            negative_count += 1

            _export_subset(train_images, train_img_dir, train_lbl_dir)
            _export_subset(val_images, val_img_dir, val_lbl_dir)

            classes_file = os.path.join(dataset_dir, 'classes.txt')
            sorted_classes = sorted(self.classes, key=lambda x: x['id'])
            with open(classes_file, 'w', encoding='utf-8') as f:
                for cls in sorted_classes:
                    f.write(f"{cls['name']}\n")

            yaml_file = os.path.join(dataset_dir, 'data.yaml')
            if structure_mode == "server":
                # 服务器结构省略本机绝对路径，上传远端服务器后可直接用于训练
                yaml_content = (
                    f"train: {yaml_train_rel}\n"
                    f"val: {yaml_val_rel}\n\n"
                    f"nc: {len(sorted_classes)}\n"
                    f"names: {[cls['name'] for cls in sorted_classes]}\n"
                )
            else:
                yaml_content = (
                    f"path: {os.path.abspath(dataset_dir)}\n"
                    f"train: {yaml_train_rel}\n"
                    f"val: {yaml_val_rel}\n\n"
                    f"nc: {len(sorted_classes)}\n"
                    f"names: {[cls['name'] for cls in sorted_classes]}\n"
                )
            with open(yaml_file, 'w', encoding='utf-8') as f:
                f.write(yaml_content)

            QMessageBox.information(
                self,
                "导出成功",
                f"已导出 YOLO 完整数据集到\n{dataset_dir}\n\n"
                f"- 目录结构: {structure_desc}\n"
                f"- 划分比例: {train_ratio}:{val_ratio}（训练集 {len(train_images)} 张 / 验证集 {len(val_images)} 张）\n"
                f"- 导出标签: {exported_count} 个（含负样本空标签 {negative_count} 个）"
            )
                
        except Exception as e:
            QMessageBox.critical(self, "导出失败", f"导出过程中出错:\n{str(e)}")
