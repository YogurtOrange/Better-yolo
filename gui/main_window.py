# -*- coding: utf-8 -*-
"""Better-YOLO desktop workspace and global display controls."""

from PyQt6.QtWidgets import (
    QApplication, QAbstractSpinBox, QMainWindow, QWidget, QHBoxLayout,
    QVBoxLayout, QStackedWidget, QLabel, QPushButton, QFrame, QScrollArea,
    QSpinBox, QSizePolicy, QBoxLayout,
)
from PyQt6.QtCore import Qt, QSettings, QSize, QPoint, QSignalBlocker
from PyQt6.QtGui import QKeySequence, QShortcut

from gui.styles import get_full_stylesheet
from gui.ui_scale import UiScaleManager, ui_pixels
from gui.pages.import_page import ImportPage
from gui.pages.annotate_page import AnnotatePage
from gui.pages.ai_label_page import AiLabelPage
from gui.pages.train_page import TrainPage
from gui.pages.result_page import ResultPage
from gui.pages.test_page import TestPage
from gui.pages.settings_page import SettingsPage
from gui.pages.about_page import AboutPage
from models.database import db


class MainWindow(QMainWindow):
    """Keep workflow state in the pages and presentation state in the shell."""

    PAGE_INFO = (
        ("数据导入", "创建项目，导入图片、视频与已有标注"),
        ("图像标注", "手动精修，结合 YOLO 与 SAM 辅助完成标注"),
        ("AI 标注", "用自然语言描述目标，持续优化自动标注结果"),
        ("模型训练", "配置训练参数，实时观察模型表现"),
        ("训练结果", "查看评估指标、分析结果与导出模型"),
        ("模型测试", "使用图片或视频验证模型的实际效果"),
        ("偏好设置", "调整显示比例、保存行为与工作路径"),
        ("关于", "Better-YOLO · 本地视觉模型工作空间"),
    )

    def __init__(self):
        super().__init__()
        self.settings = QSettings("Better-YOLO", "MainWindow")
        self.scale_manager = UiScaleManager(QApplication.instance(), get_full_stylesheet(), self)
        self.init_ui()
        self.load_window_state()
        self.init_scaling()

    def init_ui(self):
        self.setWindowTitle("Better-YOLO · 视觉模型工作空间")
        self.setMinimumSize(1000, 680)
        central = QWidget()
        central.setObjectName("workspaceRoot")
        self.setCentralWidget(central)
        layout = QHBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.sidebar = self.create_sidebar()
        layout.addWidget(self.sidebar)

        workspace = QWidget()
        workspace_layout = QVBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)
        workspace_layout.setSpacing(0)
        workspace_layout.addWidget(self.create_workspace_header())
        self.page_card = QFrame()
        self.page_card.setObjectName("pageCard")
        card_layout = QVBoxLayout(self.page_card)
        card_layout.setContentsMargins(1, 1, 1, 1)
        self.content_stack = self.create_content_area()
        card_layout.addWidget(self.content_stack)
        content_layout = QVBoxLayout()
        content_layout.setContentsMargins(18, 18, 18, 18)
        content_layout.addWidget(self.page_card)
        workspace_layout.addLayout(content_layout, 1)
        layout.addWidget(workspace, 1)
        self.settings_page.theme_changed.connect(self.on_theme_changed)
        self.import_page.project_combo.currentIndexChanged.connect(self.update_header)
        self.update_header()
        self.sync_database_files()

    def create_sidebar(self):
        sidebar = QWidget()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(224)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(14, 24, 14, 18)
        layout.setSpacing(12)
        brand = QHBoxLayout()
        brand.setSpacing(10)
        mark = QLabel("BY")
        mark.setObjectName("brandMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        mark.setFixedSize(42, 42)
        brand.addWidget(mark)
        brand_text = QVBoxLayout()
        brand_text.setSpacing(3)
        title = QLabel("Better-YOLO")
        title.setObjectName("brandTitle")
        subtitle = QLabel("视觉模型工作空间")
        subtitle.setObjectName("brandSubtitle")
        brand_text.addWidget(title)
        brand_text.addWidget(subtitle)
        brand.addLayout(brand_text)
        layout.addLayout(brand)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        contents = QWidget()
        contents.setObjectName("sidebarContents")
        nav_layout = QVBoxLayout(contents)
        nav_layout.setContentsMargins(0, 0, 0, 0)
        nav_layout.setSpacing(6)
        self.nav_buttons = []
        section = QLabel("工作流程")
        section.setObjectName("sectionLabel")
        nav_layout.addWidget(section)
        labels = ("01   数据导入", "02   图像标注", "AI   智能标注", "03   模型训练", "04   训练结果", "05   模型测试")
        for index, text in enumerate(labels):
            button = self.create_nav_button(text, index)
            self.nav_buttons.append(button)
            nav_layout.addWidget(button)
        nav_layout.addStretch(1)
        section = QLabel("工作空间")
        section.setObjectName("sectionLabel")
        nav_layout.addWidget(section)
        for index, text in ((6, "偏好设置"), (7, "关于 Better-YOLO")):
            button = self.create_nav_button(text, index)
            self.nav_buttons.append(button)
            nav_layout.addWidget(button)
        scroll.setWidget(contents)
        layout.addWidget(scroll, 1)
        badge = QLabel("●  本地工作空间")
        badge.setObjectName("localBadge")
        layout.addWidget(badge)
        return sidebar

    def create_nav_button(self, text, index):
        button = QPushButton(text)
        button.setObjectName("nav_button")
        button.setCheckable(True)
        button.setCursor(Qt.CursorShape.PointingHandCursor)
        button.clicked.connect(lambda checked, idx=index: self.switch_page(idx))
        return button

    def create_workspace_header(self):
        header = QFrame()
        header.setObjectName("workspaceHeader")
        layout = QHBoxLayout(header)
        self.header_layout = layout
        layout.setContentsMargins(24, 17, 24, 17)
        layout.setSpacing(20)
        heading = QVBoxLayout()
        heading.setSpacing(4)
        self.breadcrumb = QLabel("WORKSPACE / 数据导入")
        self.breadcrumb.setObjectName("breadcrumb")
        self.breadcrumb.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        self.page_heading = QLabel()
        self.page_heading.setObjectName("pageHeading")
        self.page_description = QLabel()
        self.page_description.setObjectName("pageDescription")
        self.page_description.setWordWrap(True)
        heading.addWidget(self.breadcrumb)
        heading.addWidget(self.page_heading)
        heading.addWidget(self.page_description)
        layout.addLayout(heading, 1)

        controls = QVBoxLayout()
        controls.setSpacing(6)
        caption = QLabel("界面缩放")
        caption.setObjectName("subtitle")
        controls.addWidget(caption)
        zoom_frame = QFrame()
        self.zoom_frame = zoom_frame
        zoom_frame.setObjectName("zoomControls")
        zoom_frame.setMinimumWidth(190)
        zoom_layout = QHBoxLayout(zoom_frame)
        zoom_layout.setContentsMargins(4, 3, 4, 3)
        zoom_layout.setSpacing(2)
        self.zoom_out_button = QPushButton("−")
        self.zoom_out_button.setObjectName("zoomButton")
        self.zoom_out_button.setAccessibleName("缩小界面")
        self.zoom_out_button.setToolTip("缩小界面（Ctrl+-）")
        self.zoom_out_button.setFixedSize(30, 30)
        self.zoom_out_button.clicked.connect(lambda: self.change_ui_scale(-10))
        self.zoom_value = QSpinBox()
        self.zoom_value.setObjectName("zoomValue")
        self.zoom_value.setAccessibleName("界面缩放百分比")
        self.zoom_value.setRange(UiScaleManager.MIN_PERCENT, UiScaleManager.MAX_PERCENT)
        self.zoom_value.setSingleStep(10)
        self.zoom_value.setSuffix("%")
        self.zoom_value.setValue(100)
        self.zoom_value.setFixedWidth(62)
        self.zoom_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.zoom_value.setButtonSymbols(QAbstractSpinBox.ButtonSymbols.NoButtons)
        self.zoom_value.setKeyboardTracking(False)
        self.zoom_in_button = QPushButton("+")
        self.zoom_in_button.setObjectName("zoomButton")
        self.zoom_in_button.setAccessibleName("放大界面")
        self.zoom_in_button.setToolTip("放大界面（Ctrl++ / Ctrl+=）")
        self.zoom_in_button.setFixedSize(30, 30)
        self.zoom_in_button.clicked.connect(lambda: self.change_ui_scale(10))
        reset = QPushButton("重置")
        reset.setObjectName("zoomReset")
        reset.setToolTip("恢复 100%（Ctrl+0）")
        reset.clicked.connect(lambda: self.scale_manager.set_scale(100))
        for widget in (self.zoom_out_button, self.zoom_value, self.zoom_in_button, reset):
            widget.setCursor(Qt.CursorShape.PointingHandCursor)
            zoom_layout.addWidget(widget)
        controls.addWidget(zoom_frame)
        layout.addLayout(controls)
        return header

    def init_scaling(self):
        self.zoom_value.valueChanged.connect(self.scale_manager.set_scale)
        self.settings_page.ui_scale_changed.connect(self.scale_manager.set_scale)
        self.scale_manager.scale_changed.connect(self.on_ui_scale_changed)
        self.zoom_shortcuts = []
        for key, callback in (
            ("Ctrl+=", lambda: self.change_ui_scale(10)),
            ("Ctrl++", lambda: self.change_ui_scale(10)),
            ("Ctrl+-", lambda: self.change_ui_scale(-10)),
            ("Ctrl+0", lambda: self.scale_manager.set_scale(100)),
        ):
            shortcut = QShortcut(QKeySequence(key), self)
            shortcut.activated.connect(callback)
            self.zoom_shortcuts.append(shortcut)
        try:
            percent = int(self.settings.value("ui_scale", 100))
        except (TypeError, ValueError):
            percent = 100
        self.scale_manager.set_scale(percent)

    def change_ui_scale(self, delta):
        self.scale_manager.set_scale(self.scale_manager.percent + delta)

    def on_ui_scale_changed(self, percent):
        with QSignalBlocker(self.zoom_value):
            self.zoom_value.setValue(percent)
        self.settings_page.set_ui_scale(percent)
        self.zoom_out_button.setEnabled(percent > UiScaleManager.MIN_PERCENT)
        self.zoom_in_button.setEnabled(percent < UiScaleManager.MAX_PERCENT)
        self.settings.setValue("ui_scale", percent)
        self.annotate_page.refresh_toolbar_button_layout()
        self.train_page.apply_ui_scale(percent / 100.0)
        self.update_header_layout()

    def update_header_layout(self):
        if not hasattr(self, 'header_layout'):
            return
        compact = self.width() - self.sidebar.width() < ui_pixels(700)
        direction = QBoxLayout.Direction.TopToBottom if compact else QBoxLayout.Direction.LeftToRight
        self.header_layout.setDirection(direction)
        self.header_layout.setStretch(0, 0 if compact else 1)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_header_layout()

    def update_header(self, *_):
        index = self.content_stack.currentIndex()
        title, description = self.PAGE_INFO[index]
        self.page_heading.setText(title)
        self.page_description.setText(description)
        name = self.import_page.project_combo.currentText() if self.import_page.current_project_id else "本地工作空间"
        self.breadcrumb.setText(f"{name}  /  {title}")

    def add_scrollable_page(self, stack, page):
        # Native pages keep their minimum content size; overflow stays reachable.
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        area.setWidget(page)
        stack.addWidget(area)

    def create_content_area(self) -> QStackedWidget:
        """创建内容区域"""
        stack = QStackedWidget()
        
        # 导入页面
        self.import_page = ImportPage()
        self.add_scrollable_page(stack, self.import_page)
        
        # 标注页面
        self.annotate_page = AnnotatePage()
        self.add_scrollable_page(stack, self.annotate_page)

        # AI标注页面
        self.ai_label_page = AiLabelPage()
        self.add_scrollable_page(stack, self.ai_label_page)

        # 训练页面
        self.train_page = TrainPage()
        self.add_scrollable_page(stack, self.train_page)
        
        # 训练结果页面
        self.result_page = ResultPage()
        self.add_scrollable_page(stack, self.result_page)
        
        # 测试页面
        self.test_page = TestPage()
        self.add_scrollable_page(stack, self.test_page)
        
        # 设置页面
        self.settings_page = SettingsPage()
        self.add_scrollable_page(stack, self.settings_page)

        # AI标注页添加/修改标签后，标注页类别列表实时同步
        self.ai_label_page.classes_changed.connect(
            self._on_ai_classes_changed
        )

        # 关于页面
        self.about_page = AboutPage()
        self.add_scrollable_page(stack, self.about_page)
        
        # 默认选中第一个页面
        self.nav_buttons[0].setChecked(True)
        
        return stack
    
    def switch_page(self, index: int):
        """切换页面"""
        # 更新按钮状态
        for i, btn in enumerate(self.nav_buttons):
            btn.setChecked(i == index)
        
        # 切换页面
        self.content_stack.setCurrentIndex(index)
        self.update_header()
        
        # 同步项目信息到各页面
        current_project_id = self.import_page.current_project_id
        
        if index == 1:  # 标注页面
            if current_project_id:
                self.annotate_page.set_project(current_project_id)
            else:
                # 如果没有选择项目，提示用户
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")

        elif index == 2:  # AI标注页面
            if current_project_id:
                self.ai_label_page.set_project(current_project_id)
            else:
                # 如果没有选择项目，提示用户
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")

        elif index == 3:  # 训练页面
            if current_project_id:
                self.train_page.set_project(current_project_id)
            else:
                # 如果没有选择项目，提示用户
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")

        elif index == 4:  # 训练结果页面
            if current_project_id:
                self.result_page.set_project(current_project_id)
            else:
                # 如果没有选择项目，提示用户
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")

        elif index == 5:  # 测试页面
            if current_project_id:
                self.test_page.set_project(current_project_id)
            else:
                # 如果没有选择项目，提示用户
                from PyQt6.QtWidgets import QMessageBox
                QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")

        elif index == 6:  # 设置页面
            # 设置页面不需要项目信息
            pass

        elif index == 7:  # 关于页面
            # 关于页面不需要项目信息
            pass
    
    def load_window_state(self):
        """加载窗口状态"""
        # 恢复窗口大小
        size = self.settings.value("size", QSize(1400, 900))
        self.resize(size)
        
        # 恢复窗口位置
        pos = self.settings.value("pos", QPoint(100, 100))
        self.move(pos)
        
        # 恢复窗口状态（最大化等）
        state = self.settings.value("windowState")
        if state:
            self.restoreState(state)
    
    def save_window_state(self):
        """保存窗口状态"""
        self.settings.setValue("size", self.size())
        self.settings.setValue("pos", self.pos())
        self.settings.setValue("windowState", self.saveState())
    
    def on_theme_changed(self, theme_key):
        """Reapply the shared dark palette while preserving display scale."""
        self.scale_manager.stylesheet = get_full_stylesheet('dark')
        self.scale_manager.set_scale(self.scale_manager.percent)
        self.settings.setValue("theme", "深色主题")

    def _on_ai_classes_changed(self):
        """AI标注页标签同步到DB后，刷新标注页的类别列表"""
        try:
            if (self.annotate_page.current_project_id ==
                    self.ai_label_page.current_project_id):
                self.annotate_page.refresh_classes()
        except RuntimeError:
            pass  # 页面销毁中，忽略

    def closeEvent(self, event):
        """关闭事件"""
        ai_page = getattr(self, "ai_label_page", None)
        if ai_page is not None:
            ai_page._config_timer.stop()
            ai_page.save_task_config()
        self.save_window_state()
        event.accept()

    def sync_database_files(self):
        """同步数据库与真实文件"""
        try:
            # 调用数据库同步方法
            result = db.sync_files_with_database()
            
            deleted_db_count = result.get('deleted_db_count', 0)
            deleted_file_count = result.get('deleted_file_count', 0)
            total_deleted = result.get('total_deleted', 0)
            
            if total_deleted > 0:
                print(f"[同步] 删除了 {deleted_db_count} 个数据库中不存在的文件记录")
                print(f"[同步] 删除了 {deleted_file_count} 个文件夹中不存在于数据库的文件")
                print(f"[同步] 总共删除了 {total_deleted} 个项目")
                # 可以在这里添加一个通知，告知用户同步结果
                # 例如：QMessageBox.information(self, "同步完成", f"删除了 {deleted_file_count} 个不存在于数据库的文件和 {deleted_db_count} 个无效记录")
            else:
                print("[同步] 数据库与文件系统一致，无需删除")
        except Exception as e:
            print(f"[同步] 同步过程中出现错误: {str(e)}")
