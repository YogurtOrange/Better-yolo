# -*- coding: utf-8 -*-
"""
关于页面
"""

from html import escape

from PyQt6.QtWidgets import QWidget, QVBoxLayout, QLabel, QGroupBox, QFormLayout
from PyQt6.QtCore import Qt
from config.app_info import APP_VERSION, AUTHOR_NAME, GITHUB_REPOSITORY, LAST_UPDATED
from gui.styles import COLORS


class AboutPage(QWidget):
    """关于页面"""
    
    def __init__(self):
        super().__init__()
        self.setObjectName("pageContainer")
        self.init_ui()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 标题
        title = QLabel("关于 Better-YOLO")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 20px; font-weight: bold;")
        main_layout.addWidget(title)
        
        # 作者信息
        author_group = self.create_author_group()
        main_layout.addWidget(author_group)
        
        # 项目仓库
        contact_group = self.create_contact_group()
        main_layout.addWidget(contact_group)
        
        # 版本信息
        version_group = self.create_version_group()
        main_layout.addWidget(version_group)
        
        # 底部
        main_layout.addStretch()
    
    def create_author_group(self) -> QGroupBox:
        """创建作者信息组"""
        group = QGroupBox("作者信息")
        
        layout = QFormLayout(group)
        
        author_label = QLabel(AUTHOR_NAME)
        author_label.setStyleSheet(f"background-color: {COLORS['background']}; padding: 8px; border-radius: 6px;")
        author_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        layout.addRow("作者:", author_label)
        
        return group
    
    def create_contact_group(self) -> QGroupBox:
        """只展示项目仓库地址，支持打开和复制链接。"""
        group = QGroupBox("项目仓库")
        
        layout = QFormLayout(group)
        
        repository_url = escape(GITHUB_REPOSITORY, quote=True)
        repository_label = QLabel(
            f'<a href="{repository_url}" style="color: {COLORS["primary"]};">'
            f'{repository_url}</a>'
        )
        repository_label.setStyleSheet(f"background-color: {COLORS['background']}; padding: 8px; border-radius: 6px;")
        repository_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextBrowserInteraction)
        repository_label.setOpenExternalLinks(True)
        repository_label.setWordWrap(True)
        layout.addRow("GitHub:", repository_label)
        
        return group
    
    def create_version_group(self) -> QGroupBox:
        """创建版本信息组"""
        group = QGroupBox("版本信息")
        
        layout = QFormLayout(group)
        
        version_label = QLabel(APP_VERSION)
        version_label.setStyleSheet(f"background-color: {COLORS['background']}; padding: 8px; border-radius: 6px;")
        layout.addRow("版本:", version_label)
        
        update_label = QLabel(LAST_UPDATED)
        update_label.setStyleSheet(f"background-color: {COLORS['background']}; padding: 8px; border-radius: 6px;")
        layout.addRow("更新日期:", update_label)
        
        return group
