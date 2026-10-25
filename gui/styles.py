# -*- coding: utf-8 -*-
"""Shared colors and native Qt widget styles for the desktop workspace."""

import html
import re
from pathlib import Path

DARK_COLORS = {
    'background': '#101722', 'sidebar': '#141E2C', 'panel': '#1B2738',
    'selected': '#223E63', 'primary': '#528FFF', 'secondary': '#8192AD',
    'success': '#54CCAF', 'warning': '#E6BD73', 'error': '#F07C86',
    'text_primary': '#E5ECF6', 'text_secondary': '#96A8C0',
    'text_disabled': '#5D708A', 'border': '#2C3C52', 'hover': '#223248',
}
LIGHT_COLORS = {
    'background': '#F3F6FC', 'sidebar': '#FFFFFF', 'panel': '#FFFFFF',
    'selected': '#E5EEFF', 'primary': '#3877E8', 'secondary': '#6F819B',
    'success': '#198C72', 'warning': '#AF7A1E', 'error': '#D44A5B',
    'text_primary': '#203047', 'text_secondary': '#63758E',
    'text_disabled': '#9CAAC0', 'border': '#D8E2EF', 'hover': '#EDF3FC',
}
COLORS = DARK_COLORS


def format_log_html(message, timestamp=""):
    """按日志内容语义着色为 HTML，供各页面日志终端 QTextEdit.append 使用。

    分类优先级: 错误(红) > 警告(橙) > 成功(绿) > 阶段标题(蓝,加粗) > 普通
    """
    text = str(message)
    escaped = html.escape(text).replace("\n", "<br>")
    prefix = f"[{timestamp}] " if timestamp else ""

    if text.startswith("[警告]"):
        color, weight = COLORS['warning'], 'normal'
    elif any(k in text for k in ("✗", "错误", "出错", "失败", "Traceback", "异常", "Error")):
        color, weight = COLORS['error'], 'normal'
    elif any(k in text for k in ("警告", "Warning", "未检测到", "不存在", "请先", "跳过", "暂停", "取消", "停止")):
        color, weight = COLORS['warning'], 'normal'
    elif any(k in text for k in ("✓", "完成", "成功")):
        color, weight = COLORS['success'], 'normal'
    elif (re.search(r"[=\-*#]{3,}", text.strip())
          or "[Epoch" in text
          or any(k in text for k in ("开始训练", "开始推理", "开始标注"))):
        color, weight = COLORS['primary'], 'bold'
    else:
        return f"{prefix}{escaped}"

    return f'{prefix}<span style="color:{color}; font-weight:{weight};">{escaped}</span>'


def generate_stylesheet(c):
    assets = Path(__file__).with_name('assets').as_posix()
    return f"""
    QWidget {{
        color: {c['text_primary']};
        font-family: "Noto Sans CJK SC", "Microsoft YaHei", "Segoe UI", sans-serif;
        font-size: 14px;
        selection-background-color: {c['selected']};
        selection-color: {c['text_primary']};
    }}
    QMainWindow, QDialog, QWidget#workspaceRoot, QWidget#pageContainer {{
        background: {c['background']};
    }}
    QLabel {{ background: transparent; border: none; }}
    QLabel#title {{ font-size: 21px; font-weight: 600; }}
    QLabel#subtitle {{ color: {c['text_secondary']}; font-size: 13px; }}
    QFrame {{ border: none; }}
    QWidget#sidebar {{ background: {c['sidebar']}; border-right: 1px solid {c['border']}; }}
    QWidget#sidebarContents {{ background: {c['sidebar']}; }}
    QLabel#brandMark {{
        background: {c['primary']}; color: white; border-radius: 12px;
        font-size: 17px; font-weight: 700;
    }}
    QLabel#brandTitle {{ font-size: 19px; font-weight: 700; }}
    QLabel#brandSubtitle {{ font-size: 11px; color: {c['text_secondary']}; }}
    QLabel#sectionLabel {{
        font-size: 11px; color: {c['text_secondary']}; padding: 15px 12px 7px 12px;
    }}
    QPushButton#nav_button {{
        background: transparent; color: {c['text_secondary']}; text-align: left;
        padding: 13px 15px; border: 1px solid transparent; border-radius: 9px;
        font-weight: 500;
    }}
    QPushButton#nav_button:hover {{ background: {c['hover']}; color: {c['text_primary']}; }}
    QPushButton#nav_button:checked {{
        background: {c['selected']}; color: {c['text_primary']};
        border: 1px solid {c['primary']}; font-weight: 600;
    }}
    QLabel#localBadge {{
        color: {c['success']}; background: {c['panel']}; border: 1px solid {c['border']};
        border-radius: 8px; padding: 10px; font-size: 12px;
    }}
    QFrame#workspaceHeader {{ background: {c['sidebar']}; border-bottom: 1px solid {c['border']}; }}
    QLabel#breadcrumb {{ color: {c['text_secondary']}; font-size: 11px; }}
    QLabel#pageHeading {{ font-size: 19px; font-weight: 600; }}
    QLabel#pageDescription {{ color: {c['text_secondary']}; font-size: 12px; }}
    QFrame#zoomControls {{
        background: {c['background']}; border: 1px solid {c['border']}; border-radius: 9px;
    }}
    QPushButton#zoomButton, QPushButton#zoomReset {{
        background: transparent; color: {c['text_secondary']}; padding: 5px 8px;
        border: none; border-radius: 6px;
    }}
    QPushButton#zoomButton {{ font-size: 19px; }}
    QPushButton#zoomButton:hover, QPushButton#zoomReset:hover {{
        background: {c['hover']}; color: {c['text_primary']};
    }}
    QSpinBox#zoomValue {{ background: transparent; border: none; padding: 4px 0px; font-weight: 600; }}
    QFrame#pageCard {{ background: {c['background']}; border: 1px solid {c['border']}; border-radius: 12px; }}
    QScrollArea, QStackedWidget {{ background: transparent; border: none; }}
    QScrollArea > QWidget > QWidget {{ background: {c['background']}; }}
    QPushButton, QToolButton {{
        background: {c['primary']}; color: white; border: 1px solid transparent;
        padding: 8px 14px; border-radius: 7px; font-weight: 500;
    }}
    QPushButton:hover, QToolButton:hover {{ background: #679FFF; }}
    QPushButton:pressed, QToolButton:pressed {{ background: #3879E8; }}
    QPushButton:disabled, QToolButton:disabled {{
        background: {c['panel']}; color: {c['text_disabled']}; border-color: {c['border']};
    }}
    QPushButton#secondary {{
        background: {c['panel']}; color: {c['text_primary']}; border: 1px solid {c['border']};
    }}
    QPushButton#secondary:hover {{ background: {c['hover']}; border-color: {c['text_secondary']}; }}
    QPushButton:focus, QToolButton:focus {{ border: 1px solid {c['text_secondary']}; }}
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit, QPlainTextEdit {{
        background: {c['background']}; color: {c['text_primary']};
        border: 1px solid {c['border']}; padding: 7px 9px; border-radius: 6px;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus,
    QTextEdit:focus, QPlainTextEdit:focus {{ border-color: {c['primary']}; }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{ color: {c['text_disabled']}; }}
    QComboBox {{ padding-right: 24px; }}
    QComboBox::drop-down {{ border: none; width: 23px; }}
    QComboBox::down-arrow {{ image: url("{assets}/chevron-down.svg"); width: 14px; height: 14px; }}
    QSpinBox, QDoubleSpinBox {{ padding-right: 23px; }}
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border; subcontrol-position: top right;
        width: 20px; background: transparent; border: none;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border; subcontrol-position: bottom right;
        width: 20px; background: transparent; border: none;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        image: url("{assets}/chevron-up.svg"); width: 12px; height: 12px;
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        image: url("{assets}/chevron-down.svg"); width: 12px; height: 12px;
    }}
    QComboBox QAbstractItemView {{
        background: {c['panel']}; color: {c['text_primary']}; border: 1px solid {c['border']};
        selection-background-color: {c['selected']}; padding: 4px;
    }}
    QGroupBox {{
        background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 9px;
        margin-top: 14px; padding: 18px 10px 10px 10px; font-weight: 600;
    }}
    QGroupBox::title {{
        subcontrol-origin: margin; left: 13px; padding: 0px 7px; color: {c['text_secondary']};
    }}
    QListWidget, QTreeView, QTableView {{
        background: {c['background']}; alternate-background-color: {c['panel']};
        border: 1px solid {c['border']}; border-radius: 7px;
        outline: none; gridline-color: {c['border']};
    }}
    QListWidget::item {{ padding: 9px 10px; border-radius: 5px; }}
    QListWidget::item:hover {{ background: {c['hover']}; }}
    QListWidget::item:selected, QTreeView::item:selected, QTableView::item:selected {{
        background: {c['selected']}; color: {c['text_primary']};
    }}
    QHeaderView::section {{
        background: {c['panel']}; color: {c['text_secondary']}; border: none;
        border-bottom: 1px solid {c['border']}; padding: 8px; font-weight: 500;
    }}
    QTableCornerButton::section {{ background: {c['panel']}; border: none; }}
    QTabWidget::pane {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 8px; }}
    QTabBar::tab {{
        background: transparent; color: {c['text_secondary']};
        padding: 10px 16px; border-bottom: 2px solid transparent;
    }}
    QTabBar::tab:selected {{ color: {c['primary']}; border-bottom-color: {c['primary']}; }}
    QTabBar::tab:hover:!selected {{ background: {c['hover']}; }}
    QProgressBar {{
        background: {c['background']}; border: 1px solid {c['border']}; border-radius: 6px;
        text-align: center; min-height: 18px;
    }}
    QProgressBar::chunk {{ background: {c['primary']}; border-radius: 5px; }}
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{ background: {c['border']}; border-radius: 3px; min-height: 28px; }}
    QScrollBar::handle:horizontal {{ background: {c['border']}; border-radius: 3px; min-width: 28px; }}
    QScrollBar::handle:hover {{ background: {c['text_disabled']}; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ width: 0px; height: 0px; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QSplitter::handle {{ background: {c['border']}; }}
    QSplitter::handle:hover {{ background: {c['primary']}; }}
    QCheckBox, QRadioButton {{ spacing: 7px; background: transparent; }}
    QCheckBox::indicator, QRadioButton::indicator {{ width: 16px; height: 16px; }}
    QCheckBox::indicator {{ background: {c['background']}; border: 1px solid {c['text_disabled']}; border-radius: 4px; }}
    QCheckBox::indicator:checked {{
        background: {c['primary']}; border-color: {c['primary']}; image: url("{assets}/check.svg");
    }}
    QRadioButton::indicator {{ background: {c['background']}; border: 1px solid {c['text_disabled']}; border-radius: 8px; }}
    QRadioButton::indicator:checked {{ background: {c['primary']}; border-color: {c['primary']}; }}
    QSlider::groove:horizontal {{ height: 5px; background: {c['border']}; border-radius: 2px; }}
    QSlider::sub-page:horizontal {{ background: {c['primary']}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {c['text_primary']}; width: 15px; margin: -5px 0px; border-radius: 7px;
    }}
    QMenu {{ background: {c['panel']}; border: 1px solid {c['border']}; border-radius: 7px; padding: 5px; }}
    QMenu::item {{ padding: 8px 22px; border-radius: 4px; }}
    QMenu::item:selected {{ background: {c['selected']}; }}
    QToolTip {{
        background: {c['panel']}; color: {c['text_primary']};
        border: 1px solid {c['border']}; padding: 7px; border-radius: 5px;
    }}
    """


def get_full_stylesheet(theme='dark'):
    return generate_stylesheet(LIGHT_COLORS if theme == 'light' else DARK_COLORS)
