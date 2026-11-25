# -*- coding: utf-8 -*-
"""SAM 记忆标注对象管理弹窗。"""

from PyQt6.QtCore import pyqtSignal
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QLabel, QListWidget, QPushButton, QVBoxLayout


class SAMMemoryObjectsDialog(QDialog):
    """记忆标注对象管理弹窗（可在画布上交互后添加对象）。"""

    add_requested = pyqtSignal()
    delete_requested = pyqtSignal(int)
    save_requested = pyqtSignal()
    closed = pyqtSignal()  # 窗口关闭信号

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("SAM记忆对象管理")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        self.info_label = QLabel("请在图片上用SAM提示（点/框）后点击“添加对象”。")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)

        self.count_label = QLabel("当前对象数: 0")
        layout.addWidget(self.count_label)

        self.obj_list = QListWidget()
        layout.addWidget(self.obj_list)

        row = QHBoxLayout()
        self.btn_add = QPushButton("添加对象")
        self.btn_del = QPushButton("删除对象")
        self.btn_save = QPushButton("保存并更新记忆")
        row.addWidget(self.btn_add)
        row.addWidget(self.btn_del)
        row.addWidget(self.btn_save)
        layout.addLayout(row)

        self.btn_add.clicked.connect(self.add_requested.emit)
        self.btn_del.clicked.connect(self._on_delete_clicked)
        self.btn_save.clicked.connect(self.save_requested.emit)

    def closeEvent(self, event):
        """窗口关闭时发送信号"""
        self.closed.emit()
        super().closeEvent(event)

    def _on_delete_clicked(self):
        row = self.obj_list.currentRow()
        if row >= 0:
            self.delete_requested.emit(row)

    def update_objects(self, objects: list):
        self.obj_list.clear()
        for obj in objects:
            desc = (
                f"ID={obj.get('obj_id')}  "
                f"点={len(obj.get('points', []))}  "
                f"框={len(obj.get('bboxes', []))}"
            )
            self.obj_list.addItem(desc)
        self.count_label.setText(f"当前对象数: {len(objects)}")
