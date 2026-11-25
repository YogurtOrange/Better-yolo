# -*- coding: utf-8 -*-
"""
数据导出配置对话框

点击「导出完整数据集」或「导出标注文件」后弹出，用于选择:
  - 导出格式: YOLO格式 / COCO格式
  - 目录结构 (完整数据集 + YOLO格式):
      * 默认结构: images/{train,val} + labels/{train,val}
      * 服务器结构: train/{images,labels} + val/{images,labels}
  - 划分比例: 默认 训练集 8 : 验证集 2（实时预览比例百分比与预计张数）
  - 随机打乱划分开关

避免在标注页窄右侧栏常驻展示过多表单项而挤压上方「类别列表」显示空间。
"""

from PyQt6.QtCore import QSettings, Qt
from PyQt6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from gui.styles import COLORS

DEFAULT_TREE_TEXT = (
    "├── images\n"
    "│   ├── train\n"
    "│   └── val\n"
    "└── labels\n"
    "    ├── train\n"
    "    └── val"
)

SERVER_TREE_TEXT = (
    "├── train\n"
    "│   ├── images\n"
    "│   └── labels\n"
    "└── val\n"
    "    ├── images\n"
    "    └── labels"
)


class DatasetExportDialog(QDialog):
    """点击导出按钮后弹出的导出格式、目录结构与划分比例配置对话框"""

    def __init__(
        self,
        parent=None,
        mode: str = "dataset",  # "dataset" | "annotations"
        image_count: int = 0,
        default_format: str = "YOLO格式",
    ):
        super().__init__(parent)
        self.mode = mode
        self.image_count = max(0, int(image_count))
        self.settings = QSettings("Better-YOLO", "ExportDataset")

        if mode == "dataset":
            self.setWindowTitle("导出完整数据集设置")
            self.setMinimumWidth(460)
        else:
            self.setWindowTitle("导出标注文件设置")
            self.setMinimumWidth(380)

        self._init_ui(default_format)
        self._load_saved_preferences(default_format)
        self._refresh_visibility()
        self._update_preview()

    def _group_style(self) -> str:
        return f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 10px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """

    def _init_ui(self, default_format: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 1. 导出格式
        fmt_group = QGroupBox("导出格式")
        fmt_group.setStyleSheet(self._group_style())
        fmt_form = QFormLayout(fmt_group)
        self.cb_format = QComboBox()
        self.cb_format.addItems(["YOLO格式", "COCO格式"])
        idx = self.cb_format.findText(default_format)
        if idx >= 0:
            self.cb_format.setCurrentIndex(idx)
        self.cb_format.currentTextChanged.connect(self._on_format_changed)
        fmt_form.addRow("选择格式:", self.cb_format)
        layout.addWidget(fmt_group)

        # 2. YOLO 目录结构与训练/验证划分（仅在 mode=="dataset" 且 YOLO格式 时显示）
        self.yolo_options_container = QWidget()
        yolo_vlayout = QVBoxLayout(self.yolo_options_container)
        yolo_vlayout.setContentsMargins(0, 0, 0, 0)
        yolo_vlayout.setSpacing(12)

        struct_group = QGroupBox("数据集目录结构")
        struct_group.setStyleSheet(self._group_style())
        struct_layout = QVBoxLayout(struct_group)

        radio_row = QHBoxLayout()
        self.rb_default_struct = QRadioButton("默认结构 (images/train)")
        self.rb_server_struct = QRadioButton("服务器结构 (train/images)")
        self.rb_default_struct.setChecked(True)
        self.rb_default_struct.toggled.connect(self._update_preview)
        radio_row.addWidget(self.rb_default_struct)
        radio_row.addWidget(self.rb_server_struct)
        radio_row.addStretch(1)
        struct_layout.addLayout(radio_row)

        self.lbl_tree_preview = QLabel(DEFAULT_TREE_TEXT)
        self.lbl_tree_preview.setStyleSheet(f"""
            QLabel {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px 12px;
                font-family: Consolas, Monaco, monospace;
                font-size: 12px;
                color: {COLORS['text_primary']};
            }}
        """)
        struct_layout.addWidget(self.lbl_tree_preview)
        yolo_vlayout.addWidget(struct_group)

        split_group = QGroupBox("训练集 / 验证集划分比例")
        split_group.setStyleSheet(self._group_style())
        split_form = QFormLayout(split_group)
        split_form.setVerticalSpacing(10)

        ratio_row_widget = QWidget()
        ratio_row = QHBoxLayout(ratio_row_widget)
        ratio_row.setContentsMargins(0, 0, 0, 0)
        ratio_row.setSpacing(8)

        self.sb_train_ratio = QSpinBox()
        self.sb_train_ratio.setRange(1, 99)
        self.sb_train_ratio.setValue(8)
        self.sb_train_ratio.setPrefix("训练 ")
        self.sb_train_ratio.valueChanged.connect(self._update_preview)

        self.sb_val_ratio = QSpinBox()
        self.sb_val_ratio.setRange(0, 99)
        self.sb_val_ratio.setValue(2)
        self.sb_val_ratio.setPrefix("验证 ")
        self.sb_val_ratio.valueChanged.connect(self._update_preview)

        ratio_row.addWidget(self.sb_train_ratio, 1)
        ratio_row.addWidget(QLabel(":"))
        ratio_row.addWidget(self.sb_val_ratio, 1)
        split_form.addRow("划分比例:", ratio_row_widget)

        self.lbl_split_summary = QLabel("")
        self.lbl_split_summary.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.lbl_split_summary.setWordWrap(True)
        split_form.addRow("预计分配:", self.lbl_split_summary)

        self.chk_shuffle = QCheckBox("导出前随机打乱顺序（推荐，防止连续帧集中）")
        self.chk_shuffle.setChecked(True)
        split_form.addRow("", self.chk_shuffle)

        yolo_vlayout.addWidget(split_group)
        layout.addWidget(self.yolo_options_container)

        # 3. 底部按钮
        btn_row = QHBoxLayout()
        if self.mode == "dataset":
            btn_reset = QPushButton("恢复默认 (8:2)")
            btn_reset.setObjectName("secondary")
            btn_reset.clicked.connect(self._reset_defaults)
            btn_row.addWidget(btn_reset)
        btn_row.addStretch(1)

        btn_cancel = QPushButton("取消")
        btn_cancel.setObjectName("secondary")
        btn_cancel.clicked.connect(self.reject)
        self.btn_ok = QPushButton("选择目录并导出")
        self.btn_ok.clicked.connect(self._on_accept)
        btn_row.addWidget(btn_cancel)
        btn_row.addWidget(self.btn_ok)
        layout.addLayout(btn_row)

    def _load_saved_preferences(self, default_format: str):
        saved_fmt = str(self.settings.value("format", default_format))
        idx = self.cb_format.findText(saved_fmt)
        if idx >= 0:
            self.cb_format.setCurrentIndex(idx)

        saved_struct = str(self.settings.value("structure", "default"))
        if saved_struct == "server":
            self.rb_server_struct.setChecked(True)
        else:
            self.rb_default_struct.setChecked(True)

        try:
            self.sb_train_ratio.setValue(max(1, int(self.settings.value("train_ratio", 8))))
            self.sb_val_ratio.setValue(max(0, int(self.settings.value("val_ratio", 2))))
        except (TypeError, ValueError):
            self.sb_train_ratio.setValue(8)
            self.sb_val_ratio.setValue(2)

        self.chk_shuffle.setChecked(self.settings.value("shuffle", True, type=bool))

    def _reset_defaults(self):
        self.cb_format.setCurrentText("YOLO格式")
        self.rb_default_struct.setChecked(True)
        self.sb_train_ratio.setValue(8)
        self.sb_val_ratio.setValue(2)
        self.chk_shuffle.setChecked(True)
        self._update_preview()

    def _on_format_changed(self, _fmt: str):
        self._refresh_visibility()
        self.adjustSize()

    def _refresh_visibility(self):
        show_yolo_split = (
            self.mode == "dataset" and self.cb_format.currentText() == "YOLO格式"
        )
        self.yolo_options_container.setVisible(show_yolo_split)

    def _update_preview(self):
        is_server = self.rb_server_struct.isChecked()
        self.lbl_tree_preview.setText(SERVER_TREE_TEXT if is_server else DEFAULT_TREE_TEXT)

        t = self.sb_train_ratio.value()
        v = self.sb_val_ratio.value()
        total_w = max(1, t + v)
        t_pct = round(t * 100 / total_w)
        v_pct = 100 - t_pct

        if self.image_count <= 0:
            self.lbl_split_summary.setText(f"比例 {t}:{v}（约 {t_pct}% : {v_pct}%）")
            return

        if v <= 0 or self.image_count == 1:
            t_cnt, v_cnt = self.image_count, 0
        else:
            v_cnt = max(1, min(self.image_count - 1, round(self.image_count * v / total_w)))
            t_cnt = self.image_count - v_cnt

        self.lbl_split_summary.setText(
            f"比例 {t}:{v}（{t_pct}% : {v_pct}%） → 预计训练集 {t_cnt} 张，验证集 {v_cnt} 张"
        )

    def _on_accept(self):
        self.settings.setValue("format", self.cb_format.currentText())
        self.settings.setValue("structure", self.get_structure_mode())
        self.settings.setValue("train_ratio", self.sb_train_ratio.value())
        self.settings.setValue("val_ratio", self.sb_val_ratio.value())
        self.settings.setValue("shuffle", self.chk_shuffle.isChecked())
        self.accept()

    def get_export_format(self) -> str:
        return self.cb_format.currentText()

    def get_structure_mode(self) -> str:
        return "server" if self.rb_server_struct.isChecked() else "default"

    def get_config(self) -> dict:
        return {
            "export_format": self.get_export_format(),
            "structure_mode": self.get_structure_mode(),
            "train_ratio": self.sb_train_ratio.value(),
            "val_ratio": self.sb_val_ratio.value(),
            "shuffle": self.chk_shuffle.isChecked(),
        }
