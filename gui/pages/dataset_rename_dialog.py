# -*- coding: utf-8 -*-
"""
一键重命名数据集图片与标签对话框

支持:
  - 当前打开项目内的数据一键重命名（同步图片文件、数据库记录与导出标签名）
  - 本地外部 YOLO/VOC/COCO 数据集文件夹一键配对重命名（图片 + 同名 .txt/.xml/.json 标签文件）
  - 自定义前缀（默认 img_）、起始编号（默认 1 -> img_001）、最少补零位数（默认 3 位）及排序规则
  - 实时预览重命名前后映射效果
"""

from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QProgressDialog,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.dataset_renamer import (
    preview_external_folder_rename,
    preview_project_rename,
    rename_external_folder_dataset,
    rename_project_dataset,
)
from gui.styles import COLORS
from models.database import db


class DatasetRenameDialog(QDialog):
    """数据集图片与标签一键重命名弹窗"""

    project_renamed = pyqtSignal(int)

    def __init__(self, parent=None, project_id: Optional[int] = None):
        super().__init__(parent)
        self.project_id = project_id
        self.project = db.get_project(project_id) if project_id else None

        self.setWindowTitle("一键重命名数据集图片与标签")
        self.setMinimumSize(580, 520)
        self.resize(620, 560)

        self._init_ui()
        self._refresh_mode_ui()
        self.update_preview()

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

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # 1. 目标范围选择
        scope_group = QGroupBox("重命名对象")
        scope_group.setStyleSheet(self._group_style())
        scope_layout = QVBoxLayout(scope_group)

        proj_title = (
            f"当前项目：{self.project.get('name', '')}"
            if self.project
            else "当前项目（未打开项目）"
        )
        self.rb_project = QRadioButton(f"{proj_title}（同步重命名项目图片与关联标注/导出标签名）")
        self.rb_external = QRadioButton("本地外部数据集文件夹（配对重命名图片与同名 .txt / .xml / .json 标签）")

        if self.project:
            self.rb_project.setChecked(True)
        else:
            self.rb_project.setEnabled(False)
            self.rb_external.setChecked(True)

        self.rb_project.toggled.connect(self._on_mode_changed)
        scope_layout.addWidget(self.rb_project)
        scope_layout.addWidget(self.rb_external)

        self.folder_row_widget = QWidget()
        folder_row = QHBoxLayout(self.folder_row_widget)
        folder_row.setContentsMargins(22, 4, 0, 0)
        self.le_folder = QLineEdit()
        self.le_folder.setPlaceholderText("选择数据集根目录（支持同目录混放或 images/ + labels/ 结构，自动跳过 classes.txt）")
        self.le_folder.textChanged.connect(self.update_preview)
        self.btn_browse = QPushButton("浏览...")
        self.btn_browse.clicked.connect(self._browse_folder)
        folder_row.addWidget(self.le_folder, 1)
        folder_row.addWidget(self.btn_browse)
        scope_layout.addWidget(self.folder_row_widget)

        layout.addWidget(scope_group)

        # 2. 命名规则配置
        rule_group = QGroupBox("命名格式设置")
        rule_group.setStyleSheet(self._group_style())
        form = QFormLayout(rule_group)
        form.setVerticalSpacing(10)

        self.le_prefix = QLineEdit("img_")
        self.le_prefix.setPlaceholderText("默认 img_（如 img_001），可改为 sample_ 等或留空")
        self.le_prefix.textChanged.connect(self.update_preview)
        form.addRow("名称前缀:", self.le_prefix)

        num_row = QHBoxLayout()
        self.sb_start_index = QSpinBox()
        self.sb_start_index.setRange(0, 999999)
        self.sb_start_index.setValue(1)
        self.sb_start_index.setToolTip("默认从 1 开始（即 img_001），可自由指定任意起始数字")
        self.sb_start_index.valueChanged.connect(self.update_preview)
        num_row.addWidget(self.sb_start_index, 1)

        num_row.addSpacing(12)
        num_row.addWidget(QLabel("最少位数:"))
        self.sb_min_digits = QSpinBox()
        self.sb_min_digits.setRange(1, 8)
        self.sb_min_digits.setValue(3)
        self.sb_min_digits.setSuffix(" 位")
        self.sb_min_digits.setToolTip("不足该位数时自动左侧补 0（默认 3 位：001）；超出时自动扩容不截断")
        self.sb_min_digits.valueChanged.connect(self.update_preview)
        num_row.addWidget(self.sb_min_digits)
        form.addRow("起始编号:", num_row)

        self.cb_sort_mode = QComboBox()
        self.cb_sort_mode.currentIndexChanged.connect(self.update_preview)
        form.addRow("排序方式:", self.cb_sort_mode)

        layout.addWidget(rule_group)

        # 3. 实时预览
        preview_group = QGroupBox("重命名实时预览")
        preview_group.setStyleSheet(self._group_style())
        preview_layout = QVBoxLayout(preview_group)

        self.lbl_summary = QLabel("")
        self.lbl_summary.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.lbl_summary.setWordWrap(True)
        preview_layout.addWidget(self.lbl_summary)

        self.preview_list = QListWidget()
        self.preview_list.setAlternatingRowColors(True)
        preview_layout.addWidget(self.preview_list, 1)

        layout.addWidget(preview_group, 1)

        # 4. 底部操作按钮
        btn_row = QHBoxLayout()
        btn_reset = QPushButton("恢复默认 (img_001)")
        btn_reset.setObjectName("secondary")
        btn_reset.clicked.connect(self._reset_defaults)
        btn_row.addWidget(btn_reset)
        btn_row.addStretch(1)

        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.setObjectName("secondary")
        self.btn_cancel.clicked.connect(self.reject)
        self.btn_execute = QPushButton("开始一键重命名")
        self.btn_execute.clicked.connect(self._on_execute)
        btn_row.addWidget(self.btn_cancel)
        btn_row.addWidget(self.btn_execute)
        layout.addLayout(btn_row)

    def _reset_defaults(self):
        self.le_prefix.setText("img_")
        self.sb_start_index.setValue(1)
        self.sb_min_digits.setValue(3)
        self.cb_sort_mode.setCurrentIndex(0)
        self.update_preview()

    def _on_mode_changed(self):
        self._refresh_mode_ui()
        self.update_preview()

    def _refresh_mode_ui(self):
        is_proj = self.rb_project.isChecked()
        self.folder_row_widget.setVisible(not is_proj)

        self.cb_sort_mode.blockSignals(True)
        self.cb_sort_mode.clear()
        if is_proj:
            self.cb_sort_mode.addItem("按项目导入顺序（默认）", "import_order")
            self.cb_sort_mode.addItem("按原文件名自然排序", "natural_name")
        else:
            self.cb_sort_mode.addItem("按原文件名自然排序（默认）", "natural_name")
            self.cb_sort_mode.addItem("按文件修改时间排序", "mtime")
        self.cb_sort_mode.blockSignals(False)

    def _browse_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self,
            "选择要重命名的数据集目录",
            self.le_folder.text().strip() or str(Path.home()),
        )
        if folder:
            self.le_folder.setText(folder)

    def update_preview(self):
        """实时计算并刷新预览列表"""
        self.preview_list.clear()
        prefix = self.le_prefix.text()
        start_idx = self.sb_start_index.value()
        min_digits = self.sb_min_digits.value()
        sort_mode = self.cb_sort_mode.currentData() or "natural_name"

        try:
            if self.rb_project.isChecked():
                if not self.project_id:
                    self.lbl_summary.setText("当前未选择项目。")
                    self.btn_execute.setEnabled(False)
                    return
                plan = preview_project_rename(
                    self.project_id,
                    prefix=prefix,
                    start_index=start_idx,
                    min_digits=min_digits,
                    sort_mode=sort_mode,
                )
                if not plan:
                    self.lbl_summary.setText("当前项目中还没有导入任何图片。")
                    self.btn_execute.setEnabled(False)
                    return

                labeled_cnt = sum(1 for it in plan if it['has_db_label'] or it['sidecar_moves'])
                self.lbl_summary.setText(
                    f"共 {len(plan)} 张项目图片（含已标注图片 {labeled_cnt} 张，导出与训练时自动对应生成同名 .txt 标签）\n"
                    f"编号范围：{plan[0]['new_filename']}  ~  {plan[-1]['new_filename']}"
                )
                self._populate_preview_items(plan, is_project=True)
                self.btn_execute.setEnabled(True)
            else:
                folder = self.le_folder.text().strip()
                if not folder:
                    self.lbl_summary.setText("请先选择本地外部数据集文件夹。")
                    self.btn_execute.setEnabled(False)
                    return
                plan = preview_external_folder_rename(
                    folder,
                    prefix=prefix,
                    start_index=start_idx,
                    min_digits=min_digits,
                    sort_mode=sort_mode,
                )
                if not plan:
                    self.lbl_summary.setText("所选文件夹中未找到支持的图片文件。")
                    self.btn_execute.setEnabled(False)
                    return

                lbl_cnt = sum(len(it['label_moves']) for it in plan)
                self.lbl_summary.setText(
                    f"共扫描到 {len(plan)} 张图片，自动匹配到 {lbl_cnt} 个同名标签文件（classes.txt 等配置自动跳过）\n"
                    f"编号范围：{plan[0]['new_filename']}  ~  {plan[-1]['new_filename']}"
                )
                self._populate_preview_items(plan, is_project=False)
                self.btn_execute.setEnabled(True)

        except Exception as exc:
            self.lbl_summary.setText(f"参数或路径提示: {exc}")
            self.btn_execute.setEnabled(False)

    def _populate_preview_items(self, plan: list, is_project: bool):
        head_count = min(6, len(plan))
        for item in plan[:head_count]:
            self.preview_list.addItem(self._format_plan_row(item, is_project))
        if len(plan) > head_count + 1:
            self.preview_list.addItem(f"    ... 省略中间 {len(plan) - head_count - 1} 项 ...")
        if len(plan) > head_count:
            self.preview_list.addItem(self._format_plan_row(plan[-1], is_project))

    @staticmethod
    def _format_plan_row(item: dict, is_project: bool) -> str:
        old_fn = item['old_filename']
        new_fn = item['new_filename']
        new_stem = item['new_stem']
        if is_project:
            if item.get('has_db_label') or item.get('sidecar_moves'):
                return f"{old_fn} (+标注)   →   {new_fn}  &  {new_stem}.txt"
            return f"{old_fn}   →   {new_fn}"
        else:
            lbls = item.get('label_moves') or []
            if lbls:
                exts = "/".join(dst.name for _, dst in lbls)
                return f"{old_fn} (+{len(lbls)}个标签)   →   {new_fn}  &  {exts}"
            return f"{old_fn} (无同名标签)   →   {new_fn}"

    def _on_execute(self):
        prefix = self.le_prefix.text()
        start_idx = self.sb_start_index.value()
        min_digits = self.sb_min_digits.value()
        sort_mode = self.cb_sort_mode.currentData() or "natural_name"
        is_proj = self.rb_project.isChecked()

        target_desc = (
            f"当前项目「{self.project.get('name', '')}」"
            if is_proj and self.project
            else f"本地目录「{self.le_folder.text().strip()}」"
        )
        reply = QMessageBox.question(
            self,
            "确认一键重命名",
            f"即将对 {target_desc} 执行批量重命名。\n"
            "图片与匹配的标签文件将同步改名，是否立即执行？",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        progress = QProgressDialog("正在批量重命名图片与标签...", None, 0, 100, self)
        progress.setWindowTitle("正在重命名")
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)

        def on_prog(pct: int, msg: str):
            progress.setValue(pct)
            progress.setLabelText(msg)

        try:
            if is_proj:
                res = rename_project_dataset(
                    self.project_id,
                    prefix=prefix,
                    start_index=start_idx,
                    min_digits=min_digits,
                    sort_mode=sort_mode,
                    progress_callback=on_prog,
                )
                progress.close()
                self.project_renamed.emit(self.project_id)
                QMessageBox.information(
                    self,
                    "重命名完成",
                    f"已完成当前项目重命名！\n\n"
                    f"• 重命名图片：{res['renamed_images']} 张\n"
                    f"• 同步已标注记录：{res['synced_db_labels']} 张（导出时自动生成同名 .txt）\n"
                    f"• 编号范围：{res.get('first_name', '')} ~ {res.get('last_name', '')}",
                )
            else:
                folder = self.le_folder.text().strip()
                res = rename_external_folder_dataset(
                    folder,
                    prefix=prefix,
                    start_index=start_idx,
                    min_digits=min_digits,
                    sort_mode=sort_mode,
                    progress_callback=on_prog,
                )
                progress.close()
                QMessageBox.information(
                    self,
                    "重命名完成",
                    f"已完成外部数据集目录重命名！\n\n"
                    f"• 重命名图片：{res['renamed_images']} 张\n"
                    f"• 配对重命名标签：{res['renamed_label_files']} 个\n"
                    f"• 编号范围：{res.get('first_name', '')} ~ {res.get('last_name', '')}",
                )
            self.accept()
        except Exception as exc:
            progress.close()
            QMessageBox.critical(self, "重命名失败", str(exc))
