# -*- coding: utf-8 -*-
"""
训练结果页面
显示runs文件夹中的训练结果图
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QListWidget, 
    QListWidgetItem, QPushButton, QSplitter, QScrollArea,
    QGroupBox, QFormLayout, QGridLayout, QMessageBox, QFileDialog,
    QDialog, QSpinBox, QDoubleSpinBox, QCheckBox, QComboBox, QLineEdit,
    QTabWidget, QTextEdit, QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QSignalBlocker
from PyQt6.QtGui import QPixmap, QImage
from datetime import datetime
import os
import shutil
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple

from gui.styles import COLORS
from gui.widgets.image_preview import ImagePreview
from gui.widgets.flow_layout import FlowLayout


def load_pixmap_unicode_safe(filepath: str) -> QPixmap:
    """安全加载包含中文或特殊字符路径的图像为 QPixmap。"""
    if not filepath or not os.path.exists(filepath):
        return QPixmap()

    pixmap = QPixmap(filepath)
    if not pixmap.isNull():
        return pixmap

    try:
        raw = np.fromfile(filepath, dtype=np.uint8)
        if raw.size == 0:
            return QPixmap()
        img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if img is None:
            return QPixmap()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        qimg = QImage(
            img.data, width, height, width * channel, QImage.Format.Format_RGB888
        ).copy()
        return QPixmap.fromImage(qimg)
    except Exception:
        return QPixmap()


def _run_dir_mtime(run_dir: str) -> float:
    """计算训练实验目录的最新修改时间（优先取 results.csv 与 best.pt）。"""
    candidates = [
        run_dir,
        os.path.join(run_dir, "results.csv"),
        os.path.join(run_dir, "weights", "best.pt"),
        os.path.join(run_dir, "weights", "last.pt"),
    ]
    mtimes = []
    for path in candidates:
        if os.path.exists(path):
            try:
                mtimes.append(os.path.getmtime(path))
            except OSError:
                pass
    return max(mtimes) if mtimes else 0.0


def discover_run_directories(
    runs_dir: str, project_id: Optional[int]
) -> Tuple[List[Dict], bool]:
    """扫描 runs 目录下所有包含 weights 的训练实验，按修改时间倒序排列。

    Returns:
        (runs, is_fallback_all):
            runs: 列表项包含 rel_path, abs_path, mtime, display_name
            is_fallback_all: 当前项目无专属实验、回退显示全部实验时为 True
    """
    if not runs_dir or not os.path.exists(runs_dir):
        return [], False

    all_runs: List[Dict] = []
    for root, dirs, _files in os.walk(runs_dir):
        if 'weights' in dirs:
            rel_path = os.path.relpath(root, runs_dir)
            mtime = _run_dir_mtime(root)
            mtime_str = (
                datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                if mtime > 0
                else ""
            )
            display_name = f"{rel_path}  ({mtime_str})" if mtime_str else rel_path
            all_runs.append({
                'rel_path': rel_path,
                'abs_path': root,
                'mtime': mtime,
                'display_name': display_name,
            })

    if not all_runs:
        return [], False

    # 按修改时间倒序，最新实验在前
    all_runs.sort(key=lambda item: (item['mtime'], item['rel_path']), reverse=True)

    if project_id is None:
        return all_runs, False

    prefix = f"exp_{project_id}"
    exact_matches = [
        item for item in all_runs
        if os.path.basename(item['rel_path']).startswith(prefix)
        or item['rel_path'].startswith(prefix)
    ]
    if exact_matches:
        return exact_matches, False

    fuzzy_matches = [
        item for item in all_runs
        if str(project_id) in item['rel_path']
    ]
    if fuzzy_matches:
        return fuzzy_matches, False

    return all_runs, True


class ModelExportThread(QThread):
    """后台执行模型与结果文件夹导出，避免阻塞主界面。"""

    progress_message = pyqtSignal(str)
    export_finished = pyqtSignal(bool, str)

    def __init__(
        self,
        export_mode: str,
        project_dir: str,
        save_path: str,
        format_type: str = "",
        export_config: Optional[Dict] = None,
        run_name: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self.export_mode = export_mode
        self.project_dir = project_dir
        self.save_path = save_path
        self.format_type = format_type
        self.export_config = dict(export_config or {})
        self.run_name = run_name

    def run(self):
        try:
            if self.export_mode == 'pt':
                best_model = os.path.join(self.project_dir, "weights", "best.pt")
                if not os.path.exists(best_model):
                    self.export_finished.emit(False, "未找到 best.pt 模型文件")
                    return
                self.progress_message.emit("正在复制 best.pt 权重文件...")
                shutil.copy2(best_model, self.save_path)
                self.export_finished.emit(True, f"模型已导出为 {self.save_path}")
                return

            if self.export_mode == 'folder':
                run_folder_name = self.run_name or os.path.basename(self.project_dir)
                target_dir = os.path.join(self.save_path, run_folder_name)
                self.progress_message.emit(f"正在复制结果文件夹到 {target_dir}...")
                if os.path.exists(target_dir):
                    shutil.rmtree(target_dir)
                shutil.copytree(self.project_dir, target_dir)
                self.export_finished.emit(True, f"结果文件夹已导出到 {target_dir}")
                return

            if self.export_mode == 'ultralytics':
                best_model = os.path.join(self.project_dir, "weights", "best.pt")
                if not os.path.exists(best_model):
                    self.export_finished.emit(False, "未找到 best.pt 模型文件")
                    return

                self.progress_message.emit(
                    f"正在导出为 {self.format_type.upper()}，请稍候..."
                )
                from ultralytics import YOLO

                model = YOLO(best_model)
                export_format = (
                    'engine' if self.format_type == 'tensorrt' else self.format_type
                )
                export_kwargs = {'format': export_format}
                if self.format_type in ('onnx', 'tensorrt'):
                    export_kwargs.update(self.export_config)
                    export_kwargs = {
                        k: v for k, v in export_kwargs.items() if v is not None
                    }

                print(f"[导出] 参数: {export_kwargs}")
                exported = model.export(**export_kwargs)

                candidates = []
                if isinstance(exported, (str, os.PathLike)) and os.path.exists(exported):
                    candidates.append(str(exported))
                if self.format_type == 'onnx':
                    candidates.append(os.path.join(self.project_dir, "weights", "best.onnx"))
                elif self.format_type == 'tensorrt':
                    candidates.append(os.path.join(self.project_dir, "weights", "best.engine"))
                elif self.format_type == 'torchscript':
                    candidates.append(
                        os.path.join(self.project_dir, "weights", "best.torchscript.pt")
                    )

                export_path = next((p for p in candidates if os.path.exists(p)), None)
                if export_path:
                    if os.path.abspath(export_path) != os.path.abspath(self.save_path):
                        shutil.move(export_path, self.save_path)
                    self.export_finished.emit(True, f"模型已导出为 {self.save_path}")
                else:
                    self.export_finished.emit(False, "导出失败，未生成目标文件，请检查日志")
                return

            self.export_finished.emit(False, f"不支持的导出模式: {self.export_mode}")
        except Exception as e:
            self.export_finished.emit(False, f"导出出错: {str(e)}")


class ONNXExportDialog(QDialog):
    """ONNX导出配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ONNX导出配置")
        self.setMinimumWidth(400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 基本参数
        basic_group = QGroupBox("基本参数")
        basic_layout = QFormLayout(basic_group)
        
        # imgsz
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 4096)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        basic_layout.addRow("输入尺寸 (imgsz):", self.spin_imgsz)
        
        # batch
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(1)
        basic_layout.addRow("批量大小 (batch):", self.spin_batch)
        
        # opset
        self.spin_opset = QSpinBox()
        self.spin_opset.setRange(7, 17)
        self.spin_opset.setValue(12)
        basic_layout.addRow("ONNX Opset:", self.spin_opset)
        
        # device
        self.combo_device = QComboBox()
        self.combo_device.addItems(["cpu", "0", "1", "2", "3"])
        basic_layout.addRow("设备 (device):", self.combo_device)
        
        layout.addWidget(basic_group)
        
        # 选项参数
        options_group = QGroupBox("选项")
        options_layout = QVBoxLayout(options_group)
        
        self.chk_half = QCheckBox("半精度 (half) - FP16")
        self.chk_half.setChecked(False)
        options_layout.addWidget(self.chk_half)
        
        self.chk_dynamic = QCheckBox("动态轴 (dynamic)")
        self.chk_dynamic.setChecked(False)
        options_layout.addWidget(self.chk_dynamic)
        
        self.chk_simplify = QCheckBox("简化模型 (simplify)")
        self.chk_simplify.setChecked(True)
        options_layout.addWidget(self.chk_simplify)
        
        self.chk_nms = QCheckBox("包含NMS (nms)")
        self.chk_nms.setChecked(False)
        options_layout.addWidget(self.chk_nms)
        
        layout.addWidget(options_group)
        
        # NMS参数组（仅在NMS选中时启用）
        self.nms_group = QGroupBox("NMS参数")
        self.nms_group.setEnabled(False)
        nms_layout = QFormLayout(self.nms_group)
        
        # conf
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        nms_layout.addRow("置信度阈值 (conf):", self.spin_conf)
        
        # iou
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setValue(0.45)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        nms_layout.addRow("IoU阈值 (iou):", self.spin_iou)
        
        # agnostic_nms
        self.chk_agnostic_nms = QCheckBox("类别无关NMS (agnostic_nms)")
        self.chk_agnostic_nms.setChecked(False)
        nms_layout.addRow(self.chk_agnostic_nms)
        
        layout.addWidget(self.nms_group)
        
        # NMS选中时启用NMS参数组
        self.chk_nms.stateChanged.connect(
            lambda state: self.nms_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
    
    def get_config(self) -> dict:
        """获取配置"""
        config = {
            'imgsz': self.spin_imgsz.value(),
            'half': self.chk_half.isChecked(),
            'dynamic': self.chk_dynamic.isChecked(),
            'simplify': self.chk_simplify.isChecked(),
            'opset': self.spin_opset.value(),
            'nms': self.chk_nms.isChecked(),
            'batch': self.spin_batch.value(),
            'device': self.combo_device.currentText()
        }
        
        # 如果启用NMS，添加NMS参数
        if self.chk_nms.isChecked():
            config['conf'] = self.spin_conf.value()
            config['iou'] = self.spin_iou.value()
            config['agnostic_nms'] = self.chk_agnostic_nms.isChecked()
        
        return config


class TensorRTExportDialog(QDialog):
    """TensorRT导出配置对话框"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("TensorRT导出配置")
        self.setMinimumWidth(400)
        self.init_ui()
    
    def init_ui(self):
        layout = QVBoxLayout(self)
        
        # 基本参数
        basic_group = QGroupBox("基本参数")
        basic_layout = QFormLayout(basic_group)
        
        # imgsz
        self.spin_imgsz = QSpinBox()
        self.spin_imgsz.setRange(32, 4096)
        self.spin_imgsz.setValue(640)
        self.spin_imgsz.setSingleStep(32)
        basic_layout.addRow("输入尺寸 (imgsz):", self.spin_imgsz)
        
        # batch
        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(1)
        basic_layout.addRow("批量大小 (batch):", self.spin_batch)
        
        # workspace
        self.spin_workspace = QSpinBox()
        self.spin_workspace.setRange(1, 16)
        self.spin_workspace.setValue(4)
        basic_layout.addRow("工作空间 (workspace GB):", self.spin_workspace)
        
        # device
        self.combo_device = QComboBox()
        self.combo_device.addItems(["0", "1", "2", "3"])
        basic_layout.addRow("设备 (device):", self.combo_device)
        
        layout.addWidget(basic_group)
        
        # 选项参数
        options_group = QGroupBox("选项")
        options_layout = QVBoxLayout(options_group)
        
        self.chk_half = QCheckBox("半精度 (half) - FP16")
        self.chk_half.setChecked(True)
        options_layout.addWidget(self.chk_half)
        
        self.chk_int8 = QCheckBox("INT8量化 (int8)")
        self.chk_int8.setChecked(False)
        options_layout.addWidget(self.chk_int8)
        
        self.chk_dynamic = QCheckBox("动态轴 (dynamic)")
        self.chk_dynamic.setChecked(False)
        options_layout.addWidget(self.chk_dynamic)
        
        self.chk_simplify = QCheckBox("简化模型 (simplify)")
        self.chk_simplify.setChecked(True)
        options_layout.addWidget(self.chk_simplify)
        
        self.chk_nms = QCheckBox("包含NMS (nms)")
        self.chk_nms.setChecked(False)
        options_layout.addWidget(self.chk_nms)
        
        layout.addWidget(options_group)
        
        # NMS参数组（仅在NMS选中时启用）
        self.nms_group = QGroupBox("NMS参数")
        self.nms_group.setEnabled(False)
        nms_layout = QFormLayout(self.nms_group)
        
        # conf
        self.spin_conf = QDoubleSpinBox()
        self.spin_conf.setRange(0.01, 1.0)
        self.spin_conf.setValue(0.25)
        self.spin_conf.setDecimals(2)
        self.spin_conf.setSingleStep(0.05)
        nms_layout.addRow("置信度阈值 (conf):", self.spin_conf)
        
        # iou
        self.spin_iou = QDoubleSpinBox()
        self.spin_iou.setRange(0.1, 1.0)
        self.spin_iou.setValue(0.45)
        self.spin_iou.setDecimals(2)
        self.spin_iou.setSingleStep(0.05)
        nms_layout.addRow("IoU阈值 (iou):", self.spin_iou)
        
        # agnostic_nms
        self.chk_agnostic_nms = QCheckBox("类别无关NMS (agnostic_nms)")
        self.chk_agnostic_nms.setChecked(False)
        nms_layout.addRow(self.chk_agnostic_nms)
        
        layout.addWidget(self.nms_group)
        
        # NMS选中时启用NMS参数组
        self.chk_nms.stateChanged.connect(
            lambda state: self.nms_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # INT8校准参数（仅在INT8选中时启用）
        self.int8_group = QGroupBox("INT8校准参数")
        self.int8_group.setEnabled(False)
        int8_layout = QFormLayout(self.int8_group)
        
        self.edit_data = QLineEdit()
        self.edit_data.setPlaceholderText("校准数据集路径（如：coco128.yaml）")
        int8_layout.addRow("校准数据 (data):", self.edit_data)
        
        self.spin_fraction = QDoubleSpinBox()
        self.spin_fraction.setRange(0.1, 1.0)
        self.spin_fraction.setValue(1.0)
        self.spin_fraction.setSingleStep(0.1)
        int8_layout.addRow("数据比例 (fraction):", self.spin_fraction)
        
        layout.addWidget(self.int8_group)
        
        # INT8选中时启用校准参数
        self.chk_int8.stateChanged.connect(
            lambda state: self.int8_group.setEnabled(state == Qt.CheckState.Checked.value)
        )
        
        # 按钮
        btn_layout = QHBoxLayout()
        self.btn_ok = QPushButton("确定")
        self.btn_ok.clicked.connect(self.accept)
        self.btn_cancel = QPushButton("取消")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_ok)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
    
    def get_config(self) -> dict:
        """获取配置"""
        config = {
            'imgsz': self.spin_imgsz.value(),
            'half': self.chk_half.isChecked(),
            'dynamic': self.chk_dynamic.isChecked(),
            'simplify': self.chk_simplify.isChecked(),
            'workspace': self.spin_workspace.value(),
            'int8': self.chk_int8.isChecked(),
            'nms': self.chk_nms.isChecked(),
            'batch': self.spin_batch.value(),
            'device': self.combo_device.currentText()
        }
        
        # 如果启用NMS，添加NMS参数
        if self.chk_nms.isChecked():
            config['conf'] = self.spin_conf.value()
            config['iou'] = self.spin_iou.value()
            config['agnostic_nms'] = self.chk_agnostic_nms.isChecked()
        
        # 如果启用INT8，添加校准参数
        if self.chk_int8.isChecked():
            config['data'] = self.edit_data.text() or None
            config['fraction'] = self.spin_fraction.value()
        
        return config


class ResultPage(QWidget):
    """训练结果页面"""
    
    def __init__(self):
        super().__init__()
        self.current_project_id = None
        self.current_images = []
        self.current_image_idx = -1
        self.runs_dir = "runs"
        self.available_runs: List[Dict] = []
        self.current_run_dir: Optional[str] = None
        self.current_run_name: str = ""
        self._export_thread: Optional[ModelExportThread] = None
        
        self.init_ui()
        # 初始时不自动扫描，等待设置项目
    
    def set_project(self, project_id: int):
        """设置当前项目"""
        self.current_project_id = project_id
        if project_id:
            print(f"[ResultPage] 已切换到项目: {project_id}")
            self.scan_runs_directory()
        else:
            print("[ResultPage] 项目已取消选择")
            # 清空显示
            self.available_runs = []
            self.current_run_dir = None
            self.current_run_name = ""
            with QSignalBlocker(self.run_selector):
                self.run_selector.clear()
                self.run_selector.setEnabled(False)
            self.image_list.clear()
            self.current_images = []
            self.image_label.setText("请先选择一个项目")
            self._reset_metrics()
    
    def init_ui(self):
        """初始化界面"""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)
        
        # 标题
        title = QLabel("训练结果")
        title.setObjectName("title")
        title.setStyleSheet("font-size: 16px; font-weight: bold; margin-bottom: 8px;")
        main_layout.addWidget(title)
        
        # 主分割器
        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter = splitter
        splitter.setChildrenCollapsible(False)
        
        # 左侧：图像列表
        left_panel = self.create_image_list_panel()
        left_panel.setMinimumWidth(220)
        splitter.addWidget(left_panel)
        
        # 右侧：图像显示
        right_panel = self.create_image_display_panel()
        right_panel.setMinimumWidth(400)
        splitter.addWidget(right_panel)
        
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([300, 900])
        main_layout.addWidget(splitter, 1)
        
        # 底部：指标和导出
        bottom_panel = self.create_metrics_export_panel()
        main_layout.addWidget(bottom_panel)
    
    def create_image_list_panel(self) -> QWidget:
        """创建图像列表面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        list_title = QLabel("结果图像")
        list_title.setStyleSheet("font-weight: bold; margin-bottom: 4px;")
        layout.addWidget(list_title)

        # 历史训练实验选择栏
        run_label = QLabel("训练实验:")
        run_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        layout.addWidget(run_label)

        run_row = QHBoxLayout()
        run_row.setSpacing(6)
        self.run_selector = QComboBox()
        self.run_selector.setObjectName("run_selector")
        self.run_selector.setEnabled(False)
        self.run_selector.setToolTip("选择要查看或导出的历史训练实验（按修改时间由新到旧排序）")
        self.run_selector.currentIndexChanged.connect(self.on_run_selected)
        run_row.addWidget(self.run_selector, 1)

        # 刷新按钮
        self.btn_refresh_runs = QPushButton("🔄 刷新")
        self.btn_refresh_runs.clicked.connect(self.scan_runs_directory)
        run_row.addWidget(self.btn_refresh_runs)
        layout.addLayout(run_row)
        
        # 图像列表
        self.image_list = QListWidget()
        self.image_list.setObjectName("image_list")
        
        # 简单样式
        self.image_list.setStyleSheet('''
            QListWidget {
                background-color: #252526;
                border: 1px solid #3e3e42;
                border-radius: 6px;
                padding: 4px;
            }
            QListWidget::item {
                padding: 8px;
                border-radius: 4px;
            }
            QListWidget::item:selected {
                background-color: #007ACC;
                color: white;
            }
        ''')
        
        self.image_list.itemClicked.connect(self.on_image_selected)
        layout.addWidget(self.image_list)
        
        return panel
    
    def create_image_display_panel(self) -> QWidget:
        """创建图像显示面板"""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 标题
        display_title = QLabel("图像预览")
        display_title.setStyleSheet("font-weight: bold; margin-bottom: 8px;")
        layout.addWidget(display_title)
        
        # 图像容器
        self.image_container = QWidget()
        self.image_container.setObjectName("image_container")
        
        self.image_container.setStyleSheet(f"""
            QWidget#image_container {{
                background-color: {COLORS['sidebar']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)
        self.image_layout = QVBoxLayout(self.image_container)
        self.image_layout.setContentsMargins(12, 12, 12, 12)
        self.image_label = ImagePreview("选择左侧图像查看预览")
        self.image_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        self.image_layout.addWidget(self.image_label, 1)
        layout.addWidget(self.image_container, 1)

        return panel
    
    def create_metrics_export_panel(self) -> QWidget:
        """创建指标和导出面板"""
        panel = QGroupBox("训练指标与导出")
        
        panel.setStyleSheet('''
            QGroupBox {
                border: 1px solid #3e3e42;
                border-radius: 6px;
                padding: 10px;
                margin-top: 10px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
        ''')
        
        layout = QVBoxLayout(panel)
        
        # 指标网格
        metrics_grid = QGridLayout()
        metrics_grid.setSpacing(10)
        
        # 指标标签
        metrics = ['mAP50', 'mAP50_95', '精确率', '召回率']
        self.metric_labels = {}
        
        for i, metric in enumerate(metrics):
            label = QLabel(metric + ':')
            value_label = QLabel('--')
            value_label.setObjectName(f"metric_{metric.lower()}")
            value_label.setStyleSheet("font-weight: bold;")
            self.metric_labels[metric] = value_label
            
            metrics_grid.addWidget(label, i, 0)
            metrics_grid.addWidget(value_label, i, 1)
        
        layout.addLayout(metrics_grid)
        
        # 导出按钮
        export_layout = FlowLayout(spacing=10)
        
        self.btn_export_onnx = QPushButton("导出为ONNX")
        self.btn_export_onnx.clicked.connect(lambda: self.export_model('onnx'))
        export_layout.addWidget(self.btn_export_onnx)
        
        self.btn_export_tensorrt = QPushButton("导出为TensorRT")
        self.btn_export_tensorrt.clicked.connect(lambda: self.export_model('tensorrt'))
        export_layout.addWidget(self.btn_export_tensorrt)
        
        self.btn_export_torchscript = QPushButton("导出为TorchScript")
        self.btn_export_torchscript.clicked.connect(lambda: self.export_model('torchscript'))
        export_layout.addWidget(self.btn_export_torchscript)
        
        self.btn_export_pt = QPushButton("导出为pt")
        self.btn_export_pt.clicked.connect(self.export_model_pt)
        export_layout.addWidget(self.btn_export_pt)
        
        # 导出结果文件夹按钮（使用不同颜色）
        self.btn_export_folder = QPushButton("导出结果文件夹")
        self.btn_export_folder.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['primary']};
                color: white;
                font-weight: bold;
            }}
        """)
        self.btn_export_folder.clicked.connect(self.export_result_folder)
        export_layout.addWidget(self.btn_export_folder)
        
        layout.addLayout(export_layout)

        # 后台导出进度反馈条（默认隐藏）
        status_row = QHBoxLayout()
        status_row.setSpacing(8)
        self.export_progress_bar = QProgressBar()
        self.export_progress_bar.setRange(0, 0)
        self.export_progress_bar.setMaximumHeight(14)
        self.export_progress_bar.setMaximumWidth(180)
        self.export_progress_bar.setVisible(False)
        status_row.addWidget(self.export_progress_bar)

        self.export_status_label = QLabel("")
        self.export_status_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 12px;")
        self.export_status_label.setVisible(False)
        status_row.addWidget(self.export_status_label, 1)
        layout.addLayout(status_row)
        
        return panel

    def _reset_metrics(self):
        """将指标显示重置为占位符"""
        for label in self.metric_labels.values():
            label.setText("--")

    def _set_export_busy(self, busy: bool, message: str = ""):
        """切换后台导出状态，禁用导出按钮并展示进度提示"""
        export_buttons = (
            self.btn_export_onnx,
            self.btn_export_tensorrt,
            self.btn_export_torchscript,
            self.btn_export_pt,
            self.btn_export_folder,
        )
        for btn in export_buttons:
            btn.setEnabled(not busy)
        self.export_progress_bar.setVisible(busy)
        self.export_status_label.setVisible(busy and bool(message))
        self.export_status_label.setText(message if busy else "")

    def _get_selected_run(self) -> Optional[Dict]:
        """获取当前下拉框选中的训练实验信息"""
        idx = self.run_selector.currentIndex()
        if 0 <= idx < len(self.available_runs):
            return self.available_runs[idx]
        return None

    def scan_runs_directory(self, show_dialogs: bool = True):
        """扫描 runs 目录并填充历史训练实验下拉列表"""
        self.image_list.clear()
        self.current_images = []
        self._reset_metrics()

        runs_dir = self.runs_dir
        if not os.path.exists(runs_dir):
            self.available_runs = []
            self.current_run_dir = None
            self.current_run_name = ""
            with QSignalBlocker(self.run_selector):
                self.run_selector.clear()
                self.run_selector.setEnabled(False)
            self.image_label.setText("runs 目录不存在")
            if show_dialogs:
                QMessageBox.warning(self, "提示", "runs目录不存在")
            return

        previous_rel = self.current_run_name
        runs, is_fallback_all = discover_run_directories(runs_dir, self.current_project_id)

        self.available_runs = runs
        with QSignalBlocker(self.run_selector):
            self.run_selector.clear()
            for run in runs:
                self.run_selector.addItem(run['display_name'], run['abs_path'])
            self.run_selector.setEnabled(bool(runs))

        if not runs:
            self.current_run_dir = None
            self.current_run_name = ""
            self.image_label.setText("未找到训练结果")
            if show_dialogs:
                QMessageBox.warning(self, "提示", "没有找到训练项目")
            return

        if is_fallback_all and show_dialogs:
            QMessageBox.information(
                self,
                "提示",
                f"未找到与项目 {self.current_project_id} 对应的训练结果，显示所有可用结果",
            )

        # 尽量保持之前选中的实验；否则默认选择最新（索引 0）
        target_index = 0
        if previous_rel:
            for idx, run in enumerate(runs):
                if run['rel_path'] == previous_rel:
                    target_index = idx
                    break

        with QSignalBlocker(self.run_selector):
            self.run_selector.setCurrentIndex(target_index)
        self.on_run_selected(target_index)

    def on_run_selected(self, index: int):
        """切换历史训练实验并加载对应结果图像与指标"""
        self.image_list.clear()
        self.current_images = []
        self._reset_metrics()

        if index < 0 or index >= len(self.available_runs):
            self.current_run_dir = None
            self.current_run_name = ""
            self.image_label.setText("请选择左侧训练实验")
            return

        run_info = self.available_runs[index]
        project_dir = run_info['abs_path']
        run_name = run_info['rel_path']
        self.current_run_dir = project_dir
        self.current_run_name = run_name
        self.run_selector.setToolTip(f"{run_name}\n路径: {project_dir}")

        # 找到所有图像文件并排序
        image_extensions = ('.png', '.jpg', '.jpeg')
        image_files = []
        for root, _, files in os.walk(project_dir):
            for file in files:
                if file.lower().endswith(image_extensions):
                    image_files.append((file, os.path.join(root, file)))
        image_files.sort(key=lambda pair: pair[0].lower())

        self.current_images = image_files
        for filename, filepath in image_files:
            item = QListWidgetItem(filename)
            item.setData(Qt.ItemDataRole.UserRole, filepath)
            self.image_list.addItem(item)

        if image_files:
            self.image_list.setCurrentRow(0)
            self.display_image(image_files[0][1])
        else:
            self.image_label.setText("当前实验暂无结果图像")

        # 更新标题并读取指标
        self.setWindowTitle(f"训练结果 - 项目 {self.current_project_id} - {run_name}")
        self.read_training_metrics(project_dir)
    
    def on_image_selected(self, item):
        """选择图像"""
        filepath = item.data(Qt.ItemDataRole.UserRole)
        if filepath and os.path.exists(filepath):
            self.display_image(filepath)
    
    def display_image(self, filepath):
        """显示图像（兼容中文与特殊字符路径）"""
        try:
            pixmap = load_pixmap_unicode_safe(filepath)
            if pixmap.isNull():
                self.image_label.setText("无法加载图像")
                return
            self.image_label.setPixmap(pixmap)
        except Exception as e:
            self.image_label.setText(f"加载失败: {str(e)}")

    def _ensure_run_selected_for_export(self) -> Optional[Dict]:
        """校验当前项目、导出线程状态及选中的训练实验"""
        if self._export_thread is not None and self._export_thread.isRunning():
            QMessageBox.information(self, "提示", "已有导出任务正在后台运行，请等待完成")
            return None

        if not self.current_project_id:
            QMessageBox.warning(self, "提示", "请先选择一个项目")
            return None

        run_info = self._get_selected_run()
        if run_info is None:
            self.scan_runs_directory(show_dialogs=False)
            run_info = self._get_selected_run()

        if run_info is None or not os.path.exists(run_info['abs_path']):
            QMessageBox.warning(self, "提示", "没有找到可导出的训练实验")
            return None

        return run_info

    def _start_export_thread(self, worker: ModelExportThread, initial_msg: str):
        """启动后台导出线程并绑定状态回调"""
        self._export_thread = worker
        self._set_export_busy(True, initial_msg)
        worker.progress_message.connect(
            lambda msg: self._set_export_busy(True, msg)
        )
        worker.export_finished.connect(self._on_export_finished)
        worker.start()

    def _on_export_finished(self, success: bool, message: str):
        """后台导出完成回调"""
        self._set_export_busy(False)
        if self._export_thread is not None:
            self._export_thread.deleteLater()
            self._export_thread = None

        if success:
            QMessageBox.information(self, "成功", message)
        else:
            QMessageBox.warning(self, "导出失败", message)
    
    def export_model(self, format_type):
        """后台异步导出模型（ONNX / TensorRT / TorchScript）"""
        run_info = self._ensure_run_selected_for_export()
        if not run_info:
            return

        project_dir = run_info['abs_path']
        best_model = os.path.join(project_dir, "weights", "best.pt")
        if not os.path.exists(best_model):
            QMessageBox.warning(self, "提示", "未找到best.pt模型文件")
            return

        # 显示导出配置对话框
        export_config = {}
        if format_type == 'onnx':
            dialog = ONNXExportDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            export_config = dialog.get_config()
        elif format_type == 'tensorrt':
            dialog = TensorRTExportDialog(self)
            if dialog.exec() != QDialog.DialogCode.Accepted:
                return
            export_config = dialog.get_config()

        # 选择导出路径
        file_filter = ""
        if format_type == 'onnx':
            file_filter = "ONNX files (*.onnx)"
        elif format_type == 'tensorrt':
            file_filter = "TensorRT files (*.engine)"
        elif format_type == 'torchscript':
            file_filter = "TorchScript files (*.pt)"

        default_name = f"best.{format_type}"
        if format_type == 'tensorrt':
            default_name = "best.engine"

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            f"导出为{format_type.upper()}",
            os.path.join(project_dir, default_name),
            file_filter,
        )
        if not save_path:
            return

        worker = ModelExportThread(
            export_mode='ultralytics',
            project_dir=project_dir,
            save_path=save_path,
            format_type=format_type,
            export_config=export_config,
            run_name=run_info['rel_path'],
            parent=self,
        )
        self._start_export_thread(worker, f"正在后台导出 {format_type.upper()}...")
    
    def export_model_pt(self):
        """后台导出为 pt 文件"""
        run_info = self._ensure_run_selected_for_export()
        if not run_info:
            return

        project_dir = run_info['abs_path']
        best_model = os.path.join(project_dir, "weights", "best.pt")
        if not os.path.exists(best_model):
            QMessageBox.warning(self, "提示", "未找到best.pt模型文件")
            return

        save_path, _ = QFileDialog.getSaveFileName(
            self,
            "导出为pt",
            os.path.join(project_dir, "best.pt"),
            "PyTorch files (*.pt)",
        )
        if not save_path:
            return

        worker = ModelExportThread(
            export_mode='pt',
            project_dir=project_dir,
            save_path=save_path,
            run_name=run_info['rel_path'],
            parent=self,
        )
        self._start_export_thread(worker, "正在导出 best.pt...")
    
    def export_result_folder(self):
        """后台导出结果文件夹"""
        run_info = self._ensure_run_selected_for_export()
        if not run_info:
            return

        project_dir = run_info['abs_path']
        save_dir = QFileDialog.getExistingDirectory(
            self,
            "选择导出目标文件夹",
            os.path.dirname(project_dir),
        )
        if not save_dir:
            return

        worker = ModelExportThread(
            export_mode='folder',
            project_dir=project_dir,
            save_path=save_dir,
            run_name=run_info['rel_path'].replace(os.sep, "_"),
            parent=self,
        )
        self._start_export_thread(worker, "正在导出结果文件夹...")
    
    def update_metrics(self, metrics: Dict):
        """更新指标"""
        for metric, value in metrics.items():
            if metric in self.metric_labels:
                self.metric_labels[metric].setText(f"{value:.4f}")
    
    def read_training_metrics(self, project_dir):
        """从 results.csv 读取训练指标（自动去除列名首尾空白）"""
        results_csv = os.path.join(project_dir, "results.csv")
        
        if not os.path.exists(results_csv):
            return
        
        try:
            import csv
            
            with open(results_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                if not rows:
                    return
                
                # 读取最后一行数据，并清洗 Ultralytics CSV 列名中的多余空格
                last_row = {
                    (k.strip() if isinstance(k, str) else k): (
                        v.strip() if isinstance(v, str) else v
                    )
                    for k, v in rows[-1].items()
                }
                
                # 提取指标
                precision = float(last_row.get('metrics/precision(B)', 0) or 0)
                recall = float(last_row.get('metrics/recall(B)', 0) or 0)
                map50 = float(last_row.get('metrics/mAP50(B)', 0) or 0)
                map50_95 = float(last_row.get('metrics/mAP50-95(B)', 0) or 0)
                
                # 更新指标显示
                if 'mAP50' in self.metric_labels:
                    self.metric_labels['mAP50'].setText(f"{map50:.4f}")
                if 'mAP50_95' in self.metric_labels:
                    self.metric_labels['mAP50_95'].setText(f"{map50_95:.4f}")
                if '精确率' in self.metric_labels:
                    self.metric_labels['精确率'].setText(f"{precision:.4f}")
                if '召回率' in self.metric_labels:
                    self.metric_labels['召回率'].setText(f"{recall:.4f}")
                    
        except Exception as e:
            print(f"读取指标失败: {e}")


if __name__ == "__main__":
    from PyQt6.QtWidgets import QApplication
    import sys
    
    app = QApplication(sys.argv)
    window = ResultPage()
    window.show()
    sys.exit(app.exec())
