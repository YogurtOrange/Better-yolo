# -*- coding: utf-8 -*-
"""
AI标注页面 - Qwen-VL 自动标注
移植自 ai_yolo_annotator_gui.py 的GUI逻辑，适配 Better-YOLO 项目体系。

功能:
  - 任务描述提示词(内置类别表) + 负样本排除
  - few-shot 示例图片
  - 模型选择(带在线搜索)
  - 多轮检测取平均 / 二次放大精修
  - 试标数量 + 断点续标
  - AI生成/迭代任务提示词
  - 结果直接写入项目数据库，同步显示在标注页面
"""

import os
import re
import io
import json
import threading
from pathlib import Path

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFormLayout, QProgressBar, QTextEdit, QLineEdit,
    QFileDialog, QMessageBox, QCheckBox, QSpinBox, QListWidget,
    QListWidgetItem,
    QAbstractItemView, QDialog, QSplitter, QScrollArea,
    QTableWidget, QTableWidgetItem, QHeaderView
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QPixmap

from gui.styles import COLORS, format_log_html
from models.database import db
from core.qwen_annotator import (
    StopRequested,
    run_ai_annotation,
    generate_task_prompt,
    fetch_available_models,
    clear_breakpoint,
    parse_category_list,
    export_manual_examples,
    register_classes,
    DEFAULT_MODEL,
    DEFAULT_DETECT_ROUNDS,
    DEFAULT_ZOOM_REFINE,
    DEFAULT_ZOOM_ITERS,
    DEFAULT_UPLOAD_MEGAPIXELS,
    DEFAULT_MAX_RETRY,
    DEFAULT_MAX_WORKERS,
    IMAGE_EXTENSIONS,
)

# LLM配置文件(与自动打标签弹窗共用api_key)
LLM_CONFIG_FILE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
    'config', 'llm_config.json'
)


def load_llm_api_config():
    """读取config/llm_config.json中的api_key与base_url"""
    config = {}
    if os.path.exists(LLM_CONFIG_FILE):
        try:
            with open(LLM_CONFIG_FILE, 'r', encoding='utf-8') as f:
                config = json.load(f) or {}
        except Exception:
            config = {}
    return config


# ==================== 后台线程 ====================

class AiAnnotateWorker(QThread):
    """AI标注后台线程"""
    log_signal = pyqtSignal(str)
    progress_signal = pyqtSignal(int, int)
    finished_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.stop_event = threading.Event()

    def run(self):
        try:
            run_ai_annotation(
                self.params,
                log=self.log_signal.emit,
                stop_event=self.stop_event,
                progress=lambda d, t: self.progress_signal.emit(d, t)
            )
            self.finished_signal.emit()
        except StopRequested:
            self.log_signal.emit("已停止")
            self.finished_signal.emit()
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self.stop_event.set()


class PromptGenWorker(QThread):
    """AI生成提示词后台线程"""
    log_signal = pyqtSignal(str)
    result_signal = pyqtSignal(str, object)
    error_signal = pyqtSignal(str)

    def __init__(self, params):
        super().__init__()
        self.params = params
        self.stop_event = threading.Event()

    def run(self):
        try:
            prompt, negative = generate_task_prompt(
                self.params["model"],
                self.params["sample_images"],
                self.params["example_images"],
                self.params["previous_prompt"],
                self.params["feedback"],
                self.params["megapixels"],
                self.params["max_retry"],
                log=self.log_signal.emit,
                stop_event=self.stop_event,
                api_key=self.params.get("api_key"),
                base_url=self.params.get("base_url"),
                extra_note=self.params.get("extra_note", ""),
                comparison_pairs=self.params.get("comparison_pairs"),
            )
            self.result_signal.emit(prompt, negative)
        except StopRequested:
            self.log_signal.emit("已停止")
        except Exception as e:
            self.error_signal.emit(str(e))

    def stop(self):
        self.stop_event.set()


class ModelFetchWorker(QThread):
    """后台拉取模型列表线程"""
    fetched = pyqtSignal(list)
    error = pyqtSignal(str)

    def __init__(self, api_key=None, base_url=None):
        super().__init__()
        self.api_key = api_key
        self.base_url = base_url

    def run(self):
        try:
            ids = fetch_available_models(
                api_key=self.api_key, base_url=self.base_url
            )
            self.fetched.emit(ids)
        except Exception as e:
            self.error.emit(str(e))


# ==================== 对话框 ====================

class ModelSearchDialog(QDialog):
    """可用模型搜索对话框"""

    model_selected = pyqtSignal(str)

    def __init__(self, parent=None, api_key=None, base_url=None):
        super().__init__(parent)

        self.setWindowTitle("搜索可用模型")
        self.resize(480, 520)

        self.all_models = []
        self.worker = None
        self.api_key = api_key
        self.base_url = base_url

        layout = QVBoxLayout(self)

        tip = QLabel(
            "默认显示视觉模型(名称含vl)，输入关键词过滤全部模型；"
            "双击条目选择"
        )
        tip.setWordWrap(True)
        layout.addWidget(tip)

        # 搜索框: 默认过滤"vl"，清空则显示全部
        self.search_edit = QLineEdit("vl")
        self.search_edit.setPlaceholderText(
            "输入关键词过滤，例如 qwen3-vl / max / flash"
        )
        self.search_edit.textChanged.connect(self.apply_filter)
        layout.addWidget(self.search_edit)

        self.list_widget = QListWidget()
        self.list_widget.setAlternatingRowColors(True)
        self.list_widget.itemDoubleClicked.connect(self.pick)
        layout.addWidget(self.list_widget, 1)

        status_row = QHBoxLayout()
        self.status_label = QLabel("正在拉取模型列表...")
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self.start_fetch)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.refresh_btn)
        layout.addLayout(status_row)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("选择")
        ok_btn.clicked.connect(self.pick_current)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

        self.start_fetch()

    def start_fetch(self):
        """后台拉取账号可用模型列表"""
        if self.worker is not None and self.worker.isRunning():
            return

        self.list_widget.clear()
        self.status_label.setText("正在拉取模型列表...")
        self.refresh_btn.setEnabled(False)

        self.worker = ModelFetchWorker(self.api_key, self.base_url)
        self.worker.fetched.connect(self.on_fetched)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_fetched(self, ids):
        self.all_models = ids
        self.refresh_btn.setEnabled(True)
        self.apply_filter(self.search_edit.text())

    def on_error(self, msg):
        self.refresh_btn.setEnabled(True)
        self.status_label.setText(f"拉取失败: {msg}")
        QMessageBox.warning(self, "错误", f"获取模型列表失败:\n{msg}")

    def apply_filter(self, text):
        """按关键词过滤（空=全部）"""
        self.list_widget.clear()

        key = (text or "").strip().lower()

        for mid in self.all_models:
            if not key or key in mid.lower():
                self.list_widget.addItem(mid)

        n = self.list_widget.count()
        total = len(self.all_models)
        self.status_label.setText(
            f"{n}/{total} 个模型"
            + ("" if n else "（无匹配，可清空关键词）")
        )

    def pick(self, item):
        self.model_selected.emit(item.text())
        self.accept()

    def pick_current(self):
        item = self.list_widget.currentItem()
        if item is None:
            QMessageBox.information(self, "提示", "请先选择一个模型")
            return
        self.pick(item)


class PromptGenDialog(QDialog):
    """AI生成提示词对话框

    流程:
      1. 先手动标注少量示例图(添加到示例图片列表)
      2. AI观察采样图+示例图 -> 生成任务描述
      3. 用生成的描述试标，在"标注"页面查看效果，
         若框不符合预期，写明哪里不对 -> AI修正提示词
    """

    def __init__(self, parent, sample_count, locked_images=None):
        super().__init__(parent)

        self.setWindowTitle("AI生成任务提示词")
        self.resize(560, 560)

        layout = QVBoxLayout(self)

        info = QLabel(
            f"AI将观察: 项目采样图 {sample_count} 张、"
            f"示例图片列表中的参考示例"
            + "\n生成的提示词将填入任务描述框，可再人工修改。"
        )
        info.setWordWrap(True)
        layout.addWidget(info)

        # 已确认(锁定)图片作为反馈参考(默认全选)
        # 元素: (img_id, filename, modified) modified=人工修正过
        locked_images = locked_images or []
        self.locked_list = None

        if locked_images:
            n_modified = sum(1 for it in locked_images if len(it) > 2 and it[2])
            tip = QLabel(
                f"检测到 {len(locked_images)} 张已人工确认(锁定)的图片，"
                "可作为标准答案重新喂给AI学习(推荐):"
                + (
                    f"\n图例: ✏️=人工修正过({n_modified}张，将自动附带"
                    "\"AI原始结果(红框) vs 修正结果(绿框)\"对比图组)  "
                    "🔒=已确认"
                    if n_modified else ""
                )
            )
            tip.setWordWrap(True)
            layout.addWidget(tip)

            self.locked_box = QCheckBox("全部不使用")
            self.locked_box.setTristate(True)
            layout.addWidget(self.locked_box)

            locked_list = QListWidget()
            locked_list.setMaximumHeight(150)
            for img in locked_images:
                img_id, filename = img[0], img[1]
                modified = len(img) > 2 and bool(img[2])
                badge = "✏️🔒 " if modified else "🔒 "
                item = QListWidgetItem(f"{badge}{filename}")
                item.setFlags(
                    item.flags() | Qt.ItemFlag.ItemIsUserCheckable
                )
                item.setCheckState(Qt.CheckState.Checked)
                item.setData(Qt.ItemDataRole.UserRole, img_id)
                locked_list.addItem(item)
            layout.addWidget(locked_list)

            self.locked_list = locked_list

            def _on_master_toggle(state):
                # 仅响应全选/全不选，半选(部分勾选)不动作
                if state not in (0, 2):
                    return
                checked = (
                    Qt.CheckState.Checked if state == 2
                    else Qt.CheckState.Unchecked
                )
                for i in range(locked_list.count()):
                    locked_list.item(i).setCheckState(checked)

            self.locked_box.stateChanged.connect(_on_master_toggle)

            def _sync_master():
                states = [
                    locked_list.item(i).checkState()
                    for i in range(locked_list.count())
                ]
                n_on = states.count(Qt.CheckState.Checked)
                if n_on == 0:
                    self.locked_box.setCheckState(Qt.CheckState.Unchecked)
                elif n_on == len(states):
                    self.locked_box.setCheckState(Qt.CheckState.Checked)
                else:
                    self.locked_box.setCheckState(
                        Qt.CheckState.PartiallyChecked
                    )

            self._sync_master = _sync_master
            locked_list.itemChanged.connect(lambda _item: _sync_master())
        else:
            hint = QLabel(
                "提示: 暂无锁定图片。若AI大规模标错，可先在标注页"
                "修正少量图片(修正后自动锁定)，再用它们重新生成提示词"
            )
            hint.setWordWrap(True)
            hint.setStyleSheet("color: #8a8a8a;")
            layout.addWidget(hint)

        layout.addWidget(QLabel("上一轮结果反馈(首次生成可留空):"))
        self.feedback_edit = QTextEdit()
        self.feedback_edit.setPlaceholderText(
            "例如: 框比目标大了一圈，把靶板也框进去了；"
            "漏掉了右下角的目标..."
        )
        self.feedback_edit.setMaximumHeight(140)
        layout.addWidget(self.feedback_edit)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        cancel_btn = QPushButton("取消")
        cancel_btn.clicked.connect(self.reject)
        ok_btn = QPushButton("开始生成")
        ok_btn.clicked.connect(self.accept)
        btn_row.addWidget(cancel_btn)
        btn_row.addWidget(ok_btn)
        layout.addLayout(btn_row)

    def get_feedback(self):
        return self.feedback_edit.toPlainText().strip()

    def get_selected_locked_ids(self):
        """返回勾选的锁定图片id列表(无锁定图或未勾选则空)"""
        if self.locked_list is None:
            return []
        return [
            self.locked_list.item(i).data(Qt.ItemDataRole.UserRole)
            for i in range(self.locked_list.count())
            if self.locked_list.item(i).checkState()
            == Qt.CheckState.Checked
        ]


class ModifiedImagesDialog(QDialog):
    """修正记录查看器: 左侧列出人工修改过的图片，
    右侧上下并排预览 AI原始结果(红框) vs 修正后结果(绿框)"""

    def __init__(self, parent, project_id):
        super().__init__(parent)
        self.setWindowTitle("修正记录 (人工修改过的图片)")
        self.resize(980, 640)

        self._preview_cache = {}

        layout = QHBoxLayout(self)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧: 修正过的图片列表
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(QLabel("✏️ 人工修改过的图片:"))
        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_select_image)
        left_layout.addWidget(self.image_list, 1)
        splitter.addWidget(left)

        # 右侧: 前后对比预览
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.before_label = QLabel()
        self.before_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.after_label = QLabel()
        self.after_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label = QLabel("")
        self.hint_label.setWordWrap(True)

        right_layout.addWidget(QLabel("AI上一轮原始结果 (红框):"))
        right_layout.addWidget(self.before_label, 1)
        right_layout.addWidget(QLabel("用户人工修正后 (绿框):"))
        right_layout.addWidget(self.after_label, 1)
        right_layout.addWidget(self.hint_label)
        splitter.addWidget(right)

        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 5)
        layout.addWidget(splitter)

        self._load_images(project_id)

    # -------------------- 数据与渲染 --------------------

    def _load_images(self, project_id):
        images = [
            i for i in db.get_project_images(project_id)
            if i.get('modified')
        ]
        for img in images:
            has_snap = bool(img.get('ai_original'))
            badge = "✏️ " if has_snap else "✏️(无AI快照) "
            item = QListWidgetItem(f"{badge}{img['filename']}")
            item.setData(Qt.ItemDataRole.UserRole, img['id'])
            item.setData(
                Qt.ItemDataRole.UserRole + 1, img['storage_path']
            )
            self.image_list.addItem(item)

        if not images:
            self.hint_label.setText(
                "还没有人工修改过的图片。\n"
                "在\"标注\"页面对AI标注结果进行增/删/改框后，"
                "这里就会出现记录。"
            )
        else:
            self.image_list.setCurrentRow(0)

    def _render_pixmap(self, storage_path, bboxes, color):
        """渲染带框图并转为QPixmap(带缓存，避免重复画)"""
        key = (storage_path, tuple(bboxes), color)
        if key not in self._preview_cache:
            from core.qwen_annotator import _render_boxes
            pil_img = _render_boxes(storage_path, bboxes, color)
            buf = io.BytesIO()
            pil_img.save(buf, "JPEG", quality=88)
            pix = QPixmap()
            pix.loadFromData(buf.getvalue())
            # 限制宽度，保持纵横比
            if pix.width() > 560:
                pix = pix.scaledToWidth(
                    560, Qt.TransformationMode.SmoothTransformation
                )
            self._preview_cache[key] = pix
        return self._preview_cache[key]

    def on_select_image(self, cur, _prev):
        self.before_label.clear()
        self.after_label.clear()
        self.hint_label.clear()
        if cur is None:
            return

        img_id = cur.data(Qt.ItemDataRole.UserRole)
        storage_path = cur.data(Qt.ItemDataRole.UserRole + 1)

        from core.qwen_annotator import (
            _get_ai_snapshot_bboxes, _get_current_bboxes,
        )

        before = _get_ai_snapshot_bboxes(img_id)
        after = _get_current_bboxes(img_id)

        RED = (220, 40, 40)
        GREEN = (40, 180, 60)

        try:
            if before:
                self.before_label.setPixmap(
                    self._render_pixmap(storage_path, before, RED)
                )
                self.before_label.setText("")
            else:
                self.before_label.setText("(无AI快照: 该图未经过AI标注，"
                                          "或早于快照功能锁定)")
            if after:
                self.after_label.setPixmap(
                    self._render_pixmap(storage_path, after, GREEN)
                )
                self.after_label.setText("")
            else:
                self.after_label.setText("(该图当前没有标注)")
        except Exception as e:
            self.hint_label.setText(f"预览失败: {e}")
            return

        if before and after:
            self.hint_label.setText(
                f"共 {len(before)} 个AI原始框 → {len(after)} 个修正后框"
            )
        elif before:
            self.hint_label.setText("用户清空了全部标注")
        elif after:
            self.hint_label.setText(
                f"用户手动标注了 {len(after)} 个框(无AI原始结果可对比)"
            )


# ==================== 主页面 ====================

class AiLabelPage(QWidget):
    """AI标注页面"""

    # 类别表变动并已同步到项目DB时发出(标注页监听刷新类别列表)
    classes_changed = pyqtSignal()

    def __init__(self):
        super().__init__()

        self.current_project_id = None
        self.current_project = None
        self.worker = None
        self.prompt_worker = None

        # 任务配置持久化(随项目保存)
        self._loading_config = False
        self._config_timer = QTimer(self)
        self._config_timer.setSingleShot(True)
        self._config_timer.setInterval(800)
        self._config_timer.timeout.connect(self.save_task_config)

        self.init_ui()

        # 表格/附加规则/负样本编辑后自动保存(防抖)
        self.category_table.itemChanged.connect(self._on_config_changed)
        self.te_extra_rules.textChanged.connect(self._on_config_changed)
        self.le_negative.textChanged.connect(self._on_config_changed)

    # -------------------- UI --------------------

    def init_ui(self):
        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        # 左侧配置面板(可滚动)
        self.left_scroll = QScrollArea()
        left_scroll = self.left_scroll
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_scroll.setWidget(left_panel)

        left_layout.addWidget(self.create_project_group())
        left_layout.addWidget(self.create_api_group())
        left_layout.addWidget(self.create_prompt_group())
        left_layout.addWidget(self.create_example_group())
        left_layout.addWidget(self.create_params_group())
        left_layout.addWidget(self.create_run_group())
        left_layout.addStretch()

        # 右侧日志
        right_panel = self.create_log_panel()

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_panel)
        # 左右比例约 6:4 (窗口缩放时保持该比例)
        splitter.setStretchFactor(0, 6)
        splitter.setStretchFactor(1, 4)
        left_scroll.setMinimumWidth(620)
        splitter.setSizes([1160, 760])

        main_layout.addWidget(splitter)

    def _group_style(self):
        return f"""
            QGroupBox {{
                font-weight: bold;
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 10px;
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }}
        """

    def create_project_group(self):
        group = QGroupBox("当前项目")
        group.setStyleSheet(self._group_style())

        layout = QVBoxLayout(group)

        self.project_label = QLabel("未选择项目 (请先在导入页面选择)")
        self.project_label.setWordWrap(True)
        layout.addWidget(self.project_label)

        self.project_stats_label = QLabel("")
        self.project_stats_label.setWordWrap(True)
        layout.addWidget(self.project_stats_label)

        # 修正记录查看器入口
        self.btn_modified = QPushButton("✏️ 修正记录")
        self.btn_modified.setToolTip(
            "查看所有人工修改过标注的图片，"
            "对比AI原始结果(红框)与修正后结果(绿框)"
        )
        self.btn_modified.setEnabled(False)
        self.btn_modified.clicked.connect(self.on_show_modified)
        layout.addWidget(self.btn_modified)

        return group

    def create_api_group(self):
        group = QGroupBox("API配置")
        group.setStyleSheet(self._group_style())

        layout = QFormLayout(group)

        # API Key
        self.le_api_key = QLineEdit()
        self.le_api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.le_api_key.setPlaceholderText(
            "留空则使用环境变量 DASHSCOPE_API_KEY"
        )
        layout.addRow("API Key:", self.le_api_key)

        # Base URL
        self.le_base_url = QLineEdit()
        layout.addRow("Base URL:", self.le_base_url)

        # 模型名
        model_row = QHBoxLayout()
        self.le_model = QLineEdit(DEFAULT_MODEL)
        btn_search_model = QPushButton("搜索")
        btn_search_model.clicked.connect(self.on_search_model)
        model_row.addWidget(self.le_model, 1)
        model_row.addWidget(btn_search_model)
        layout.addRow("模型:", model_row)

        # 加载已保存的配置
        config = load_llm_api_config()
        if config.get('api_key'):
            self.le_api_key.setText(config.get('api_key', ''))
        self.le_base_url.setText(
            config.get('base_url', '') or
            "https://dashscope.aliyuncs.com/compatible-mode/v1"
        )
        if config.get('model_name', '').startswith('qwen'):
            self.le_model.setText(config['model_name'])

        return group

    def create_prompt_group(self):
        group = QGroupBox("标注类别 (每行一个类别)")
        group.setStyleSheet(self._group_style())

        layout = QVBoxLayout(group)

        # 类别表格: 序号(仅排序) | 标签名(classes.txt中的类别名) | 描述
        self.category_table = QTableWidget(0, 3)
        self.category_table.setHorizontalHeaderLabels(["序号", "标签名", "描述"])
        self.category_table.horizontalHeaderItem(0).setToolTip(
            "仅用于排序，不写入标注；类别名以标签名列为准"
        )
        self.category_table.horizontalHeaderItem(1).setToolTip(
            "写入classes.txt的实际类别名"
        )
        self.category_table.verticalHeader().setVisible(False)
        self.category_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        header = self.category_table.horizontalHeader()
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.category_table.setColumnWidth(0, 70)
        self.category_table.setColumnWidth(1, 150)
        self.category_table.verticalHeader().setDefaultSectionSize(42)
        self.category_table.setMinimumHeight(220)
        layout.addWidget(self.category_table)

        # 表格操作按钮(右对齐)
        table_btn_row = QHBoxLayout()
        table_btn_row.addStretch(1)
        btn_add_cat = QPushButton("添加类别")
        btn_add_cat.clicked.connect(self.on_add_category)
        btn_import_cls = QPushButton("导入classes.txt")
        btn_import_cls.setToolTip(
            "读取已有classes.txt，逐行作为标签名填入表格"
            "(描述列需自行补充)"
        )
        btn_import_cls.clicked.connect(self.on_import_classes)
        btn_del_cat = QPushButton("删除选中")
        btn_del_cat.clicked.connect(self.on_remove_category)
        table_btn_row.addWidget(btn_add_cat)
        table_btn_row.addWidget(btn_import_cls)
        table_btn_row.addWidget(btn_del_cat)
        layout.addLayout(table_btn_row)

        # 附加规则(非类别的补充要求，逐行一条) + AI生成提示词按钮同行
        extra_label_row = QHBoxLayout()
        extra_label_row.addWidget(
            QLabel("附加规则 (遮挡处理/排除背景/单框要求等，逐行一条):")
        )
        extra_label_row.addStretch(1)
        self.btn_gen_prompt = QPushButton("AI生成提示词")
        self.btn_gen_prompt.clicked.connect(self.on_generate_prompt)
        extra_label_row.addWidget(self.btn_gen_prompt)
        layout.addLayout(extra_label_row)

        self.te_extra_rules = QTextEdit()
        self.te_extra_rules.setPlaceholderText(
            "若目标被部分遮挡，仅框出可见部分的完整外轮廓。\n"
            "除以上类别外，白色墙面、地面、支架等均不是目标，不要标注。\n"
            "每个目标只输出一个框，不要重复标注。"
        )
        self.te_extra_rules.setMaximumHeight(100)
        layout.addWidget(self.te_extra_rules)

        # 负样本排除
        neg_row = QHBoxLayout()
        neg_row.addWidget(QLabel("负样本排除:"))
        self.le_negative = QLineEdit()
        self.le_negative.setPlaceholderText(
            "画面里存在但不要标的东西，顿号分隔，如: 靶板、地面、阴影"
        )
        neg_row.addWidget(self.le_negative, 1)
        layout.addLayout(neg_row)

        return group

    # -------------------- 类别表格 --------------------

    def on_add_category(self):
        """添加一行空类别，序号自动递增"""
        row = self.category_table.rowCount()
        self.category_table.insertRow(row)

        # 序号自动补齐: 已有序号最大值+1，三位数格式
        max_seq = 0
        for r in range(row):
            item = self.category_table.item(r, 0)
            if item and item.text().strip().isdigit():
                max_seq = max(max_seq, int(item.text().strip()))
        self.category_table.setItem(
            row, 0, QTableWidgetItem(str(max_seq + 1))
        )

        self.category_table.setFocus()
        self.category_table.setCurrentCell(row, 1)

    def on_remove_category(self):
        """删除选中行(按行选择，支持多选)"""
        rows = sorted(
            {idx.row() for idx in self.category_table.selectedIndexes()},
            reverse=True
        )
        if not rows:
            QMessageBox.information(self, "提示", "请先选中要删除的行")
            return
        for row in rows:
            self.category_table.removeRow(row)

    def on_import_classes(self):
        """导入已有classes.txt: 每行标签名填入表格(描述留空)"""
        path, _ = QFileDialog.getOpenFileName(
            self, "选择classes.txt", "",
            "classes.txt (*.txt);;所有文件 (*)"
        )
        if not path:
            return

        try:
            with open(path, encoding='utf-8-sig') as f:
                names = [ln.strip() for ln in f if ln.strip()]
        except OSError as e:
            QMessageBox.warning(self, "错误", f"读取失败: {e}")
            return

        if not names:
            QMessageBox.information(self, "提示", "文件为空")
            return

        # 已存在的标签名跳过，避免重复
        existing = {
            self.category_table.item(r, 1).text().strip()
            for r in range(self.category_table.rowCount())
            if self.category_table.item(r, 1)
        }

        added = 0
        for name in names:
            if name in existing:
                continue
            self.on_add_category()
            row = self.category_table.rowCount() - 1
            self.category_table.setItem(
                row, 1, QTableWidgetItem(name)
            )
            existing.add(name)
            added += 1

        QMessageBox.information(
            self, "导入完成",
            f"新增 {added} 个类别，跳过重复 {len(names) - added} 个\n"
            f"描述列为空，请逐行补充描述"
        )

    def on_generate_manual_examples(self):
        """把手动标注过的图片导出为带框示例图，加入示例列表"""
        if not self.current_project:
            QMessageBox.warning(self, "提示", "请先选择项目")
            return

        paths = export_manual_examples(
            self.current_project['id'],
            log=self.append_log
        )

        if not paths:
            QMessageBox.information(
                self, "提示",
                "项目中没有已标注的图片。\n"
                "请先在\"标注\"页面手动标注几张图片再生成示例"
            )
            return

        # 加入示例列表(去重)
        existing = {
            self.example_list.item(i).text()
            for i in range(self.example_list.count())
        }
        added = 0
        for p in paths:
            if p not in existing:
                self.example_list.addItem(p)
                added += 1

        self.append_log(
            f"已从手动标注生成 {len(paths)} 张示例图"
            f"(新增{added}张)，可点击\"AI生成提示词\"让AI参考"
        )

    def build_prompt_text(self):
        """把表格+附加规则组装成发给AI的任务描述文本

        序号仅用于排序(有数字序号的行按序号排，其余按行序)，
        类别行格式为 "标签名：描述"，标签名即classes.txt中的类别名。
        """
        rows_data = []

        for row in range(self.category_table.rowCount()):

            def _cell(col):
                item = self.category_table.item(row, col)
                return item.text().strip() if item else ""

            seq = _cell(0)
            name = _cell(1)
            desc = _cell(2)

            if not name or not desc:
                continue

            rows_data.append((seq, name, desc, row))

        # 按序号排序: 数字序号按数值排，无序号/非数字的按行序排在后面
        rows_data.sort(
            key=lambda t: (0, int(t[0]), t[3])
            if t[0].isdigit() else (1, 0, t[3])
        )

        lines = [f"{name}：{desc}" for _, name, desc, _ in rows_data]

        extra = self.te_extra_rules.toPlainText().strip()

        if extra:
            lines.append(extra)

        return "\n".join(lines).strip()

    def fill_table_from_prompt(self, prompt):
        """把AI生成的提示词文本回填到表格

        表格已有标签名时(导入的classes.txt或手动添加):
        标名列原样保留，只更新描述列——AI不允许改标签名；
        生成的描述按标签名匹配，匹配不上但数量一致时按行序对应。
        表格为空时: 类别行 -> 表格行(标签名+描述)，其余行 -> 附加规则
        """
        cats = parse_category_list(prompt)

        # 收集当前表格已有的标签名
        existing = []
        for row in range(self.category_table.rowCount()):
            item = self.category_table.item(row, 1)
            if item and item.text().strip():
                existing.append((row, item.text().strip()))

        if existing:
            # 已有标签: 只更新描述列
            cat_map = {name: desc for name, desc in cats}

            matched = 0
            for row, label in existing:
                if label in cat_map:
                    self.category_table.setItem(
                        row, 2, QTableWidgetItem(cat_map[label])
                    )
                    matched += 1

            # 标签名一个都没匹配上但数量一致 -> 按行序对应更新描述
            if matched == 0 and len(cats) == len(existing):
                for (row, _), (_, desc) in zip(existing, cats):
                    self.category_table.setItem(
                        row, 2, QTableWidgetItem(desc)
                    )
                matched = len(existing)

            if matched == 0:
                # 诊断: 帮助定位AI输出格式问题
                sample = next(iter(cat_map), None)
                self.append_log(
                    f"[警告] 生成内容未能匹配任何标签"
                    f"(AI返回{len(cats)}个类别, 示例标签名: {sample})，"
                    "描述未更新，可点AI生成提示词重试或手动填写"
                )
            else:
                self.append_log(
                    f"已更新 {matched}/{len(existing)} 个标签的描述"
                    "(标签名保持不变)"
                )
            return

        # 表格为空: 类别行进表格，其余行进附加规则
        self.category_table.setRowCount(0)
        extra_lines = []

        cat_names = {name for name, _ in cats}

        for raw in prompt.splitlines():
            line = raw.strip()
            if not line:
                continue

            # 判断是否类别行(与核心解析的分隔符一致)
            head = re.split(r"[：:=]", line, 1)[0].strip()
            if head in cat_names:
                continue  # 类别行，由cats统一回填
            extra_lines.append(line)

        # 按类别表声明顺序回填
        for idx, (name, desc) in enumerate(cats):
            row = self.category_table.rowCount()
            self.category_table.insertRow(row)
            self.category_table.setItem(
                row, 0, QTableWidgetItem(str(idx + 1))
            )
            self.category_table.setItem(
                row, 1, QTableWidgetItem(name)
            )
            self.category_table.setItem(
                row, 2, QTableWidgetItem(desc)
            )

        self.te_extra_rules.setPlainText("\n".join(extra_lines))

    def create_example_group(self):
        group = QGroupBox("示例图片 (few-shot，已画好框和标签的参考图)")
        group.setStyleSheet(self._group_style())

        layout = QVBoxLayout(group)

        self.example_list = QListWidget()
        self.example_list.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.example_list.setMaximumHeight(110)
        layout.addWidget(self.example_list)

        btn_row = QHBoxLayout()
        btn_row.addStretch(1)
        btn_examples = QPushButton("从手动标注生成")
        btn_examples.setToolTip(
            "把项目中手动标注过的图片(含你画的框)导出为示例图，"
            "供AI生成提示词时参考"
        )
        btn_examples.clicked.connect(self.on_generate_manual_examples)
        btn_add = QPushButton("添加图片")
        btn_add.clicked.connect(self.on_add_example)
        btn_remove = QPushButton("移除选中")
        btn_remove.clicked.connect(self.on_remove_example)
        btn_row.addWidget(btn_examples)
        btn_row.addWidget(btn_add)
        btn_row.addWidget(btn_remove)
        btn_row.addStretch(1)
        layout.addLayout(btn_row)

        return group

    def create_params_group(self):
        group = QGroupBox("标注参数")
        group.setStyleSheet(self._group_style())

        layout = QFormLayout(group)

        # 检测轮数
        self.sb_detect_rounds = QSpinBox()
        self.sb_detect_rounds.setRange(1, 10)
        self.sb_detect_rounds.setValue(DEFAULT_DETECT_ROUNDS)
        self.sb_detect_rounds.setToolTip(
            ">1时同一张图检测多次取平均，减少框抖动(消耗更多token)"
        )
        layout.addRow("检测轮数:", self.sb_detect_rounds)

        # 并发数
        self.sb_workers = QSpinBox()
        self.sb_workers.setRange(1, 50)
        self.sb_workers.setValue(DEFAULT_MAX_WORKERS)
        layout.addRow("并发请求数:", self.sb_workers)

        # 上传像素
        self.sb_megapixels = QSpinBox()
        self.sb_megapixels.setRange(50, 2000)
        self.sb_megapixels.setValue(DEFAULT_UPLOAD_MEGAPIXELS)
        self.sb_megapixels.setSuffix(" 万")
        layout.addRow("上传最大像素:", self.sb_megapixels)

        # 重试
        self.sb_retry = QSpinBox()
        self.sb_retry.setRange(1, 20)
        self.sb_retry.setValue(DEFAULT_MAX_RETRY)
        layout.addRow("API重试次数:", self.sb_retry)

        # 二次放大精修
        zoom_row = QHBoxLayout()
        self.chk_zoom = QCheckBox("二次放大精修")
        self.chk_zoom.setChecked(DEFAULT_ZOOM_REFINE)
        self.chk_zoom.setToolTip(
            "对每个检出目标裁剪放大后让AI重新定位，核心精度手段"
        )
        zoom_row.addWidget(self.chk_zoom)
        zoom_row.addWidget(QLabel("迭代轮数:"))
        self.sb_zoom_iters = QSpinBox()
        self.sb_zoom_iters.setRange(1, 5)
        self.sb_zoom_iters.setValue(DEFAULT_ZOOM_ITERS)
        zoom_row.addWidget(self.sb_zoom_iters)
        zoom_row.addStretch(1)
        layout.addRow(zoom_row)

        # 试标数量
        self.sb_limit = QSpinBox()
        self.sb_limit.setRange(0, 99999)
        self.sb_limit.setValue(0)
        self.sb_limit.setToolTip(
            ">0时本次只标注前N张未完成的图片，用于小批量验证效果；"
            "满意后改回0继续剩余图片(断点自动续标)"
        )
        layout.addRow("试标数量(0=全部):", self.sb_limit)

        # 覆盖已有标注
        self.chk_overwrite = QCheckBox("覆盖图片已有标注")
        self.chk_overwrite.setToolTip(
            "不勾选(推荐): 每次重标只替换上一轮AI生成的框，手动标注保留\n"
            "勾选: 清空图片上所有标注(含手动标注)再写入"
        )
        layout.addRow(self.chk_overwrite)

        return group

    def create_run_group(self):
        group = QGroupBox("运行")
        group.setStyleSheet(self._group_style())

        layout = QVBoxLayout(group)

        btn_row = QHBoxLayout()
        self.btn_start = QPushButton("开始标注")
        self.btn_start.clicked.connect(self.on_start)
        self.btn_stop = QPushButton("停止")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self.on_stop)
        self.btn_clear_bp = QPushButton("清除断点")
        self.btn_clear_bp.clicked.connect(self.on_clear_breakpoint)
        btn_row.addWidget(self.btn_start)
        btn_row.addWidget(self.btn_stop)
        btn_row.addWidget(self.btn_clear_bp)
        layout.addLayout(btn_row)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        layout.addWidget(self.progress_bar)

        return group

    def create_log_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)

        title = QLabel("运行日志")
        title.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(title)

        self.log_edit = QTextEdit()
        self.log_edit.setReadOnly(True)
        layout.addWidget(self.log_edit, 1)

        return panel

    # -------------------- 项目 --------------------

    def set_project(self, project_id: int):
        """设置当前项目(切换页面时由主窗口调用)"""
        if project_id == self.current_project_id:
            return

        # 切换前先把上一个项目的任务配置落盘
        if self.current_project_id is not None:
            self._config_timer.stop()
            self.save_task_config()

        self.current_project_id = project_id

        if project_id:
            self.current_project = db.get_project(project_id)
            if self.current_project:
                name = self.current_project.get('name', 'Unknown')
                self.project_label.setText(
                    f"项目: {name} (ID: {project_id})"
                )
                self.refresh_project_stats()
                self.load_task_config()
        else:
            self.current_project = None
            self.project_label.setText("未选择项目 (请先在导入页面选择)")
            self.project_stats_label.setText("")

    # -------------------- 任务配置持久化 --------------------

    def _task_config_path(self):
        if not self.current_project:
            return None
        storage = self.current_project.get('storage_path')
        if not storage:
            return None
        return Path(storage) / "ai_task_config.json"

    def _on_config_changed(self, *_):
        if not self._loading_config:
            self._config_timer.start()

    def save_task_config(self):
        """把类别表格/附加规则/负样本保存到项目目录"""
        path = self._task_config_path()
        if not path:
            return

        categories = []
        for row in range(self.category_table.rowCount()):

            def _cell(col):
                item = self.category_table.item(row, col)
                return item.text().strip() if item else ""

            categories.append({
                "seq": _cell(0),
                "name": _cell(1),
                "desc": _cell(2),
            })

        data = {
            "categories": categories,
            "extra_rules": self.te_extra_rules.toPlainText(),
            "negative": self.le_negative.text().strip(),
        }

        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8"
            )
        except OSError as e:
            self.append_log(f"[警告] 任务配置保存失败: {e}")

        # 表格中的标签名同步注册到项目类别表
        self._sync_classes_to_project()

    def _sync_classes_to_project(self):
        """把类别表格中的标签名注册进项目classes表

        新标签会立即出现在"标注"页面的类别列表中，可直接手动画框
        """
        if not self.current_project:
            return

        # 收集表格中的标签名(去空、去重、保序)
        names = []
        seen = set()
        for row in range(self.category_table.rowCount()):
            item = self.category_table.item(row, 1)
            if item:
                name = item.text().strip()
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)

        if not names:
            return

        try:
            before_project = db.get_project(self.current_project['id'])
            before = self._parse_project_classes(before_project)
            register_classes(self.current_project['id'], names)
            after = self._parse_project_classes(
                db.get_project(self.current_project['id'])
            )
        except Exception as e:
            self.append_log(f"[警告] 标签同步到项目失败: {e}")
            return

        added = after - before
        if added:
            self.append_log(
                f"已同步 {len(added)} 个新标签到项目: "
                f"{', '.join(sorted(added))}"
            )
            self.classes_changed.emit()

    @staticmethod
    def _parse_project_classes(project):
        """解析项目classes字段(JSON字符串或列表) -> {name}集合"""
        if not project:
            return set()
        classes = project.get('classes') or []
        if isinstance(classes, str):
            try:
                classes = json.loads(classes)
            except (ValueError, TypeError):
                classes = []
        return {c.get('name') for c in classes if isinstance(c, dict)}

    def load_task_config(self):
        """从项目目录恢复任务配置(无配置文件则清空)"""
        self._loading_config = True
        try:
            self.category_table.setRowCount(0)
            self.te_extra_rules.clear()
            self.le_negative.clear()

            path = self._task_config_path()
            if not (path and path.exists()):
                return

            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as e:
                self.append_log(f"[警告] 任务配置读取失败: {e}")
                return

            for cat in data.get("categories", []):
                row = self.category_table.rowCount()
                self.category_table.insertRow(row)
                self.category_table.setItem(
                    row, 0, QTableWidgetItem(cat.get("seq", ""))
                )
                self.category_table.setItem(
                    row, 1, QTableWidgetItem(cat.get("name", ""))
                )
                self.category_table.setItem(
                    row, 2, QTableWidgetItem(cat.get("desc", ""))
                )

            self.te_extra_rules.setPlainText(data.get("extra_rules", ""))
            self.le_negative.setText(data.get("negative", ""))
            self.append_log("已恢复本项目的标注任务配置")
        finally:
            self._loading_config = False

    def refresh_project_stats(self):
        """刷新项目图片统计"""
        if not self.current_project_id:
            self.btn_modified.setEnabled(False)
            return
        images = db.get_project_images(self.current_project_id)
        annotated = sum(1 for i in images if i.get('status') == 'annotated')
        modified = sum(1 for i in images if i.get('modified'))

        # 断点信息
        bp_count = 0
        if self.current_project:
            bp_file = Path(self.current_project.get('storage_path', '')) / 'ai_progress.json'
            if bp_file.exists():
                try:
                    with open(bp_file, 'r', encoding='utf-8') as f:
                        bp_count = len(json.load(f).get('completed', []))
                except Exception:
                    pass

        self.project_stats_label.setText(
            f"图片: {len(images)} 张 (已标注 {annotated} 张)"
            + (f" | 已人工修正 {modified} 张" if modified else "")
            + (f" | AI断点: {bp_count} 张已完成" if bp_count else "")
        )
        self.btn_modified.setEnabled(True)
        self.btn_modified.setText(
            f"✏️ 修正记录({modified})" if modified else "✏️ 修正记录"
        )

    def on_show_modified(self):
        """打开修正记录查看器"""
        if not self.current_project_id:
            QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")
            return
        dlg = ModifiedImagesDialog(self, self.current_project_id)
        dlg.exec()

    def _get_api_key(self):
        return self.le_api_key.text().strip() or os.getenv("DASHSCOPE_API_KEY", "")

    # -------------------- 示例图片 --------------------

    def on_add_example(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "选择示例图片(已画好框和标签)", "",
            "Images (*.jpg *.jpeg *.png *.webp)"
        )
        for p in paths:
            self.example_list.addItem(p)

    def on_remove_example(self):
        for item in self.example_list.selectedItems():
            self.example_list.takeItem(self.example_list.row(item))

    # -------------------- 模型搜索 --------------------

    def on_search_model(self):
        api_key = self._get_api_key()
        if not api_key:
            QMessageBox.warning(
                self, "提示", "请先填写API Key或设置 DASHSCOPE_API_KEY 环境变量"
            )
            return

        dlg = ModelSearchDialog(
            self, api_key=api_key,
            base_url=self.le_base_url.text().strip()
        )
        dlg.model_selected.connect(self.le_model.setText)
        dlg.exec()

    # -------------------- AI生成提示词 --------------------

    def on_generate_prompt(self):
        if not self.current_project_id:
            QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")
            return

        if not self._get_api_key():
            QMessageBox.warning(self, "提示", "请先填写API Key")
            return

        # 采样图: 项目图片均匀采样
        images = db.get_project_images(self.current_project_id)
        image_paths = [
            Path(i['storage_path']) for i in images
            if Path(i.get('storage_path') or '').suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not image_paths:
            QMessageBox.information(self, "提示", "项目中没有图片")
            return

        sample_count = min(4, len(image_paths))
        step = len(image_paths) / sample_count
        sample_images = [
            image_paths[int(i * step)] for i in range(sample_count)
        ]

        example_images = [
            Path(self.example_list.item(i).text())
            for i in range(self.example_list.count())
        ]

        dlg = PromptGenDialog(
            self, sample_count,
            locked_images=[
                (i['id'], i['filename'], i.get('modified'))
                for i in images if i.get('locked')
            ],
        )
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return

        # 已确认(锁定)图片导出为带框反馈参考图，喂给AI重新学习
        extra_note = ""
        locked_ids = dlg.get_selected_locked_ids()
        comparison_pairs = []
        if locked_ids:
            from core.qwen_annotator import export_locked_feedback
            fb_paths, fb_count = export_locked_feedback(
                self.current_project_id, only_ids=set(locked_ids),
                log=self.append_log,
            )
            if fb_paths:
                example_images = example_images + fb_paths
                extra_note = (
                    f"重要: 本次参考示例中有 {fb_count} 张来自用户"
                    "已人工确认(锁定)的标注，其中的框和标签都是正确"
                    "答案。请重点依据这些图片修正类别描述和框的松紧"
                    "标准，先逐一观察每个目标的外观特征再动笔。"
                )
                self.append_log(
                    f"已将 {fb_count} 张确认图片(含修正后的框)导出为"
                    "反馈参考，AI将据此重新学习"
                )

            # 人工修正过的图片: 额外生成"AI原始(红框) vs 修正后(绿框)"
            # 对比图组，让AI定位自己上一轮的错误模式
            from core.qwen_annotator import export_modified_comparison
            modified_set = {
                i['id'] for i in images
                if i.get('modified') and i.get('ai_original')
            }
            pair_ids = set(locked_ids) & modified_set
            if pair_ids:
                comparison_pairs = export_modified_comparison(
                    self.current_project_id, only_ids=pair_ids,
                    log=self.append_log,
                )
                if comparison_pairs:
                    self.append_log(
                        f"已生成 {len(comparison_pairs)} 组修正对比图"
                        "(AI原始红框 vs 修正后绿框)，AI将对比分析错误"
                        "模式并重新优化负样本排除"
                    )

        self.btn_gen_prompt.setEnabled(False)

        self.prompt_worker = PromptGenWorker({
            "model": self.le_model.text().strip() or DEFAULT_MODEL,
            "sample_images": sample_images,
            "example_images": example_images,
            "previous_prompt": self.build_prompt_text(),
            "feedback": dlg.get_feedback(),
            "megapixels": self.sb_megapixels.value(),
            "max_retry": self.sb_retry.value(),
            "api_key": self._get_api_key(),
            "base_url": self.le_base_url.text().strip(),
            "extra_note": extra_note,
            "comparison_pairs": comparison_pairs,
        })
        self.prompt_worker.log_signal.connect(self.append_log)
        self.prompt_worker.result_signal.connect(self.on_prompt_generated)
        self.prompt_worker.error_signal.connect(self.on_prompt_error)
        self.prompt_worker.finished.connect(
            lambda: self.btn_gen_prompt.setEnabled(True)
        )
        self.prompt_worker.start()

        self.append_log("开始生成任务提示词...")

    def on_prompt_generated(self, prompt, negative=None):
        old_negative = self.le_negative.text().strip()
        self.fill_table_from_prompt(prompt)
        # AI重新优化了负样本排除 -> 覆盖写回(空则保留原值)
        if negative:
            self.le_negative.setText(negative)
            if old_negative:
                self.append_log(
                    f"负样本排除已由AI更新:\n  旧值: {old_negative}\n"
                    f"  新值: {negative}"
                )
            else:
                self.append_log(f"负样本排除已由AI生成: {negative}")
        self.append_log("提示词已生成并填入类别表格，可人工微调")

    def on_prompt_error(self, msg):
        self.append_log(f"[错误] 生成提示词失败: {msg}")
        QMessageBox.critical(self, "错误", f"生成提示词失败:\n{msg}")

    # -------------------- 运行控制 --------------------

    def on_start(self):
        if not self.current_project_id:
            QMessageBox.information(self, "提示", "请先在导入页面选择一个项目")
            return

        if self.worker is not None and self.worker.isRunning():
            return

        prompt = self.build_prompt_text()
        example_images = [
            self.example_list.item(i).text()
            for i in range(self.example_list.count())
        ]

        if not prompt and not example_images:
            QMessageBox.warning(self, "提示", "请在类别表格中添加类别（含描述）或添加示例图片")
            return

        api_key = self._get_api_key()
        if not api_key:
            QMessageBox.warning(
                self, "提示",
                "请先填写API Key或设置 DASHSCOPE_API_KEY 环境变量"
            )
            return

        images = db.get_project_images(self.current_project_id)

        if not images:
            QMessageBox.information(self, "提示", "项目中没有图片，请先导入")
            return

        params = {
            "project_id": self.current_project_id,
            "images": images,
            "project_dir": self.current_project.get('storage_path', '.'),
            "prompt": prompt,
            "negative": self.le_negative.text().strip(),
            "example_images": example_images,
            "model": self.le_model.text().strip() or DEFAULT_MODEL,
            "api_key": api_key,
            "base_url": self.le_base_url.text().strip(),
            "detect_rounds": self.sb_detect_rounds.value(),
            "max_workers": self.sb_workers.value(),
            "upload_megapixels": self.sb_megapixels.value(),
            "max_retry": self.sb_retry.value(),
            "zoom_refine": self.chk_zoom.isChecked(),
            "zoom_iters": self.sb_zoom_iters.value(),
            "limit": self.sb_limit.value(),
            "overwrite": self.chk_overwrite.isChecked(),
        }

        self.worker = AiAnnotateWorker(params)
        self.worker.log_signal.connect(self.append_log)
        self.worker.progress_signal.connect(self.on_progress)
        self.worker.finished_signal.connect(self.on_finished)
        self.worker.error_signal.connect(self.on_error)
        self.worker.start()

        self.btn_start.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.progress_bar.setValue(0)

    def on_stop(self):
        if self.worker is not None:
            self.worker.stop()
            self.append_log("正在停止... (等待进行中的请求完成)")

    def on_clear_breakpoint(self):
        if not self.current_project:
            return

        reply = QMessageBox.question(
            self, "确认",
            "确定清除断点吗？清除断点后重新标注会覆盖之前的进度记录"
            "(已写入的标注框不会被删除)。",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        if reply != QMessageBox.StandardButton.Yes:
            return

        project_dir = self.current_project.get('storage_path', '.')
        if clear_breakpoint(project_dir):
            self.append_log("断点已清除")
        else:
            self.append_log("没有找到断点记录")
        self.refresh_project_stats()

    def on_progress(self, done, total):
        if total > 0:
            self.progress_bar.setMaximum(total)
            self.progress_bar.setValue(done)

    def on_finished(self):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.refresh_project_stats()

    def on_error(self, msg):
        self.btn_start.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.append_log(f"[错误] {msg}")
        QMessageBox.critical(self, "错误", f"标注失败:\n{msg}")

    # -------------------- 日志 --------------------

    def append_log(self, msg):
        self.log_edit.append(format_log_html(msg))

        # 同步滚动到底部
        sb = self.log_edit.verticalScrollBar()
        sb.setValue(sb.maximum())
