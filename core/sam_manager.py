# -*- coding: utf-8 -*-
"""SAM / SAM2 / SAM3 / FastSAM 模型缓存管理器与后台推理线程。"""

import gc
import os
import threading

from PyQt6.QtCore import QThread, pyqtSignal

from core.model_manager import get_pretrained_dir

SAM3_DOWNLOAD_URL = "https://huggingface.co/1038lab/sam3/discussions/1"


class SAMModelManager:
    """SAM模型缓存管理器：同配置复用，切换配置自动释放重载。"""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._model = None
        self._model_key = None

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_model(self, sam_type: str, model_path: str, device: str):
        """获取可复用模型；配置变化时释放旧模型并重载。"""
        model_key = (sam_type, model_path, str(device))
        with self._lock:
            if self._model is not None and self._model_key == model_key:
                return self._model

            self._release_model_locked()

            if sam_type == "FastSAM":
                from ultralytics import FastSAM
                self._model = FastSAM(model_path)
            else:
                # SAM/SAM2/MobileSAM/SAM3 均使用 SAM 类
                from ultralytics import SAM
                self._model = SAM(model_path)

            self._model_key = model_key
            return self._model

    def release_model(self):
        """主动释放当前缓存模型。"""
        with self._lock:
            self._release_model_locked()

    def _release_model_locked(self):
        if self._model is not None:
            try:
                del self._model
            except Exception:
                pass
            self._model = None
            self._model_key = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class SAMMemoryPredictorManager:
    """SAM2/SAM3 记忆预测器管理：按配置复用，切换配置释放。"""

    _instance = None
    _instance_lock = threading.Lock()

    def __init__(self):
        self._lock = threading.Lock()
        self._predictor = None
        self._predictor_key = None

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def get_predictor(self, sam_config: dict):
        sam_type = sam_config.get("sam_type", "SAM")
        model_file = sam_config.get("model_file", "sam2_t.pt")
        device = sam_config.get("device", "cpu")
        imgsz = int(sam_config.get("imgsz", 1024))
        # 动态记忆分割对阈值更敏感，使用较低conf避免把有效mask过滤空
        conf = 0.01
        model_key = (sam_type, model_file, str(device), imgsz)

        with self._lock:
            if self._predictor is not None and self._predictor_key == model_key:
                return self._predictor

            self._release_locked()

            from ultralytics.models.sam import SAM2DynamicInteractivePredictor
            resolved_model = model_file
            candidate_pretrained = get_pretrained_dir() / model_file
            if not os.path.exists(resolved_model) and candidate_pretrained.exists():
                resolved_model = str(candidate_pretrained)
            overrides = dict(
                conf=conf,
                task="segment",
                mode="predict",
                imgsz=imgsz,
                model=resolved_model,
                save=False,
                device=device,
                verbose=False,
            )
            self._predictor = SAM2DynamicInteractivePredictor(overrides=overrides, max_obj_num=20)
            self._predictor_key = model_key
            return self._predictor

    def clear(self):
        with self._lock:
            self._release_locked()

    def _release_locked(self):
        if self._predictor is not None:
            try:
                del self._predictor
            except Exception:
                pass
            self._predictor = None
            self._predictor_key = None
            gc.collect()
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass


class SAMInferenceWorker(QThread):
    """SAM推理工作线程"""

    inference_finished = pyqtSignal(bool, str, object)  # 成功, 消息, mask

    def __init__(self, sam_config: dict, image_path: str, points: list = None, bboxes: list = None):
        super().__init__()
        self.sam_config = sam_config
        self.image_path = image_path
        self.points = points or []
        self.bboxes = bboxes or []

    def _build_predict_kwargs(self) -> dict:
        """构建统一推理参数，确保UI配置生效。"""
        kwargs = {"verbose": False}
        if self.sam_config.get("imgsz"):
            kwargs["imgsz"] = int(self.sam_config["imgsz"])
        if self.sam_config.get("device") is not None:
            kwargs["device"] = self.sam_config["device"]
        if self.sam_config.get("retina_masks") is not None:
            kwargs["retina_masks"] = bool(self.sam_config["retina_masks"])
        return kwargs

    def run(self):
        """运行SAM推理"""
        try:
            sam_type = self.sam_config.get("sam_type", "SAM")
            model_file = self.sam_config.get("model_file", "sam_b.pt")
            device = self.sam_config.get("device", "cpu")

            # 检查模型文件是否存在（含偏好设置中的预训练模型路径）
            model_paths = [
                model_file,
                str(get_pretrained_dir() / model_file),
                os.path.join("models", model_file),
                os.path.join(os.path.expanduser("~"), ".cache", "ultralytics", model_file),
            ]

            model_path = None
            for path in model_paths:
                if os.path.exists(path):
                    model_path = path
                    break

            if model_path is None:
                if sam_type == "SAM3":
                    self.inference_finished.emit(
                        False,
                        "SAM3模型未找到: sam3.pt\n"
                        "SAM3不支持自动下载。\n"
                        "请到以下页面下载 sam3.pt 后放到项目根目录：\n"
                        f"{SAM3_DOWNLOAD_URL}",
                        None,
                    )
                else:
                    self.inference_finished.emit(
                        False, f"模型文件不存在: {model_file}\n请下载模型后重试", None
                    )
                return

            # 模型缓存复用：同配置仅加载一次，切换配置时自动释放重载
            model = SAMModelManager.instance().get_model(sam_type, model_path, device)

            predict_kwargs = self._build_predict_kwargs()
            if sam_type == "FastSAM":
                predict_kwargs["conf"] = float(self.sam_config.get("conf", 0.4))
                predict_kwargs["iou"] = float(self.sam_config.get("iou", 0.9))

            # 准备提示
            if self.points:
                points_array = [[p[0], p[1]] for p in self.points]
                labels = [1] * len(points_array)  # 1表示前景
                results = model(self.image_path, points=points_array, labels=labels, **predict_kwargs)
            elif self.bboxes:
                bbox = self.bboxes[-1]
                results = model(self.image_path, bboxes=[bbox], **predict_kwargs)
            else:
                self.inference_finished.emit(False, "没有提供提示", None)
                return

            # 提取mask
            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, "masks") and result.masks is not None:
                    masks = (
                        result.masks.data.cpu().numpy()
                        if hasattr(result.masks.data, "cpu")
                        else result.masks.data
                    )
                    scores = None
                    if (
                        hasattr(result, "boxes")
                        and result.boxes is not None
                        and hasattr(result.boxes, "conf")
                    ):
                        confs = result.boxes.conf
                        if confs is not None:
                            scores = confs.cpu().numpy() if hasattr(confs, "cpu") else confs
                    self.inference_finished.emit(True, "推理成功", {"masks": masks, "scores": scores})
                else:
                    self.inference_finished.emit(False, "未检测到分割结果", None)
            else:
                self.inference_finished.emit(False, "推理无结果", None)

        except Exception as e:
            sam_type = self.sam_config.get("sam_type", "SAM")
            if sam_type == "SAM3":
                self.inference_finished.emit(
                    False,
                    "SAM3加载/推理失败。\n"
                    "请确认 sam3.pt 已放在项目根目录且文件可用。\n"
                    f"下载说明: {SAM3_DOWNLOAD_URL}\n"
                    f"错误详情: {str(e)}",
                    None,
                )
            else:
                self.inference_finished.emit(False, f"推理出错: {str(e)}", None)
