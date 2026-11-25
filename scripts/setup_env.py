#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Better-YOLO 一键环境配置与健康体检脚本（仅依赖 Python 标准库）。

用法示例:
    python scripts/setup_env.py --check                  # 仅体检当前环境
    python scripts/setup_env.py --install                # 自动检测 GPU/CPU 并安装全部依赖
    python scripts/setup_env.py --install --cpu          # 强制安装 CPU 版依赖
    python scripts/setup_env.py --install --mirror tuna  # 使用清华 PyPI 镜像加速
    python scripts/setup_env.py --install --dry-run      # 仅预览安装命令，不实际执行
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import importlib
import importlib.metadata as importlib_metadata
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Optional, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REQUIREMENTS_PATH = PROJECT_ROOT / "requirements.txt"

PYPI_MIRRORS: Dict[str, str] = {
    "tuna": "https://pypi.tuna.tsinghua.edu.cn/simple",
    "aliyun": "https://mirrors.aliyun.com/pypi/simple/",
    "ustc": "https://pypi.mirrors.ustc.edu.cn/simple/",
}

CUDA_INDEX_URLS: Dict[str, str] = {
    "cu128": "https://download.pytorch.org/whl/cu128",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu118": "https://download.pytorch.org/whl/cu118",
    "cpu": "https://download.pytorch.org/whl/cpu",
}

CORE_MODULES: List[Tuple[str, str, bool]] = [
    # (import_name, display_name, required)
    ("PyQt6", "PyQt6 (桌面界面)", True),
    ("matplotlib", "Matplotlib (训练曲线)", True),
    ("numpy", "NumPy (数值计算)", True),
    ("cv2", "OpenCV (图像/视频处理)", True),
    ("PIL", "Pillow (图像编解码)", True),
    ("torch", "PyTorch (深度学习框架)", True),
    ("torchvision", "TorchVision", True),
    ("ultralytics", "Ultralytics (YOLO / SAM)", True),
    ("openai", "OpenAI SDK (Qwen-VL AI标注)", True),
    ("onnx", "ONNX (模型导出)", False),
    ("onnxslim", "ONNXSlim (ONNX 简化)", False),
    ("onnxruntime", "ONNX Runtime (ONNX 推理)", False),
    ("tensorrt", "TensorRT (可选加速)", False),
]


@dataclass
class CheckItem:
    name: str
    status: str  # "ok" | "warn" | "error"
    detail: str
    required: bool = True


def detect_nvidia_gpu() -> Dict[str, Optional[str]]:
    """通过 nvidia-smi 检测本机 NVIDIA 显卡与驱动支持的 CUDA 版本。"""
    result: Dict[str, Optional[str]] = {
        "has_gpu": False,
        "gpu_name": None,
        "cuda_version": None,
    }
    if not shutil.which("nvidia-smi"):
        return result

    try:
        proc = subprocess.run(
            ["nvidia-smi"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if proc.returncode != 0:
            return result
        output = proc.stdout or ""
        cuda_match = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", output)
        if cuda_match:
            result["cuda_version"] = cuda_match.group(1)

        name_proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
        if name_proc.returncode == 0 and name_proc.stdout.strip():
            result["gpu_name"] = name_proc.stdout.strip().splitlines()[0].strip()
            result["has_gpu"] = True
        elif cuda_match:
            result["has_gpu"] = True
    except Exception:
        pass

    return result


def recommend_cuda_tag(cuda_version: Optional[str]) -> str:
    """根据驱动 CUDA 版本号推荐匹配的 PyTorch wheel 标签。"""
    if not cuda_version:
        return "cpu"
    try:
        parts = [int(p) for p in cuda_version.strip().split(".")[:2]]
        major, minor = parts[0], (parts[1] if len(parts) > 1 else 0)
    except ValueError:
        return "cu124"

    if (major, minor) >= (12, 8):
        return "cu128"
    if (major, minor) >= (12, 6):
        return "cu126"
    if (major, minor) >= (12, 4):
        return "cu124"
    if (major, minor) >= (12, 1):
        return "cu121"
    if (major, minor) >= (11, 8):
        return "cu118"
    return "cpu"


def parse_requirements(requirements_path: Path = REQUIREMENTS_PATH) -> List[str]:
    """解析 requirements.txt，过滤注释和空行。"""
    if not requirements_path.exists():
        return []
    pkgs: List[str] = []
    for raw_line in requirements_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        pkgs.append(line)
    return pkgs


def resolve_mirror_url(mirror: Optional[str]) -> Optional[str]:
    if not mirror:
        return None
    key = mirror.strip().lower()
    return PYPI_MIRRORS.get(key, mirror.strip())


def build_install_commands(
    python_exe: str,
    device: str = "auto",
    cuda_tag: Optional[str] = None,
    mirror: Optional[str] = None,
    requirements_path: Path = REQUIREMENTS_PATH,
    gpu_info: Optional[Dict[str, Optional[str]]] = None,
) -> List[List[str]]:
    """构建有序的 pip 安装命令列表（自动处理 PyTorch CUDA/CPU 与 ONNXRuntime 变体）。"""
    gpu = gpu_info if gpu_info is not None else detect_nvidia_gpu()
    if device == "cpu":
        use_gpu = False
        chosen_tag = "cpu"
    elif device == "gpu":
        use_gpu = True
        chosen_tag = cuda_tag or recommend_cuda_tag(gpu.get("cuda_version"))
        if chosen_tag == "cpu":
            chosen_tag = "cu128"
    else:
        use_gpu = bool(gpu.get("has_gpu"))
        chosen_tag = (
            (cuda_tag or recommend_cuda_tag(gpu.get("cuda_version")))
            if use_gpu
            else "cpu"
        )
        if chosen_tag == "cpu":
            use_gpu = False

    reqs = parse_requirements(requirements_path)
    torch_pkgs: List[str] = []
    other_pkgs: List[str] = []

    for spec in reqs:
        pkg_name = re.split(r"[=<>!~\[]", spec, maxsplit=1)[0].strip().lower()
        if pkg_name in ("torch", "torchvision"):
            torch_pkgs.append(spec)
        elif pkg_name == "onnxruntime-gpu":
            other_pkgs.append(spec if use_gpu else "onnxruntime==1.23.2")
        elif pkg_name == "onnxruntime":
            other_pkgs.append("onnxruntime-gpu==1.22.0" if use_gpu else spec)
        else:
            other_pkgs.append(spec)

    if not torch_pkgs:
        torch_pkgs = ["torch", "torchvision"]

    commands: List[List[str]] = []
    mirror_url = resolve_mirror_url(mirror)

    # 1. 安装 PyTorch + TorchVision（按 CUDA/CPU 选择官方 wheel 源）
    torch_cmd = [python_exe, "-m", "pip", "install", *torch_pkgs]
    index_url = CUDA_INDEX_URLS.get(chosen_tag)
    if index_url:
        torch_cmd.extend(["--index-url", index_url])
    commands.append(torch_cmd)

    # 2. 安装其余项目依赖
    if other_pkgs:
        other_cmd = [python_exe, "-m", "pip", "install", *other_pkgs]
        if mirror_url:
            other_cmd.extend(["-i", mirror_url])
        commands.append(other_cmd)

    return commands


def ensure_workspace_dirs(root: Path = PROJECT_ROOT) -> List[Path]:
    """创建运行所需的基础目录（不会覆盖已有数据）。"""
    created: List[Path] = []
    for rel in ("data", "pretrained", "projects", "outputs", "config"):
        target = root / rel
        if not target.exists():
            target.mkdir(parents=True, exist_ok=True)
            created.append(target)
    return created


def _installed_dist_version(dist_name: str) -> Optional[str]:
    try:
        return importlib_metadata.version(dist_name)
    except importlib_metadata.PackageNotFoundError:
        return None


def run_environment_checks(root: Path = PROJECT_ROOT) -> List[CheckItem]:
    """执行完整环境体检，返回结构化检查项列表。"""
    items: List[CheckItem] = []

    # 1. Python 版本
    py_ver = sys.version_info
    py_str = f"{py_ver.major}.{py_ver.minor}.{py_ver.micro} ({sys.executable})"
    if (py_ver.major, py_ver.minor) >= (3, 10):
        items.append(CheckItem("Python 版本", "ok", py_str, required=True))
    else:
        items.append(
            CheckItem(
                "Python 版本",
                "error",
                f"{py_str}，要求 Python >= 3.10",
                required=True,
            )
        )

    # 2. 虚拟环境检查
    in_venv = (
        bool(os.environ.get("CONDA_DEFAULT_ENV"))
        or sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    )
    env_name = os.environ.get("CONDA_DEFAULT_ENV") or (
        Path(sys.prefix).name if in_venv else "系统全局 Python"
    )
    items.append(
        CheckItem(
            "运行环境",
            "ok" if in_venv else "warn",
            f"{env_name}" + ("" if in_venv else "（建议使用 conda 或 venv 虚拟环境）"),
            required=False,
        )
    )

    # 3. 硬件 GPU 检测
    gpu = detect_nvidia_gpu()
    if gpu["has_gpu"]:
        items.append(
            CheckItem(
                "NVIDIA GPU",
                "ok",
                f"{gpu['gpu_name'] or 'Detected'} (驱动 CUDA {gpu['cuda_version'] or '未知'})",
                required=False,
            )
        )
    else:
        items.append(
            CheckItem(
                "NVIDIA GPU",
                "warn",
                "未检测到 nvidia-smi（将使用 CPU 模式，标注与轻量推理仍可正常运行）",
                required=False,
            )
        )

    # 4. Python 依赖检查
    torch_mod = None
    for mod_name, display, required in CORE_MODULES:
        try:
            mod = importlib.import_module(mod_name)
            ver = getattr(mod, "__version__", None) or _installed_dist_version(mod_name) or "已安装"
            items.append(CheckItem(display, "ok", str(ver), required=required))
            if mod_name == "torch":
                torch_mod = mod
        except Exception:
            items.append(
                CheckItem(
                    display,
                    "error" if required else "warn",
                    "未安装" if required else "未安装（可选）",
                    required=required,
                )
            )

    # 5. PyTorch CUDA 可用性匹配检查
    if torch_mod is not None:
        cuda_ok = bool(torch_mod.cuda.is_available())
        if cuda_ok:
            dev_name = torch_mod.cuda.get_device_name(0)
            torch_cuda = getattr(torch_mod.version, "cuda", "未知")
            items.append(
                CheckItem(
                    "PyTorch CUDA",
                    "ok",
                    f"可用 — {dev_name} (torch CUDA {torch_cuda})",
                    required=False,
                )
            )
        elif gpu["has_gpu"]:
            tag = recommend_cuda_tag(gpu.get("cuda_version"))
            items.append(
                CheckItem(
                    "PyTorch CUDA",
                    "warn",
                    f"检测到 NVIDIA GPU 但当前 PyTorch 为 CPU 版；建议运行: python scripts/setup_env.py --install --cuda {tag}",
                    required=False,
                )
            )
        else:
            items.append(
                CheckItem("PyTorch CUDA", "ok", "CPU 模式运行", required=False)
            )

    # 6. ONNX Runtime 冲突检测
    ort_cpu = _installed_dist_version("onnxruntime")
    ort_gpu = _installed_dist_version("onnxruntime-gpu")
    if ort_cpu and ort_gpu:
        items.append(
            CheckItem(
                "ONNXRuntime 冲突",
                "warn",
                f"同时安装了 onnxruntime({ort_cpu}) 与 onnxruntime-gpu({ort_gpu})，建议仅保留其一",
                required=False,
            )
        )

    # 7. AI 标注 API Key 配置提示
    llm_cfg = root / "config" / "llm_config.json"
    has_key = bool(os.environ.get("DASHSCOPE_API_KEY")) or llm_cfg.exists()
    items.append(
        CheckItem(
            "AI 标注配置",
            "ok" if has_key else "warn",
            "已检测到配置或环境变量"
            if has_key
            else "尚未配置 API Key（启动后可在「AI 标注」页填写）",
            required=False,
        )
    )

    return items


def print_report(items: List[CheckItem]) -> int:
    icons = {"ok": "✅", "warn": "⚠️ ", "error": "❌"}
    print("\n================ Better-YOLO 环境体检报告 ================")
    for item in items:
        print(f"  {icons.get(item.status, '•')} {item.name:<24s} : {item.detail}")
    print("==========================================================")

    errors = [i for i in items if i.status == "error" and i.required]
    warns = [i for i in items if i.status == "warn"]
    if errors:
        print(f"结果: 缺少 {len(errors)} 项必需依赖。可执行一键安装:")
        print("  python scripts/setup_env.py --install")
        return 1
    if warns:
        print(f"结果: 核心依赖齐全（另有 {len(warns)} 项提示），可直接运行: python main.py")
    else:
        print("结果: 环境完全就绪，可直接运行: python main.py")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Better-YOLO 一键环境配置与体检工具")
    parser.add_argument("--check", action="store_true", help="仅检查当前环境状态")
    parser.add_argument("--install", action="store_true", help="一键安装项目依赖")
    parser.add_argument(
        "--device",
        choices=("auto", "gpu", "cpu"),
        default="auto",
        help="安装设备模式（默认 auto 自动检测 NVIDIA GPU）",
    )
    parser.add_argument("--cpu", action="store_true", help="--device cpu 的简写")
    parser.add_argument(
        "--cuda",
        choices=tuple(k for k in CUDA_INDEX_URLS if k != "cpu"),
        default=None,
        help="指定 PyTorch CUDA wheel 版本（如 cu128 / cu124）",
    )
    parser.add_argument(
        "--mirror",
        default=None,
        help="PyPI 镜像源（支持快捷名: tuna / aliyun / ustc，或完整 URL）",
    )
    parser.add_argument("--dry-run", action="store_true", help="仅打印安装命令，不实际执行")
    parser.add_argument("--json", action="store_true", help="以 JSON 格式输出体检报告")

    args = parser.parse_args(argv)
    device = "cpu" if args.cpu else args.device

    if args.install:
        created = ensure_workspace_dirs()
        if created:
            print("已初始化运行目录:", ", ".join(p.name for p in created))

        commands = build_install_commands(
            python_exe=sys.executable,
            device=device,
            cuda_tag=args.cuda,
            mirror=args.mirror,
        )
        print("将执行以下安装步骤:")
        for idx, cmd in enumerate(commands, start=1):
            print(f"  [{idx}/{len(commands)}] {' '.join(cmd)}")

        if args.dry_run:
            return 0

        for idx, cmd in enumerate(commands, start=1):
            print(f"\n>>> 执行步骤 [{idx}/{len(commands)}]...")
            proc = subprocess.run(cmd, check=False)
            if proc.returncode != 0:
                print(f"❌ 步骤 {idx} 失败（退出码 {proc.returncode}）", file=sys.stderr)
                return proc.returncode

        print("\n✅ 依赖安装完成，开始自动体检...")

    items = run_environment_checks()
    if args.json:
        print(json.dumps([asdict(i) for i in items], ensure_ascii=False, indent=2))
        return 1 if any(i.status == "error" and i.required for i in items) else 0
    return print_report(items)


if __name__ == "__main__":
    raise SystemExit(main())
