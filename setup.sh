#!/usr/bin/env bash
# Better-YOLO Linux / macOS 一键环境配置脚本
# 用法:
#   ./setup.sh                  # 创建/复用虚拟环境并自动安装依赖（自动检测 GPU/CPU）
#   ./setup.sh --mirror tuna    # 使用清华镜像源加速
#   ./setup.sh --cpu            # 强制安装 CPU 版本
#   ./setup.sh --check          # 仅做环境体检

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="${BETTER_YOLO_ENV:-better-yolo}"
PYTHON_BIN=""

# 若仅检查环境，优先使用当前激活的 Python
for arg in "$@"; do
    if [[ "$arg" == "--check" ]]; then
        exec "${PYTHON:-python3}" scripts/setup_env.py "$@"
    fi
done

if [[ -n "${CONDA_PREFIX:-}" ]] && command -v python >/dev/null 2>&1; then
    echo ">>> 检测到已激活 Conda 环境: ${CONDA_DEFAULT_ENV:-$CONDA_PREFIX}"
    PYTHON_BIN="$(command -v python)"
elif [[ -n "${VIRTUAL_ENV:-}" ]] && command -v python >/dev/null 2>&1; then
    echo ">>> 检测到已激活虚拟环境: $VIRTUAL_ENV"
    PYTHON_BIN="$(command -v python)"
elif command -v conda >/dev/null 2>&1; then
    echo ">>> 检测到 Conda，准备创建/复用环境: $ENV_NAME (Python 3.10)"
    if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
        conda create -y -n "$ENV_NAME" python=3.10
    fi
    exec conda run --no-capture-output -n "$ENV_NAME" python scripts/setup_env.py --install "$@"
else
    VENV_DIR="$SCRIPT_DIR/.venv"
    if [[ ! -d "$VENV_DIR" ]]; then
        echo ">>> 创建本地虚拟环境: $VENV_DIR"
        python3 -m venv "$VENV_DIR"
    fi
    PYTHON_BIN="$VENV_DIR/bin/python"
fi

echo ">>> 使用解释器: $PYTHON_BIN"
exec "$PYTHON_BIN" scripts/setup_env.py --install "$@"
