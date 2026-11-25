# Better-YOLO 新手一键安装与环境配置教程

本教程面向**零基础或首次部署**的用户，通过内置的一键配置脚本，在 **Windows / Linux / macOS** 上快速完成 Better-YOLO 的运行环境搭建，避免手动折腾 PyTorch、CUDA 和依赖冲突。

---

## 一、安装前准备（只需 2 分钟检查）

在运行脚本前，先确认电脑具备以下最基础的条件：

1. **Python 3.10 及以上**（或已安装 **Miniconda / Anaconda**）
   - 推荐安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)（最省心，脚本会自动帮你创建干净的 `Python 3.10` 虚拟环境，不会污染系统环境）。
   - 如果不想装 Conda，也可以直接安装 [Python 3.10](https://www.python.org/downloads/)（Windows 安装时务必勾选 **`Add python.exe to PATH`**）。
2. **显卡说明（有无 NVIDIA 显卡都能用）**
   - **有 NVIDIA 独立显卡**：确保已安装最新 NVIDIA 显卡驱动（命令行输入 `nvidia-smi` 能看到显卡型号即可），脚本会**自动识别驱动 CUDA 版本**并安装 GPU 加速版 PyTorch。
   - **无独显 / 核显 / macOS**：同样可以使用数据导入、手动标注、AI 大模型标注、轻量模型推理等功能，脚本会自动或通过 `--cpu` 参数安装 CPU 版依赖。

---

## 二、获取项目代码

```bash
git clone https://github.com/YogurtOrange/Better-yolo.git
cd Better-yolo
```

> 没装 Git 也可以在 GitHub 页面点击 **Code → Download ZIP**，解压到本地英文路径（**建议路径中不要包含空格**）。

---

## 三、一键安装（按操作系统选择）

### 🪟 Windows 用户

1. 进入解压后的 `Better-yolo` 文件夹。
2. 在文件夹空白处按住 `Shift` + 右键，选择 **「在此处打开终端 / PowerShell / 命令提示符」**。
3. 运行以下任意一条命令：

```bat
:: 国内用户推荐（使用清华镜像加速，自动识别 GPU/CPU）
setup.bat --mirror tuna

:: 海外或直连网络
setup.bat

:: 没有 NVIDIA 显卡，强制安装 CPU 版本
setup.bat --cpu --mirror tuna
```

> 也可以直接**双击 `setup.bat`** 运行默认一键安装。

### 🐧 Linux / 🍎 macOS 用户

在项目根目录打开终端执行：

```bash
# 国内用户推荐（使用清华镜像加速，自动识别 GPU/CPU）
./setup.sh --mirror tuna

# 海外或直连网络
./setup.sh

# 没有 NVIDIA 显卡，强制安装 CPU 版本
./setup.sh --cpu --mirror tuna
```

> 如果提示权限不足，先执行一次 `chmod +x setup.sh` 再运行即可。

### 脚本会自动帮你做什么？

1. **自动准备虚拟环境**：
   - 如果你已经 `conda activate` 了某个环境，直接装进当前环境；
   - 如果本机装了 Conda 但未激活环境，自动创建并复用名为 `better-yolo` 的 `Python 3.10` 环境；
   - 如果没装 Conda，自动在项目目录下创建 `.venv` 虚拟环境。
2. **自动匹配 PyTorch 与 ONNXRuntime**：
   - 检测到 NVIDIA GPU 时，按驱动 CUDA 版本自动选择对应的 PyTorch CUDA wheel（如 `cu128` / `cu124`），并安装 `onnxruntime-gpu`；
   - 无 GPU 或指定 `--cpu` 时，安装 CPU 版 PyTorch 与 `onnxruntime`，避免报错。
3. **自动创建运行目录并体检**：
   - 初始化 `data/`、`pretrained/`、`projects/`、`outputs/`、`config/` 目录；
   - 安装结束后自动输出一份 **环境体检报告**。

---

## 四、启动 Better-YOLO

看到体检报告最后提示 **`环境完全就绪 / 核心依赖齐全，可直接运行: python main.py`** 后，按你使用的环境类型启动软件：

### 如果由 Conda 自动创建了 `better-yolo` 环境
```bash
conda activate better-yolo
python main.py
```

### 如果由 `.venv` 本地虚拟环境安装
```bash
# Linux / macOS
source .venv/bin/activate
python main.py
```
```bat
:: Windows
.venv\Scripts\activate
python main.py
```

### 如果安装在你原本已激活的环境中
```bash
python main.py
```

---

## 五、常用进阶参数速查

所有参数都可以通过 `./setup.sh`、`setup.bat` 或直接调用 `python scripts/setup_env.py` 使用：

| 场景 | 命令示例 | 说明 |
| :--- | :--- | :--- |
| **仅体检当前环境** | `python scripts/setup_env.py --check` | 只读检查 Python、GPU/CUDA、全部依赖与冲突，不修改任何包 |
| **预览将执行的命令** | `python scripts/setup_env.py --install --dry-run --mirror tuna` | 不实际下载，先看清楚具体会装哪些包和源 |
| **切换国内镜像源** | `./setup.sh --mirror aliyun` 或 `--mirror ustc` | 支持 `tuna`（清华）、`aliyun`（阿里云）、`ustc`（中科大）或自定义镜像 URL |
| **手动指定 CUDA 版本** | `./setup.sh --cuda cu124 --mirror tuna` | 可选 `cu128` / `cu126` / `cu124` / `cu121` / `cu118` |
| **自定义 Conda 环境名** | `BETTER_YOLO_ENV=my-yolo ./setup.sh --mirror tuna` | Linux/macOS 可通过环境变量指定新建的 Conda 环境名 |
| **导出 JSON 体检报告** | `python scripts/setup_env.py --check --json` | 方便在排错或反馈 Issue 时直接粘贴完整环境信息 |

---

## 六、配置 AI 智能标注（可选）

如果你需要使用左侧的 **「AI 智能标注」**（Qwen-VL 大模型自动画框与对比纠错）：

1. 启动软件后切换到 **「AI 标注」** 页面；
2. 在 **API 配置** 中填入你的 DashScope（阿里云百炼）或其他 OpenAI 兼容接口的 **API Key**（配置会自动保存在本地 `config/llm_config.json`，不会上传）；
3. 也可以通过系统环境变量配置：
   ```bash
   # Linux / macOS
   export DASHSCOPE_API_KEY="你的API_KEY"
   ```
   ```bat
   :: Windows PowerShell
   $env:DASHSCOPE_API_KEY="你的API_KEY"
   ```

---

## 七、新手常见问题排查

### 1. 体检报告提示「检测到 NVIDIA GPU 但当前 PyTorch 为 CPU 版」
说明之前误装了 CPU 版本的 PyTorch。直接执行体检报告末尾给出的修复命令即可，例如：
```bash
python scripts/setup_env.py --install --cuda cu128 --mirror tuna
```

### 2. 体检报告提示「同时安装了 onnxruntime 与 onnxruntime-gpu」
这两个包共享同一个模块名 `onnxruntime`，混装会导致 GPU 加速失效。按以下两步清理（保留 GPU 版）：
```bash
pip uninstall -y onnxruntime
pip install --force-reinstall --no-deps onnxruntime-gpu==1.22.0
```
如果是无独显电脑，则反过来卸载 `onnxruntime-gpu` 并重装 `onnxruntime==1.23.2`。

### 3. 下载 PyTorch 很慢或超时
PyTorch CUDA 包体积较大（约 2~3 GB），若下载中断，直接**重新执行一次相同的安装命令**即可（pip 会自动复用已下载缓存）；建议配合 `--mirror tuna` 加速其余依赖下载。

### 4. Linux 启动 PyQt6 报错 `Could not load the Qt platform plugin "xcb"`
Ubuntu/Debian 桌面环境缺少 XCB 系统库，安装以下系统依赖后重试：
```bash
sudo apt-get update && sudo apt-get install -y libxcb-cursor0 libxcb-xinerama0 libgl1
```

### 5. 下一步看哪里？
环境配置完成后，建议继续阅读 [docs/TUTORIAL.md](TUTORIAL.md) 图文使用教程，跟着演示完成从**新建项目 → AI 标注 → 模型训练 → 推理测试**的完整流程。
