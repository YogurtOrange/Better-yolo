@echo off
chcp 65001 >nul
setlocal enabledelayedexpansion

REM Better-YOLO Windows 一键环境配置脚本
REM 用法:
REM   setup.bat                  自动检测 GPU/CPU 并安装依赖
REM   setup.bat --mirror tuna    使用清华 PyPI 镜像加速
REM   setup.bat --cpu            强制安装 CPU 版本
REM   setup.bat --check          仅体检当前环境

cd /d "%~dp0"

set "IS_CHECK=0"
for %%A in (%*) do (
    if /I "%%~A"=="--check" set "IS_CHECK=1"
)

if "%IS_CHECK%"=="1" (
    python scripts\setup_env.py %*
    exit /b %ERRORLEVEL%
)

if defined CONDA_PREFIX (
    echo ^>^>^> 检测到已激活 Conda 环境: %CONDA_DEFAULT_ENV%
    python scripts\setup_env.py --install %*
    exit /b %ERRORLEVEL%
)

if defined VIRTUAL_ENV (
    echo ^>^>^> 检测到已激活虚拟环境: %VIRTUAL_ENV%
    python scripts\setup_env.py --install %*
    exit /b %ERRORLEVEL%
)

where conda >nul 2>nul
if %ERRORLEVEL%==0 (
    set "ENV_NAME=better-yolo"
    echo ^>^>^> 检测到 Conda，准备创建/复用环境: !ENV_NAME! ^(Python 3.10^)
    conda env list | findstr /R /C:"^!ENV_NAME! " >nul
    if errorlevel 1 (
        conda create -y -n !ENV_NAME! python=3.10
    )
    conda run --no-capture-output -n !ENV_NAME! python scripts\setup_env.py --install %*
    exit /b %ERRORLEVEL%
)

if not exist ".venv\Scripts\python.exe" (
    echo ^>^>^> 创建本地虚拟环境: .venv
    python -m venv .venv
)

.venv\Scripts\python.exe scripts\setup_env.py --install %*
exit /b %ERRORLEVEL%
