@echo off
chcp 65001 >nul
title 本地Whisper语音识别工具

echo 🤖 本地Whisper语音识别工具
echo ================================

:: 检查Python环境
if exist "venv\subtitle_tools-cpu\Scripts\python.exe" (
    echo ✅ 找到虚拟环境，使用虚拟环境Python...
    set PYTHON_EXE=venv\subtitle_tools-cpu\Scripts\python.exe
) else (
    echo ⚠️ 未找到虚拟环境，使用系统Python
    set PYTHON_EXE=python
)

:: 检查主程序文件
if not exist "local_whisper_app.py" (
    echo ❌ 错误：找不到主程序文件 local_whisper_app.py
    pause
    exit /b 1
)

echo 🚀 启动语音识别工具...
echo.

:: 启动主程序
%PYTHON_EXE% local_whisper_app.py

:: 如果程序异常退出，暂停显示错误信息
if errorlevel 1 (
    echo.
    echo ❌ 程序异常退出，错误代码：%errorlevel%
    pause
) 