@echo off
chcp 65001 >nul
title Local Whisper Speech Recognition Tool

echo ============================================
echo Local Whisper Speech Recognition Tool
echo ============================================

:: Check Python environment
if exist "py\python.exe" (
    echo [OK] Found built-in Python environment
    set PYTHON_EXE=py\python.exe
) else (
    echo [ERROR] Built-in Python environment not found
    echo Please ensure py\python.exe exists
    pause
    exit /b 1
)

:: Check main program file
if not exist "whisper_app.py" (
    echo [ERROR] Main program file whisper_app.py not found
    pause
    exit /b 1
)

:: Check models directory
if not exist "models" (
    echo [WARNING] Models directory not found
    echo Please ensure model files are downloaded to models directory
)

echo [INFO] Starting offline speech recognition tool...
echo [INFO] This version runs completely offline, no network required
echo.

:: Start main program
"%PYTHON_EXE%" whisper_app.py

:: If program exits abnormally, pause to show error
if errorlevel 1 (
    echo.
    echo [ERROR] Program exited abnormally, error code: %errorlevel%
    pause
) 