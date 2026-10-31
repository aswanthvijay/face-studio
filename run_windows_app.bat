@echo off
setlocal
cd /d "%~dp0"

set "STUDIO_PYTHON=D:\Image AI\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe"
if not exist "%STUDIO_PYTHON%" set "STUDIO_PYTHON=D:\Image AI L model\runtime\python.exe"
if not exist "%STUDIO_PYTHON%" (
    echo Face Studio's Python runtime was not found.
    echo Run install_windows_app.bat after installing the inference runtime.
    pause
    exit /b 1
)

"%STUDIO_PYTHON%" desktop_app.py
if errorlevel 1 pause
