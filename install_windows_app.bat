@echo off
setlocal
cd /d "%~dp0"

set "STUDIO_PYTHON=D:\Image AI\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe"
if not exist "%STUDIO_PYTHON%" set "STUDIO_PYTHON=D:\Image AI L model\runtime\python.exe"
if not exist "%STUDIO_PYTHON%" (
    echo Face Studio's Python runtime was not found.
    echo Install Python 3.12 and the Face Studio inference dependencies first.
    pause
    exit /b 1
)

"%STUDIO_PYTHON%" -m pip install -r requirements-desktop.txt
if errorlevel 1 (
    echo Desktop app dependency installation failed.
    pause
    exit /b 1
)

echo Windows desktop app dependency installed.
pause
