@echo off
title Face Studio AI - Port 7870
cd /d "%~dp0"

echo ======================================================================
echo           Starting Face Studio ^& Photovance AI (Port 7870)
echo ======================================================================
echo.
echo Launching Web Studio at http://127.0.0.1:7870 ...
start http://127.0.0.1:7870

if exist "D:\Image AI\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe" (
    "D:\Image AI\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe" -u server.py
) else (
    "D:\Image AI L model\runtime\python.exe" server.py
)
pause
