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

"%STUDIO_PYTHON%" -m pip install -r requirements-windows-build.txt
if errorlevel 1 (
    echo Windows app build dependencies could not be installed.
    pause
    exit /b 1
)

"%STUDIO_PYTHON%" -m PyInstaller --noconfirm --clean --windowed --onedir ^
    --name FaceStudio ^
    --distpath dist\windows ^
    --workpath build\windows ^
    --specpath build\windows ^
    --collect-all webview ^
    --add-data "%~dp0desktop;desktop" ^
    --add-data "%~dp0server.py;." ^
    --add-data "%~dp0core;core" ^
    --add-data "%~dp0static;static" ^
    --add-data "%~dp0models;models" ^
    --add-data "%~dp0extra_models.yaml;." ^
    --add-data "%~dp0privacy.html;." ^
    desktop_app.py
if errorlevel 1 (
    echo Windows desktop packaging failed.
    pause
    exit /b 1
)

echo.
echo Windows app packaged at dist\windows\FaceStudio\FaceStudio.exe
echo Copy the entire FaceStudio folder when distributing the app.
set "ISCC=%ProgramFiles(x86)%\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC.exe"
if not exist "%ISCC%" set "ISCC=%LOCALAPPDATA%\Programs\Inno Setup 6\ISCC\ISCC.exe"
if exist "%ISCC%" (
    "%ISCC%" windows_installer.iss
    if errorlevel 1 (
        echo Windows installer creation failed.
        pause
        exit /b 1
    )
    echo Windows installer created at dist\installer\FaceStudio-Setup.exe
) else (
    echo Inno Setup was not found. The distributable installer is built by the Windows desktop release workflow.
)
pause
