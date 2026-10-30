@echo off
setlocal enabledelayedexpansion
title Face Studio — Custom LoRA Training Studio (RTX 3060 6GB)

set "PYTHON_EXE=D:\Image AI\StabilityMatrix-win-x64\Data\Packages\ComfyUI\venv\Scripts\python.exe"

if not exist "!PYTHON_EXE!" (
    echo [ERROR] Python environment not found at: !PYTHON_EXE!
    pause
    exit /b 1
)

cd /d "%~dp0\.."

:MENU
cls
echo ======================================================================
echo           Face Studio AI — Custom LoRA Training Studio
echo           Tuned for NVIDIA RTX 3060 (6GB VRAM Low-Memory)
echo ======================================================================
echo.
echo [1] Quick LoRA Validation (5 Steps on Dress Dataset, verifies 6GB VRAM)
echo [2] Download Curated 30-Dress Dataset from Web (to training\raw_images)
echo [3] Run SegFormer Dress Detection & Prepper (on training\raw_images)
echo [4] Full Fashion LoRA Training (e.g. 150-350 steps)
echo [5] Launch Web Studio (Port 7870)
echo [6] Exit
echo.
set /p CHOICE="Choose an option [1-6]: "

if "%CHOICE%"=="1" (
    echo.
    echo [*] Running 5-step dry-run training on detected dress dataset...
    "!PYTHON_EXE!" training\train_lora_sdxl.py --dataset-dir training\dataset --lora-name fs_couture --dry-run
    pause
    goto MENU
)

if "%CHOICE%"=="2" (
    echo.
    echo [*] Downloading 30 curated high-resolution dress photos...
    "!PYTHON_EXE!" training\download_training_dataset.py
    pause
    goto MENU
)

if "%CHOICE%"=="3" (
    echo.
    set /p TRIGGER="Enter Trigger Token (default: fs_couture): "
    if "!TRIGGER!"=="" set "TRIGGER=fs_couture"
    echo [*] Running SegFormer dress detection, bucketing and captioning...
    "!PYTHON_EXE!" training\detect_and_crop_dresses.py --raw-dir training\raw_images --out-dir training\dataset --trigger !TRIGGER!
    pause
    goto MENU
)

if "%CHOICE%"=="4" (
    echo.
    set /p LORA_NAME="Enter LoRA Name (default: fs_couture): "
    if "!LORA_NAME!"=="" set "LORA_NAME=fs_couture"
    
    set /p STEPS="Enter Training Steps (default: 200): "
    if "!STEPS!"=="" set "STEPS=200"

    echo.
    echo [*] Starting Fashion LoRA Training (!STEPS! steps)...
    "!PYTHON_EXE!" training\train_lora_sdxl.py --dataset-dir training\dataset --lora-name !LORA_NAME! --steps !STEPS!
    pause
    goto MENU
)

if "%CHOICE%"=="5" (
    echo.
    echo [*] Launching Face Studio Web Studio on Port 7870...
    start "" run.bat
    goto MENU
)

if "%CHOICE%"=="6" (
    exit /b 0
)

goto MENU
