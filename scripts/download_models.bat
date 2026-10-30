@echo off
title Face Studio - Download Additional XL Models
cd /d "%~dp0\.."
echo Starting download for CyberRealistic XL and epiCRealism XL...
python scripts\download_models.py
pause
