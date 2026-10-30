@echo off
title Face Studio - Android Release Compile
cd /d "%~dp0\..\android"
call scripts\build_app.bat
pause
