@echo off
title Photovance AI - Android Build Pipeline
echo ======================================================================
echo    PHOTOVANCE AI - ANDROID COMPILATION & BUNDLE GENERATION
echo ======================================================================
echo.
set "JAVA_HOME=C:\Program Files\Microsoft\jdk-17.0.20.101-hotspot"
set "ANDROID_HOME=D:\android_sdk"
set "PATH=%JAVA_HOME%\bin;%ANDROID_HOME%\platform-tools;%PATH%"

cd /d "%~dp0\.."

echo [1/2] Verifying Android SDK and Java 17...
java -version
if %ERRORLEVEL% NEQ 0 (
    echo Error: Java 17 not found.
    pause
    exit /b 1
)

echo.
echo [2/2] Compiling Photovance AI Release App Bundle (.aab) and APK...
if exist "D:\gradle-8.7\bin\gradle.bat" (
    call "D:\gradle-8.7\bin\gradle.bat" assembleRelease bundleRelease
) else (
    call gradlew.bat assembleRelease bundleRelease
)

echo.
echo ======================================================================
echo Build Complete!
echo Release APK:    android\app\build\outputs\apk\release\app-release.apk
echo Play Store AAB: android\app\build\outputs\bundle\release\app-release.aab
echo ======================================================================
pause
