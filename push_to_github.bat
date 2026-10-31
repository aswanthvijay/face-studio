@echo off
echo ======================================================================
echo    Pushing Face Studio Deep Bug Remediation to GitHub
echo ======================================================================
cd /d "%~dp0"
git add core/ server.py static/app.js test_server_security.py scripts/verify_bilateral_clamping_and_hull.py scripts/verify_reinhard_seamless_pipeline.py push_to_github.bat
git reset core/__pycache__ core/vendor/opencv_mediapipe/__pycache__ 2>nul
git commit -m "Fix all 13 root-cause defects across biometric face, skin, clothing, and pipeline layers"
git push origin main
