@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo Starting PneumoVision AI - Chest X-Ray Pneumonia Detection
echo Powered by VGG16 + VGG19 Hybrid Deep Learning Architecture
echo ================================================================
echo.

set "PY_CMD=python"
if exist "venv\Scripts\python.exe" (
    set "PY_CMD=venv\Scripts\python.exe"
    echo [*] Local virtual environment detected.
)

echo [*] Starting Flask Server on http://127.0.0.1:5000
echo [*] Open your browser and visit: http://127.0.0.1:5000
echo.

%PY_CMD% App.py

if errorlevel 1 (
    echo.
    echo [ERROR] Flask server stopped with an error.
    pause
)