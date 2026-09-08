@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo Starting PneumoVision AI (Chest X-Ray Pneumonia Detection)
echo Powered by VGG16 + VGG19 Hybrid Deep Learning Architecture
echo ================================================================
echo.

if exist "venv\Scripts\python.exe" (
    echo [*] Using local virtual environment (venv)...
    set "PY_CMD=venv\Scripts\python.exe"
) else (
    echo [*] Using system python...
    set "PY_CMD=python"
)

echo [*] Starting Flask Server on http://127.0.0.1:5000
echo [*] Open your browser and visit: http://127.0.0.1:5000
echo.

%PY_CMD% App.py
pause
