@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo Starting PneumoVision AI - Chest X-Ray Pneumonia Detection
echo Powered by VGG16 + VGG19 Hybrid Deep Learning Architecture
echo ================================================================
echo.

if not exist "venv\Scripts\python.exe" goto :need_setup
goto :launch_app

:need_setup
echo [*] Virtual environment not found. Running automated setup first...
echo.
call setup.bat
if not exist "venv\Scripts\python.exe" (
    echo.
    echo [ERROR] Setup did not complete successfully.
    pause
    exit /b 1
)

:launch_app
echo [*] Using local virtual environment venv...
echo [*] Starting Flask Server on http://127.0.0.1:5000
echo [*] Open your browser and visit: http://127.0.0.1:5000
echo.

"venv\Scripts\python.exe" App.py

if errorlevel 1 (
    echo.
    echo [ERROR] Flask server stopped with an error.
    pause
)