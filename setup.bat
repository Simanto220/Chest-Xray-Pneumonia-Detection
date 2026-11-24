@echo off
setlocal
cd /d "%~dp0"

echo ================================================================
echo PneumoVision AI - Automated Environment Setup
echo ================================================================
echo.
echo [*] Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not added to PATH!
    echo [*] Please install Python from python.org and check "Add python.exe to PATH".
    echo.
    pause
    exit /b 1
)

echo [*] Creating virtual environment (venv)...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] Failed to create virtual environment!
    pause
    exit /b 1
)

echo [*] Activating venv and installing required libraries...
echo [*] This will take a few minutes depending on your internet speed.
echo.
call venv\Scripts\activate.bat
python -m pip install --upgrade pip
pip install -r requirements.txt

echo.
echo ================================================================
echo [SUCCESS] Setup completed successfully!
echo You can now double-click "run.bat" to start the website.
echo ================================================================
echo.
pause