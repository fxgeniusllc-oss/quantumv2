@echo off
REM ========================================
REM Quantum Trading System - Full System Install and Deploy
REM Windows Batch Script
REM ========================================

setlocal enabledelayedexpansion

echo ========================================
echo Quantum Trading System Setup
echo Full Installation and Deployment
echo ========================================
echo.

REM Check if running as administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo Warning: Not running as administrator. Some operations may fail.
    echo It is recommended to run this script as administrator.
    echo.
    pause
)

REM Configuration
set PYTHON_MIN_VERSION=3.9
set NODE_MIN_VERSION=16
set ENVIRONMENT=%1
if "%ENVIRONMENT%"=="" set ENVIRONMENT=development

echo Environment: %ENVIRONMENT%
echo.

REM ========================================
REM Step 1: Check Prerequisites
REM ========================================
echo [1/8] Checking prerequisites...
echo.

REM Check Python
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://www.python.org/downloads/
    goto :error
)

for /f "tokens=2" %%i in ('python --version 2^>^&1') do set PYTHON_VERSION=%%i
echo Python version: %PYTHON_VERSION%

REM Check Node.js
node --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Node.js is not installed or not in PATH
    echo Please install Node.js 16+ from https://nodejs.org/
    goto :error
)

for /f "tokens=1" %%i in ('node --version 2^>^&1') do set NODE_VERSION=%%i
echo Node.js version: %NODE_VERSION%

REM Check Yarn
yarn --version >nul 2>&1
if %errorLevel% neq 0 (
    echo Yarn is not installed. Installing yarn globally...
    call npm install -g yarn
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install Yarn
        goto :error
    )
)

for /f "tokens=1" %%i in ('yarn --version 2^>^&1') do set YARN_VERSION=%%i
echo Yarn version: %YARN_VERSION%

REM Check Git
git --version >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Git is not installed or not in PATH
    echo Please install Git from https://git-scm.com/download/win
    goto :error
)

echo.
echo All prerequisites are met!
echo.

REM ========================================
REM Step 2: Create Virtual Environment
REM ========================================
echo [2/8] Setting up Python virtual environment...
echo.

if not exist "venv" (
    python -m venv venv
    if %errorLevel% neq 0 (
        echo ERROR: Failed to create virtual environment
        goto :error
    )
    echo Virtual environment created successfully
) else (
    echo Virtual environment already exists
)

REM Activate virtual environment
call venv\Scripts\activate.bat
if %errorLevel% neq 0 (
    echo ERROR: Failed to activate virtual environment
    goto :error
)

echo Virtual environment activated
echo.

REM ========================================
REM Step 3: Upgrade pip
REM ========================================
echo [3/8] Upgrading pip...
echo.

python -m pip install --upgrade pip
if %errorLevel% neq 0 (
    echo ERROR: Failed to upgrade pip
    goto :error
)

echo pip upgraded successfully
echo.

REM ========================================
REM Step 4: Install Python Dependencies
REM ========================================
echo [4/8] Installing Python dependencies...
echo.

if not exist "requirements.txt" (
    echo ERROR: requirements.txt not found
    goto :error
)

pip install -r requirements.txt
if %errorLevel% neq 0 (
    echo ERROR: Failed to install Python dependencies
    goto :error
)

echo Python dependencies installed successfully
echo.

REM ========================================
REM Step 5: Install Node.js Dependencies with Yarn
REM ========================================
echo [5/8] Installing Node.js dependencies with Yarn...
echo.

if not exist "package.json" (
    echo Warning: package.json not found, skipping Node.js dependencies
) else (
    yarn install
    if %errorLevel% neq 0 (
        echo ERROR: Failed to install Node.js dependencies with Yarn
        goto :error
    )
    echo Node.js dependencies installed successfully
)

echo.

REM ========================================
REM Step 6: Create Necessary Directories
REM ========================================
echo [6/8] Creating necessary directories...
echo.

if not exist "secrets" mkdir secrets
if not exist "logs" mkdir logs
if not exist "data" mkdir data
if not exist "backups" mkdir backups

echo Directories created successfully
echo.

REM ========================================
REM Step 7: Setup Configuration
REM ========================================
echo [7/8] Setting up configuration...
echo.

if not exist ".env" (
    if exist ".env.example" (
        copy ".env.example" ".env"
        echo .env file created from template
        echo IMPORTANT: Please edit .env with your API keys and configuration!
    ) else (
        echo Warning: .env.example not found
    )
) else (
    echo .env file already exists
)

REM Load environment-specific config
if exist ".env.%ENVIRONMENT%" (
    echo Loading environment-specific configuration from .env.%ENVIRONMENT%
    copy /Y ".env.%ENVIRONMENT%" ".env"
)

echo.

REM ========================================
REM Step 8: Run Tests
REM ========================================
echo [8/8] Running system tests...
echo.

python -m pytest tests/ -v --tb=short
if %errorLevel% neq 0 (
    echo Warning: Some tests failed. Please review the output above.
    echo You can continue with deployment, but it's recommended to fix failing tests.
    set /p continue="Continue anyway? (y/n): "
    if /i not "!continue!"=="y" goto :error
) else (
    echo All tests passed successfully!
)

echo.

REM ========================================
REM Build Node.js Components
REM ========================================
echo Building Node.js components with Yarn...
echo.

if exist "package.json" (
    REM Check if there's a build script in package.json
    findstr /C:"\"build\"" package.json >nul 2>&1
    if %errorLevel% equ 0 (
        yarn build
        if %errorLevel% neq 0 (
            echo Warning: Yarn build failed
        ) else (
            echo Build completed successfully
        )
    ) else (
        echo No build script found in package.json, skipping build
    )
)

echo.

REM ========================================
REM Deployment Summary
REM ========================================
echo ========================================
echo Installation and Setup Completed!
echo ========================================
echo.
echo System Information:
echo   Python: %PYTHON_VERSION%
echo   Node.js: %NODE_VERSION%
echo   Yarn: %YARN_VERSION%
echo   Environment: %ENVIRONMENT%
echo.
echo Next Steps:
echo   1. Edit .env with your API keys and configuration
echo   2. Review and customize settings in .env
echo   3. Run the system:
echo      - For Quantum Trading: python main.py
echo      - For DeFi strategies: python defi_main.py
echo.
echo To activate virtual environment:
echo   venv\Scripts\activate.bat
echo.
echo To run tests:
echo   python -m pytest tests/ -v
echo.
echo To view logs:
echo   type logs\output.log
echo.
echo For deployment to production, run with 'production' parameter:
echo   install-and-deploy.bat production
echo.
echo ========================================
echo.

REM Deactivate virtual environment
call venv\Scripts\deactivate.bat 2>nul

echo Installation complete! Press any key to exit...
pause >nul
exit /b 0

:error
echo.
echo ========================================
echo ERROR: Installation failed!
echo ========================================
echo Please review the error messages above and fix any issues.
echo.
pause
exit /b 1
