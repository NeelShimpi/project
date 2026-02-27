@echo off
REM Driver Drowsiness Detection System - Setup Script for Windows

echo ======================================================
echo Driver Drowsiness Detection System - Setup
echo ======================================================
echo.

REM Check Python installation
echo [1/6] Checking Python version...
python --version 2>nul
if errorlevel 1 (
    echo Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://www.python.org/downloads/
    pause
    exit /b 1
)

python --version
echo [OK] Python check passed
echo.

REM Create virtual environment
echo [2/6] Creating virtual environment...
if not exist "venv" (
    python -m venv venv
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate virtual environment
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
echo [OK] Virtual environment activated
echo.

REM Upgrade pip
echo [4/6] Upgrading pip...
python -m pip install --upgrade pip
echo [OK] pip upgraded
echo.

REM Install PyTorch
echo [5/6] Installing PyTorch...
echo Which version would you like to install?
echo 1) CUDA 11.8 (recommended for most NVIDIA GPUs)
echo 2) CUDA 12.1 (for newer NVIDIA GPUs)
echo 3) CPU only (no GPU acceleration)
set /p cuda_choice="Enter choice [1-3]: "

if "%cuda_choice%"=="1" (
    echo Installing PyTorch with CUDA 11.8...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
) else if "%cuda_choice%"=="2" (
    echo Installing PyTorch with CUDA 12.1...
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
) else if "%cuda_choice%"=="3" (
    echo Installing PyTorch (CPU only)...
    pip install torch torchvision
) else (
    echo Invalid choice. Exiting.
    pause
    exit /b 1
)

echo [OK] PyTorch installed
echo.

REM Install other dependencies
echo [6/6] Installing other dependencies...
pip install -r requirements.txt

if errorlevel 1 (
    echo [WARNING] Some dependencies failed to install
    echo Try installing them manually:
    echo pip install opencv-python dlib imutils scipy scikit-learn matplotlib einops tqdm
) else (
    echo [OK] All dependencies installed
)
echo.

REM Download dlib shape predictor
echo Downloading dlib shape predictor...
if not exist "shape_predictor_68_face_landmarks.dat" (
    echo Please download shape_predictor_68_face_landmarks.dat manually from:
    echo http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    echo.
    echo After downloading:
    echo 1. Extract the .bz2 file
    echo 2. Place shape_predictor_68_face_landmarks.dat in this directory
    echo.
) else (
    echo [OK] Shape predictor already exists
)

echo.
echo ======================================================
echo Setup Complete!
echo ======================================================
echo.

REM Ask to run tests
set /p run_tests="Would you like to run system tests? (y/n): "
if /i "%run_tests%"=="y" (
    echo.
    echo Running system tests...
    python test_system.py --test all
)

echo.
echo ======================================================
echo Next Steps:
echo ======================================================
echo 1. Activate virtual environment: venv\Scripts\activate
echo 2. Run detection: python drowsiness_detection.py
echo 3. Run tests: python test_system.py
echo.
echo For training your own model:
echo 1. Prepare dataset in dataset\train and dataset\val
echo 2. Run: python train_vit.py
echo.
echo Happy coding! Stay alert, stay safe! 🚗💤🚨
echo.
pause
