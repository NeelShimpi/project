#!/bin/bash

# Driver Drowsiness Detection System - Setup Script
# This script automates the installation and setup process

echo "======================================================"
echo "Driver Drowsiness Detection System - Setup"
echo "======================================================"
echo ""

# Color codes
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check Python version
echo -e "${YELLOW}[1/6] Checking Python version...${NC}"
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 is not installed. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python check passed${NC}"
echo ""

# Create virtual environment
echo -e "${YELLOW}[2/6] Creating virtual environment...${NC}"
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo -e "${GREEN}✓ Virtual environment created${NC}"
else
    echo -e "${GREEN}✓ Virtual environment already exists${NC}"
fi
echo ""

# Activate virtual environment
echo -e "${YELLOW}[3/6] Activating virtual environment...${NC}"
source venv/bin/activate
echo -e "${GREEN}✓ Virtual environment activated${NC}"
echo ""

# Upgrade pip
echo -e "${YELLOW}[4/6] Upgrading pip...${NC}"
pip install --upgrade pip
echo -e "${GREEN}✓ pip upgraded${NC}"
echo ""

# Install PyTorch with CUDA
echo -e "${YELLOW}[5/6] Installing PyTorch...${NC}"
echo "Which version would you like to install?"
echo "1) CUDA 11.8 (recommended for most systems)"
echo "2) CUDA 12.1"
echo "3) CPU only (no GPU acceleration)"
read -p "Enter choice [1-3]: " cuda_choice

case $cuda_choice in
    1)
        echo "Installing PyTorch with CUDA 11.8..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
        ;;
    2)
        echo "Installing PyTorch with CUDA 12.1..."
        pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
        ;;
    3)
        echo "Installing PyTorch (CPU only)..."
        pip install torch torchvision
        ;;
    *)
        echo -e "${RED}Invalid choice. Exiting.${NC}"
        exit 1
        ;;
esac

echo -e "${GREEN}✓ PyTorch installed${NC}"
echo ""

# Install other dependencies
echo -e "${YELLOW}[6/6] Installing other dependencies...${NC}"
pip install -r requirements.txt

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ All dependencies installed${NC}"
else
    echo -e "${RED}✗ Some dependencies failed to install${NC}"
    echo "Try installing them manually:"
    echo "pip install opencv-python dlib imutils scipy scikit-learn matplotlib einops tqdm"
fi
echo ""

# Download dlib shape predictor
echo -e "${YELLOW}Downloading dlib shape predictor...${NC}"
if [ ! -f "shape_predictor_68_face_landmarks.dat" ]; then
    echo "Downloading shape_predictor_68_face_landmarks.dat.bz2..."
    wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
    
    if [ -f "shape_predictor_68_face_landmarks.dat.bz2" ]; then
        echo "Extracting..."
        bunzip2 shape_predictor_68_face_landmarks.dat.bz2
        echo -e "${GREEN}✓ Shape predictor downloaded and extracted${NC}"
    else
        echo -e "${YELLOW}⚠ Failed to download automatically. Please download manually from:${NC}"
        echo "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
    fi
else
    echo -e "${GREEN}✓ Shape predictor already exists${NC}"
fi
echo ""

# Run system tests
echo "======================================================"
echo "Setup Complete!"
echo "======================================================"
echo ""
echo "Would you like to run system tests? (y/n)"
read -p "Run tests: " run_tests

if [ "$run_tests" = "y" ] || [ "$run_tests" = "Y" ]; then
    echo ""
    echo "Running system tests..."
    python test_system.py --test all
fi

echo ""
echo "======================================================"
echo "Next Steps:"
echo "======================================================"
echo "1. Activate virtual environment: source venv/bin/activate"
echo "2. Run detection: python drowsiness_detection.py"
echo "3. Run tests: python test_system.py"
echo ""
echo "For training your own model:"
echo "1. Prepare dataset in dataset/train and dataset/val"
echo "2. Run: python train_vit.py"
echo ""
echo -e "${GREEN}Happy coding! Stay alert, stay safe! 🚗💤🚨${NC}"
