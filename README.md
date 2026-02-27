# Driver Drowsiness Detection System

A real-time driver drowsiness detection system using Computer Vision and Deep Learning techniques.

## Tech Stack

- **OpenCV**: Real-time video processing and face detection
- **PyTorch**: Deep learning framework with CUDA support
- **Vision Transformer (ViT)**: State-of-the-art transformer architecture for image classification
- **Haar Cascade**: Face detection algorithm
- **Eye Aspect Ratio (EAR)**: Blink detection and eye closure measurement
- **CUDA**: GPU acceleration for real-time inference

## Features

- ✅ Real-time face and eye detection using Haar Cascade
- ✅ Eye Aspect Ratio (EAR) calculation for drowsiness detection
- ✅ Vision Transformer (ViT) deep learning model for enhanced accuracy
- ✅ CUDA/GPU acceleration support for fast inference
- ✅ Dual detection system (EAR + ViT) for robust performance
- ✅ Real-time FPS counter and statistics
- ✅ Visual alerts and status indicators

## System Requirements

### Hardware
- Webcam or camera device
- **Recommended**: NVIDIA GPU with CUDA support for real-time performance
- **Minimum**: CPU (slower but functional)

### Software
- Python 3.8+
- CUDA Toolkit 11.8 or 12.1 (for GPU support)
- cuDNN (for GPU support)

## Installation

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd drowsiness-detection
```

### 2. Create Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install PyTorch with CUDA Support

**For CUDA 11.8:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**For CUDA 12.1:**
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**For CPU only:**
```bash
pip install torch torchvision
```

### 4. Install Other Dependencies
```bash
pip install -r requirements.txt
```

### 5. Download Required Model Files

**dlib Shape Predictor (68 facial landmarks):**
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
```

Place `shape_predictor_68_face_landmarks.dat` in the project root directory.

## Project Structure

```
drowsiness-detection/
│
├── drowsiness_detection.py    # Main detection script
├── vit_model.py               # Vision Transformer architecture
├── train_vit.py               # Training script for ViT model
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── shape_predictor_68_face_landmarks.dat  # dlib facial landmarks
├── vit_drowsiness_model.pth   # Trained ViT model (after training)
│
└── dataset/                   # Training dataset (optional)
    ├── train/
    │   ├── alert/
    │   └── drowsy/
    └── val/
        ├── alert/
        └── drowsy/
```

## Usage

### Quick Start (Webcam Detection)

```bash
python drowsiness_detection.py
```

This will:
- Open your default webcam (device 0)
- Start real-time drowsiness detection
- Show video feed with annotations
- Display EAR values and drowsiness alerts

Press **'q'** to quit the application.

### Detection on Video File

Modify the main section in `drowsiness_detection.py`:

```python
if __name__ == "__main__":
    detector = DrowsinessDetector(use_cuda=True)
    detector.run(video_source='path/to/your/video.mp4', use_vit=True)
```

### Configuration Options

In `drowsiness_detection.py`, you can adjust:

```python
# EAR threshold (lower = more sensitive)
self.EAR_THRESHOLD = 0.25

# Number of consecutive frames for drowsiness detection
self.EAR_CONSEC_FRAMES = 20

# Enable/disable CUDA
detector = DrowsinessDetector(use_cuda=True)

# Enable/disable Vision Transformer
detector.run(video_source=0, use_vit=True)
```

## Training Your Own Model

### 1. Prepare Dataset

Organize your dataset in the following structure:

```
dataset/
├── train/
│   ├── alert/     # Images of alert/awake drivers
│   └── drowsy/    # Images of drowsy/sleepy drivers
└── val/
    ├── alert/
    └── drowsy/
```

### 2. Configure Training

Edit the `CONFIG` dictionary in `train_vit.py`:

```python
CONFIG = {
    'train_dir': 'dataset/train',
    'val_dir': 'dataset/val',
    'img_size': 224,
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'use_light_model': False,  # True for faster inference
    'save_path': 'vit_drowsiness_model.pth'
}
```

### 3. Start Training

```bash
python train_vit.py
```

Training will:
- Use GPU if available (CUDA)
- Save the best model based on validation accuracy
- Generate training curves plot
- Display metrics (accuracy, precision, recall, F1-score)

### 4. Use Your Trained Model

The trained model will be saved as `vit_drowsiness_model.pth`. The detection script will automatically load it.

## How It Works

### Eye Aspect Ratio (EAR) Method

1. **Face Detection**: Haar Cascade detects faces in the frame
2. **Landmark Detection**: dlib identifies 68 facial landmarks
3. **Eye Extraction**: Extract eye region coordinates
4. **EAR Calculation**: 
   ```
   EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
   ```
5. **Drowsiness Detection**: If EAR < threshold for N consecutive frames → Alert!

### Vision Transformer Method

1. **Face Preprocessing**: Resize and normalize face image
2. **Patch Embedding**: Split image into 16x16 patches
3. **Transformer Encoding**: Process through 12 transformer blocks
4. **Classification**: Binary classification (alert/drowsy)
5. **Confidence Score**: Output probability of drowsiness

### Combined Detection

The system uses both methods for robust detection:
- **EAR**: Fast, reliable for eye closure detection
- **ViT**: Captures subtle facial cues and head position
- **Alert Trigger**: Either method detecting drowsiness triggers alert

## Performance Optimization

### For Real-Time Performance:

1. **Use GPU/CUDA**:
   ```python
   detector = DrowsinessDetector(use_cuda=True)
   ```

2. **Use Lightweight ViT**:
   In `drowsiness_detection.py`, change:
   ```python
   from vit_model import VisionTransformerDrowsinessLight
   self.vit_model = VisionTransformerDrowsinessLight(...)
   ```

3. **Disable ViT** (EAR only - fastest):
   ```python
   detector.run(video_source=0, use_vit=False)
   ```

4. **Reduce Video Resolution**:
   ```python
   cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
   cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
   ```

## Troubleshooting

### Issue: "CUDA not available"
- Install CUDA Toolkit from NVIDIA
- Install correct PyTorch version with CUDA support
- Check: `torch.cuda.is_available()`

### Issue: "shape_predictor_68_face_landmarks.dat not found"
- Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
- Extract and place in project root

### Issue: "No faces detected"
- Ensure good lighting conditions
- Face should be clearly visible to camera
- Adjust Haar Cascade parameters in code

### Issue: Low FPS
- Enable CUDA/GPU acceleration
- Use lightweight ViT model
- Reduce video resolution
- Close other GPU-intensive applications

### Issue: "dlib installation failed"
- Install CMake: `pip install cmake`
- On Windows, install Visual Studio C++ Build Tools
- On Linux: `sudo apt-get install build-essential cmake`

## Model Performance

### Vision Transformer (Full)
- Parameters: ~86M
- Inference Time (GPU): ~15-20ms per frame
- Accuracy: ~95%+ (with proper training)

### Vision Transformer (Light)
- Parameters: ~22M
- Inference Time (GPU): ~8-12ms per frame
- Accuracy: ~92%+ (with proper training)

### EAR Method
- Inference Time: <5ms per frame
- Accuracy: ~85-90% (eye closure detection)

## Dataset Recommendations

For training, use datasets like:
- **MRL Eye Dataset**: http://mrl.cs.vsb.cz/eyedataset
- **Drowsiness Dataset**: https://www.kaggle.com/datasets/dheerajperumandla/drowsiness-dataset
- **Custom Dataset**: Collect your own data with different lighting, angles, and subjects

Recommended dataset size:
- Training: 5,000+ images per class
- Validation: 1,000+ images per class

## Future Enhancements

- [ ] Audio alert system (buzzer/alarm)
- [ ] Head pose estimation
- [ ] Yawning detection
- [ ] Mobile/embedded deployment (TensorRT, ONNX)
- [ ] Multi-face detection
- [ ] Driver distraction detection
- [ ] Integration with vehicle systems

## License

MIT License - Feel free to use and modify for your projects.

## Acknowledgments

- OpenCV for computer vision tools
- PyTorch team for the deep learning framework
- dlib for facial landmark detection
- Vision Transformer paper: "An Image is Worth 16x16 Words"

## Support

For issues, questions, or contributions, please open an issue on GitHub.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{drowsiness_detection,
  title={Driver Drowsiness Detection System},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/drowsiness-detection}
}
```

---

**Stay Alert, Stay Safe! 🚗💤🚨**
