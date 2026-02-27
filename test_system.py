"""
Utility script for testing and evaluating the drowsiness detection system
"""

import torch
import cv2
import numpy as np
from vit_model import VisionTransformerDrowsiness, VisionTransformerDrowsinessLight
from drowsiness_detection import DrowsinessDetector
import time
import argparse
from pathlib import Path


def test_cuda():
    """Test CUDA availability and performance"""
    print("=" * 60)
    print("CUDA/GPU TEST")
    print("=" * 60)
    
    print(f"\nCUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"PyTorch Version: {torch.__version__}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  Memory Allocated: {torch.cuda.memory_allocated(i) / 1024**2:.2f} MB")
            print(f"  Memory Reserved: {torch.cuda.memory_reserved(i) / 1024**2:.2f} MB")
            print(f"  Max Memory Allocated: {torch.cuda.max_memory_allocated(i) / 1024**2:.2f} MB")
        
        # Benchmark
        print("\n" + "-" * 60)
        print("Performance Benchmark:")
        print("-" * 60)
        
        device = torch.device('cuda')
        model = VisionTransformerDrowsiness().to(device)
        model.eval()
        
        # Warmup
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        with torch.no_grad():
            for _ in range(10):
                _ = model(dummy_input)
        
        # Benchmark
        num_iterations = 100
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_iterations):
                _ = model(dummy_input)
                torch.cuda.synchronize()
        
        end_time = time.time()
        avg_time = (end_time - start_time) / num_iterations * 1000
        
        print(f"Average inference time (GPU): {avg_time:.2f} ms")
        print(f"Approximate FPS: {1000/avg_time:.1f}")
    else:
        print("\nCUDA is not available. Using CPU.")
        print("Install CUDA-enabled PyTorch for GPU acceleration:")
        print("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118")


def test_model_inference():
    """Test model inference speed"""
    print("\n" + "=" * 60)
    print("MODEL INFERENCE SPEED TEST")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test full ViT
    print("\nTesting Full Vision Transformer:")
    model_full = VisionTransformerDrowsiness().to(device)
    model_full.eval()
    
    test_input = torch.randn(1, 3, 224, 224).to(device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model_full(test_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Benchmark
    num_iterations = 100
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model_full(test_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    avg_time_full = (time.time() - start_time) / num_iterations * 1000
    
    print(f"  Average inference time: {avg_time_full:.2f} ms")
    print(f"  Approximate FPS: {1000/avg_time_full:.1f}")
    print(f"  Parameters: {sum(p.numel() for p in model_full.parameters()):,}")
    
    # Test light ViT
    print("\nTesting Lightweight Vision Transformer:")
    model_light = VisionTransformerDrowsinessLight().to(device)
    model_light.eval()
    
    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model_light(test_input)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    
    # Benchmark
    start_time = time.time()
    
    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model_light(test_input)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
    
    avg_time_light = (time.time() - start_time) / num_iterations * 1000
    
    print(f"  Average inference time: {avg_time_light:.2f} ms")
    print(f"  Approximate FPS: {1000/avg_time_light:.1f}")
    print(f"  Parameters: {sum(p.numel() for p in model_light.parameters()):,}")
    
    print(f"\nSpeed improvement: {avg_time_full/avg_time_light:.2f}x faster")


def test_camera():
    """Test camera availability"""
    print("\n" + "=" * 60)
    print("CAMERA TEST")
    print("=" * 60)
    
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                height, width = frame.shape[:2]
                print(f"\n✓ Camera {i} is available")
                print(f"  Resolution: {width}x{height}")
                print(f"  FPS: {cap.get(cv2.CAP_PROP_FPS)}")
            cap.release()
        else:
            if i == 0:
                print(f"\n✗ Camera {i} is not available")
                print("  Please check if camera is connected and not in use")


def test_haarcascade():
    """Test Haar Cascade availability"""
    print("\n" + "=" * 60)
    print("HAAR CASCADE TEST")
    print("=" * 60)
    
    haar_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    
    try:
        face_cascade = cv2.CascadeClassifier(haar_path)
        if not face_cascade.empty():
            print(f"\n✓ Haar Cascade loaded successfully")
            print(f"  Path: {haar_path}")
        else:
            print(f"\n✗ Failed to load Haar Cascade")
    except Exception as e:
        print(f"\n✗ Error loading Haar Cascade: {e}")


def test_dlib():
    """Test dlib shape predictor"""
    print("\n" + "=" * 60)
    print("DLIB SHAPE PREDICTOR TEST")
    print("=" * 60)
    
    try:
        import dlib
        predictor_path = 'shape_predictor_68_face_landmarks.dat'
        
        if Path(predictor_path).exists():
            predictor = dlib.shape_predictor(predictor_path)
            print(f"\n✓ dlib shape predictor loaded successfully")
            print(f"  Path: {predictor_path}")
        else:
            print(f"\n✗ shape_predictor_68_face_landmarks.dat not found")
            print("  Download from:")
            print("  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
    except ImportError:
        print("\n✗ dlib not installed")
        print("  Install with: pip install dlib")
    except Exception as e:
        print(f"\n✗ Error: {e}")


def benchmark_full_pipeline(video_source=0, duration=30):
    """Benchmark the full detection pipeline"""
    print("\n" + "=" * 60)
    print("FULL PIPELINE BENCHMARK")
    print("=" * 60)
    print(f"\nRunning benchmark for {duration} seconds...")
    
    detector = DrowsinessDetector(use_cuda=True)
    cap = cv2.VideoCapture(video_source)
    
    if not cap.isOpened():
        print("✗ Could not open video source")
        return
    
    frame_count = 0
    start_time = time.time()
    ear_times = []
    vit_times = []
    total_times = []
    
    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_start = time.time()
        
        # EAR detection
        ear_start = time.time()
        _, _ = detector.detect_drowsiness_ear(frame)
        ear_time = time.time() - ear_start
        ear_times.append(ear_time * 1000)
        
        # ViT detection
        vit_start = time.time()
        _, _ = detector.detect_drowsiness_vit(frame)
        vit_time = time.time() - vit_start
        vit_times.append(vit_time * 1000)
        
        frame_time = time.time() - frame_start
        total_times.append(frame_time * 1000)
        
        frame_count += 1
    
    cap.release()
    
    actual_duration = time.time() - start_time
    avg_fps = frame_count / actual_duration
    
    print(f"\n✓ Benchmark completed")
    print(f"  Total frames processed: {frame_count}")
    print(f"  Duration: {actual_duration:.2f} seconds")
    print(f"  Average FPS: {avg_fps:.2f}")
    print(f"\nTiming breakdown (average):")
    print(f"  EAR detection: {np.mean(ear_times):.2f} ms")
    print(f"  ViT detection: {np.mean(vit_times):.2f} ms")
    print(f"  Total per frame: {np.mean(total_times):.2f} ms")
    print(f"  Theoretical max FPS: {1000/np.mean(total_times):.2f}")


def run_all_tests():
    """Run all system tests"""
    test_cuda()
    test_model_inference()
    test_camera()
    test_haarcascade()
    test_dlib()
    
    print("\n" + "=" * 60)
    print("SYSTEM TEST COMPLETE")
    print("=" * 60)
    print("\nIf all tests passed, you're ready to run the detection system!")
    print("Run: python drowsiness_detection.py")


def main():
    parser = argparse.ArgumentParser(description='Test and evaluate drowsiness detection system')
    parser.add_argument('--test', type=str, choices=['all', 'cuda', 'model', 'camera', 
                                                      'haarcascade', 'dlib', 'benchmark'],
                       default='all', help='Test to run')
    parser.add_argument('--duration', type=int, default=30,
                       help='Benchmark duration in seconds (default: 30)')
    parser.add_argument('--source', type=int, default=0,
                       help='Video source for benchmark (default: 0)')
    
    args = parser.parse_args()
    
    if args.test == 'all':
        run_all_tests()
    elif args.test == 'cuda':
        test_cuda()
    elif args.test == 'model':
        test_model_inference()
    elif args.test == 'camera':
        test_camera()
    elif args.test == 'haarcascade':
        test_haarcascade()
    elif args.test == 'dlib':
        test_dlib()
    elif args.test == 'benchmark':
        benchmark_full_pipeline(args.source, args.duration)


if __name__ == "__main__":
    main()
