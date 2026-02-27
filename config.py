"""
Configuration file for Drowsiness Detection System
Modify these parameters to customize the detection behavior
"""

# ==================== DETECTION PARAMETERS ====================

# Eye Aspect Ratio (EAR) Configuration
EAR_CONFIG = {
    'threshold': 0.25,              # EAR threshold (lower = more sensitive)
    'consecutive_frames': 20,        # Frames below threshold to trigger alert
    'eye_ar_consec_frames': 3,      # Frames to confirm eye is closed
}

# ==================== MODEL PARAMETERS ====================

# Vision Transformer Configuration
VIT_CONFIG = {
    'img_size': 224,                 # Input image size
    'patch_size': 16,                # Patch size for ViT
    'num_classes': 2,                # Number of classes (alert/drowsy)
    'dim': 768,                      # Embedding dimension
    'depth': 12,                     # Number of transformer blocks
    'heads': 12,                     # Number of attention heads
    'mlp_dim': 3072,                 # MLP hidden dimension
    'dropout': 0.1,                  # Dropout rate
    'use_light_model': False,        # Use lightweight version for speed
}

# Lightweight ViT Configuration (for faster inference)
VIT_LIGHT_CONFIG = {
    'img_size': 224,
    'patch_size': 16,
    'num_classes': 2,
    'dim': 384,                      # Reduced dimension
    'depth': 6,                      # Reduced depth
    'heads': 6,                      # Reduced heads
    'mlp_dim': 1536,                 # Reduced MLP dimension
    'dropout': 0.1,
}

# ==================== FILE PATHS ====================

PATHS = {
    'haar_cascade': 'haarcascade_frontalface_default.xml',
    'shape_predictor': 'shape_predictor_68_face_landmarks.dat',
    'vit_model': 'vit_drowsiness_model.pth',
    'vit_light_model': 'vit_drowsiness_light_model.pth',
}

# ==================== HARDWARE CONFIGURATION ====================

HARDWARE_CONFIG = {
    'use_cuda': True,                # Enable CUDA/GPU acceleration
    'device': 'cuda',                # 'cuda' or 'cpu'
    'pin_memory': True,              # Pin memory for faster GPU transfer
}

# ==================== CAMERA CONFIGURATION ====================

CAMERA_CONFIG = {
    'source': 0,                     # Camera index (0 for default webcam)
    'width': 640,                    # Camera resolution width
    'height': 480,                   # Camera resolution height
    'fps': 30,                       # Desired FPS (if supported)
}

# ==================== DISPLAY CONFIGURATION ====================

DISPLAY_CONFIG = {
    'show_fps': True,                # Show FPS counter
    'show_ear': True,                # Show EAR value
    'show_landmarks': True,          # Show facial landmarks
    'show_eye_contours': True,       # Show eye contours
    'show_vit_prob': True,           # Show ViT drowsiness probability
    'alert_color': (0, 0, 255),      # BGR color for alerts (red)
    'normal_color': (0, 255, 0),     # BGR color for normal state (green)
    'text_color': (255, 255, 255),   # BGR color for text (white)
}

# ==================== ALERT CONFIGURATION ====================

ALERT_CONFIG = {
    'visual_alert': True,            # Show visual alert on screen
    'audio_alert': False,            # Play audio alert (requires setup)
    'alert_duration': 2.0,           # Alert duration in seconds
    'blink_alert': True,             # Blink the alert message
}

# ==================== TRAINING CONFIGURATION ====================

TRAINING_CONFIG = {
    'train_dir': 'dataset/train',
    'val_dir': 'dataset/val',
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'num_workers': 4,
    'save_best_only': True,
    'early_stopping_patience': 10,
}

# ==================== DATA AUGMENTATION ====================

AUGMENTATION_CONFIG = {
    'horizontal_flip': True,
    'flip_probability': 0.5,
    'rotation_range': 10,
    'brightness_range': 0.2,
    'contrast_range': 0.2,
    'saturation_range': 0.2,
}

# ==================== LOGGING CONFIGURATION ====================

LOGGING_CONFIG = {
    'log_level': 'INFO',             # DEBUG, INFO, WARNING, ERROR
    'save_logs': True,
    'log_file': 'drowsiness_detection.log',
    'log_statistics': True,
}

# ==================== PERFORMANCE TUNING ====================

PERFORMANCE_CONFIG = {
    'use_vit': True,                 # Enable ViT model (slower but more accurate)
    'vit_batch_inference': False,    # Batch process for ViT (if applicable)
    'skip_frames': 0,                # Skip N frames for faster processing (0 = no skip)
    'face_detection_frequency': 1,   # Run face detection every N frames
    'optimize_for_speed': False,     # Optimize for speed over accuracy
}

# ==================== FACIAL LANDMARKS ====================

FACIAL_LANDMARKS = {
    'left_eye_start': 42,
    'left_eye_end': 48,
    'right_eye_start': 36,
    'right_eye_end': 42,
    'mouth_start': 48,
    'mouth_end': 68,
    'jaw_start': 0,
    'jaw_end': 17,
}

# ==================== THRESHOLDS ====================

THRESHOLDS = {
    'ear_alert': 0.25,               # EAR threshold for alert
    'ear_drowsy': 0.20,              # EAR threshold for drowsy
    'vit_drowsy': 0.70,              # ViT probability threshold for drowsy
    'vit_alert': 0.30,               # ViT probability threshold for alert
}

# ==================== EXPORT CONFIGURATION ====================

EXPORT_CONFIG = {
    'save_detections': False,        # Save detection results to file
    'output_dir': 'outputs',
    'save_video': False,             # Save processed video
    'video_codec': 'mp4v',           # Video codec for saving
}


# ==================== HELPER FUNCTIONS ====================

def get_config(config_name):
    """Get configuration by name"""
    configs = {
        'ear': EAR_CONFIG,
        'vit': VIT_CONFIG,
        'vit_light': VIT_LIGHT_CONFIG,
        'paths': PATHS,
        'hardware': HARDWARE_CONFIG,
        'camera': CAMERA_CONFIG,
        'display': DISPLAY_CONFIG,
        'alert': ALERT_CONFIG,
        'training': TRAINING_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'performance': PERFORMANCE_CONFIG,
        'landmarks': FACIAL_LANDMARKS,
        'thresholds': THRESHOLDS,
        'export': EXPORT_CONFIG,
    }
    return configs.get(config_name, {})


def print_config():
    """Print all configurations"""
    print("=" * 60)
    print("DROWSINESS DETECTION SYSTEM - CONFIGURATION")
    print("=" * 60)
    
    configs = [
        ('EAR Configuration', EAR_CONFIG),
        ('ViT Configuration', VIT_CONFIG),
        ('Hardware Configuration', HARDWARE_CONFIG),
        ('Camera Configuration', CAMERA_CONFIG),
        ('Display Configuration', DISPLAY_CONFIG),
        ('Alert Configuration', ALERT_CONFIG),
        ('Performance Configuration', PERFORMANCE_CONFIG),
    ]
    
    for name, config in configs:
        print(f"\n{name}:")
        for key, value in config.items():
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    print_config()
