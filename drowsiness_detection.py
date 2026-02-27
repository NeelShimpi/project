import cv2
import torch
import torch.nn as nn
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib
import time
import warnings
warnings.filterwarnings('ignore')

# Eye Aspect Ratio calculation
def eye_aspect_ratio(eye):
    """
    Calculate the Eye Aspect Ratio (EAR)
    EAR = (||p2-p6|| + ||p3-p5||) / (2||p1-p4||)
    """
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear


class DrowsinessDetector:
    def __init__(self, 
                 haar_cascade_path='haarcascade_frontalface_default.xml',
                 shape_predictor_path='shape_predictor_68_face_landmarks.dat',
                 vit_model_path=r"C:\Users\neels\Desktop\AIML repositories\project\drowsiness_model.pth",
                 use_cuda=True):
        """
        Initialize the drowsiness detection system
        
        Args:
            haar_cascade_path: Path to Haar Cascade XML file
            shape_predictor_path: Path to dlib's facial landmark predictor
            vit_model_path: Path to trained Vision Transformer model
            use_cuda: Whether to use CUDA for GPU acceleration
        """
        # CUDA setup
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + haar_cascade_path)
        
        # Dlib for facial landmarks
        self.detector = dlib.get_frontal_face_detector()
        try:
            self.predictor = dlib.shape_predictor(shape_predictor_path)
        except:
            print("Warning: dlib shape predictor not found. Download from:")
            print("http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")
            self.predictor = None
        
        # Eye indices for 68-point facial landmarks
        self.LEFT_EYE_START, self.LEFT_EYE_END = 42, 48
        self.RIGHT_EYE_START, self.RIGHT_EYE_END = 36, 42
        
        # EAR thresholds
        self.EAR_THRESHOLD = 0.25
        self.EAR_CONSEC_FRAMES = 20
        self.frame_counter = 0
        self.drowsy_flag = False
        
        # Load Vision Transformer model
        from vit_model import VisionTransformerDrowsiness
        self.vit_model = VisionTransformerDrowsiness(
            img_size=224,
            patch_size=16,
            num_classes=4,
            dim=768,
            depth=12,
            heads=12,
            mlp_dim=3072
        ).to(self.device)
        
        # Load pretrained weights if available
        try:
            self.vit_model.load_state_dict(torch.load(vit_model_path, map_location=self.device))
            self.vit_model.eval()
            print("Vision Transformer model loaded successfully")
        except:
            print("Warning: No pretrained ViT model found. Using random initialization.")
            print("For production, train the model first using train_vit.py")
        
        # Statistics
        self.total_frames = 0
        self.drowsy_frames = 0
        self.alert_frames = 0
        self.true_positives = 0
        self.true_negatives = 0
        self.false_positives = 0
        self.false_negatives = 0
        
        # Performance metrics
        self.ear_history = []
        self.vit_confidence_history = []
        self.detection_accuracy = 0.0
        
    
    def detect_drowsiness_ear(self, frame):
        """
        Detect drowsiness using Eye Aspect Ratio method
        
        Args:
            frame: Input video frame
            
        Returns:
            frame: Annotated frame
            is_drowsy: Boolean indicating drowsiness
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces using Haar Cascade
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
        
        is_drowsy = False
        
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            if self.predictor is None:
                continue
            
            # Get facial landmarks
            rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
            shape = self.predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            
            # Extract eye coordinates
            left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
            right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
            
            # Calculate EAR for both eyes
            left_ear = eye_aspect_ratio(left_eye)
            right_ear = eye_aspect_ratio(right_eye)
            ear = (left_ear + right_ear) / 2.0
            
            # Draw eye contours
            left_eye_hull = cv2.convexHull(left_eye)
            right_eye_hull = cv2.convexHull(right_eye)
            cv2.drawContours(frame, [left_eye_hull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [right_eye_hull], -1, (0, 255, 0), 1)
            
            # Check for drowsiness
            if ear < self.EAR_THRESHOLD:
                self.frame_counter += 1
                
                if self.frame_counter >= self.EAR_CONSEC_FRAMES:
                    is_drowsy = True
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                self.frame_counter = 0
            
            # Display EAR value
            cv2.putText(frame, f"EAR: {ear:.2f}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return frame, is_drowsy
    
    
    def detect_drowsiness_vit(self, frame):
        """
        Detect drowsiness using Vision Transformer
        
        Args:
            frame: Input video frame
            
        Returns:
            prediction: Drowsiness probability
            class_label: 0 (alert) or 1 (drowsy)
        """
        # Preprocess frame for ViT
        face_img = cv2.resize(frame, (224, 224))
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype(np.float32) / 255.0
        
        # Normalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        face_img = (face_img - mean) / std
        
        # Convert to tensor and add batch dimension
        face_tensor = torch.from_numpy(face_img).permute(2, 0, 1).unsqueeze(0).float()
        face_tensor = face_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.vit_model(face_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            drowsy_prob = probabilities[0][1].item()
            class_label = torch.argmax(probabilities, dim=1).item()
        
        return drowsy_prob, class_label
    
    
    def print_final_statistics(self):
        """Print comprehensive final statistics"""
        print("\n" + "="*60)
        print("FINAL DETECTION STATISTICS")
        print("="*60)
        
        if self.total_frames > 0:
            drowsy_percentage = (self.drowsy_frames / self.total_frames) * 100
            alert_percentage = (self.alert_frames / self.total_frames) * 100
            
            print(f"\nTotal Frames Processed: {self.total_frames}")
            print(f"Alert Frames: {self.alert_frames} ({alert_percentage:.2f}%)")
            print(f"Drowsy Frames: {self.drowsy_frames} ({drowsy_percentage:.2f}%)")
            
            if self.ear_history:
                print(f"\nEye Aspect Ratio Statistics:")
                print(f"  Average EAR: {np.mean(self.ear_history):.3f}")
                print(f"  Min EAR: {np.min(self.ear_history):.3f}")
                print(f"  Max EAR: {np.max(self.ear_history):.3f}")
                print(f"  Std Dev: {np.std(self.ear_history):.3f}")
            
            if self.vit_confidence_history:
                print(f"\nViT Confidence Statistics:")
                print(f"  Average Confidence: {np.mean(self.vit_confidence_history):.3f}")
                print(f"  Min Confidence: {np.min(self.vit_confidence_history):.3f}")
                print(f"  Max Confidence: {np.max(self.vit_confidence_history):.3f}")
            
            # Detection performance
            if drowsy_percentage > 50:
                print(f"\n⚠️  WARNING: High drowsiness detected ({drowsy_percentage:.1f}%)")
                print("   Consider taking a break!")
            elif drowsy_percentage > 20:
                print(f"\n⚠️  CAUTION: Moderate drowsiness detected ({drowsy_percentage:.1f}%)")
                print("   Stay alert!")
            else:
                print(f"\n✓  GOOD: Low drowsiness level ({drowsy_percentage:.1f}%)")
                print("   Keep up the good driving!")
        else:
            print("\nNo frames processed.")
        
        print("\n" + "="*60 + "\n")
    
    
    def draw_statistics_overlay(self, frame, ear_value, vit_prob, is_drowsy, fps):
        """
        Draw statistics overlay on the frame
        
        Args:
            frame: Input frame
            ear_value: Current EAR value
            vit_prob: ViT drowsiness probability
            is_drowsy: Current drowsiness state
            fps: Current FPS
        """
        # Create semi-transparent overlay
        overlay = frame.copy()
        height, width = frame.shape[:2]
        
        # Stats panel dimensions
        panel_width = 300
        panel_height = 250
        panel_x = width - panel_width - 10
        panel_y = 10
        
        # Draw panel background
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        
        # Blend overlay
        alpha = 0.6
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Calculate statistics
        if self.total_frames > 0:
            drowsy_percentage = (self.drowsy_frames / self.total_frames) * 100
            alert_percentage = 100 - drowsy_percentage
            detection_rate = (self.alert_frames / self.total_frames) * 100
        else:
            drowsy_percentage = 0
            alert_percentage = 100
            detection_rate = 0
        
        # Store history for accuracy calculation
        if len(self.ear_history) > 100:
            self.ear_history.pop(0)
        if len(self.vit_confidence_history) > 100:
            self.vit_confidence_history.pop(0)
        
        self.ear_history.append(ear_value)
        self.vit_confidence_history.append(vit_prob)
        
        # Calculate average confidence
        avg_ear = np.mean(self.ear_history) if self.ear_history else 0
        avg_vit_conf = np.mean(self.vit_confidence_history) if self.vit_confidence_history else 0
        
        # Draw statistics text
        y_offset = panel_y + 25
        line_height = 30
        
        # Title
        cv2.putText(frame, "STATISTICS", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += line_height
        
        # FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        # Current state with color
        state_text = "DROWSY" if is_drowsy else "ALERT"
        state_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.putText(frame, f"State: {state_text}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, state_color, 2)
        y_offset += line_height
        
        # EAR value with bar
        cv2.putText(frame, f"EAR: {ear_value:.3f}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # EAR bar (normalized to 0-1)
        bar_length = int(min(ear_value / 0.4, 1.0) * 150)
        bar_color = (0, 255, 0) if ear_value > self.EAR_THRESHOLD else (0, 0, 255)
        cv2.rectangle(frame, (panel_x + 120, y_offset - 10), 
                     (panel_x + 120 + bar_length, y_offset), bar_color, -1)
        y_offset += line_height
        
        # ViT confidence with bar
        cv2.putText(frame, f"ViT Conf: {vit_prob:.3f}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # ViT confidence bar
        bar_length = int(vit_prob * 150)
        bar_color = (0, 0, 255) if vit_prob > 0.7 else (0, 255, 0)
        cv2.rectangle(frame, (panel_x + 120, y_offset - 10), 
                     (panel_x + 120 + bar_length, y_offset), bar_color, -1)
        y_offset += line_height
        
        # Detection accuracy
        cv2.putText(frame, f"Alert: {alert_percentage:.1f}%", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        y_offset += line_height
        
        cv2.putText(frame, f"Drowsy: {drowsy_percentage:.1f}%", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        y_offset += line_height
        
        # Total frames
        cv2.putText(frame, f"Frames: {self.total_frames}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame
    
    
    def run(self, video_source=0, use_vit=True, show_stats=True):
        """
        Run the drowsiness detection system
        
        Args:
            video_source: Camera index or video file path
            use_vit: Whether to use Vision Transformer in addition to EAR
            show_stats: Whether to show statistics overlay
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("Starting drowsiness detection...")
        print("Press 'q' to quit")
        print("Press 's' to toggle statistics overlay")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        current_ear = 0.0
        current_vit_prob = 0.0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            self.total_frames += 1
            
            # EAR-based detection
            frame, ear_drowsy = self.detect_drowsiness_ear(frame)
            
            # Get current EAR value from the last detected face
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            
            if len(faces) > 0 and self.predictor is not None:
                x, y, w, h = faces[0]
                rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                shape = self.predictor(gray, rect)
                shape = face_utils.shape_to_np(shape)
                
                left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
                right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                current_ear = (left_ear + right_ear) / 2.0
            
            # ViT-based detection
            if use_vit:
                current_vit_prob, vit_class = self.detect_drowsiness_vit(frame)
                vit_drowsy = vit_class == 1
            else:
                vit_drowsy = False
                current_vit_prob = 0.0
            
            # Combined decision (either method detects drowsiness)
            is_drowsy = ear_drowsy or vit_drowsy
            
            if is_drowsy:
                self.drowsy_frames += 1
                self.drowsy_flag = True
                # Large warning text
                cv2.putText(frame, "*** DROWSINESS ALERT! ***", (10, 60),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
                cv2.putText(frame, "*** WAKE UP! ***", (10, 110),
                           cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)
            else:
                self.alert_frames += 1
            
            # Calculate and display FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Draw statistics overlay
            if show_stats:
                frame = self.draw_statistics_overlay(frame, current_ear, current_vit_prob, is_drowsy, fps)
            else:
                # Just show FPS if stats are hidden
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Driver Drowsiness Detection', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                show_stats = not show_stats
                print(f"Statistics overlay: {'ON' if show_stats else 'OFF'}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_final_statistics()


if __name__ == "__main__":
    # Initialize detector
    detector = DrowsinessDetector(use_cuda=True)
    
    # Run detection on webcam (0) or video file path
    detector.run(video_source=0, use_vit=True)

import os

print("Model exists:", os.path.exists(vit_model_path))
print("Model path:", vit_model_path)