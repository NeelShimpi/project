"""
Enhanced Drowsiness Detection with Live Accuracy Tracking
This version includes real-time performance metrics, accuracy display, and visual graphs
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
from drowsiness_detection import DrowsinessDetector, eye_aspect_ratio
import dlib


class EnhancedDrowsinessDetector(DrowsinessDetector):
    """
    Enhanced version with live accuracy metrics and visualization
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extended metrics
        self.ear_timeline = deque(maxlen=300)  # Last 10 seconds at 30 FPS
        self.vit_timeline = deque(maxlen=300)
        self.drowsy_timeline = deque(maxlen=300)
        self.fps_timeline = deque(maxlen=300)
        
        # Accuracy tracking
        self.detection_confidence = deque(maxlen=100)
        self.avg_detection_confidence = 0.0
        
        # Performance metrics
        self.inference_times = {
            'ear': deque(maxlen=100),
            'vit': deque(maxlen=100),
            'total': deque(maxlen=100)
        }
        
    def calculate_detection_accuracy(self):
        """
        Calculate detection accuracy based on confidence and consistency
        """
        if len(self.detection_confidence) == 0:
            return 0.0
        
        # Calculate confidence-based accuracy
        confidence_score = np.mean(self.detection_confidence)
        
        # Calculate consistency score (how stable the detections are)
        if len(self.drowsy_timeline) > 10:
            recent_detections = list(self.drowsy_timeline)[-30:]
            changes = sum(1 for i in range(len(recent_detections)-1) 
                         if recent_detections[i] != recent_detections[i+1])
            consistency_score = 1.0 - (changes / len(recent_detections))
        else:
            consistency_score = 1.0
        
        # Combined accuracy score
        accuracy = (confidence_score * 0.6 + consistency_score * 0.4) * 100
        return min(accuracy, 100.0)
    
    def draw_enhanced_statistics(self, frame, ear_value, vit_prob, is_drowsy, fps, 
                                 ear_time, vit_time, total_time):
        """
        Draw comprehensive statistics with graphs
        """
        height, width = frame.shape[:2]
        
        # Update timelines
        self.ear_timeline.append(ear_value)
        self.vit_timeline.append(vit_prob)
        self.drowsy_timeline.append(1 if is_drowsy else 0)
        self.fps_timeline.append(fps)
        
        # Calculate detection confidence
        confidence = 1.0 - abs(ear_value - self.EAR_THRESHOLD) / self.EAR_THRESHOLD
        confidence = max(0.0, min(1.0, confidence))
        self.detection_confidence.append(confidence)
        
        # Update inference times
        self.inference_times['ear'].append(ear_time)
        self.inference_times['vit'].append(vit_time)
        self.inference_times['total'].append(total_time)
        
        # Calculate accuracy
        accuracy = self.calculate_detection_accuracy()
        
        # Create two panels
        # Panel 1: Statistics (left side)
        self._draw_statistics_panel(frame, ear_value, vit_prob, is_drowsy, fps, 
                                    accuracy, ear_time, vit_time)
        
        # Panel 2: Real-time graphs (bottom)
        if len(self.ear_timeline) > 10:
            self._draw_timeline_graphs(frame, width, height)
        
        return frame
    
    def _draw_statistics_panel(self, frame, ear_value, vit_prob, is_drowsy, fps,
                               accuracy, ear_time, vit_time):
        """Draw the main statistics panel"""
        overlay = frame.copy()
        
        # Panel dimensions
        panel_width = 350
        panel_height = 400
        panel_x = 10
        panel_y = 10
        
        # Draw panel background
        cv2.rectangle(overlay, (panel_x, panel_y), 
                     (panel_x + panel_width, panel_y + panel_height),
                     (0, 0, 0), -1)
        
        # Blend overlay
        alpha = 0.7
        frame_blend = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        frame[:] = frame_blend[:]
        
        # Calculate percentages
        if self.total_frames > 0:
            drowsy_percentage = (self.drowsy_frames / self.total_frames) * 100
            alert_percentage = (self.alert_frames / self.total_frames) * 100
        else:
            drowsy_percentage = 0
            alert_percentage = 100
        
        # Average metrics
        avg_ear = np.mean(self.ear_timeline) if self.ear_timeline else 0
        avg_vit = np.mean(self.vit_timeline) if self.vit_timeline else 0
        avg_fps = np.mean(self.fps_timeline) if self.fps_timeline else 0
        
        # Draw text
        y_offset = panel_y + 30
        line_height = 35
        
        # Title
        cv2.putText(frame, "PERFORMANCE METRICS", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        y_offset += line_height + 5
        
        # Detection Accuracy
        accuracy_color = (0, 255, 0) if accuracy > 80 else (0, 165, 255) if accuracy > 60 else (0, 0, 255)
        cv2.putText(frame, f"Detection Accuracy: {accuracy:.1f}%", 
                   (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, accuracy_color, 2)
        # Accuracy bar
        bar_length = int(accuracy * 2.5)
        cv2.rectangle(frame, (panel_x + 10, y_offset + 5), 
                     (panel_x + 10 + bar_length, y_offset + 15), accuracy_color, -1)
        y_offset += line_height
        
        # Current State
        state_text = "🔴 DROWSY" if is_drowsy else "🟢 ALERT"
        state_color = (0, 0, 255) if is_drowsy else (0, 255, 0)
        cv2.putText(frame, f"State: {state_text}", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
        y_offset += line_height
        
        # FPS
        fps_color = (0, 255, 0) if fps > 20 else (0, 165, 255) if fps > 10 else (0, 0, 255)
        cv2.putText(frame, f"FPS: {fps:.1f} (avg: {avg_fps:.1f})", 
                   (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2)
        y_offset += line_height
        
        # EAR Metrics
        cv2.putText(frame, "EYE ASPECT RATIO", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        ear_color = (0, 255, 0) if ear_value > self.EAR_THRESHOLD else (0, 0, 255)
        cv2.putText(frame, f"Current: {ear_value:.3f}", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, ear_color, 1)
        # EAR bar
        bar_length = int(min(ear_value / 0.4, 1.0) * 200)
        cv2.rectangle(frame, (panel_x + 150, y_offset - 12), 
                     (panel_x + 150 + bar_length, y_offset - 2), ear_color, -1)
        y_offset += 25
        
        cv2.putText(frame, f"Average: {avg_ear:.3f}", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Time: {ear_time:.1f}ms", (panel_x + 200, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        # ViT Metrics
        cv2.putText(frame, "VISION TRANSFORMER", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        vit_color = (0, 0, 255) if vit_prob > 0.7 else (0, 255, 0)
        cv2.putText(frame, f"Confidence: {vit_prob:.3f}", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, vit_color, 1)
        # ViT confidence bar
        bar_length = int(vit_prob * 200)
        cv2.rectangle(frame, (panel_x + 150, y_offset - 12), 
                     (panel_x + 150 + bar_length, y_offset - 2), vit_color, -1)
        y_offset += 25
        
        cv2.putText(frame, f"Average: {avg_vit:.3f}", (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        cv2.putText(frame, f"Time: {vit_time:.1f}ms", (panel_x + 200, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += line_height
        
        # Session Statistics
        cv2.putText(frame, "SESSION STATS", (panel_x + 10, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        y_offset += 25
        
        cv2.putText(frame, f"Total Frames: {self.total_frames}", 
                   (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        y_offset += 25
        
        cv2.putText(frame, f"Alert: {alert_percentage:.1f}%  Drowsy: {drowsy_percentage:.1f}%", 
                   (panel_x + 20, y_offset),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
    
    def _draw_timeline_graphs(self, frame, width, height):
        """Draw timeline graphs at the bottom"""
        graph_height = 120
        graph_y = height - graph_height - 10
        graph_width = width - 20
        
        # Create semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, graph_y), (width - 10, height - 10), (0, 0, 0), -1)
        alpha = 0.7
        frame_blend = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        frame[:] = frame_blend[:]
        
        # Draw EAR timeline
        self._draw_single_graph(frame, list(self.ear_timeline), 
                               10, graph_y + 10, graph_width // 2 - 10, 50,
                               "EAR Timeline", (0, 255, 0), self.EAR_THRESHOLD)
        
        # Draw ViT confidence timeline
        self._draw_single_graph(frame, list(self.vit_timeline),
                               width // 2, graph_y + 10, graph_width // 2 - 10, 50,
                               "ViT Confidence", (255, 255, 0), 0.7)
        
        # Draw drowsiness state timeline
        drowsy_states = [x * 100 for x in self.drowsy_timeline]
        self._draw_single_graph(frame, drowsy_states,
                               10, graph_y + 70, graph_width - 10, 40,
                               "Detection State (Red=Drowsy, Green=Alert)", (0, 165, 255))
    
    def _draw_single_graph(self, frame, data, x, y, width, height, title, color, threshold=None):
        """Draw a single timeline graph"""
        if len(data) < 2:
            return
        
        # Draw title
        cv2.putText(frame, title, (x + 5, y - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Draw border
        cv2.rectangle(frame, (x, y), (x + width, y + height), (100, 100, 100), 1)
        
        # Normalize data
        data_array = np.array(data)
        if data_array.max() > data_array.min():
            normalized = (data_array - data_array.min()) / (data_array.max() - data_array.min())
        else:
            normalized = np.ones_like(data_array) * 0.5
        
        # Scale to graph height
        scaled = (1 - normalized) * height
        
        # Draw points and lines
        points = []
        for i, value in enumerate(scaled):
            px = x + int((i / len(data)) * width)
            py = y + int(value)
            points.append((px, py))
        
        # Draw lines
        for i in range(len(points) - 1):
            # Color based on value
            if len(data) == len(self.drowsy_timeline):
                # Drowsy timeline - red for drowsy, green for alert
                line_color = (0, 0, 255) if data[i] > 50 else (0, 255, 0)
            else:
                line_color = color
            
            cv2.line(frame, points[i], points[i + 1], line_color, 2)
        
        # Draw threshold line if provided
        if threshold is not None and data_array.max() > 0:
            threshold_normalized = (threshold - data_array.min()) / (data_array.max() - data_array.min())
            threshold_y = y + int((1 - threshold_normalized) * height)
            cv2.line(frame, (x, threshold_y), (x + width, threshold_y), (0, 0, 255), 1, cv2.LINE_AA)
    
    def run_enhanced(self, video_source=0, use_vit=True):
        """
        Run enhanced detection with live metrics
        """
        cap = cv2.VideoCapture(video_source)
        
        if not cap.isOpened():
            print("Error: Could not open video source")
            return
        
        print("="*60)
        print("ENHANCED DROWSINESS DETECTION WITH LIVE METRICS")
        print("="*60)
        print("\nControls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("\nStarting detection...\n")
        
        fps_start_time = time.time()
        fps_counter = 0
        fps = 0
        
        while True:
            frame_start = time.time()
            ret, frame = cap.read()
            if not ret:
                break
            
            self.total_frames += 1
            
            # EAR detection with timing
            ear_start = time.time()
            frame, ear_drowsy = self.detect_drowsiness_ear(frame)
            ear_time = (time.time() - ear_start) * 1000
            
            # Get current EAR
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
            current_ear = 0.0
            
            if len(faces) > 0 and self.predictor is not None:
                x, y, w, h = faces[0]
                rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))
                shape = self.predictor(gray, rect)
                from imutils import face_utils
                shape = face_utils.shape_to_np(shape)
                
                left_eye = shape[self.LEFT_EYE_START:self.LEFT_EYE_END]
                right_eye = shape[self.RIGHT_EYE_START:self.RIGHT_EYE_END]
                
                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                current_ear = (left_ear + right_ear) / 2.0
            
            # ViT detection with timing
            vit_start = time.time()
            if use_vit:
                current_vit_prob, vit_class = self.detect_drowsiness_vit(frame)
                vit_drowsy = vit_class == 1
            else:
                current_vit_prob = 0.0
                vit_drowsy = False
            vit_time = (time.time() - vit_start) * 1000
            
            # Combined decision
            is_drowsy = ear_drowsy or vit_drowsy
            
            if is_drowsy:
                self.drowsy_frames += 1
            else:
                self.alert_frames += 1
            
            # Calculate FPS
            fps_counter += 1
            if (time.time() - fps_start_time) > 1:
                fps = fps_counter / (time.time() - fps_start_time)
                fps_counter = 0
                fps_start_time = time.time()
            
            # Total frame time
            total_time = (time.time() - frame_start) * 1000
            
            # Draw enhanced statistics
            frame = self.draw_enhanced_statistics(frame, current_ear, current_vit_prob,
                                                  is_drowsy, fps, ear_time, vit_time, total_time)
            
            # Display
            cv2.imshow('Enhanced Drowsiness Detection', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"drowsiness_screenshot_{int(time.time())}.png"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        # Print final statistics
        self.print_final_statistics()


if __name__ == "__main__":
    # Create enhanced detector
    detector = EnhancedDrowsinessDetector(use_cuda=True)
    
    # Run with enhanced visualization
    detector.run_enhanced(video_source=0, use_vit=True)
