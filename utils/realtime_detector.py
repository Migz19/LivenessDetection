"""
Real-time face detection streaming using MediaPipe
Useful for webcam or video streaming scenarios
"""

import cv2
import mediapipe as mp
import numpy as np
from typing import Optional, Tuple, Callable

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils


class RealTimeFaceDetector:
    """
    Real-time face detection for webcam/video streaming
    Uses MediaPipe's optimized streaming approach
    """
    
    def __init__(self, model_selection: int = 0, min_detection_confidence: float = 0.5):
        """
        Initialize real-time detector
        Args:
            model_selection: 0 for short-range, 1 for full-range
            min_detection_confidence: Confidence threshold (0-1)
        """
        self.face_detection = mp_face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence
        )
    
    def process_webcam(self, 
                      callback: Optional[Callable] = None,
                      show_fps: bool = True,
                      flip_horizontal: bool = True) -> None:
        """
        Process webcam stream with real-time face detection
        Args:
            callback: Optional function to call on each frame (frame_with_detections, faces)
            show_fps: Display FPS counter
            flip_horizontal: Flip image horizontally (selfie view)
        """
        cap = cv2.VideoCapture(0)
        
        # FPS tracking
        frame_count = 0
        fps = 0
        prev_frame_time = 0
        
        print("Starting webcam... Press 'q' to quit")
        
        while cap.isOpened():
            success, image = cap.read()
            
            if not success:
                print("Ignoring empty camera frame.")
                continue
            
            # Get frame dimensions
            h, w, c = image.shape
            
            # Process frame
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            # Convert back to BGR
            image.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw detections
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image_bgr, detection)
            
            # Extract face bboxes for callback
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * w)
                    y1 = int(bbox.ymin * h)
                    x2 = int((bbox.xmin + bbox.width) * w)
                    y2 = int((bbox.ymin + bbox.height) * h)
                    
                    # Add padding
                    padding = int(0.1 * (x2 - x1))
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(w, x2 + padding)
                    y2 = min(h, y2 + padding)
                    
                    faces.append((x1, y1, x2, y2))
            
            # Call callback if provided
            if callback:
                image_bgr = callback(image_bgr, faces)
            
            # Calculate and display FPS
            if show_fps:
                current_time = cv2.getTickCount()
                frame_count += 1
                
                if frame_count % 10 == 0:
                    fps = 10.0 * cv2.getTickFrequency() / (current_time - prev_frame_time)
                    prev_frame_time = current_time
                
                cv2.putText(image_bgr, f"FPS: {fps:.1f}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(image_bgr, f"Faces: {len(faces)}", (10, 70),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Flip for selfie view
            if flip_horizontal:
                image_bgr = cv2.flip(image_bgr, 1)
            
            # Display
            cv2.imshow('MediaPipe Face Detection', image_bgr)
            
            # Break on 'q'
            if cv2.waitKey(5) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed")
    
    def process_video_file(self, 
                          video_path: str,
                          callback: Optional[Callable] = None,
                          show_fps: bool = True,
                          save_output: Optional[str] = None) -> None:
        """
        Process video file with face detection
        Args:
            video_path: Path to video file
            callback: Optional function to call on each frame
            show_fps: Display FPS counter
            save_output: Optional path to save output video
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            return
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Processing video: {width}x{height} @ {fps} FPS ({total_frames} frames)")
        
        # Output video writer if specified
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
        
        frame_count = 0
        
        while cap.isOpened():
            success, image = cap.read()
            
            if not success:
                break
            
            frame_count += 1
            
            # Process frame
            image.flags.writeable = False
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            results = self.face_detection.process(image_rgb)
            
            # Convert back to BGR
            image.flags.writeable = True
            image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            
            # Draw detections
            if results.detections:
                for detection in results.detections:
                    mp_drawing.draw_detection(image_bgr, detection)
            
            # Extract face bboxes
            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    x1 = int(bbox.xmin * width)
                    y1 = int(bbox.ymin * height)
                    x2 = int((bbox.xmin + bbox.width) * width)
                    y2 = int((bbox.ymin + bbox.height) * height)
                    
                    padding = int(0.1 * (x2 - x1))
                    x1 = max(0, x1 - padding)
                    y1 = max(0, y1 - padding)
                    x2 = min(width, x2 + padding)
                    y2 = min(height, y2 + padding)
                    
                    faces.append((x1, y1, x2, y2))
            
            # Call callback
            if callback:
                image_bgr = callback(image_bgr, faces)
            
            # Add progress info
            if show_fps:
                cv2.putText(image_bgr, f"Frame: {frame_count}/{total_frames}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(image_bgr, f"Faces: {len(faces)}", (10, 65),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Write to output file
            if writer:
                writer.write(image_bgr)
            
            # Display (optional)
            cv2.imshow('Face Detection', image_bgr)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if writer:
            writer.release()
        cv2.destroyAllWindows()
        
        print(f"Processing complete. Processed {frame_count} frames")
        if save_output:
            print(f"Output saved to: {save_output}")
    
    def __del__(self):
        """Cleanup"""
        try:
            self.face_detection.close()
        except:
            pass


# Example usage functions
def example_webcam():
    """Run webcam example"""
    detector = RealTimeFaceDetector(model_selection=0, min_detection_confidence=0.5)
    
    def custom_callback(frame, faces):
        """Custom callback to draw additional info"""
        for idx, (x1, y1, x2, y2) in enumerate(faces):
            cv2.putText(frame, f"Face {idx+1}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
    
    detector.process_webcam(callback=custom_callback, show_fps=True, flip_horizontal=True)


def example_video(video_path: str):
    """Run video example"""
    detector = RealTimeFaceDetector(model_selection=0, min_detection_confidence=0.5)
    detector.process_video_file(video_path, show_fps=True)


if __name__ == "__main__":
    # Run webcam example
    example_webcam()
