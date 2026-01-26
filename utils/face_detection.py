"""
Face detection using DeepFace
Supports face detection, verification, and processing from images/videos
"""

import cv2
import numpy as np
from typing import Tuple, List, Optional, Dict
from deepface import DeepFace
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class FaceDetector:
    """
    Face detection using DeepFace
    Supports multiple detector backends: opencv, mtcnn, retinaface, mediapipe, ssd
    """
    def __init__(self, detector_backend: str = 'opencv', enforce_detection: bool = False):
        """
        Initialize FaceDetector
        Args:
            detector_backend: 'opencv', 'mtcnn', 'retinaface', 'mediapipe', or 'ssd'
                            'retinaface' is most accurate, 'opencv' is fastest
            enforce_detection: Raise error if no face detected
        """
        self.detector_backend = detector_backend
        self.enforce_detection = enforce_detection
        print(f"✓ DeepFace FaceDetector initialized with backend: {detector_backend}")
    
    def detect_faces(self, image: np.ndarray) -> List[dict]:
        """
        Detect faces in image using DeepFace
        Args:
            image: Input image in BGR format
        Returns:
            List of detected faces with bounding boxes and confidence
        """
        try:
            # Convert BGR to RGB for DeepFace
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Use DeepFace extract_faces for detection
            detected_faces = DeepFace.extract_faces(
                img_path=image_rgb,
                detector_backend=self.detector_backend,
                enforce_detection=False,  # Don't raise error if no face
                align=True  # Align faces to canonical position
            )
            
            faces = []
            for face_data in detected_faces:
                # Get bounding box from detected face
                x = face_data['facial_area']['x']
                y = face_data['facial_area']['y']
                w = face_data['facial_area']['w']
                h = face_data['facial_area']['h']
                
                # Convert to (x1, y1, x2, y2) format
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                
                # Add padding around face
                padding_x = int(0.1 * w)
                padding_y = int(0.1 * h)
                
                x1 = max(0, x1 - padding_x)
                y1 = max(0, y1 - padding_y)
                x2 = min(image.shape[1], x2 + padding_x)
                y2 = min(image.shape[0], y2 + padding_y)
                
                # Get confidence score
                confidence = face_data.get('confidence', 0.95)
                
                faces.append({
                    'bbox': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'face_data': face_data  # Store full face data for verification
                })
            
            return faces
            
        except Exception as e:
            print(f"Warning: Face detection error: {e}")
            return []
    
    def verify_faces(self, img1: np.ndarray, img2: np.ndarray, 
                    model_name: str = 'ArcFace', 
                    distance_metric: str = 'cosine') -> Dict:
        """
        Verify if two faces belong to the same person
        Args:
            img1: First image (BGR)
            img2: Second image (BGR)
            model_name: 'ArcFace', 'Facenet', 'VGG-Face', 'DeepID', 'ArcFace2', 'SFace'
            distance_metric: 'cosine', 'euclidean', 'euclidean_l2'
        Returns:
            Verification result with verified boolean and distance
        """
        try:
            # Convert BGR to RGB
            img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            result = DeepFace.verify(
                img1_path=img1_rgb,
                img2_path=img2_rgb,
                model_name=model_name,
                detector_backend=self.detector_backend,
                distance_metric=distance_metric,
                enforce_detection=False
            )
            
            return result
            
        except Exception as e:
            print(f"Warning: Face verification error: {e}")
            return {'verified': False, 'distance': float('inf')}
    
    def recognize_face(self, face_image: np.ndarray, 
                      reference_faces: List[np.ndarray],
                      model_name: str = 'ArcFace',
                      distance_metric: str = 'cosine',
                      threshold: float = 0.6) -> Tuple[bool, float, int]:
        """
        Recognize a face against multiple reference faces
        Args:
            face_image: Face to recognize (BGR)
            reference_faces: List of reference face images (BGR)
            model_name: Face recognition model
            distance_metric: Distance metric to use
            threshold: Similarity threshold (lower is more similar)
        Returns:
            Tuple of (is_match, min_distance, matched_index)
        """
        if not reference_faces:
            return False, float('inf'), -1
        
        try:
            face_rgb = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            min_distance = float('inf')
            matched_index = -1
            
            for idx, ref_face in enumerate(reference_faces):
                ref_face_rgb = cv2.cvtColor(ref_face, cv2.COLOR_BGR2RGB)
                
                result = DeepFace.verify(
                    img1_path=face_rgb,
                    img2_path=ref_face_rgb,
                    model_name=model_name,
                    detector_backend=self.detector_backend,
                    distance_metric=distance_metric,
                    enforce_detection=False
                )
                
                distance = result['distance']
                if distance < min_distance:
                    min_distance = distance
                    matched_index = idx
            
            is_match = min_distance < threshold
            return is_match, min_distance, matched_index
            
        except Exception as e:
            print(f"Warning: Face recognition error: {e}")
            return False, float('inf'), -1
    
    def crop_face(self, image: np.ndarray, bbox: Tuple) -> np.ndarray:
        """
        Crop face region from image
        Args:
            image: Input image
            bbox: Bounding box (x1, y1, x2, y2)
        Returns:
            Cropped face image
        """
        x1, y1, x2, y2 = bbox
        return image[y1:y2, x1:x2].copy()
    
    def draw_faces(self, image: np.ndarray, faces: List[dict], 
                  confidences: Optional[List[float]] = None) -> np.ndarray:
        """
        Draw bounding boxes on image
        Args:
            image: Input image
            faces: List of detected faces with bboxes
            confidences: Optional list of confidence scores for each face
        Returns:
            Image with drawn bounding boxes
        """
        image_copy = image.copy()
        h, w = image_copy.shape[:2]
        
        for idx, face in enumerate(faces):
            x1, y1, x2, y2 = face['bbox']
            
            # Determine color based on confidence
            if confidences and idx < len(confidences):
                conf = confidences[idx]
                color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)  # Green for Live, Red for Fake
                label = f"Live: {conf:.2%}" if conf > 0.5 else f"Fake: {(1-conf):.2%}"
            else:
                color = (0, 255, 0)
                label = f"Face {idx + 1}"
            
            # Draw rectangle
            cv2.rectangle(image_copy, (x1, y1), (x2, y2), color, 3)
            
            # Draw label with background
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]
            
            # Background rectangle for text
            cv2.rectangle(image_copy, 
                         (x1, y1 - text_size[1] - 10),
                         (x1 + text_size[0], y1),
                         color, -1)
            
            # Text
            cv2.putText(image_copy, label, (x1, y1 - 5),
                       font, font_scale, (255, 255, 255), thickness)
            
            # DeepFace confidence if available
            if 'confidence' in face:
                conf_text = f"Conf: {face['confidence']:.2%}"
                cv2.putText(image_copy, conf_text, (x1, y2 + 25),
                           font, 0.6, color, 2)
        
        return image_copy
    
    def get_face_quality(self, image: np.ndarray, bbox: Tuple) -> float:
        """
        Assess face quality based on image properties
        Args:
            image: Input image
            bbox: Face bounding box
        Returns:
            Quality score (0-1)
        """
        face = self.crop_face(image, bbox)
        
        if face.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
        
        # Estimate blur using Laplacian variance
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = min(1.0, laplacian_var / 100.0)  # Normalized
        
        # Estimate brightness
        brightness = np.mean(gray) / 255.0
        brightness_score = 1.0 - abs(brightness - 0.5) * 2  # Optimal at 0.5
        
        # Estimate contrast
        contrast = np.std(gray) / 128.0
        contrast_score = min(1.0, contrast)
        
        # Combine scores
        quality = (blur_score * 0.4 + brightness_score * 0.3 + contrast_score * 0.3)
        return quality


class MultiiFaceProcessor:
    """
    Process multiple faces in an image/frame using DeepFace
    """
    def __init__(self, detector_backend: str = 'opencv'):
        self.face_detector = FaceDetector(detector_backend=detector_backend)
    
    def process_frame(self, frame: np.ndarray) -> Tuple[List[np.ndarray], List[Tuple]]:
        """
        Extract all faces from frame
        Args:
            frame: Input frame
        Returns:
            Tuple of (face crops, bounding boxes)
        """
        faces = self.face_detector.detect_faces(frame)
        
        face_crops = []
        bboxes = []
        for face in faces:
            bbox = face['bbox']
            crop = self.face_detector.crop_face(frame, bbox)
            face_crops.append(crop)
            bboxes.append(bbox)
        
        return face_crops, bboxes
    
    def process_video(self, video_path: str, frame_skip: int = 1) -> Tuple[List[np.ndarray], List[List[Tuple]]]:
        """
        Extract faces from video frames
        Args:
            video_path: Path to video file
            frame_skip: Process every Nth frame (1 = all frames, 2 = every 2nd frame, etc.)
        Returns:
            Tuple of (all frames, face bboxes per frame)
        """
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return [], []
        
        frames = []
        all_bboxes = []
        frame_count = 0
        
        print(f"Processing video: {video_path}")
        
        while cap.isOpened():
            ret, frame = cap.read()
            
            if not ret:
                break
            
            frame_count += 1
            
            # Skip frames if specified
            if frame_count % frame_skip != 0:
                continue
            
            # Detect faces in frame
            faces = self.face_detector.detect_faces(frame)
            bboxes = [face['bbox'] for face in faces]
            
            frames.append(frame)
            all_bboxes.append(bboxes)
            
            if frame_count % 30 == 0:
                print(f"  Processed {frame_count} frames, extracted {len(bboxes)} faces")
        
        cap.release()
        
        print(f"✓ Video processing complete: {frame_count} total frames, {len(frames)} processed")
        
        return frames, all_bboxes
    
    def assess_all_faces(self, frame: np.ndarray) -> List[dict]:
        """
        Assess quality of all detected faces
        Args:
            frame: Input frame
        Returns:
            List of face info with quality scores
        """
        faces = self.face_detector.detect_faces(frame)
        
        face_info = []
        for face in faces:
            quality = self.face_detector.get_face_quality(frame, face['bbox'])
            face_info.append({
                'bbox': face['bbox'],
                'confidence': face['confidence'],
                'quality': quality
            })
        
        return face_info


# Example usage
if __name__ == "__main__":
    # Example 1: Detect faces in an image
    print("Example 1: Detect faces in image")
    detector = FaceDetector(detector_backend='retinaface')
    # image = cv2.imread('test_image.jpg')
    # faces = detector.detect_faces(image)
    # print(f"Found {len(faces)} faces")
    
    # Example 2: Process video and get faces
    print("\nExample 2: Process video")
    processor = MultiiFaceProcessor(detector_backend='opencv')
    # frames, all_bboxes = processor.process_video('test_video.mp4', frame_skip=5)
    # print(f"Extracted {len(frames)} frames from video")
