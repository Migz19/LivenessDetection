"""
Integrated inference with Temporal Smoothing.

Add to existing inference.py or use directly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import cv2

# Import temporal smoother
try:
    from temporal_smoother import TemporalSmoothingPipeline
except ImportError:
    TemporalSmoothingPipeline = None


class SmoothedLivenessInference:
    """
    CNN-based inference + Temporal Smoothing.
    
    Replaces mean pooling with attention-weighted smoothing.
    Frozen transformer learns which frames are reliable.
    """
    
    def __init__(self, 
                 model, 
                 preprocessor, 
                 device='cpu', 
                 use_enhanced_features=False,
                 use_temporal_smoothing=True,
                 window_size=8,
                 confidence_threshold=0.5):
        """
        Args:
            model: CNN model
            preprocessor: Image preprocessor
            device: 'cpu' or 'cuda'
            use_enhanced_features: Use liveness features
            use_temporal_smoothing: Enable temporal smoothing
            window_size: Frames to smooth over
            confidence_threshold: Decision threshold
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.use_enhanced_features = use_enhanced_features
        self.use_temporal_smoothing = use_temporal_smoothing and TemporalSmoothingPipeline is not None
        self.confidence_threshold = confidence_threshold
        
        self.model.eval()
        self.model.to(device)
        
        # Create temporal smoother if enabled
        if self.use_temporal_smoothing:
            self.smoother_pipeline = TemporalSmoothingPipeline(
                window_size=window_size,
                confidence_threshold=confidence_threshold
            )
            self.smoother_pipeline.smoother.to(device)
        else:
            self.smoother_pipeline = None
    
    def _get_frame_confidence(self, image_array: np.ndarray, 
                             face_bbox: Optional[Tuple] = None) -> float:
        """Get CNN confidence for single frame"""
        # Preprocess
        if self.use_enhanced_features:
            tensors = self.preprocessor.preprocess_with_liveness_features(
                image_array, face_bbox
            )
            tensors = tuple(t.to(self.device) for t in tensors)
        else:
            if hasattr(self.preprocessor, 'preprocess_array'):
                tensor = self.preprocessor.preprocess_array(image_array, face_bbox)
            else:
                tensor, *_ = self.preprocessor.preprocess_with_liveness_features(
                    image_array, face_bbox
                )
            tensor = tensor.to(self.device)
            tensors = (tensor,)
        
        # Inference
        with torch.no_grad():
            if len(tensors) > 1:
                output = self.model(*tensors)
            else:
                output = self.model(tensors[0])
            
            # Get confidence
            if output.shape[-1] == 1:
                # Binary output
                conf = torch.sigmoid(output).item()
            else:
                # Softmax output
                probs = torch.softmax(output, dim=1)
                conf = probs[0, 1].item()  # Live class
        
        return conf
    
    def predict_single(self, image_path: str = None, 
                      image_array: np.ndarray = None,
                      face_bbox: Optional[Tuple] = None) -> Tuple[str, float]:
        """Single image prediction (no smoothing)"""
        if image_path:
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise ValueError(f"Failed to load image from {image_path}")
        
        if image_array is None:
            raise ValueError("Either image_path or image_array must be provided")
        
        conf = self._get_frame_confidence(image_array, face_bbox)
        prediction = "Live" if conf > self.confidence_threshold else "Fake"
        
        return prediction, conf
    
    def predict_video(self, video_path: str, 
                     face_bboxes: Optional[List[Tuple]] = None,
                     sample_rate: int = 1) -> Dict[str, Any]:
        """
        Predict video with temporal smoothing.
        
        Args:
            video_path: Path to video
            face_bboxes: Optional list of face bboxes per frame
            sample_rate: Sample every Nth frame (1 = all frames)
            
        Returns:
            Dict with prediction, confidence, attention weights, etc.
        """
        # Read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frames.append(frame)
            
            frame_idx += 1
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames in video: {video_path}")
        
        # Get CNN confidences
        confidences = []
        for i, frame in enumerate(frames):
            bbox = None
            if face_bboxes and i < len(face_bboxes):
                bbox = face_bboxes[i]
            
            conf = self._get_frame_confidence(frame, bbox)
            confidences.append(conf)
        
        confidences = np.array(confidences, dtype=np.float32)
        
        # Apply temporal smoothing if enabled
        if self.use_temporal_smoothing:
            result = self.smoother_pipeline.process_video(confidences.tolist())
            result['num_frames'] = len(frames)
            result['sampled_frames'] = len(confidences)
            result['raw_confidence_all'] = confidences
        else:
            # Fallback: mean pooling
            mean_conf = confidences.mean()
            prediction = "Live" if mean_conf > self.confidence_threshold else "Fake"
            result = {
                'prediction': prediction,
                'smoothed_confidence': float(mean_conf),
                'raw_confidence': float(mean_conf),
                'num_frames': len(frames),
                'sampled_frames': len(confidences),
                'raw_confidence_all': confidences,
            }
        
        return result
    
    def predict_frame_stream(self, frames: List[np.ndarray],
                            face_bboxes: Optional[List[Tuple]] = None) -> Dict[str, Any]:
        """
        Process frames with streaming smoothing.
        
        Args:
            frames: List of frame arrays
            face_bboxes: Optional face boxes
            
        Returns:
            Dict with smoothed prediction
        """
        # Get confidences
        confidences = []
        for i, frame in enumerate(frames):
            bbox = None
            if face_bboxes and i < len(face_bboxes):
                bbox = face_bboxes[i]
            
            conf = self._get_frame_confidence(frame, bbox)
            confidences.append(conf)
        
        # Smooth
        if self.use_temporal_smoothing:
            result = self.smoother_pipeline.process_video(confidences)
            result['num_frames'] = len(frames)
        else:
            mean_conf = np.mean(confidences)
            prediction = "Live" if mean_conf > self.confidence_threshold else "Fake"
            result = {
                'prediction': prediction,
                'smoothed_confidence': float(mean_conf),
                'raw_confidence': float(mean_conf),
                'num_frames': len(frames),
            }
        
        return result
