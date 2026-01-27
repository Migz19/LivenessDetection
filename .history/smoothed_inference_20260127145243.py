"""
Integration: Combine existing CNN + temporal smoother

This replaces the inference pipeline with:
1. CNN predicts each frame
2. Temporal smoother stabilizes confidences
3. Return smooth predictions
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
import cv2
from temporal_smoother import TemporalSmoother, TemporalSmoothingPipeline


class SmoothedLivenessDetector:
    """
    Wraps existing CNN model with temporal smoothing.
    
    Usage:
        detector = SmoothedLivenessDetector(cnn_model, preprocessor)
        result = detector.predict_video('video.mp4')
    """
    
    def __init__(self, 
                 cnn_model,
                 preprocessor,
                 device='cpu',
                 use_enhanced_features=False,
                 window_size=8,
                 confidence_threshold=0.5):
        """
        Args:
            cnn_model: Existing EfficientNet or similar model
            preprocessor: Image preprocessor
            device: 'cpu' or 'cuda'
            use_enhanced_features: If True, use liveness features
            window_size: Frames to smooth over
            confidence_threshold: Decision threshold
        """
        self.cnn_model = cnn_model
        self.preprocessor = preprocessor
        self.device = device
        self.use_enhanced_features = use_enhanced_features
        
        # Setup CNN
        self.cnn_model.eval()
        self.cnn_model.to(device)
        
        # Create temporal smoother (frozen, random init)
        self.smoother = TemporalSmoother(
            hidden_dim=128,
            num_heads=4,
            num_layers=2,
            max_seq_length=64,
        )
        self.smoother.eval()
        self.smoother.to(device)
        
        # Pipeline
        self.pipeline = TemporalSmoothingPipeline(
            smoother=self.smoother,
            window_size=window_size,
            confidence_threshold=confidence_threshold,
        )
    
    def _get_frame_confidences(self, frames: List[np.ndarray]) -> np.ndarray:
        """
        Get CNN confidence for each frame.
        
        Returns: (num_frames,) array of confidences [0, 1]
        """
        confidences = []
        
        with torch.no_grad():
            for frame in frames:
                # Preprocess
                if self.use_enhanced_features:
                    tensor, *_ = self.preprocessor.preprocess_with_liveness_features(frame)
                else:
                    tensor = self.preprocessor.preprocess(frame)
                
                tensor = tensor.to(self.device)
                
                # CNN inference
                output = self.cnn_model(tensor)
                
                # Get confidence (sigmoid output or softmax)
                if output.shape[-1] == 1:
                    # Binary output (sigmoid already applied)
                    conf = torch.sigmoid(output).item()
                else:
                    # Softmax output
                    probs = torch.softmax(output, dim=1)
                    conf = probs[0, 1].item()  # Live class
                
                confidences.append(conf)
        
        return np.array(confidences, dtype=np.float32)
    
    def predict_video(self, video_path: str) -> Dict:
        """
        Predict liveness for video with temporal smoothing.
        
        Returns:
            {
                'prediction': 'Live' or 'Spoof',
                'smoothed_confidence': float,
                'raw_confidence': float,
                'variance_reduction': float,
                'num_frames': int,
                'stable': bool,
                'details': {...}
            }
        """
        # Read video
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        
        cap.release()
        
        if not frames:
            raise ValueError(f"No frames in video: {video_path}")
        
        # Get CNN confidences for all frames
        cnn_confidences = self._get_frame_confidences(frames)
        
        # Smooth with temporal attention
        result = self.pipeline.process_video(cnn_confidences.tolist())
        result['num_frames'] = len(frames)
        
        return result
    
    def predict_frame(self, frame: np.ndarray) -> Tuple[str, float]:
        """
        Quick single-frame prediction (no smoothing).
        
        Returns:
            (prediction_label, confidence)
        """
        # Preprocess
        if self.use_enhanced_features:
            tensor, *_ = self.preprocessor.preprocess_with_liveness_features(frame)
        else:
            tensor = self.preprocessor.preprocess(frame)
        
        tensor = tensor.to(self.device)
        
        # CNN inference
        with torch.no_grad():
            output = self.cnn_model(tensor)
            
            if output.shape[-1] == 1:
                conf = torch.sigmoid(output).item()
            else:
                probs = torch.softmax(output, dim=1)
                conf = probs[0, 1].item()
        
        pred = "Live" if conf > 0.5 else "Spoof"
        return pred, conf
    
    def predict_stream(self, frame: np.ndarray, 
                      frame_buffer: List[np.ndarray]) -> Dict:
        """
        Real-time streaming prediction.
        
        Usage:
            buffer = []
            while True:
                ret, frame = cap.read()
                if not ret: break
                result = detector.predict_stream(frame, buffer)
                buffer.append(frame)
        """
        # Add frame to buffer
        frame_buffer.append(frame)
        
        # Trim buffer to max size
        max_buffer = 64
        if len(frame_buffer) > max_buffer:
            frame_buffer = frame_buffer[-max_buffer:]
        
        # Need at least window_size frames
        if len(frame_buffer) < self.pipeline.window_size:
            # Quick prediction on latest frame
            pred, conf = self.predict_frame(frame)
            return {
                'prediction': pred,
                'confidence': conf,
                'smoothed': False,
                'frames_buffered': len(frame_buffer),
            }
        
        # Get confidences for entire buffer
        cnn_confidences = self._get_frame_confidences(frame_buffer)
        
        # Smooth entire buffer
        result = self.pipeline.process_video(cnn_confidences.tolist())
        result['smoothed'] = True
        result['frames_buffered'] = len(frame_buffer)
        
        return result
