"""
Enhanced Liveness Inference with Hybrid Approach
Combines model predictions with texture and motion analysis
"""

import torch
import numpy as np
from typing import List, Tuple, Dict
from .liveness_features import LivenessPreprocessor, MotionBasedLivenessDetector, get_liveness_features_summary


class EnhancedLivenessInference:
    """
    Enhanced inference combining:
    1. Deep learning model predictions
    2. LBP texture features (good at detecting spoofs)
    3. Motion detection (for videos)
    4. Frequency domain analysis
    """
    
    def __init__(self, model, device='cpu'):
        self.model = model
        self.device = device
        self.liveness_preprocessor = LivenessPreprocessor()
        self.motion_detector = MotionBasedLivenessDetector()
    
    def predict_single_with_features(self, image: np.ndarray, 
                                    face_bbox: Tuple = None) -> Dict:
        """
        Predict liveness with additional feature analysis
        """
        # Get model prediction
        tensor = self.liveness_preprocessor._preprocess_face(
            self.liveness_preprocessor._crop_face(image, face_bbox)
        )
        tensor = tensor.to(self.device)
        
        with torch.no_grad():
            output = self.model(tensor)
            probs = torch.softmax(output, dim=1)
            pred = torch.argmax(probs, dim=1).item()
            model_conf = probs[0, pred].item()
        
        # Get liveness features
        features = get_liveness_features_summary(image, face_bbox)
        
        # Adjust confidence based on features
        adjusted_conf = self._adjust_confidence_by_features(
            model_conf, features, pred
        )
        
        prediction = "Live" if pred == 1 else "Fake"
        
        return {
            'prediction': prediction,
            'model_confidence': model_conf,
            'adjusted_confidence': adjusted_conf,
            'features': features
        }
    
    def predict_batch_with_features(self, image_arrays: List[np.ndarray],
                                   face_bboxes: List[Tuple] = None) -> Dict:
        """
        Batch prediction with feature analysis
        """
        predictions = []
        confidences = []
        features_list = []
        
        for idx, image in enumerate(image_arrays):
            bbox = None
            if face_bboxes:
                if len(face_bboxes) == 1:
                    bbox = face_bboxes[0]
                elif idx < len(face_bboxes):
                    bbox = face_bboxes[idx]
            
            result = self.predict_single_with_features(image, bbox)
            predictions.append(result['prediction'])
            confidences.append(result['adjusted_confidence'])
            features_list.append(result['features'])
        
        return {
            'predictions': predictions,
            'confidences': confidences,
            'features': features_list
        }
    
    def predict_video_with_motion(self, frames: List[np.ndarray],
                                 face_bboxes: List[Tuple] = None) -> Dict:
        """
        Predict liveness for video with motion analysis
        Primary: Model predictions on frames
        Secondary: Motion for confirmation/tiebreaking
        """
        # Model-based predictions on individual frames
        frame_results = self.predict_batch_with_features(frames, face_bboxes)
        
        # Motion-based liveness detection
        motion_pred, motion_conf, motion_features = self.motion_detector.detect_from_frames(
            frames, face_bboxes if face_bboxes else [(0, 0, frames[0].shape[1], frames[0].shape[0])]
        )
        
        # Combine predictions
        live_count = sum(1 for p in frame_results['predictions'] if p == "Live")
        fake_count = len(frame_results['predictions']) - live_count
        
        # Calculate average model confidence
        avg_model_conf = np.mean(frame_results['confidences'])
        
        # PRIMARY DECISION: Model-based majority voting
        if live_count > fake_count:
            # Models say more Live than Fake
            final_prediction = "Live"
            # Confidence is based on the proportion of Live frames
            base_confidence = live_count / len(frame_results['predictions'])
            
            # Motion can boost or slightly reduce confidence
            if motion_pred == "Live":
                # Motion confirms = higher confidence
                final_confidence = min(0.98, (base_confidence + motion_conf) / 2)
            elif motion_pred == "Uncertain":
                # Motion uncertain = use model confidence
                final_confidence = min(0.98, base_confidence * 0.95)
            else:
                # Motion says Fake but frames say Live = slightly reduce confidence
                final_confidence = min(0.98, base_confidence * 0.8)
        
        elif fake_count > live_count:
            # Models say more Fake than Live
            final_prediction = "Fake"
            base_confidence = fake_count / len(frame_results['predictions'])
            
            # Motion can boost or slightly reduce confidence
            if motion_pred == "Fake":
                # Motion confirms = higher confidence
                final_confidence = min(0.98, (base_confidence + motion_conf) / 2)
            elif motion_pred == "Uncertain":
                # Motion uncertain = use model confidence
                final_confidence = min(0.98, base_confidence * 0.95)
            else:
                # Motion says Live but frames say Fake = slightly reduce confidence
                final_confidence = min(0.98, base_confidence * 0.8)
        
        else:
            # Equal split - use motion to decide
            if motion_pred == "Live":
                final_prediction = "Live"
                final_confidence = motion_conf * 0.8
            elif motion_pred == "Fake":
                final_prediction = "Fake"
                final_confidence = motion_conf * 0.8
            else:
                # Both inconclusive
                final_prediction = "Uncertain"
                final_confidence = 0.5
        
        return {
            'predictions': frame_results['predictions'],
            'confidences': frame_results['confidences'],
            'live_count': live_count,
            'fake_count': fake_count,
            'motion_prediction': motion_pred,
            'motion_confidence': motion_conf,
            'final_prediction': final_prediction,
            'final_confidence': min(0.99, final_confidence),
            'features': frame_results['features']
        }
    
    def _adjust_confidence_by_features(self, model_conf: float, 
                                      features: Dict, pred: int) -> float:
        """
        Adjust model confidence based on image quality features
        """
        adjusted = model_conf
        
        # Brightness factor (spoof detection)
        brightness = features['brightness']
        if brightness < 30 or brightness > 225:
            # Too dark or too bright - likely problematic
            adjusted *= 0.8
        
        # Blur factor (spoofed images often have blur)
        blur = features['blurriness']
        if blur < 50:  # Very blurry
            # Reduce confidence in LIVE prediction, increase in FAKE
            if pred == 1:  # Live prediction
                adjusted *= 0.7
            else:  # Fake prediction
                adjusted = min(1.0, adjusted * 1.2)
        
        # Face size factor
        face_size = features['face_size']
        if face_size < 5000:  # Too small face
            adjusted *= 0.85
        
        # Contrast factor (important for liveness)
        contrast = features['contrast']
        if contrast < 20:  # Low contrast
            if pred == 1:
                adjusted *= 0.8
            else:
                adjusted = min(1.0, adjusted * 1.1)
        
        return min(1.0, adjusted)


class AdaptiveLivenessDetector:
    """
    Adaptive detector that chooses the best method based on input
    """
    
    def __init__(self, model, device='cpu'):
        self.enhanced_inference = EnhancedLivenessInference(model, device)
        self.motion_detector = MotionBasedLivenessDetector(threshold=0.03)
    
    def detect(self, frames: List[np.ndarray], 
               face_bboxes: List[Tuple] = None, is_video: bool = False) -> Dict:
        """
        Adaptive detection:
        - Single frame: Use model + quality features
        - Video: Use model + motion + texture analysis
        """
        
        if is_video and len(frames) > 1:
            # Use video-specific detection
            return self.enhanced_inference.predict_video_with_motion(frames, face_bboxes)
        else:
            # Use single frame detection
            result = self.enhanced_inference.predict_single_with_features(
                frames[0], face_bboxes[0] if face_bboxes else None
            )
            return {
                'prediction': result['prediction'],
                'confidence': result['adjusted_confidence'],
                'features': result['features']
            }
