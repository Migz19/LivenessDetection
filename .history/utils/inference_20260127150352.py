import torch
import numpy as np
from typing import Tuple, List, Optional, Dict, Any
import cv2
import sys
import os

# Add parent dir to path for temporal_smoother
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from temporal_smoother import TemporalSmoothingPipeline
    TEMPORAL_SMOOTHER_AVAILABLE = True
except ImportError:
    TEMPORAL_SMOOTHER_AVAILABLE = False


class LivenessInference:
    """
    Enhanced inference pipeline for liveness detection
    Supports both standard CNN models and multi-feature fusion models
    """
    def __init__(self, model, preprocessor, device='cpu', use_enhanced_features=False,
                 use_temporal_smoothing=True, temporal_window_size=8):
        """
        Args:
            model: Loaded model
            preprocessor: LivenessPreprocessor or ImagePreprocessor instance
            device: Device to run inference on
            use_enhanced_features: Whether model expects enhanced liveness features
            use_temporal_smoothing: Enable temporal smoothing (default True)
            temporal_window_size: Window size for smoothing
        """
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.use_enhanced_features = use_enhanced_features
        self.use_temporal_smoothing = use_temporal_smoothing and TEMPORAL_SMOOTHER_AVAILABLE
        self.model.eval()
        self.model.to(device)
        
        if self.use_temporal_smoothing:
            self.temporal_smoother = TemporalSmoothingPipeline(
                window_size=temporal_window_size,
                confidence_threshold=0.5
            )
            self.temporal_smoother.smoother.to(device)
    
    def predict_single(self, image_path: str = None, image_array: np.ndarray = None,
                      face_bbox: Optional[Tuple] = None) -> Tuple[str, float]:
        """
        Predict liveness for single image
        Args:
            image_path: Path to image file
            image_array: Numpy array of image (BGR format)
            face_bbox: Optional face bounding box (x1, y1, x2, y2)
        Returns:
            Tuple of (prediction, confidence)
        """
        # Load image if path provided
        if image_path:
            image_array = cv2.imread(image_path)
            if image_array is None:
                raise ValueError(f"Failed to load image from {image_path}")
        
        if image_array is None:
            raise ValueError("Either image_path or image_array must be provided")
        
        # Preprocess based on feature type
        if self.use_enhanced_features:
            tensors = self.preprocessor.preprocess_with_liveness_features(
                image_array, face_bbox
            )
            # Move all tensors to device
            tensors = tuple(t.to(self.device) for t in tensors)
        else:
            # Standard preprocessing
            if hasattr(self.preprocessor, 'preprocess_array'):
                tensor = self.preprocessor.preprocess_array(image_array, face_bbox)
            else:
                # Fallback for LivenessPreprocessor
                tensor, *_ = self.preprocessor.preprocess_with_liveness_features(
                    image_array, face_bbox
                )
            tensor = tensor.to(self.device)
            tensors = (tensor,)
        
        # Inference
        with torch.no_grad():
            if len(tensors) > 1:
                # Multi-feature model
                output = self.model(*tensors)
            else:
                # Single input model
                output = self.model(tensors[0])
            
            probabilities = torch.softmax(output, dim=1)
            pred_class = torch.argmax(probabilities, dim=1).item()
            confidence = probabilities[0, pred_class].item()
        
        prediction = "Live" if pred_class == 1 else "Fake"
        return prediction, confidence
    
    def predict_batch(self, image_arrays: List[np.ndarray], 
                     face_bboxes: Optional[List[Tuple]] = None) -> Tuple[List[str], List[float]]:
        """
        Predict liveness for multiple images (batch)
        Args:
            image_arrays: List of numpy arrays (BGR format)
            face_bboxes: Optional list of face bounding boxes
        Returns:
            Tuple of (predictions list, confidences list)
        """
        if self.use_enhanced_features:
            tensors = self.preprocessor.extract_batch_liveness_features(
                image_arrays, face_bboxes
            )
            # Move all tensors to device
            tensors = tuple(t.to(self.device) for t in tensors)
        else:
            # Standard preprocessing
            if hasattr(self.preprocessor, 'preprocess_batch'):
                tensor = self.preprocessor.preprocess_batch(image_arrays, face_bboxes)
            else:
                # Fallback
                tensors_list = []
                for idx, img in enumerate(image_arrays):
                    bbox = None
                    if face_bboxes:
                        bbox = face_bboxes[idx] if idx < len(face_bboxes) else face_bboxes[0]
                    t, *_ = self.preprocessor.preprocess_with_liveness_features(img, bbox)
                    tensors_list.append(t.squeeze(0))
                tensor = torch.stack(tensors_list)
            tensor = tensor.to(self.device)
            tensors = (tensor,)
        
        # Inference
        with torch.no_grad():
            if len(tensors) > 1:
                # Multi-feature model
                output = self.model(*tensors)
            else:
                # Single input model
                output = self.model(tensors[0])
            
            probabilities = torch.softmax(output, dim=1)
            pred_classes = torch.argmax(probabilities, dim=1).cpu().numpy()
            confidences = probabilities.max(dim=1)[0].cpu().numpy()
        
        predictions = ["Live" if pred == 1 else "Fake" for pred in pred_classes]
        return predictions, confidences.tolist()
    
    def predict_video_frames(self, frames: List[np.ndarray],
                            face_bboxes: Optional[List[Tuple]] = None,
                            use_motion_detector: bool = True) -> Dict[str, Any]:
        """
        Predict liveness for video frames with motion analysis and aggregation
        Args:
            frames: List of video frames (BGR format)
            face_bboxes: Optional list of face bounding boxes
            use_motion_detector: Whether to use motion-based detection
        Returns:
            Dictionary with comprehensive results
        """
        # Standard frame-by-frame predictions
        predictions, confidences = self.predict_batch(frames, face_bboxes)
        
        # Motion-based detection
        motion_result = None
        if use_motion_detector and len(frames) >= 2:
            try:
                from enhanced_liveness_preprocessing import MotionBasedLivenessDetector
                motion_detector = MotionBasedLivenessDetector()
                
                # Use first bbox for all frames if available
                bboxes = face_bboxes if face_bboxes else [None]
                motion_label, motion_conf, motion_features = motion_detector.detect_from_frames(
                    frames, bboxes
                )
                motion_result = {
                    'label': motion_label,
                    'confidence': motion_conf,
                    'features': motion_features
                }
            except Exception as e:
                print(f"Motion detection failed: {e}")
                motion_result = None
        
        # Aggregate frame predictions
        live_count = sum(1 for p in predictions if p == "Live")
        fake_count = len(predictions) - live_count
        
        # Frame-based prediction
        frame_prediction = "Live" if live_count > fake_count else "Fake"
        frame_confidence = max(live_count, fake_count) / len(predictions)
        
        # Combine frame and motion predictions
        if motion_result and motion_result['label'] != "Uncertain":
            # Weight: 60% frame predictions, 40% motion detection
            if frame_prediction == motion_result['label']:
                # Agreement - boost confidence
                overall_prediction = frame_prediction
                overall_confidence = min(0.98, 0.6 * frame_confidence + 0.4 * motion_result['confidence'])
            else:
                # Disagreement - use higher confidence
                if frame_confidence > motion_result['confidence']:
                    overall_prediction = frame_prediction
                    overall_confidence = frame_confidence * 0.8  # Reduce due to disagreement
                else:
                    overall_prediction = motion_result['label']
                    overall_confidence = motion_result['confidence'] * 0.8
        else:
            # No motion result, use frame-based only
            overall_prediction = frame_prediction
            overall_confidence = frame_confidence
        
        # Calculate per-class statistics
        live_confidences = [c for p, c in zip(predictions, confidences) if p == "Live"]
        fake_confidences = [c for p, c in zip(predictions, confidences) if p == "Fake"]
        
        avg_live_conf = float(np.mean(live_confidences)) if live_confidences else 0.0
        avg_fake_conf = float(np.mean(fake_confidences)) if fake_confidences else 0.0
        
        # Temporal consistency
        consistency_score = self._calculate_temporal_consistency(predictions)
        
        return {
            'overall_prediction': overall_prediction,
            'overall_confidence': float(overall_confidence),
            'frame_prediction': frame_prediction,
            'frame_confidence': float(frame_confidence),
            'motion_result': motion_result,
            'predictions': predictions,
            'confidences': confidences,
            'live_count': live_count,
            'fake_count': fake_count,
            'avg_live_confidence': avg_live_conf,
            'avg_fake_confidence': avg_fake_conf,
            'temporal_consistency': consistency_score,
            'num_frames': len(frames)
        }
    
    def _calculate_temporal_consistency(self, predictions: List[str]) -> float:
        """
        Calculate how consistent predictions are across frames
        High consistency = likely real (stable prediction)
        Low consistency = uncertain or manipulation
        """
        if len(predictions) < 2:
            return 1.0
        
        # Count transitions
        transitions = sum(1 for i in range(len(predictions) - 1) 
                         if predictions[i] != predictions[i + 1])
        
        # Consistency score (fewer transitions = more consistent)
        consistency = 1.0 - (transitions / (len(predictions) - 1))
        return float(consistency)
    
    def predict_with_uncertainty(self, image_path: str = None, 
                                image_array: np.ndarray = None,
                                face_bbox: Optional[Tuple] = None,
                                num_augmentations: int = 5) -> Dict[str, Any]:
        """
        Predict with uncertainty estimation using test-time augmentation
        Args:
            image_path: Path to image
            image_array: Numpy array
            face_bbox: Face bounding box
            num_augmentations: Number of augmented samples
        Returns:
            Dictionary with predictions and uncertainty metrics
        """
        if image_path:
            image_array = cv2.imread(image_path)
        
        predictions = []
        confidences = []
        
        # Original prediction
        pred, conf = self.predict_single(image_array=image_array, face_bbox=face_bbox)
        predictions.append(pred)
        confidences.append(conf)
        
        # Augmented predictions
        for _ in range(num_augmentations - 1):
            # Apply random augmentations
            aug_image = self._augment_image(image_array)
            pred, conf = self.predict_single(image_array=aug_image, face_bbox=face_bbox)
            predictions.append(pred)
            confidences.append(conf)
        
        # Calculate metrics
        live_count = sum(1 for p in predictions if p == "Live")
        fake_count = num_augmentations - live_count
        
        final_pred = "Live" if live_count > fake_count else "Fake"
        agreement_ratio = max(live_count, fake_count) / num_augmentations
        
        # Uncertainty: higher when predictions disagree
        uncertainty = 1.0 - agreement_ratio
        
        # Average confidence
        avg_confidence = float(np.mean(confidences))
        
        return {
            'prediction': final_pred,
            'confidence': avg_confidence,
            'agreement_ratio': float(agreement_ratio),
            'uncertainty': float(uncertainty),
            'all_predictions': predictions,
            'all_confidences': confidences,
            'live_count': live_count,
            'fake_count': fake_count
        }
    
    def _augment_image(self, image: np.ndarray) -> np.ndarray:
        """
        Apply random augmentation to image
        """
        aug = image.copy()
        
        # Random brightness adjustment
        if np.random.rand() > 0.5:
            factor = np.random.uniform(0.8, 1.2)
            aug = np.clip(aug * factor, 0, 255).astype(np.uint8)
        
        # Random contrast adjustment
        if np.random.rand() > 0.5:
            mean = np.mean(aug)
            factor = np.random.uniform(0.8, 1.2)
            aug = np.clip((aug - mean) * factor + mean, 0, 255).astype(np.uint8)
        
        # Random Gaussian noise
        if np.random.rand() > 0.5:
            noise = np.random.normal(0, 5, aug.shape)
            aug = np.clip(aug + noise, 0, 255).astype(np.uint8)
        
        return aug
    
    def get_detailed_analysis(self, image_array: np.ndarray, 
                            face_bbox: Optional[Tuple] = None) -> Dict[str, Any]:
        """
        Get detailed liveness analysis including all features
        """
        from enhanced_liveness_preprocessing import (
            get_liveness_features_summary, 
            compute_photo_artifacts_score
        )
        
        # Model prediction
        prediction, confidence = self.predict_single(image_array=image_array, face_bbox=face_bbox)
        
        # Feature analysis
        features = get_liveness_features_summary(image_array, face_bbox)
        artifact_score = compute_photo_artifacts_score(image_array, face_bbox)
        
        # Combined assessment
        # If artifact score is high (>0.7) and model says Live, flag as suspicious
        suspicious = (artifact_score > 0.7 and prediction == "Live")
        
        # Adjust confidence if suspicious
        adjusted_confidence = confidence
        if suspicious:
            adjusted_confidence = confidence * (1 - artifact_score * 0.5)
        
        return {
            'prediction': prediction,
            'confidence': float(confidence),
            'adjusted_confidence': float(adjusted_confidence),
            'artifact_score': float(artifact_score),
            'suspicious': suspicious,
            'features': features,
            'recommendation': self._get_recommendation(
                prediction, adjusted_confidence, artifact_score, suspicious
            )
        }
    
    def _get_recommendation(self, prediction: str, confidence: float, 
                          artifact_score: float, suspicious: bool) -> str:
        """
        Generate human-readable recommendation
        """
        if suspicious:
            return "High artifact score detected. Recommend additional verification."
        
        if confidence > 0.9:
            return f"High confidence {prediction} detection. Reliable result."
        elif confidence > 0.7:
            return f"Moderate confidence {prediction} detection. Generally reliable."
        elif confidence > 0.5:
            return f"Low confidence {prediction} detection. Consider additional checks."
        else:
            return "Very uncertain prediction. Multiple verification methods recommended."


def create_inference_engine(model, preprocessor, 
                           device: str = 'cpu', 
                           use_enhanced_features: bool = False):
    """
    Factory function to create inference engine
    Args:
        model: Loaded model
        preprocessor: Preprocessor instance (LivenessPreprocessor or ImagePreprocessor)
        device: Device for inference ('cpu' or 'cuda')
        use_enhanced_features: Whether model expects enhanced liveness features
    Returns:
        LivenessInference instance
    """
    return LivenessInference(model, preprocessor, device, use_enhanced_features)


def batch_inference_with_analysis(inference_engine: LivenessInference,
                                  image_arrays: List[np.ndarray],
                                  face_bboxes: Optional[List[Tuple]] = None) -> List[Dict[str, Any]]:
    """
    Run batch inference with detailed analysis for each image
    Args:
        inference_engine: LivenessInference instance
        image_arrays: List of images
        face_bboxes: Optional face bounding boxes
    Returns:
        List of detailed analysis results
    """
    results = []
    
    for idx, image in enumerate(image_arrays):
        bbox = None
        if face_bboxes:
            bbox = face_bboxes[idx] if idx < len(face_bboxes) else face_bboxes[0]
        
        result = inference_engine.get_detailed_analysis(image, bbox)
        result['image_index'] = idx
        results.append(result)
    
    return results