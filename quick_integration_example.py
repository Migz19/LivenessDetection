"""
Quick Integration Example: Adding Temporal Transformer to Existing System

This script shows how to integrate the temporal transformer with your
existing EfficientNet + feature extraction pipeline.

Key points:
1. Minimal code changes to existing inference pipeline
2. Can be used alongside existing CNN predictions
3. Confidence calibration prevents stuck-at-50% predictions
"""

import torch
import cv2
import numpy as np
from typing import Tuple, Optional
from pathlib import Path

from models.temporal_transformer import TemporalLivenessTransformer
from models.efficientnet_model import load_efficientnet_model
from inference_temporal import TemporalLivenessInference
from utils.liveness_features import LivenessPreprocessor


class EnhancedLivenessDetector:
    """
    Combined detector: EfficientNet CNN + Temporal Transformer
    
    Uses CNN for single-frame predictions and Transformer for
    video-level stability and confidence calibration.
    """
    
    def __init__(
        self,
        efficientnet_model,
        transformer_model,
        device: torch.device = 'cpu',
    ):
        """
        Args:
            efficientnet_model: Loaded EfficientNet model
            transformer_model: Trained TemporalLivenessTransformer
            device: Computation device
        """
        self.efficientnet = efficientnet_model.to(device).eval()
        self.transformer = transformer_model.to(device).eval()
        self.device = device
        
        self.preprocessor = LivenessPreprocessor(model_type='cnn')
        
        # Initialize temporal inference
        self.temporal_inference = TemporalLivenessInference(
            transformer_model,
            efficientnet_model,
            device=device,
            window_size=12,
            stride=4,
        )
    
    def predict_image(
        self,
        image: np.ndarray,
        face_bbox: Optional[Tuple] = None,
    ) -> Tuple[str, float]:
        """
        Single image prediction (CNN only, for quick inference).
        
        Args:
            image: Input image (BGR)
            face_bbox: Optional face bounding box
        
        Returns:
            (prediction, confidence) - "Live"/"Spoof", [0, 1]
        """
        # Preprocess
        face_img, lbp, freq, moire, depth = \
            self.preprocessor.preprocess_with_liveness_features(image, face_bbox)
        
        # CNN prediction
        with torch.no_grad():
            face_img = face_img.unsqueeze(0).to(self.device)
            logits = self.efficientnet(face_img)
            probs = torch.softmax(logits, dim=1)
            live_prob = probs[0, 1].item()
        
        prediction = "Live" if live_prob > 0.5 else "Spoof"
        
        return prediction, live_prob
    
    def predict_video(
        self,
        video_path: str,
        use_transformer: bool = True,
        return_details: bool = False,
    ) -> Tuple[str, float, float]:
        """
        Video prediction with temporal transformer.
        
        Args:
            video_path: Path to video file
            use_transformer: If True, use temporal transformer; else CNN only
            return_details: If True, return additional info
        
        Returns:
            (prediction, score, confidence)
        """
        
        if use_transformer:
            # Use temporal transformer
            details = self.temporal_inference.process_video(
                video_path,
                return_details=True
            )
            
            score = details['liveness_score']
            confidence = details['confidence']
            
            prediction = "Live" if score > 0.5 else "Spoof"
            
            if return_details:
                return {
                    'prediction': prediction,
                    'score': score,
                    'confidence': confidence,
                    'details': details,
                }
            
            return prediction, score, confidence
        
        else:
            # Use CNN only (baseline)
            frames = self._load_frames(video_path)
            if not frames:
                raise ValueError("Could not load video")
            
            cnn_scores = []
            
            with torch.no_grad():
                for frame in frames:
                    face_img, _, _, _, _ = \
                        self.preprocessor.preprocess_with_liveness_features(frame)
                    
                    face_img = face_img.unsqueeze(0).to(self.device)
                    logits = self.efficientnet(face_img)
                    probs = torch.softmax(logits, dim=1)
                    cnn_scores.append(probs[0, 1].item())
            
            score = np.mean(cnn_scores)
            confidence = 1.0 - np.std(cnn_scores)  # High std = low confidence
            
            prediction = "Live" if score > 0.5 else "Spoof"
            
            return prediction, score, confidence
    
    def stream_predict(
        self,
        frame: np.ndarray,
    ) -> Optional[Tuple[str, float, float]]:
        """
        Real-time stream prediction.
        
        Returns result every 12 frames, None otherwise.
        
        Args:
            frame: Input frame
        
        Returns:
            (prediction, score, confidence) or None
        """
        result = self.temporal_inference.process_frame_stream(frame, buffer_size=12)
        
        if result is not None:
            score, confidence, variance = result
            prediction = "Live" if score > 0.5 else "Spoof"
            return prediction, score, confidence
        
        return None
    
    def reset_stream(self):
        """Reset stream for new video."""
        self.temporal_inference.reset_stream()
    
    @staticmethod
    def _load_frames(video_path: str, max_frames: int = None):
        """Load frames from video."""
        try:
            cap = cv2.VideoCapture(video_path)
            frames = []
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)
                if max_frames and len(frames) >= max_frames:
                    break
            
            cap.release()
            return frames if frames else None
        except:
            return None


# ============================================================================
# USAGE EXAMPLES
# ============================================================================

def example_1_single_image():
    """Example 1: Quick single-image prediction"""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: Single Image Prediction")
    print("=" * 70)
    
    # Initialize
    efficientnet = load_efficientnet_model(device='cpu')
    transformer = TemporalLivenessTransformer()
    # transformer.load_state_dict(torch.load('temporal_transformer_best.pt'))
    
    detector = EnhancedLivenessDetector(efficientnet, transformer, device='cpu')
    
    # Load test image
    test_image = cv2.imread('path/to/test_image.jpg')
    
    if test_image is not None:
        prediction, confidence = detector.predict_image(test_image)
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.4f}")
    else:
        print("⚠ No test image found (skipping)")


def example_2_video_cnn_only():
    """Example 2: Video prediction with CNN only (baseline)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Video Prediction (CNN Baseline)")
    print("=" * 70)
    
    efficientnet = load_efficientnet_model(device='cpu')
    transformer = TemporalLivenessTransformer()
    
    detector = EnhancedLivenessDetector(efficientnet, transformer, device='cpu')
    
    # Predict with CNN only
    video_path = 'path/to/test_video.mp4'
    
    try:
        prediction, score, confidence = detector.predict_video(
            video_path,
            use_transformer=False
        )
        
        print(f"Prediction: {prediction}")
        print(f"Score: {score:.4f}")
        print(f"Confidence: {confidence:.4f}")
    except Exception as e:
        print(f"⚠ Error: {e}")


def example_3_video_with_transformer():
    """Example 3: Video prediction with Temporal Transformer"""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Video Prediction (Temporal Transformer)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    efficientnet = load_efficientnet_model(device=device)
    transformer = TemporalLivenessTransformer()
    # transformer.load_state_dict(torch.load('temporal_transformer_best.pt', map_location=device))
    
    detector = EnhancedLivenessDetector(efficientnet, transformer, device=device)
    
    # Predict with transformer
    video_path = 'path/to/test_video.mp4'
    
    try:
        result = detector.predict_video(
            video_path,
            use_transformer=True,
            return_details=True
        )
        
        print(f"Prediction: {result['prediction']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        details = result['details']
        print(f"\nDetailed Results:")
        print(f"  Avg Temporal Variance: {details['avg_variance']:.4f}")
        print(f"  Num Windows: {len(details['window_scores'])}")
        print(f"  Window Scores (mean±std): {np.mean(details['window_scores']):.4f} ± "
              f"{np.std(details['window_scores']):.4f}")
        
        # Interpretation
        if result['score'] > 0.75 and result['confidence'] > 0.7:
            print("  ✓ Confident LIVE detection")
        elif result['score'] < 0.25 and result['confidence'] > 0.7:
            print("  ✓ Confident SPOOF detection")
        elif 0.4 < result['score'] < 0.6:
            print("  ⚠ Uncertain (temporal inconsistency)")
    except Exception as e:
        print(f"⚠ Error: {e}")


def example_4_real_time_streaming():
    """Example 4: Real-time webcam streaming"""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Real-Time Streaming (Webcam)")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load models
    efficientnet = load_efficientnet_model(device=device)
    transformer = TemporalLivenessTransformer()
    
    detector = EnhancedLivenessDetector(efficientnet, transformer, device=device)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    frame_count = 0
    
    print("Starting real-time inference... (Press 'q' to quit)")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Get prediction (returns every 12 frames)
        result = detector.stream_predict(frame)
        
        if result is not None:
            prediction, score, confidence = result
            print(f"Frame {frame_count}: {prediction} (score={score:.3f}, "
                  f"confidence={confidence:.3f})")
        
        # Display
        cv2.imshow('Liveness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def example_5_comparison():
    """Example 5: Side-by-side comparison (CNN vs Transformer)"""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: CNN vs Transformer Comparison")
    print("=" * 70)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    efficientnet = load_efficientnet_model(device=device)
    transformer = TemporalLivenessTransformer()
    
    detector = EnhancedLivenessDetector(efficientnet, transformer, device=device)
    
    video_path = 'path/to/test_video.mp4'
    
    try:
        # CNN prediction
        pred_cnn, score_cnn, conf_cnn = detector.predict_video(
            video_path, use_transformer=False
        )
        
        # Transformer prediction
        pred_tf, score_tf, conf_tf = detector.predict_video(
            video_path, use_transformer=True
        )
        
        print("\nCOMPARISON RESULTS:")
        print("-" * 70)
        print(f"{'Metric':<25} {'CNN':<20} {'Transformer':<20}")
        print("-" * 70)
        print(f"{'Prediction':<25} {pred_cnn:<20} {pred_tf:<20}")
        print(f"{'Score':<25} {score_cnn:<20.4f} {score_tf:<20.4f}")
        print(f"{'Confidence':<25} {conf_cnn:<20.4f} {conf_tf:<20.4f}")
        print("-" * 70)
        
        # Analysis
        score_diff = abs(score_cnn - score_tf)
        conf_improvement = conf_tf - conf_cnn
        
        print(f"\nAnalysis:")
        print(f"  Score difference: {score_diff:.4f}")
        print(f"  Confidence improvement: {conf_improvement:+.4f}")
        
        if score_diff > 0.2:
            print(f"  → Transformer significantly changes prediction")
        if conf_improvement > 0.1:
            print(f"  → Transformer increases confidence (better stability)")
        elif conf_improvement < -0.1:
            print(f"  → Transformer reduces confidence (more conservative)")
    
    except Exception as e:
        print(f"⚠ Error: {e}")


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("ENHANCED LIVENESS DETECTION: Quick Integration Examples")
    print("=" * 70)
    print("\nThis script demonstrates integration of Temporal Transformer")
    print("with existing EfficientNet pipeline.")
    print("\nFeatures:")
    print("  • Single-image prediction (CNN)")
    print("  • Video prediction (CNN baseline)")
    print("  • Video prediction (Temporal Transformer)")
    print("  • Real-time streaming")
    print("  • Side-by-side comparison")
    print("=" * 70)
    
    # Run examples (uncomment to use)
    # example_1_single_image()
    # example_2_video_cnn_only()
    # example_3_video_with_transformer()
    # example_4_real_time_streaming()
    # example_5_comparison()
    
    print("\n✓ All examples available (uncomment in main to run)")
