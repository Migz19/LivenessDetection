"""
Pre-trained Model Inference - Option 3
No training required! Use pre-trained weights for instant inference.

This script:
1. Loads pre-trained EfficientNet (ImageNet weights)
2. Initializes Temporal Transformer (or loads pre-trained weights if available)
3. Runs inference on any video
4. No dataset required!
"""

import torch
import torch.nn as nn
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional

from models.temporal_transformer import TemporalLivenessTransformer
from models.efficientnet_model import load_efficientnet_model
from inference_temporal import TemporalLivenessInference
from utils.liveness_features import LivenessPreprocessor


class PreTrainedLivenessDetector:
    """
    Use pre-trained models without any training.
    
    Two options:
    1. Load your trained transformer weights (if you have them)
    2. Use transformer with random initialization (will work but not accurate)
    3. Download pre-trained transformer from model zoo (future)
    """
    
    def __init__(
        self,
        transformer_weights: Optional[str] = None,
        efficientnet_weights: Optional[str] = None,
        device: torch.device = None,
    ):
        """
        Args:
            transformer_weights: Path to saved transformer.pt (optional)
            efficientnet_weights: Path to saved efficientnet.pt (optional)
            device: 'cuda' or 'cpu'
        """
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.device = device
        print(f"Using device: {device}")
        
        # ===== Load EfficientNet (pre-trained on ImageNet) =====
        print("\n[1] Loading EfficientNet-B3 (pre-trained on ImageNet)...")
        self.efficientnet = load_efficientnet_model(
            weights_path=efficientnet_weights,
            device=device,
            pretrained=True  # Use ImageNet weights
        )
        print("    ✓ EfficientNet loaded (ImageNet pre-trained)")
        
        # ===== Load Temporal Transformer =====
        print("\n[2] Loading Temporal Transformer...")
        self.transformer = TemporalLivenessTransformer().to(device)
        
        if transformer_weights and Path(transformer_weights).exists():
            print(f"    Loading transformer weights from: {transformer_weights}")
            state_dict = torch.load(transformer_weights, map_location=device)
            self.transformer.load_state_dict(state_dict)
            print("    ✓ Transformer weights loaded")
        else:
            print("    ⚠ Using transformer with random initialization")
            print("      (Results won't be accurate without training)")
            print("      To get better results:")
            print("      1. Train the model with your data: python train_temporal_transformer.py")
            print("      2. Pass weights path: PreTrainedLivenessDetector(transformer_weights='model.pt')")
        
        self.transformer.eval()
        
        # ===== Initialize Preprocessor =====
        print("\n[3] Initializing feature preprocessor...")
        self.preprocessor = LivenessPreprocessor(model_type='cnn')
        print("    ✓ Preprocessor ready")
        
        # ===== Initialize Inference Pipeline =====
        print("\n[4] Setting up inference pipeline...")
        self.inference = TemporalLivenessInference(
            self.transformer,
            self.efficientnet,
            device=device,
            window_size=12,
            stride=4,
        )
        print("    ✓ Inference pipeline ready\n")
    
    def predict_video(
        self,
        video_path: str,
        return_details: bool = True,
        verbose: bool = True,
    ) -> dict:
        """
        Predict liveness for a video using pre-trained models.
        
        Args:
            video_path: Path to video file (mp4, avi, mov, etc.)
            return_details: Return detailed statistics
            verbose: Print progress
        
        Returns:
            dict with keys:
            - prediction: "Live" or "Spoof"
            - score: [0, 1] liveness probability
            - confidence: [0, 1] prediction confidence
            - details: Additional stats (if return_details=True)
        """
        
        if verbose:
            print(f"\nProcessing: {video_path}")
        
        # Check file exists
        if not Path(video_path).exists():
            print(f"✗ Error: Video file not found: {video_path}")
            return None
        
        try:
            # Run inference
            details = self.inference.process_video(
                video_path,
                return_details=return_details
            )
            
            score = details['liveness_score']
            confidence = details['confidence']
            
            prediction = "Live" if score > 0.5 else "Spoof"
            
            if verbose:
                print("\n" + "=" * 60)
                print("PREDICTION RESULTS")
                print("=" * 60)
                print(f"Prediction:  {prediction}")
                print(f"Score:       {score:.4f}")
                print(f"Confidence:  {confidence:.4f}")
                print("=" * 60)
                
                # Interpretation
                if score > 0.75:
                    status = "✓ Confident LIVE" if confidence > 0.7 else "⚠ Likely LIVE (low confidence)"
                elif score < 0.25:
                    status = "✓ Confident SPOOF" if confidence > 0.7 else "⚠ Likely SPOOF (low confidence)"
                else:
                    status = "⚠ UNCERTAIN prediction"
                
                print(f"Status: {status}\n")
                
                if return_details:
                    print(f"Additional Info:")
                    print(f"  Window scores: {len(details['window_scores'])} windows")
                    print(f"  Avg variance:  {details['avg_variance']:.4f}")
            
            return {
                'prediction': prediction,
                'score': float(score),
                'confidence': float(confidence),
                'details': details if return_details else None,
            }
        
        except Exception as e:
            print(f"✗ Error processing video: {e}")
            return None
    
    def predict_frame(self, frame: np.ndarray) -> Optional[Tuple[str, float]]:
        """
        Quick prediction on single frame using CNN only.
        (Much faster but less accurate than video-based)
        
        Args:
            frame: Input frame (BGR, numpy array)
        
        Returns:
            (prediction, confidence) or None if error
        """
        try:
            # Extract features
            face_img, _, _, _, _ = \
                self.preprocessor.preprocess_with_liveness_features(frame)
            
            # CNN inference
            with torch.no_grad():
                face_img = face_img.unsqueeze(0).to(self.device)
                logits = self.efficientnet(face_img)
                probs = torch.softmax(logits, dim=1)
                live_prob = probs[0, 1].item()
            
            prediction = "Live" if live_prob > 0.5 else "Spoof"
            
            return prediction, live_prob
        
        except Exception as e:
            print(f"Error: {e}")
            return None


def demo_with_webcam():
    """
    Demo: Real-time inference from webcam
    Press 'q' to quit
    """
    print("\n" + "=" * 70)
    print("REAL-TIME WEBCAM DEMO - Pre-trained Model")
    print("=" * 70)
    print("Press 'q' to exit\n")
    
    # Initialize detector
    detector = PreTrainedLivenessDetector(device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Could not open webcam")
        return
    
    print("Webcam opened. Processing frames...")
    print("(Each detection uses 12-frame window buffer)\n")
    
    frame_count = 0
    results = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Single-frame quick prediction (CNN only)
        pred, confidence = detector.predict_frame(frame)
        
        if pred:
            results.append((frame_count, pred, confidence))
            
            # Display on frame
            text = f"{pred}: {confidence:.2f}"
            color = (0, 255, 0) if pred == "Live" else (0, 0, 255)
            
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Pre-trained Liveness Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    # Summary
    if results:
        print("\n" + "=" * 70)
        print(f"Processed {frame_count} frames")
        live_count = sum(1 for _, p, _ in results if p == "Live")
        spoof_count = sum(1 for _, p, _ in results if p == "Spoof")
        avg_confidence = np.mean([c for _, _, c in results])
        
        print(f"Live predictions:  {live_count}")
        print(f"Spoof predictions: {spoof_count}")
        print(f"Avg confidence:    {avg_confidence:.4f}")
        print("=" * 70)


def demo_with_video_file():
    """
    Demo: Inference on a video file
    """
    print("\n" + "=" * 70)
    print("VIDEO FILE INFERENCE - Pre-trained Model")
    print("=" * 70)
    
    # Initialize detector
    detector = PreTrainedLivenessDetector()
    
    # Test video path (you can change this)
    video_path = 'test_video.mp4'
    
    # Check if test video exists, if not create a dummy one or use sample
    if not Path(video_path).exists():
        print(f"\n⚠ Video file not found: {video_path}")
        print("\nTo test, provide a video file:")
        print("  detector.predict_video('your_video.mp4')")
        print("\nOr use webcam demo instead:")
        print("  demo_with_webcam()")
        return
    
    # Predict
    result = detector.predict_video(video_path, return_details=True)
    
    if result:
        print("\n✓ Prediction complete!")


def demo_download_pretrained():
    """
    Guide: Where to get pre-trained weights
    """
    print("\n" + "=" * 70)
    print("GETTING PRE-TRAINED WEIGHTS")
    print("=" * 70)
    
    print("""
Option 1: USE YOUR OWN TRAINED MODEL
────────────────────────────────────
If you've trained the transformer:
    detector = PreTrainedLivenessDetector(
        transformer_weights='temporal_transformer_best.pt'
    )

Option 2: USE MODEL ZOO (Future)
────────────────────────────────
Download pre-trained transformers:
    - Hugging Face: huggingface.co/models
    - PyTorch Hub: pytorch.org/hub
    - Model cards: kaggle.com/models

Example (when available):
    import torch
    model = torch.hub.load('repo/liveness', 'temporal_transformer')

Option 3: FINE-TUNE ON YOUR DATA (Recommended)
──────────────────────────────────────────────
1. Download dataset: SiW, OULU-NPU, or Replay-Attack
2. Train with initial weights:
    python train_temporal_transformer.py \\
        --dataset ./siw \\
        --epochs 50 \\
        --batch-size 16 \\
        --device cuda

3. Use trained model:
    detector = PreTrainedLivenessDetector(
        transformer_weights='temporal_transformer_best.pt'
    )

Option 4: USE EFFICIENTNET ONLY (Fast Baseline)
──────────────────────────────────────────────
EfficientNet-B3 is pre-trained on ImageNet:
    from models.efficientnet_model import load_efficientnet_model
    model = load_efficientnet_model(pretrained=True)
    # Works immediately, no transformer training needed
    """)


if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("PRE-TRAINED LIVENESS DETECTION - Option 3")
    print("=" * 70)
    
    # Initialize detector once
    print("\nInitializing pre-trained models...\n")
    detector = PreTrainedLivenessDetector()
    
    # ===== CHOOSE YOUR TEST =====
    
    # Test 1: Webcam (real-time)
    print("\n[DEMO OPTIONS]")
    print("1. Real-time webcam: demo_with_webcam()")
    print("2. Video file:       detector.predict_video('your_video.mp4')")
    print("3. Single frame:     detector.predict_frame(frame)")
    print("4. Download help:    demo_download_pretrained()")
    print("\nUsage:")
    print("  demo_with_webcam()                              # Real-time from webcam")
    print("  detector.predict_video('test.mp4')              # Process video file")
    print("  detector.predict_video('test.mp4', verbose=False)  # Quiet mode")
    
    print("\n" + "=" * 70)
    print("✓ Pre-trained detector ready!")
    print("=" * 70)
    
    # Uncomment to run a demo:
    # demo_with_webcam()
    # demo_with_video_file()
    # demo_download_pretrained()
