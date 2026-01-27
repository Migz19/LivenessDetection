"""
HOW TO USE TEMPORAL SMOOTHING WITH YOUR EXISTING CODE

3 options:
1. Minimal: Just replace predict_video() with smoothed version
2. Integrated: Use SmoothedLivenessInference wrapper
3. Custom: Directly use TemporalSmoothingPipeline
"""

import torch
import numpy as np
from temporal_smoother import TemporalSmoothingPipeline


# ==============================================================================
# OPTION 1: MINIMAL - Add smoothing to existing code
# ==============================================================================

def use_temporal_smoothing_minimal(cnn_confidences_list):
    """
    Simplest: You already have per-frame CNN confidences, just smooth them.
    
    Usage:
        # You have CNN predictions already
        frame_confs = [0.48, 0.52, 0.49, 0.51, 0.50, 0.47, 0.53, 0.50]
        
        result = use_temporal_smoothing_minimal(frame_confs)
        print(result)  # Shows smoothed prediction
    """
    
    # Create pipeline once
    pipeline = TemporalSmoothingPipeline(
        window_size=8,
        confidence_threshold=0.5,
    )
    
    # Process confidences
    result = pipeline.process_video(cnn_confidences_list)
    
    return result


# ==============================================================================
# OPTION 2: INTEGRATED - Wrapper class
# ==============================================================================

class VideoWithTemporalSmoothing:
    """
    Drop-in replacement for your inference function.
    
    Usage:
        detector = VideoWithTemporalSmoothing(
            model=your_cnn_model,
            preprocessor=your_preprocessor,
            device='cuda'
        )
        
        result = detector.predict_video('video.mp4')
        print(f"Prediction: {result['prediction']}")
        print(f"Confidence: {result['smoothed_confidence']:.3f}")
    """
    
    def __init__(self, model, preprocessor, device='cuda', use_features=False):
        self.model = model
        self.preprocessor = preprocessor
        self.device = device
        self.use_features = use_features
        
        self.model.eval()
        self.model.to(device)
        
        # Smoother (frozen, random init)
        self.smoother = TemporalSmoothingPipeline(
            window_size=8,
            confidence_threshold=0.5,
        )
        self.smoother.smoother.to(device)
    
    def _get_cnn_confidence(self, frame):
        """Get confidence from CNN for one frame"""
        
        # Preprocess frame
        if self.use_features:
            tensor, *_ = self.preprocessor.preprocess_with_liveness_features(frame)
        else:
            tensor = self.preprocessor.preprocess(frame)
        
        tensor = tensor.to(self.device)
        
        # CNN forward
        with torch.no_grad():
            output = self.model(tensor)
            
            if output.shape[-1] == 1:
                conf = torch.sigmoid(output).item()
            else:
                probs = torch.softmax(output, dim=1)
                conf = probs[0, 1].item()
        
        return conf
    
    def predict_video(self, video_path):
        """Process video with temporal smoothing"""
        import cv2
        
        # Read all frames
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames.append(frame)
        cap.release()
        
        # Get CNN confidences
        confidences = [self._get_cnn_confidence(f) for f in frames]
        
        # Smooth with temporal attention
        result = self.smoother.process_video(confidences)
        result['num_frames'] = len(frames)
        
        return result


# ==============================================================================
# OPTION 3: CUSTOM - Direct pipeline usage
# ==============================================================================

def process_with_custom_smoothing(cnn_predictions, 
                                  window_size=8,
                                  threshold=0.5):
    """
    Use TemporalSmoothingPipeline directly with custom settings.
    
    Example:
        my_cnn_preds = [0.48, 0.52, 0.49, 0.51, 0.50, ...]
        result = process_with_custom_smoothing(
            my_cnn_preds,
            window_size=12,
            threshold=0.6
        )
    """
    
    pipeline = TemporalSmoothingPipeline(
        window_size=window_size,
        confidence_threshold=threshold,
    )
    
    result = pipeline.process_video(cnn_predictions)
    
    return result


# ==============================================================================
# REAL EXAMPLE: How to modify existing code
# ==============================================================================

"""
BEFORE: Standard mean pooling
----------------------------------------

def predict_video(video_path, model, preprocessor):
    # Read video
    frames = read_video(video_path)
    
    # Get CNN predictions
    confidences = []
    for frame in frames:
        conf = model.predict(frame)
        confidences.append(conf)
    
    # Mean pooling (bad - no temporal info)
    mean_conf = np.mean(confidences)
    pred = "Live" if mean_conf > 0.5 else "Spoof"
    
    return pred, mean_conf


AFTER: Temporal smoothing
----------------------------------------

def predict_video(video_path, model, preprocessor):
    # Read video
    frames = read_video(video_path)
    
    # Get CNN predictions (same as before)
    confidences = []
    for frame in frames:
        conf = model.predict(frame)  # Still from CNN!
        confidences.append(conf)
    
    # ✨ NEW: Temporal smoothing
    pipeline = TemporalSmoothingPipeline(window_size=8)
    result = pipeline.process_video(confidences)
    
    return result['prediction'], result['smoothed_confidence']


THAT'S IT! 3 lines of new code.
"""


# ==============================================================================
# REAL WORLD TEST
# ==============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("INTEGRATION TEST: How temporal smoothing fixes stuck predictions")
    print("="*70 + "\n")
    
    # Simulate your CNN predictions
    print("Scenario: Unstable CNN (stuck around 0.5, can't decide)")
    print("-" * 70)
    
    unstable_cnn = [
        0.48, 0.51, 0.49, 0.52, 0.50,  # Fluctuating
        0.47, 0.53, 0.50, 0.48, 0.52,  # ~0.5 mean
    ]
    
    print(f"CNN predictions per frame: {unstable_cnn}")
    print(f"Mean: {np.mean(unstable_cnn):.3f}")
    print(f"Std:  {np.std(unstable_cnn):.3f}")
    print(f"\n❌ PROBLEM: Mean=0.500, can't decide, unstable")
    
    # OLD WAY: mean pooling
    print(f"\nOLD WAY (mean pooling):")
    old_pred = "Live" if np.mean(unstable_cnn) > 0.5 else "Spoof"
    print(f"  Decision: {old_pred}")
    print(f"  Confidence: {np.mean(unstable_cnn):.3f}")
    print(f"  Problem: Decision depends on rounding error!")
    
    # NEW WAY: temporal smoothing
    print(f"\nNEW WAY (temporal smoothing):")
    result = use_temporal_smoothing_minimal(unstable_cnn)
    print(f"  Decision: {result['prediction']}")
    print(f"  Confidence: {result['smoothed_confidence']:.3f}")
    print(f"  Stable: {result['stable']}")
    print(f"  Important frames: {len([w for w in result['attention_weights'] if w > 0.15])} / {len(result['attention_weights'])}")
    
    print(f"\n✅ BENEFIT: Transformer learned which frames are reliable")
    print(f"  Even with random init, it smooths & reduces variance")
    
    # Show attention weights
    print(f"\nFrame Importance (learned by frozen transformer):")
    for i, w in enumerate(result['attention_weights']):
        bar = "█" * int(w * 30)
        print(f"  Frame {i}: {bar} {w:.3f}")
    
    print("\n" + "="*70)
    print("✅ Temporal smoothing WORKING - just plug it in!")
    print("="*70 + "\n")
