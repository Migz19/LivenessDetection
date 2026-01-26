#!/usr/bin/env python3
"""
Test enhanced inference with static video
"""

import numpy as np
import cv2
from utils.enhanced_inference import EnhancedLivenessInference
from models.cnn_model import load_cnn_model
import torch

def create_static_frames(num_frames=9, size=(300, 300, 3)):
    """Create static frames (like a spoofed video)"""
    # Create a single frame with face-like pattern
    base_frame = np.ones(size, dtype=np.uint8) * 128
    
    # Add some detail/texture
    cv2.circle(base_frame, (150, 150), 60, (100, 100, 150), -1)
    cv2.circle(base_frame, (130, 140), 12, (200, 150, 100), -1)
    cv2.circle(base_frame, (170, 140), 12, (200, 150, 100), -1)
    
    # Return the same frame repeated (static video)
    return [base_frame.copy() for _ in range(num_frames)]

def test_enhanced_inference():
    print("=" * 60)
    print("Testing Enhanced Inference with Static Video")
    print("=" * 60)
    
    # Load model
    print("\n1. Loading CNN model...")
    device = torch.device("cpu")
    model = load_cnn_model(device)
    print("   ✓ Model loaded")
    
    # Create frames
    print("\n2. Creating 9 static frames (spoofed video)...")
    frames = create_static_frames(num_frames=9, size=(300, 300, 3))
    print(f"   Created {len(frames)} frames")
    
    # Create enhanced inference
    print("\n3. Initializing EnhancedLivenessInference...")
    inference = EnhancedLivenessInference(model, device)
    print("   ✓ Initialized")
    
    # Create fake bboxes
    face_bboxes = [(50, 50, 250, 250)]  # Simple bbox
    
    # Predict
    print("\n4. Running prediction on static video...")
    results = inference.predict_video_with_motion(frames, face_bboxes)
    
    # Check results structure
    print("\n5. Results structure check:")
    print(f"   Keys: {list(results.keys())}")
    
    expected_keys = ['predictions', 'confidences', 'live_count', 'fake_count', 
                     'motion_prediction', 'motion_confidence', 'final_prediction', 
                     'final_confidence', 'features']
    
    for key in expected_keys:
        if key in results:
            print(f"   ✓ {key}: {type(results[key])}")
        else:
            print(f"   ✗ {key}: MISSING!")
    
    # Print results
    print("\n6. Prediction Results:")
    print(f"   Motion Prediction: {results['motion_prediction']}")
    print(f"   Motion Confidence: {results['motion_confidence']:.2%}")
    print(f"   Final Prediction: {results['final_prediction']}")
    print(f"   Final Confidence: {results['final_confidence']:.2%}")
    print(f"   Live Frames: {results['live_count']}/{len(frames)}")
    print(f"   Fake Frames: {results['fake_count']}/{len(frames)}")
    
    # Analyze individual frame predictions
    print("\n7. Individual Frame Predictions:")
    for idx, (pred, conf) in enumerate(zip(results['predictions'], results['confidences'])):
        status = "✅ Live" if pred == "Live" else "❌ Fake"
        print(f"   Frame {idx + 1}: {status} ({conf:.2%})")
    
    # Verdict
    print("\n8. Verdict:")
    if results['final_prediction'] == "Fake":
        print(f"   ✅ PASS: Correctly detected static video as FAKE!")
        print(f"      Confidence: {results['final_confidence']:.2%}")
    else:
        print(f"   ❌ FAIL: Should be FAKE but got {results['final_prediction']}")
        print(f"      Confidence: {results['final_confidence']:.2%}")
        print(f"      Motion said: {results['motion_prediction']} ({results['motion_confidence']:.2%})")
    
    print("\n" + "=" * 60)
    print("Enhanced Inference Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_enhanced_inference()
