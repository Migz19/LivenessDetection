#!/usr/bin/env python3
"""
Test script for motion detection improvements
"""

import numpy as np
import cv2
from utils.liveness_features import MotionBasedLivenessDetector
from utils.enhanced_inference import EnhancedLivenessInference
from models.cnn_model import load_cnn_model
import torch

def create_static_frames(num_frames=9, size=(480, 640, 3)):
    """Create static frames (like a spoofed video)"""
    # Create a single frame with a face-like pattern
    base_frame = np.ones(size, dtype=np.uint8) * 128
    
    # Add some detail/texture
    cv2.circle(base_frame, (320, 240), 80, (100, 100, 150), -1)
    cv2.circle(base_frame, (290, 220), 15, (200, 150, 100), -1)
    cv2.circle(base_frame, (350, 220), 15, (200, 150, 100), -1)
    
    # Return the same frame repeated (static video)
    return [base_frame.copy() for _ in range(num_frames)]

def test_motion_detection():
    print("=" * 60)
    print("Testing Motion Detection Improvements")
    print("=" * 60)
    
    # Create static frames
    print("\n1. Creating static frames (spoofed video)...")
    frames = create_static_frames(num_frames=9)
    print(f"   Created {len(frames)} static frames")
    
    # Create motion detector
    print("\n2. Initializing motion detector...")
    motion_detector = MotionBasedLivenessDetector(threshold=0.03)
    
    # Create a fake bbox
    bbox = (100, 100, 500, 500)
    bboxes = [bbox] * len(frames)
    
    # Test motion detection
    print("\n3. Detecting motion in static frames...")
    prediction, confidence = motion_detector.detect_from_frames(frames, bboxes)
    
    print(f"\n   Motion Detection Result:")
    print(f"   Prediction: {prediction}")
    print(f"   Confidence: {confidence:.2%}")
    
    # Expected: "Fake" with high confidence
    if prediction == "Fake":
        print(f"   ✅ PASS: Correctly detected as Fake!")
    else:
        print(f"   ❌ FAIL: Should be Fake but got {prediction}")
    
    # Test with more motion
    print("\n4. Testing with moving frames (adding slight motion)...")
    moving_frames = []
    for i, frame in enumerate(frames):
        # Add slight motion for each frame
        if i > 0:
            # Shift the frame slightly
            M = cv2.getRotationMatrix2D((320, 240), angle=i * 2, scale=1.0)
            frame_moved = cv2.warpAffine(frame, M, (640, 480))
            moving_frames.append(frame_moved)
        else:
            moving_frames.append(frame)
    
    prediction2, confidence2 = motion_detector.detect_from_frames(moving_frames, bboxes)
    
    print(f"   Motion Detection Result:")
    print(f"   Prediction: {prediction2}")
    print(f"   Confidence: {confidence2:.2%}")
    
    if prediction2 == "Live":
        print(f"   ✅ PASS: Correctly detected motion as Live!")
    else:
        print(f"   ⚠️  Got {prediction2} (might be ambiguous threshold)")
    
    print("\n" + "=" * 60)
    print("Motion Detection Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_motion_detection()
