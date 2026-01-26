#!/usr/bin/env python3
"""Direct test of enhanced inference without using run.py"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import cv2
from utils.enhanced_inference import EnhancedLivenessInference
from models.cnn_model import load_cnn_model
import torch

def create_static_frames(num_frames=9):
    """Create static frames"""
    base_frame = np.ones((300, 300, 3), dtype=np.uint8) * 128
    cv2.circle(base_frame, (150, 150), 60, (100, 100, 150), -1)
    cv2.circle(base_frame, (130, 140), 12, (200, 150, 100), -1)
    cv2.circle(base_frame, (170, 140), 12, (200, 150, 100), -1)
    return [base_frame.copy() for _ in range(num_frames)]

print("Testing Enhanced Inference...")
print("=" * 60)

try:
    device = torch.device("cpu")
    model = load_cnn_model(device)
    print("✓ Model loaded")
    
    frames = create_static_frames(9)
    print(f"✓ Created {len(frames)} static frames")
    
    inference = EnhancedLivenessInference(model, device)
    print("✓ Inference initialized")
    
    face_bboxes = [(50, 50, 250, 250)]
    print("✓ Running prediction...")
    
    results = inference.predict_video_with_motion(frames, face_bboxes)
    
    print("\nResults:")
    print(f"  Final Prediction: {results['final_prediction']}")
    print(f"  Final Confidence: {results['final_confidence']:.2%}")
    print(f"  Motion Prediction: {results['motion_prediction']}")
    print(f"  Motion Confidence: {results['motion_confidence']:.2%}")
    
    if results['final_prediction'] == 'Fake':
        print("\n✅ PASS: Correctly detected static video as FAKE")
    else:
        print(f"\n❌ FAIL: Should be FAKE but got {results['final_prediction']}")
        
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

print("=" * 60)
