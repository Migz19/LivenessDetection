#!/usr/bin/env python3
"""Test the fixed motion detection and video inference"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
from models.cnn_model import load_cnn_model
from utils.enhanced_inference import EnhancedLivenessInference

def create_live_frames():
    """Create frames simulating a real person with subtle motion"""
    frames = []
    for i in range(9):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
        
        # Create a face-like region that moves slightly
        offset = int(i * 1.5)  # Subtle movement
        cv2.circle(frame, (320 + offset, 240), 80, (150, 100, 200), -1)
        cv2.circle(frame, (290 + offset, 220), 15, (220, 180, 160), -1)
        cv2.circle(frame, (350 + offset, 220), 15, (220, 180, 160), -1)
        
        frames.append(frame)
    return frames

def create_static_frames():
    """Create static frames (spoofed)"""
    base_frame = np.ones((480, 640, 3), dtype=np.uint8) * 120
    cv2.circle(base_frame, (320, 240), 80, (150, 100, 200), -1)
    cv2.circle(base_frame, (290, 220), 15, (220, 180, 160), -1)
    cv2.circle(base_frame, (350, 220), 15, (220, 180, 160), -1)
    return [base_frame.copy() for _ in range(9)]

print("Testing Fixed Motion Detection & Video Inference")
print("=" * 60)

device = torch.device('cpu')

print("\n1. Loading model with trained weights...")
cnn = load_cnn_model(device=device)
print("   ✓ CNN loaded")

print("\n2. Creating inference pipeline...")
inference = EnhancedLivenessInference(cnn, device)
print("   ✓ Inference ready")

# Test 1: Live video (with subtle motion)
print("\n3. Testing LIVE video (subtle motion)...")
live_frames = create_live_frames()
face_bboxes = [(100, 100, 500, 380)] * len(live_frames)
results = inference.predict_video_with_motion(live_frames, face_bboxes)

print(f"   Frame Predictions: {results['predictions']}")
print(f"   Live Frames: {results['live_count']}/9")
print(f"   Final Prediction: {results['final_prediction']}")
print(f"   Final Confidence: {results['final_confidence']:.2%}")
print(f"   Motion Prediction: {results['motion_prediction']}")

if results['final_prediction'] == 'Live':
    print("   ✅ CORRECT: Live video detected as Live")
else:
    print(f"   ⚠️  Got {results['final_prediction']} (should be Live)")

# Test 2: Fake video (static)
print("\n4. Testing FAKE video (completely static)...")
static_frames = create_static_frames()
results2 = inference.predict_video_with_motion(static_frames, face_bboxes)

print(f"   Frame Predictions: {results2['predictions']}")
print(f"   Live Frames: {results2['live_count']}/9")
print(f"   Final Prediction: {results2['final_prediction']}")
print(f"   Final Confidence: {results2['final_confidence']:.2%}")
print(f"   Motion Prediction: {results2['motion_prediction']}")

if results2['final_prediction'] in ['Fake', 'Uncertain'] or (
    results2['final_prediction'] == 'Live' and results2['final_confidence'] < 0.7
):
    print("   ✅ GOOD: Static video not highly confident as Live")
else:
    print(f"   ⚠️  Got {results2['final_prediction']} with {results2['final_confidence']:.0%}")

print("\n" + "=" * 60)
print("Test complete! Check the predictions above.")
