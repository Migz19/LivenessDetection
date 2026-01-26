#!/usr/bin/env python3
"""Quick test of video processing with loaded weights"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import cv2
import torch
from models.cnn_model import load_cnn_model
from models.efficientnet_model import load_efficientnet_model
from utils.enhanced_inference import EnhancedLivenessInference
from utils.face_detection import FaceDetector

def create_test_video():
    """Create simple test frames"""
    frames = []
    for i in range(6):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * (100 + i*10)
        # Add a simple face region
        cv2.circle(frame, (320, 240), 80, (150, 100, 200), -1)
        cv2.circle(frame, (290, 220), 15, (220, 180, 160), -1)
        cv2.circle(frame, (350, 220), 15, (220, 180, 160), -1)
        frames.append(frame)
    return frames

print("Testing video inference with weights...\n")

device = torch.device('cpu')
print("1. Loading models...")
cnn = load_cnn_model(device=device)
eff = load_efficientnet_model(device=device)
print("   ✓ Models loaded\n")

print("2. Creating inference pipeline...")
inference = EnhancedLivenessInference(cnn, device)
print("   ✓ Inference ready\n")

print("3. Creating test video frames...")
frames = create_test_video()
print(f"   ✓ Created {len(frames)} frames\n")

print("4. Running video inference...")
try:
    face_bboxes = [(100, 100, 500, 380)] * len(frames)
    results = inference.predict_video_with_motion(frames, face_bboxes)
    
    print(f"   ✓ Inference complete")
    print(f"\n   Final Prediction: {results['final_prediction']}")
    print(f"   Final Confidence: {results['final_confidence']:.2%}")
    print(f"   Motion Prediction: {results['motion_prediction']}")
    print(f"   Live Frames: {results['live_count']}/{len(frames)}")
    print(f"   Fake Frames: {results['fake_count']}/{len(frames)}")
    print("\n✅ SUCCESS: Video processing works with your weights!")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    import traceback
    traceback.print_exc()
