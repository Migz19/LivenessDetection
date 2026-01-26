#!/usr/bin/env python
"""
Test script to verify the video processing fix
"""

import numpy as np
import torch
from utils.preprocessing import ImagePreprocessor

def test_single_bbox_multi_frames():
    """Test single bbox with multiple frames (most common case)"""
    print("Testing: Single bbox applied to multiple frames...")
    
    preprocessor = ImagePreprocessor(model_type='cnn')
    
    # Create fake frames
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
    
    # Single bbox (what the video app now uses)
    bbox_list = [(50, 50, 150, 150)]
    
    try:
        tensor = preprocessor.preprocess_batch(frames, bbox_list)
        print(f"  ✅ SUCCESS: Processed {len(frames)} frames with {len(bbox_list)} bbox")
        print(f"     Output shape: {tensor.shape}")
        assert tensor.shape[0] == 10, "Wrong batch size"
        print(f"  ✅ Batch size correct: {tensor.shape[0]}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def test_multiple_bboxes_matching_frames():
    """Test multiple bboxes matching frame count"""
    print("\nTesting: One bbox per frame (one-to-one mapping)...")
    
    preprocessor = ImagePreprocessor(model_type='cnn')
    
    # Create fake frames
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(5)]
    
    # One bbox per frame (for per-frame face tracking)
    bbox_list = [(50, 50, 150, 150), (55, 55, 155, 155), 
                 (60, 60, 160, 160), (65, 65, 165, 165), 
                 (70, 70, 170, 170)]
    
    try:
        tensor = preprocessor.preprocess_batch(frames, bbox_list)
        print(f"  ✅ SUCCESS: Processed {len(frames)} frames with {len(bbox_list)} bboxes")
        print(f"     Output shape: {tensor.shape}")
        assert tensor.shape[0] == 5, "Wrong batch size"
        print(f"  ✅ Batch size correct: {tensor.shape[0]}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def test_no_bboxes():
    """Test with no bboxes"""
    print("\nTesting: No bboxes (None)...")
    
    preprocessor = ImagePreprocessor(model_type='cnn')
    
    # Create fake frames
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(8)]
    
    try:
        tensor = preprocessor.preprocess_batch(frames, None)
        print(f"  ✅ SUCCESS: Processed {len(frames)} frames with no bboxes")
        print(f"     Output shape: {tensor.shape}")
        assert tensor.shape[0] == 8, "Wrong batch size"
        print(f"  ✅ Batch size correct: {tensor.shape[0]}")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def test_mismatch_bboxes():
    """Test with mismatched bbox count (should not crash)"""
    print("\nTesting: Mismatched bbox count (3 bboxes, 10 frames)...")
    
    preprocessor = ImagePreprocessor(model_type='cnn')
    
    # Create fake frames
    frames = [np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8) for _ in range(10)]
    
    # Only 3 bboxes for 10 frames (the old broken case)
    bbox_list = [(50, 50, 150, 150), (55, 55, 155, 155), (60, 60, 160, 160)]
    
    try:
        tensor = preprocessor.preprocess_batch(frames, bbox_list)
        print(f"  ✅ SUCCESS: Processed {len(frames)} frames with {len(bbox_list)} bboxes (safe fallback)")
        print(f"     Output shape: {tensor.shape}")
        assert tensor.shape[0] == 10, "Wrong batch size"
        print(f"  ✅ Batch size correct: {tensor.shape[0]}")
        print(f"  ℹ️  Frames 0-2 used their bbox, frames 3-9 used first bbox as fallback")
        return True
    except Exception as e:
        print(f"  ❌ FAILED: {e}")
        return False

def main():
    print("=" * 70)
    print("VIDEO PROCESSING FIX - VERIFICATION TEST SUITE")
    print("=" * 70)
    
    results = []
    results.append(("Single bbox, multiple frames", test_single_bbox_multi_frames()))
    results.append(("Multiple bboxes (per-frame)", test_multiple_bboxes_matching_frames()))
    results.append(("No bboxes", test_no_bboxes()))
    results.append(("Mismatched bbox count (OLD BREAKING CASE)", test_mismatch_bboxes()))
    
    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<50} {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 70)
    if all_passed:
        print("🎉 ALL TESTS PASSED - VIDEO PROCESSING FIX IS WORKING!")
        print("\nYou can now safely process videos without index errors.")
        print("Try uploading your video again in the Streamlit app.")
        return 0
    else:
        print("⚠️  SOME TESTS FAILED - Please review the errors above")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
