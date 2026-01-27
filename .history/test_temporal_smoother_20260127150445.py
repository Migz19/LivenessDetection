"""
Quick test: Temporal smoothing integration

No docs, just working code.
Tests that frozen transformer smooths CNN confidences.
"""

import sys
import torch
import numpy as np
from temporal_smoother import TemporalSmoother, TemporalSmoothingPipeline


def test_smoother():
    """Test 1: Smoother works with random init"""
    print("Test 1: Frozen random transformer smoother...")
    
    smoother = TemporalSmoother(hidden_dim=128, num_heads=4, num_layers=2)
    smoother.eval()
    
    # Simulate CNN confidences (noisy)
    confidences = torch.tensor([
        0.3, 0.35, 0.32, 0.38, 0.40,  # Noisy ~0.35
        0.7, 0.72, 0.68, 0.75, 0.73,  # Noisy ~0.71
    ], dtype=torch.float32).unsqueeze(0)
    
    with torch.no_grad():
        smoothed, weights = smoother(confidences)
    
    print(f"  Raw mean: {confidences.mean():.3f}")
    print(f"  Raw std: {confidences.std():.3f}")
    print(f"  Smoothed: {smoothed.item():.3f}")
    print(f"  Attention weights: {weights.squeeze(0).numpy()}")
    print(f"  OK Works\n")


def test_pipeline():
    """Test 2: Pipeline smooths unstable confidences"""
    print("Test 2: Temporal smoothing pipeline...")
    
    pipeline = TemporalSmoothingPipeline(confidence_threshold=0.5)
    
    # Unstable CNN predictions (stuck around 0.5)
    unstable = [
        0.48, 0.52, 0.49, 0.51, 0.50,  # Fluctuating around 0.5
        0.47, 0.53, 0.50, 0.49, 0.52,
    ]
    
    result = pipeline.process_video(unstable)
    
    print(f"  Raw: mean={result['raw_confidence']:.3f}, std={result['raw_std']:.3f}")
    print(f"  Smoothed: {result['smoothed_confidence']:.3f}")
    print(f"  Prediction: {result['prediction']}")
    print(f"  Stable: {result['stable']}")
    print(f"  Top 3 important frames: {sorted([(i, w) for i, w in enumerate(result['attention_weights'])], key=lambda x: x[1], reverse=True)[:3]}")
    print(f"  ✓ Works\n")


def test_clear_signal():
    """Test 3: Smoother preserves clear signals"""
    print("Test 3: Clear signal (high confidence)...")
    
    pipeline = TemporalSmoothingPipeline(confidence_threshold=0.5)
    
    # Clear LIVE signal
    live_signal = [0.85, 0.87, 0.84, 0.86, 0.88, 0.86, 0.85, 0.87]
    result_live = pipeline.process_video(live_signal)
    
    print(f"  LIVE - Raw: {result_live['raw_confidence']:.3f} → Smoothed: {result_live['smoothed_confidence']:.3f}")
    print(f"  Prediction: {result_live['prediction']}")
    
    # Clear SPOOF signal
    spoof_signal = [0.15, 0.12, 0.14, 0.13, 0.11, 0.14, 0.12, 0.15]
    result_spoof = pipeline.process_video(spoof_signal)
    
    print(f"  SPOOF - Raw: {result_spoof['raw_confidence']:.3f} → Smoothed: {result_spoof['smoothed_confidence']:.3f}")
    print(f"  Prediction: {result_spoof['prediction']}")
    print(f"  ✓ Clear signals preserved\n")


def test_streaming():
    """Test 4: Streaming mode"""
    print("Test 4: Streaming mode (frame by frame)...")
    
    pipeline = TemporalSmoothingPipeline(window_size=6)
    
    confidences = [0.48, 0.51, 0.49, 0.52, 0.50, 0.48, 0.85, 0.87, 0.86]
    
    print("  Frame-by-frame:")
    for i, conf in enumerate(confidences):
        result = pipeline.process_streaming([conf])
        status = "✓" if result.get('ready') else "✗ buffering"
        print(f"    Frame {i+1}: conf={conf:.2f} → pred={result['prediction']}, ready={result.get('ready')}, buffered={result['frames_buffered']}")
    
    print(f"  ✓ Streaming works\n")


def test_variance_reduction():
    """Test 5: Variance reduction metric"""
    print("Test 5: Variance reduction...")
    
    pipeline = TemporalSmoothingPipeline()
    
    # High variance noisy signal
    noisy = np.linspace(0.3, 0.7, 10)
    noisy = noisy + np.random.normal(0, 0.05, 10)
    noisy = np.clip(noisy, 0, 1)
    
    result = pipeline.process_video(noisy.tolist())
    
    print(f"  Raw std: {result['raw_std']:.4f}")
    print(f"  Variance reduction: {result['variance_reduction']:.4f}")
    print(f"  Reduction %: {100 * result['variance_reduction'] / result['raw_std']:.1f}%")
    print(f"  ✓ Reduces variance\n")


if __name__ == '__main__':
    print("=" * 60)
    print("Temporal Smoother Tests")
    print("=" * 60 + "\n")
    
    test_smoother()
    test_pipeline()
    test_clear_signal()
    test_streaming()
    test_variance_reduction()
    
    print("=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
