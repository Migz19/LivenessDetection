"""
Practical example: How to use temporal smoothing

3 scenarios:
1. Video prediction with smoothing
2. Real-time streaming
3. Comparison: before/after smoothing
"""

import numpy as np
from temporal_smoother import TemporalSmoothingPipeline


def example_1_video_smoothing():
    """Scenario 1: Stabilize video predictions"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Video Prediction with Temporal Smoothing")
    print("="*60)
    
    # Simulate noisy CNN predictions (stuck around 0.5)
    noisy_predictions = [
        0.48, 0.52, 0.49, 0.51, 0.50, 0.47, 0.53, 0.50,  # ~0.5 (unstable)
    ]
    
    pipeline = TemporalSmoothingPipeline(
        window_size=8,
        confidence_threshold=0.5,
    )
    
    result = pipeline.process_video(noisy_predictions)
    
    print(f"\nCNN Predictions (raw):      {noisy_predictions}")
    print(f"  Mean:           {result['raw_confidence']:.3f}")
    print(f"  Std:            {result['raw_std']:.3f}")
    print(f"  Decision:       {result['prediction']}")
    
    print(f"\nAfter Temporal Smoothing:")
    print(f"  Smoothed:       {result['smoothed_confidence']:.3f}")
    print(f"  Decision:       {result['prediction']}")
    print(f"  Stable:         {result['stable']}")
    
    print(f"\nFrame Importance (learned by attention):")
    for frame_id, importance in result['frame_importance'].items():
        bar = "█" * int(importance * 20)
        print(f"  Frame {frame_id}: {bar} {importance:.3f}")


def example_2_streaming():
    """Scenario 2: Real-time streaming"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Real-Time Streaming")
    print("="*60)
    
    pipeline = TemporalSmoothingPipeline(
        window_size=6,
        confidence_threshold=0.5,
    )
    
    # Simulate frame-by-frame predictions
    incoming_predictions = [
        0.48, 0.51, 0.49, 0.52, 0.50,  # Buffering...
        0.48,  # Ready! First smoothed prediction
        0.85, 0.87,  # Signal changed to LIVE
        0.86,  # Updated smoothed decision
    ]
    
    print("\nFrame-by-frame processing:")
    print(f"(Window size: 6 frames)\n")
    
    for i, conf in enumerate(incoming_predictions):
        result = pipeline.process_streaming([conf])
        
        status = "✓ READY" if result.get('ready') else "✗ buffering"
        pred = result['prediction']
        buffered = result['frames_buffered']
        
        print(f"  Frame {i+1} (conf={conf:.2f}): {status:10} → {pred:5} (buffered: {buffered})")


def example_3_before_after():
    """Scenario 3: Comparison before/after smoothing"""
    print("\n" + "="*60)
    print("EXAMPLE 3: Before/After Comparison")
    print("="*60)
    
    # Generate different signal types
    scenarios = {
        "Stuck around 0.5": [0.48, 0.51, 0.49, 0.52, 0.50, 0.47, 0.53, 0.50],
        "Strong LIVE": [0.82, 0.85, 0.83, 0.86, 0.84, 0.85, 0.83, 0.84],
        "Weak LIVE": [0.58, 0.62, 0.59, 0.61, 0.60, 0.59, 0.61, 0.60],
        "Strong SPOOF": [0.15, 0.12, 0.14, 0.13, 0.11, 0.14, 0.12, 0.15],
        "Noisy mixed": [0.35, 0.65, 0.40, 0.70, 0.30, 0.75, 0.25, 0.80],
    }
    
    pipeline = TemporalSmoothingPipeline(confidence_threshold=0.5)
    
    print("\n{'Scenario':<20} {'Raw':>8} {'Smoothed':>10} {'Change':>10} {'Decision':>10}")
    print("-" * 60)
    
    for scenario_name, preds in scenarios.items():
        result = pipeline.process_video(preds)
        
        raw = result['raw_confidence']
        smoothed = result['smoothed_confidence']
        change = smoothed - raw
        decision = result['prediction']
        
        print(f"{scenario_name:<20} {raw:>8.3f} {smoothed:>10.3f} {change:>+10.3f} {decision:>10}")


def example_4_variance_reduction():
    """Scenario 4: Variance reduction metric"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Variance Reduction")
    print("="*60)
    
    pipeline = TemporalSmoothingPipeline()
    
    # High-frequency noise
    base = 0.7
    noise = np.sin(np.linspace(0, 4*np.pi, 12)) * 0.1 + np.random.normal(0, 0.05, 12)
    signal = np.clip(base + noise, 0, 1)
    
    result = pipeline.process_video(signal.tolist())
    
    print(f"\nNoisy signal (base=0.7):")
    print(f"  Original std:       {result['raw_std']:.4f}")
    print(f"  Variance reduction: {result['variance_reduction']:.4f}")
    print(f"  Reduction %:        {100 * result['variance_reduction'] / result['raw_std']:.1f}%")
    print(f"\n  Raw predictions:    {[f'{x:.2f}' for x in signal]}")
    print(f"  After smoothing:    {result['smoothed_confidence']:.4f}")


def example_5_attention_analysis():
    """Scenario 5: Understanding attention weights"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Attention Analysis")
    print("="*60)
    
    pipeline = TemporalSmoothingPipeline()
    
    # Mixed quality frames
    predictions = [
        0.85,  # Sharp, clear
        0.82,  # Sharp, clear
        0.25,  # Blur or artifact
        0.84,  # Sharp, clear
        0.15,  # Blur or artifact
        0.86,  # Sharp, clear
    ]
    
    result = pipeline.process_video(predictions)
    
    print(f"\nFrame-by-frame analysis:")
    print(f"{'Frame':<8} {'Prediction':<12} {'Attention':<20} {'Interpretation':<20}")
    print("-" * 60)
    
    for i, (pred, attn) in enumerate(zip(predictions, result['attention_weights'])):
        bar = "█" * int(attn * 15)
        
        if attn > 0.20:
            interp = "IMPORTANT"
        elif attn > 0.10:
            interp = "somewhat"
        else:
            interp = "downweighted"
        
        print(f"{i:<8} {pred:<12.2f} {bar:<20} {interp:<20}")
    
    print(f"\nResult: {result['prediction']} ({result['smoothed_confidence']:.3f})")
    print(f"\n→ Attention learned to trust sharp frames (0.85, 0.82, 0.84, 0.86)")
    print(f"→ Downweighted blurry frames (0.25, 0.15)")


if __name__ == '__main__':
    print("\n" + "="*60)
    print("TEMPORAL SMOOTHING - PRACTICAL EXAMPLES")
    print("="*60)
    
    example_1_video_smoothing()
    example_2_streaming()
    example_3_before_after()
    example_4_variance_reduction()
    example_5_attention_analysis()
    
    print("\n" + "="*60)
    print("✅ All examples complete")
    print("="*60 + "\n")
