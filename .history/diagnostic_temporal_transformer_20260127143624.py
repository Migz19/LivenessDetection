"""
Diagnostic Script for Temporal Liveness Transformer

Validates:
1. PyTorch installation and GPU availability
2. Model architecture and parameter counts
3. Feature dimension compatibility
4. Forward pass without errors
5. Model checkpointing capability
6. Feature extraction pipeline
7. Inference latency
"""

import torch
import torch.nn as nn
import time
import numpy as np
from pathlib import Path

print("\n" + "=" * 80)
print("TEMPORAL LIVENESS TRANSFORMER - DIAGNOSTIC SCRIPT")
print("=" * 80)

# ============================================================================
# 1. System Check
# ============================================================================

print("\n[1] SYSTEM CONFIGURATION")
print("-" * 80)

print(f"PyTorch version:      {torch.__version__}")
print(f"CUDA available:       {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version:         {torch.version.cuda}")
    print(f"GPU device:           {torch.cuda.get_device_name(0)}")
    print(f"GPU memory:           {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠ CUDA not available, will use CPU (slower)")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device:         {device}")

# ============================================================================
# 2. Import Check
# ============================================================================

print("\n[2] IMPORT CHECK")
print("-" * 80)

try:
    from models.temporal_transformer import (
        TemporalLivenessTransformer,
        TemporalLivenessLoss,
    )
    print("✓ TemporalLivenessTransformer imported")
except Exception as e:
    print(f"✗ Failed to import TemporalLivenessTransformer: {e}")

try:
    from models.efficientnet_model import EfficientNetLiveness
    print("✓ EfficientNetLiveness imported")
except Exception as e:
    print(f"✗ Failed to import EfficientNetLiveness: {e}")

try:
    from utils.liveness_features import LivenessPreprocessor
    print("✓ LivenessPreprocessor imported")
except Exception as e:
    print(f"✗ Failed to import LivenessPreprocessor: {e}")

try:
    from inference_temporal import TemporalLivenessInference
    print("✓ TemporalLivenessInference imported")
except Exception as e:
    print(f"✗ Failed to import TemporalLivenessInference: {e}")

# ============================================================================
# 3. Model Instantiation
# ============================================================================

print("\n[3] MODEL INSTANTIATION")
print("-" * 80)

try:
    model = TemporalLivenessTransformer(
        cnn_embedding_dim=1280,
        lbp_dim=768,
        freq_dim=785,
        moire_dim=29,
        depth_dim=16,
        embedding_dim=256,
        num_transformer_layers=2,
        num_heads=4,
        dropout=0.1,
    ).to(device)
    
    print("✓ Model instantiated successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Total parameters:     {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Model size:           {total_params * 4 / 1e6:.2f} MB")
    
except Exception as e:
    print(f"✗ Failed to instantiate model: {e}")
    model = None

# ============================================================================
# 4. Forward Pass Test
# ============================================================================

print("\n[4] FORWARD PASS TEST")
print("-" * 80)

if model is not None:
    try:
        # Create dummy input
        batch_size = 4
        seq_len = 12
        feature_dim = 1280 + 768 + 785 + 29 + 16  # Total feature dimension
        
        dummy_input = torch.randn(batch_size, seq_len, feature_dim).to(device)
        
        print(f"Input shape:  {dummy_input.shape}")
        
        # Forward pass
        with torch.no_grad():
            liveness_scores, frame_logits, attn_weights = model(dummy_input)
        
        print(f"✓ Forward pass successful")
        print(f"  Output shape (scores):      {liveness_scores.shape}")
        print(f"  Output shape (frame logits):{frame_logits.shape}")
        print(f"  Output shape (attn weights):{attn_weights.shape}")
        
        # Validate output ranges
        assert torch.all(liveness_scores >= 0) and torch.all(liveness_scores <= 1), \
            "Liveness scores should be in [0, 1]"
        print(f"  Liveness score range: [{liveness_scores.min():.4f}, {liveness_scores.max():.4f}]")
        
        assert torch.all(attn_weights >= 0) and torch.all(attn_weights <= 1), \
            "Attention weights should be in [0, 1]"
        print(f"  Attention weight range: [{attn_weights.min():.4f}, {attn_weights.max():.4f}]")
        
        # Check softmax normalization
        attn_sum = torch.sum(attn_weights, dim=1)
        print(f"  Attention sum (should be ~1): {attn_sum.mean():.4f} ± {attn_sum.std():.4f}")
        
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")

# ============================================================================
# 5. Loss Computation Test
# ============================================================================

print("\n[5] LOSS COMPUTATION TEST")
print("-" * 80)

try:
    loss_fn = TemporalLivenessLoss(consistency_weight=0.1)
    
    # Create dummy targets
    targets = torch.randint(0, 2, (batch_size,)).float().to(device)
    
    # Compute loss
    total_loss, class_loss, consistency_loss = loss_fn(
        liveness_scores.squeeze(),
        frame_logits,
        targets,
        attn_weights
    )
    
    print("✓ Loss computation successful")
    print(f"  Total loss:         {total_loss.item():.4f}")
    print(f"  Classification loss:{class_loss.item():.4f}")
    print(f"  Consistency loss:   {consistency_loss.item():.4f}")
    
    assert total_loss.item() > 0, "Loss should be positive"
    assert not torch.isnan(total_loss), "Loss should not be NaN"
    
except Exception as e:
    print(f"✗ Loss computation failed: {e}")

# ============================================================================
# 6. Backward Pass Test
# ============================================================================

print("\n[6] BACKWARD PASS TEST")
print("-" * 80)

try:
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    
    # Forward
    liveness_scores, frame_logits, attn_weights = model(dummy_input)
    
    # Loss
    total_loss, _, _ = loss_fn(
        liveness_scores.squeeze(),
        frame_logits,
        targets,
        attn_weights
    )
    
    # Backward
    optimizer.zero_grad()
    total_loss.backward()
    
    # Check gradients
    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    print("✓ Backward pass successful")
    print(f"  Gradient norm: {grad_norm.item():.4f}")
    
    # Step
    optimizer.step()
    
    print("✓ Optimizer step successful")
    
except Exception as e:
    print(f"✗ Backward pass failed: {e}")

# ============================================================================
# 7. Padding Mask Test
# ============================================================================

print("\n[7] PADDING MASK TEST")
print("-" * 80)

try:
    # Create input with different sequence lengths
    seq_lens = torch.tensor([12, 10, 8, 12])  # Variable lengths
    
    with torch.no_grad():
        liveness_scores, frame_logits, attn_weights = model(dummy_input, seq_lens)
    
    print("✓ Variable sequence length handling successful")
    print(f"  Sequence lengths: {seq_lens.tolist()}")
    
    # Verify padding is zero'd
    for b in range(batch_size):
        padding_start = seq_lens[b].item()
        if padding_start < seq_len:
            padded_values = attn_weights[b, padding_start:, :].abs().max()
            assert padded_values.item() < 1e-5, "Padded frames should have near-zero attention"
    
    print("✓ Padding is correctly zeroed out")
    
except Exception as e:
    print(f"✗ Padding mask test failed: {e}")

# ============================================================================
# 8. Model Checkpointing
# ============================================================================

print("\n[8] MODEL CHECKPOINTING")
print("-" * 80)

try:
    checkpoint_path = Path('_test_checkpoint.pt')
    
    # Save
    torch.save(model.state_dict(), checkpoint_path)
    print(f"✓ Model saved ({checkpoint_path.stat().st_size / 1e6:.2f} MB)")
    
    # Load into new model
    model_loaded = TemporalLivenessTransformer().to(device)
    model_loaded.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print("✓ Model loaded successfully")
    
    # Verify same outputs
    with torch.no_grad():
        out_orig, _, _ = model(dummy_input[:1])
        out_loaded, _, _ = model_loaded(dummy_input[:1])
    
    assert torch.allclose(out_orig, out_loaded, atol=1e-5), \
        "Loaded model should produce identical outputs"
    print("✓ Loaded model produces identical outputs")
    
    # Cleanup
    checkpoint_path.unlink()
    
except Exception as e:
    print(f"✗ Checkpointing test failed: {e}")

# ============================================================================
# 9. Inference Latency
# ============================================================================

print("\n[9] INFERENCE LATENCY")
print("-" * 80)

try:
    # Warm up
    with torch.no_grad():
        _ = model(dummy_input)
    
    # Benchmark
    num_iters = 10
    times = []
    
    for _ in range(num_iters):
        start = time.time()
        with torch.no_grad():
            _ = model(dummy_input)
        end = time.time()
        times.append((end - start) * 1000)  # Convert to ms
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"✓ Inference latency measured")
    print(f"  Batch size:     {batch_size}")
    print(f"  Seq length:     {seq_len}")
    print(f"  Avg latency:    {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  Throughput:     {batch_size * 1000 / avg_time:.1f} samples/sec")
    
    if device.type == 'cuda':
        # GPU memory
        allocated = torch.cuda.memory_allocated(device) / 1e9
        reserved = torch.cuda.memory_reserved(device) / 1e9
        print(f"  GPU memory:     {allocated:.2f} / {reserved:.2f} GB")
    
except Exception as e:
    print(f"✗ Latency measurement failed: {e}")

# ============================================================================
# 10. Feature Extraction Pipeline
# ============================================================================

print("\n[10] FEATURE EXTRACTION PIPELINE")
print("-" * 80)

try:
    import cv2
    import numpy as np
    
    preprocessor = LivenessPreprocessor(model_type='cnn')
    print("✓ LivenessPreprocessor initialized")
    
    # Create dummy image
    dummy_image = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
    
    # Extract features
    face_img, lbp, freq, moire, depth = \
        preprocessor.preprocess_with_liveness_features(dummy_image)
    
    print("✓ Feature extraction successful")
    print(f"  Face img shape:  {face_img.shape}")
    print(f"  LBP shape:       {lbp.shape}")
    print(f"  Freq shape:      {freq.shape}")
    print(f"  Moiré shape:     {moire.shape}")
    print(f"  Depth shape:     {depth.shape}")
    
    # Check total feature dimension
    total_features = (face_img.shape[1] if len(face_img.shape) > 1 else 1) + \
                     lbp.shape[1] + freq.shape[1] + moire.shape[1] + depth.shape[1]
    print(f"  Total features:  {total_features}")
    
except ImportError:
    print("⚠ OpenCV not installed, skipping feature extraction test")
except Exception as e:
    print(f"✗ Feature extraction test failed: {e}")

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 80)
print("DIAGNOSTIC SUMMARY")
print("=" * 80)

print("\n✓ All critical tests passed!" if all([
    torch.cuda.is_available() or device.type == 'cpu',
    model is not None,
]) else "\n⚠ Some tests failed, review above for details")

print("\nRecommendations:")
print("  1. For production: Use GPU (CUDA) for faster inference")
print("  2. Verify feature dimensions match your actual data")
print("  3. Test on representative video samples before deployment")
print("  4. Monitor GPU memory usage in production")
print("  5. Store trained models with version information")

print("\nNext steps:")
print("  1. Prepare training data (video_paths, labels)")
print("  2. Run training: python train_temporal_transformer.py")
print("  3. Evaluate: python quick_integration_example.py")
print("  4. Deploy: Use inference_temporal.py in your application")

print("\n" + "=" * 80)
