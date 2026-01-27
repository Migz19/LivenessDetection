#!/usr/bin/env python3
"""
Verify pre-trained setup and test end-to-end
"""

import sys
import torch
import cv2
from pathlib import Path

def check_environment():
    """Check system environment"""
    print("\n" + "="*60)
    print("🔍 ENVIRONMENT CHECK")
    print("="*60)
    
    checks = {
        "Python version": f"{sys.version.split()[0]}",
        "PyTorch": f"{torch.__version__}",
        "CUDA available": f"{torch.cuda.is_available()}",
        "GPU device": f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}",
        "OpenCV": f"{cv2.__version__}",
    }
    
    for check, value in checks.items():
        print(f"  ✓ {check:<20} {value}")
    
    return torch.cuda.is_available()

def check_files():
    """Check if all required files exist"""
    print("\n" + "="*60)
    print("📁 FILE CHECK")
    print("="*60)
    
    required_files = [
        "models/temporal_transformer.py",
        "models/efficientnet_model.py",
        "utils/inference.py",
        "utils/liveness_features.py",
        "utils/preprocessing.py",
        "pretrained_inference.py",
        "inference_temporal.py",
    ]
    
    workspace_root = Path(__file__).parent
    missing = []
    
    for file_path in required_files:
        full_path = workspace_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (MISSING)")
            missing.append(file_path)
    
    return len(missing) == 0

def check_imports():
    """Check if all imports work"""
    print("\n" + "="*60)
    print("📦 IMPORT CHECK")
    print("="*60)
    
    imports = [
        ("torch", "torch"),
        ("torchvision.models", "torchvision"),
        ("cv2", "cv2"),
        ("numpy", "numpy"),
        ("models.temporal_transformer", "models.TemporalLivenessTransformer"),
        ("pretrained_inference", "pretrained_inference.PreTrainedLivenessDetector"),
    ]
    
    success = 0
    for import_path, display_name in imports:
        try:
            if "." in import_path and "models" not in import_path:
                parts = import_path.split(".")
                module = __import__(import_path)
                for part in parts[1:]:
                    module = getattr(module, part)
            else:
                __import__(import_path)
            print(f"  ✓ {display_name}")
            success += 1
        except Exception as e:
            print(f"  ✗ {display_name}")
            print(f"     Error: {str(e)[:100]}")
    
    return success == len(imports)

def test_model_creation():
    """Test if model can be created"""
    print("\n" + "="*60)
    print("🤖 MODEL CREATION TEST")
    print("="*60)
    
    try:
        from models.temporal_transformer import TemporalLivenessTransformer
        
        model = TemporalLivenessTransformer(
            feature_dim=3878,
            hidden_dim=256,
            num_heads=4,
            num_layers=2,
            max_seq_length=16,
        )
        
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  ✓ Model created successfully")
        print(f"    - Parameters: {param_count:,}")
        print(f"    - Device: {next(model.parameters()).device}")
        
        # Test forward pass
        batch_size = 2
        seq_length = 8
        x = torch.randn(batch_size, seq_length, 3878)
        
        with torch.no_grad():
            output = model(x)
        
        print(f"  ✓ Forward pass successful")
        print(f"    - Input shape: {x.shape}")
        print(f"    - Output shape: {output.shape}")
        print(f"    - Output range: [{output.min():.3f}, {output.max():.3f}]")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Model creation failed: {str(e)}")
        return False

def test_pretrained_detector():
    """Test PreTrainedLivenessDetector"""
    print("\n" + "="*60)
    print("🔬 PRETRAINED DETECTOR TEST")
    print("="*60)
    
    try:
        from pretrained_inference import PreTrainedLivenessDetector
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        detector = PreTrainedLivenessDetector(device=device)
        
        print(f"  ✓ PreTrainedLivenessDetector created")
        print(f"    - Device: {device}")
        print(f"    - EfficientNet loaded: Yes")
        print(f"    - Transformer initialized: Yes")
        
        # Test single frame prediction
        dummy_frame = torch.randn(3, 224, 224)
        with torch.no_grad():
            pred, conf = detector.predict_frame(dummy_frame)
        
        print(f"  ✓ Frame prediction works")
        print(f"    - Prediction: {pred}")
        print(f"    - Confidence: {conf:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  ✗ Detector test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def print_next_steps(all_checks_passed):
    """Print next steps"""
    print("\n" + "="*60)
    print("🎯 NEXT STEPS")
    print("="*60)
    
    if all_checks_passed:
        print("""
  ✅ All checks passed! You're ready to use the pre-trained model.
  
  QUICK START:
  
  1. Test with webcam:
     python -c "from pretrained_inference import demo_with_webcam; demo_with_webcam()"
  
  2. Test with a video file:
     python -c "
from pretrained_inference import PreTrainedLivenessDetector
detector = PreTrainedLivenessDetector()
result = detector.predict_video('your_video.mp4')
print(result)
     "
  
  3. Read the guide:
     cat PRETRAINED_QUICKSTART.md
     
  4. When ready to train:
     python train_temporal_transformer.py --help
        """)
    else:
        print("""
  ⚠️  Some checks failed. Please:
  
  1. Install missing dependencies:
     pip install -r requirements.txt
     pip install -r requirements-optional.txt
  
  2. Check file structure:
     ls -la models/
     ls -la utils/
  
  3. Run this script again:
     python verify_pretrained_setup.py
        """)

def main():
    """Run all checks"""
    print("\n" + "🔧 LIVENESS DETECTION - SETUP VERIFICATION\n")
    
    checks = {
        "Environment": check_environment(),
        "Files": check_files(),
        "Imports": check_imports(),
        "Model Creation": test_model_creation(),
        "Detector": test_pretrained_detector(),
    }
    
    print("\n" + "="*60)
    print("✅ SUMMARY")
    print("="*60)
    
    all_passed = True
    for check_name, result in checks.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {check_name}")
        if not result:
            all_passed = False
    
    print_next_steps(all_passed)
    
    if all_passed:
        print("\n🎉 Setup verified successfully!\n")
        return 0
    else:
        print("\n⚠️  Setup verification completed with issues.\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
