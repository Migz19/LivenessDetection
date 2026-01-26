"""
Test script to verify installation and functionality
"""

import sys
import subprocess
from pathlib import Path


def test_imports():
    """Test if all required modules can be imported"""
    print("\n" + "="*60)
    print("Testing Imports...")
    print("="*60)
    
    modules = {
        'torch': 'PyTorch',
        'torchvision': 'TorchVision',
        'cv2': 'OpenCV',
        'PIL': 'Pillow',
        'numpy': 'NumPy',
        'mediapipe': 'MediaPipe',
        'streamlit': 'Streamlit',
    }
    
    all_ok = True
    for module, name in modules.items():
        try:
            __import__(module)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name}")
            all_ok = False
    
    return all_ok


def test_torch():
    """Test PyTorch and CUDA"""
    print("\n" + "="*60)
    print("Testing PyTorch Configuration...")
    print("="*60)
    
    import torch
    
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"✓ CUDA version: {torch.version.cuda}")
        print(f"✓ GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ Running on CPU (slower)")
    
    return True


def test_models():
    """Test model loading"""
    print("\n" + "="*60)
    print("Testing Model Loading...")
    print("="*60)
    
    try:
        from models.cnn_model import load_cnn_model
        print("✓ CNN model import OK")
        
        cnn = load_cnn_model()
        print(f"✓ CNN model created: {cnn.__class__.__name__}")
        
    except Exception as e:
        print(f"✗ CNN model error: {e}")
        return False
    
    try:
        from models.efficientnet_model import load_efficientnet_model
        print("✓ EfficientNet model import OK")
        
        enet = load_efficientnet_model()
        print(f"✓ EfficientNet model created: {enet.__class__.__name__}")
        
    except Exception as e:
        print(f"✗ EfficientNet model error: {e}")
        return False
    
    return True


def test_preprocessing():
    """Test preprocessing"""
    print("\n" + "="*60)
    print("Testing Preprocessing...")
    print("="*60)
    
    try:
        from utils.preprocessing import ImagePreprocessor, VideoPreprocessor
        print("✓ Preprocessing imports OK")
        
        img_prep = ImagePreprocessor(model_type='cnn')
        print("✓ ImagePreprocessor (CNN) created")
        
        img_prep = ImagePreprocessor(model_type='efficientnet')
        print("✓ ImagePreprocessor (EfficientNet) created")
        
        vid_prep = VideoPreprocessor(model_type='cnn')
        print("✓ VideoPreprocessor created")
        
    except Exception as e:
        print(f"✗ Preprocessing error: {e}")
        return False
    
    return True


def test_face_detection():
    """Test face detection"""
    print("\n" + "="*60)
    print("Testing Face Detection...")
    print("="*60)
    
    try:
        from utils.face_detection import FaceDetector, MultiiFaceProcessor
        print("✓ Face detection imports OK")
        
        detector = FaceDetector()
        print("✓ FaceDetector created")
        
        processor = MultiiFaceProcessor()
        print("✓ MultiiFaceProcessor created")
        
    except Exception as e:
        print(f"✗ Face detection error: {e}")
        return False
    
    return True


def test_inference():
    """Test inference pipeline"""
    print("\n" + "="*60)
    print("Testing Inference...")
    print("="*60)
    
    try:
        from utils.inference import LivenessInference, create_inference_engine
        print("✓ Inference imports OK")
        
        from models.cnn_model import load_cnn_model
        from utils.preprocessing import ImagePreprocessor
        import torch
        
        model = load_cnn_model()
        preprocessor = ImagePreprocessor(model_type='cnn')
        device = torch.device('cpu')
        
        engine = LivenessInference(model, preprocessor, device)
        print("✓ LivenessInference created")
        
    except Exception as e:
        print(f"✗ Inference error: {e}")
        return False
    
    return True


def test_files():
    """Test if all required files exist"""
    print("\n" + "="*60)
    print("Testing Files...")
    print("="*60)
    
    required_files = [
        'app.py',
        'config.py',
        'requirements.txt',
        'models/cnn_model.py',
        'models/efficientnet_model.py',
        'utils/preprocessing.py',
        'utils/face_detection.py',
        'utils/inference.py',
    ]
    
    all_ok = True
    for file_path in required_files:
        full_path = Path(file_path)
        if full_path.exists():
            print(f"✓ {file_path}")
        else:
            print(f"✗ {file_path}")
            all_ok = False
    
    return all_ok


def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" LIVENESS DETECTION APP - INSTALLATION TEST")
    print("="*70)
    
    results = {}
    
    # File check
    results['Files'] = test_files()
    
    # Import check
    results['Imports'] = test_imports()
    
    if not results['Imports']:
        print("\n⚠ Some dependencies are missing!")
        print("Run: pip install -r requirements.txt")
        return False
    
    # PyTorch check
    results['PyTorch'] = test_torch()
    
    # Model check
    results['Models'] = test_models()
    
    # Preprocessing check
    results['Preprocessing'] = test_preprocessing()
    
    # Face detection check
    results['Face Detection'] = test_face_detection()
    
    # Inference check
    results['Inference'] = test_inference()
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{test_name:20} {status}")
        if not passed:
            all_passed = False
    
    print("="*60)
    
    if all_passed:
        print("\n✓ All tests passed! You're ready to use the app.")
        print("\nRun the app with:")
        print("  python run.py")
        print("or")
        print("  streamlit run app.py")
        return True
    else:
        print("\n✗ Some tests failed. Please check the errors above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
