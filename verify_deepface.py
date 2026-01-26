#!/usr/bin/env python
"""
Verification script for DeepFace integration
Tests that all components work together correctly
"""

import sys

def test_imports():
    """Test all critical imports"""
    print("🔍 Testing imports...")
    
    try:
        from utils.face_detection import FaceDetector, MultiiFaceProcessor
        print("  ✅ DeepFace components imported")
    except Exception as e:
        print(f"  ❌ DeepFace import failed: {e}")
        return False
    
    try:
        from models.cnn_model import load_cnn_model
        from models.efficientnet_model import load_efficientnet_model
        print("  ✅ Model classes imported")
    except Exception as e:
        print(f"  ❌ Model import failed: {e}")
        return False
    
    try:
        from utils.preprocessing import ImagePreprocessor, VideoPreprocessor
        print("  ✅ Preprocessing modules imported")
    except Exception as e:
        print(f"  ❌ Preprocessing import failed: {e}")
        return False
    
    try:
        from utils.inference import LivenessInference
        print("  ✅ Inference module imported")
    except Exception as e:
        print(f"  ❌ Inference import failed: {e}")
        return False
    
    return True

def test_detector_initialization():
    """Test FaceDetector can be initialized"""
    print("\n🔍 Testing FaceDetector initialization...")
    
    try:
        from utils.face_detection import FaceDetector
        detector = FaceDetector(detector_backend='opencv')
        print(f"  ✅ FaceDetector initialized with opencv backend")
        
        # Test available backends
        backends = ['opencv', 'retinaface', 'mtcnn']
        for backend in backends:
            try:
                det = FaceDetector(detector_backend=backend)
                print(f"    ✅ Backend '{backend}' available")
            except Exception as e:
                print(f"    ⚠️  Backend '{backend}' not available: {type(e).__name__}")
        
        return True
    except Exception as e:
        print(f"  ❌ FaceDetector initialization failed: {e}")
        return False

def test_model_loading():
    """Test that models load correctly"""
    print("\n🔍 Testing model loading...")
    
    try:
        import torch
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"  ℹ️  Using device: {device}")
        
        from models.cnn_model import load_cnn_model
        model = load_cnn_model(device=device)
        print("  ✅ CNN model loaded successfully")
        
        from models.efficientnet_model import load_efficientnet_model
        model = load_efficientnet_model(device=device, pretrained=True)
        print("  ✅ EfficientNet model loaded successfully")
        
        return True
    except Exception as e:
        print(f"  ❌ Model loading failed: {e}")
        return False

def test_processor():
    """Test MultiiFaceProcessor initialization"""
    print("\n🔍 Testing MultiiFaceProcessor...")
    
    try:
        from utils.face_detection import MultiiFaceProcessor
        processor = MultiiFaceProcessor()
        print("  ✅ MultiiFaceProcessor initialized")
        
        # Check available methods
        methods = ['extract_faces_from_video', 'extract_multiple_faces', 'save_faces']
        for method in methods:
            if hasattr(processor, method):
                print(f"    ✅ Method '{method}' available")
            else:
                print(f"    ⚠️  Method '{method}' not found")
        
        return True
    except Exception as e:
        print(f"  ❌ MultiiFaceProcessor test failed: {e}")
        return False

def main():
    print("=" * 60)
    print("DeepFace Integration Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Imports", test_imports()))
    results.append(("FaceDetector", test_detector_initialization()))
    results.append(("Models", test_model_loading()))
    results.append(("Processor", test_processor()))
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name:.<40} {status}")
    
    all_passed = all(result for _, result in results)
    
    if all_passed:
        print("\n🎉 All tests passed! DeepFace integration is ready.")
        print("\nNext steps:")
        print("  1. Run: streamlit run app.py")
        print("  2. Open: http://localhost:8501")
        print("  3. Test image, video, or webcam inputs")
        return 0
    else:
        print("\n⚠️  Some tests failed. Check errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
