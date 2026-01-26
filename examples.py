"""
Example usage of the Liveness Detection system
"""

import torch
import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# Import components
from models.cnn_model import load_cnn_model
from models.efficientnet_model import load_efficientnet_model
from utils.preprocessing import ImagePreprocessor, VideoPreprocessor
from utils.face_detection import FaceDetector
from utils.inference import LivenessInference


def example_1_single_image():
    """Example 1: Single image liveness detection"""
    print("\n" + "="*60)
    print("Example 1: Single Image Detection")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_cnn_model(device=device)
    preprocessor = ImagePreprocessor(model_type='cnn')
    face_detector = FaceDetector()
    inference = LivenessInference(model, preprocessor, device)
    
    # Create a dummy image for demo
    # In practice, load a real image: cv2.imread('image.jpg')
    dummy_image = np.ones((480, 640, 3), dtype=np.uint8) * 100
    
    print("Image shape:", dummy_image.shape)
    
    # Detect faces
    faces = face_detector.detect_faces(dummy_image)
    print(f"Faces detected: {len(faces)}")
    
    if faces:
        # Get first face bbox
        bbox = faces[0]['bbox']
        
        # Run prediction
        prediction, confidence = inference.predict_single(
            image_array=dummy_image,
            face_bbox=bbox
        )
        
        print(f"Prediction: {prediction}")
        print(f"Confidence: {confidence:.2%}")
    else:
        print("No faces found in image")


def example_2_batch_processing():
    """Example 2: Batch processing multiple images"""
    print("\n" + "="*60)
    print("Example 2: Batch Processing")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_efficientnet_model(device=device)
    preprocessor = ImagePreprocessor(model_type='efficientnet')
    inference = LivenessInference(model, preprocessor, device)
    
    # Create dummy images
    num_images = 3
    images = [np.ones((480, 640, 3), dtype=np.uint8) * (100 + i*20) 
              for i in range(num_images)]
    
    print(f"Processing {num_images} images...")
    
    # Batch prediction
    predictions, confidences = inference.predict_batch(images)
    
    for i, (pred, conf) in enumerate(zip(predictions, confidences)):
        print(f"Image {i+1}: {pred} ({conf:.2%})")


def example_3_video_frames():
    """Example 3: Video frame analysis"""
    print("\n" + "="*60)
    print("Example 3: Video Frame Analysis")
    print("="*60)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_cnn_model(device=device)
    preprocessor = ImagePreprocessor(model_type='cnn')
    video_preprocessor = VideoPreprocessor(model_type='cnn')
    face_detector = FaceDetector()
    inference = LivenessInference(model, preprocessor, device)
    
    print("Creating synthetic video frames...")
    
    # Create dummy frames (simulating video)
    num_frames = 10
    frames = [np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8) 
              for _ in range(num_frames)]
    
    print(f"Analyzing {num_frames} frames...")
    
    # Get face bboxes from first frame
    faces = face_detector.detect_faces(frames[0])
    
    if faces:
        face_bboxes = [f['bbox'] for f in faces]
        
        # Analyze video
        results = inference.predict_video_frames(frames, face_bboxes)
        
        print(f"Overall Prediction: {results['overall_prediction']}")
        print(f"Overall Confidence: {results['overall_confidence']:.2%}")
        print(f"Live Frames: {results['live_count']}")
        print(f"Fake Frames: {results['fake_count']}")
    else:
        print("No faces detected in frames")


def example_4_compare_models():
    """Example 4: Compare CNN vs EfficientNet"""
    print("\n" + "="*60)
    print("Example 4: Model Comparison")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create dummy image
    image = np.ones((480, 640, 3), dtype=np.uint8) * 128
    
    # CNN Model
    print("\nCNN Model (300x300):")
    cnn_model = load_cnn_model(device=device)
    cnn_prep = ImagePreprocessor(model_type='cnn')
    cnn_inf = LivenessInference(cnn_model, cnn_prep, device)
    
    cnn_pred, cnn_conf = cnn_inf.predict_single(image_array=image)
    print(f"  Prediction: {cnn_pred} ({cnn_conf:.2%})")
    
    # EfficientNet Model
    print("\nEfficientNet Model (224x224):")
    enet_model = load_efficientnet_model(device=device)
    enet_prep = ImagePreprocessor(model_type='efficientnet')
    enet_inf = LivenessInference(enet_model, enet_prep, device)
    
    enet_pred, enet_conf = enet_inf.predict_single(image_array=image)
    print(f"  Prediction: {enet_pred} ({enet_conf:.2%})")


def example_5_uncertainty_estimation():
    """Example 5: Uncertainty estimation using augmentation"""
    print("\n" + "="*60)
    print("Example 5: Uncertainty Estimation")
    print("="*60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_cnn_model(device=device)
    preprocessor = ImagePreprocessor(model_type='cnn')
    inference = LivenessInference(model, preprocessor, device)
    
    image = np.ones((480, 640, 3), dtype=np.uint8) * 150
    
    print("Running inference with augmentation-based uncertainty...")
    
    results = inference.predict_with_uncertainty(
        image_array=image,
        num_augmentations=5
    )
    
    print(f"Prediction: {results['prediction']}")
    print(f"Confidence: {results['confidence']:.2%}")
    print(f"Uncertainty: {results['uncertainty']:.2%}")
    print(f"All predictions: {results['all_predictions']}")
    print(f"All confidences: {[f'{c:.2%}' for c in results['all_confidences']]}")


def example_6_face_quality():
    """Example 6: Face quality assessment"""
    print("\n" + "="*60)
    print("Example 6: Face Quality Assessment")
    print("="*60)
    
    face_detector = FaceDetector()
    
    # Create test images with different qualities
    print("Testing face quality assessment...")
    
    # Clear face
    clear_face = np.ones((300, 300, 3), dtype=np.uint8) * 200
    cv2.rectangle(clear_face, (50, 50), (250, 250), (100, 100, 100), -1)
    
    # Blurry face
    blurry_face = cv2.GaussianBlur(clear_face, (21, 21), 0)
    
    # Dark face
    dark_face = (clear_face.astype(float) * 0.3).astype(np.uint8)
    
    images = [clear_face, blurry_face, dark_face]
    labels = ["Clear", "Blurry", "Dark"]
    
    for image, label in zip(images, labels):
        # Create dummy bbox
        bbox = (50, 50, 250, 250)
        quality = face_detector.get_face_quality(image, bbox)
        print(f"{label:10} - Quality: {quality:.2%}")


def example_7_multi_face():
    """Example 7: Multiple face detection and analysis"""
    print("\n" + "="*60)
    print("Example 7: Multi-Face Processing")
    print("="*60)
    
    from utils.face_detection import MultiiFaceProcessor
    
    processor = MultiiFaceProcessor()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create image with multiple faces (dummy)
    image = np.ones((480, 640, 3), dtype=np.uint8) * 120
    
    print("Assessing all faces in image...")
    
    face_info = processor.assess_all_faces(image)
    
    print(f"Total faces: {len(face_info)}")
    for i, info in enumerate(face_info):
        print(f"Face {i+1}:")
        print(f"  BBox: {info['bbox']}")
        print(f"  Confidence: {info['confidence']:.2%}")
        print(f"  Quality: {info['quality']:.2%}")


def main():
    """Run all examples"""
    print("\n" + "="*70)
    print(" LIVENESS DETECTION - USAGE EXAMPLES")
    print("="*70)
    
    try:
        # Run examples
        example_1_single_image()
        example_2_batch_processing()
        example_3_video_frames()
        example_4_compare_models()
        example_5_uncertainty_estimation()
        example_6_face_quality()
        example_7_multi_face()
        
        print("\n" + "="*70)
        print("✓ All examples completed successfully!")
        print("="*70)
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
