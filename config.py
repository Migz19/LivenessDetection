"""
Configuration file for Liveness Detection App
"""

import torch
from pathlib import Path


# ============== PATHS ==============
PROJECT_ROOT = Path(__file__).parent
MODELS_DIR = PROJECT_ROOT / 'models'
UTILS_DIR = PROJECT_ROOT / 'utils'
WEIGHTS_DIR = PROJECT_ROOT / 'weights'

# Create directories if they don't exist
WEIGHTS_DIR.mkdir(exist_ok=True)

# ============== MODEL CONFIGURATION ==============

# CNN Model Config
CNN_CONFIG = {
    'input_size': 300,
    'in_channels': 3,
    'num_classes': 2,
    'weights_path': WEIGHTS_DIR / 'cnn_livness.pt',
}

# EfficientNet Config
EFFICIENTNET_CONFIG = {
    'input_size': 224,
    'in_channels': 3,
    'num_classes': 2,
    'pretrained': True,
    'weights_path': WEIGHTS_DIR / 'efficientnet.pt',
}

# ============== PREPROCESSING ==============

# Image normalization (ImageNet)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# ============== INFERENCE ==============

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Inference settings
INFERENCE_CONFIG = {
    'confidence_threshold': 0.5,
    'num_augmentations': 5,  # For uncertainty estimation
}

# ============== VIDEO ==============

# Video processing settings
VIDEO_CONFIG = {
    'default_frames': 10,
    'min_frames': 5,
    'max_frames': 30,
    'fps': 30,
}

# ============== FACE DETECTION ==============

# Face detection settings
FACE_DETECTION_CONFIG = {
    'model_selection': 1,  # 0 = short range, 1 = full range
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5,
    'face_padding': 0.1,  # 10% padding around face
}

# ============== STREAMING ==============

# Streamlit configuration
STREAMLIT_CONFIG = {
    'page_title': 'Liveness Detection App',
    'page_icon': '🎥',
    'layout': 'wide',
    'initial_sidebar_state': 'expanded',
}

# ============== LOGGING ==============

# Logging configuration
LOGGING_CONFIG = {
    'level': 'INFO',
    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
}

# ============== CLASS LABELS ==============

CLASS_LABELS = {
    0: 'Fake',
    1: 'Live'
}

LABEL_COLORS = {
    'Live': (0, 255, 0),   # Green
    'Fake': (0, 0, 255),   # Red
}

# ============== THRESHOLDS ==============

# Confidence thresholds
CONFIDENCE_THRESHOLDS = {
    'high': 0.8,
    'medium': 0.5,
    'low': 0.0,
}

# ============== AUGMENTATION ==============

# Data augmentation probabilities
AUGMENTATION_PROBS = {
    'horizontal_flip': 0.5,
    'vertical_flip': 0.0,
    'rotation': 0.5,
    'color_jitter': 0.5,
    'gaussian_blur': 0.4,
    'random_erasing': 0.4,
}

# ============== BATCH SIZE ==============

BATCH_SIZE = 32

# ============== TRAINING ==============

TRAINING_CONFIG = {
    'num_epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'weight_decay': 1e-4,
    'num_workers': 4,
}

# ============== VALIDATION ==============

# Model evaluation metrics
METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
