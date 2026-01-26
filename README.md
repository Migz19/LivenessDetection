# Liveness Detection Application

A comprehensive Streamlit-based application for detecting facial liveness using deep learning models. The app can identify whether a face is real or spoofed/fake.

## 🎯 Features

- **Multiple Input Sources**: Image, Video, Webcam
- **Two Models Available**:
  - Custom CNN (300x300 input)
  - EfficientNet-B0 (224x224 input)
- **Multi-face Detection**: Detect and analyze multiple faces simultaneously
- **Real-time Processing**: Fast inference on CPU or GPU
- **Batch Processing**: Process multiple images at once
- **Detailed Results**: Confidence scores and frame-by-frame analysis
- **Video Analysis**: Frame aggregation for more robust predictions

## 📋 Requirements

```
streamlit==1.28.0
torch==2.0.1
torchvision==0.15.2
opencv-python==4.8.1.78
mediapipe==0.10.5
pillow==10.1.0
numpy==1.24.3
scipy==1.11.4
```

## 🚀 Installation

1. **Clone or navigate to the project directory**:
```bash
cd x:\AI\livness
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Run the application**:
```bash
streamlit run app.py
```

The app will open in your default browser at `http://localhost:8501`

## 📖 Usage Guide

### 📷 Image Detection
1. Go to the **Image Detection** tab
2. Upload an image containing faces
3. Click **Detect Liveness**
4. View results with confidence scores for each detected face

### 🎬 Video Detection
1. Go to the **Video Detection** tab
2. Upload a video file
3. Select number of frames to analyze (5-30)
4. Click **Detect Liveness**
5. Get aggregated results and frame-by-frame analysis

### 📹 Webcam Detection
1. Go to the **Webcam Detection** tab
2. Specify number of frames to capture
3. Click **Capture from Webcam**
4. Allow camera access when prompted
5. Click **Analyze Captured Frames**
6. Get instant liveness detection results

### 📊 Batch Processing
1. Go to the **Batch Processing** tab
2. Upload multiple images
3. Click **Process All Images**
4. View statistics and detailed results for all images

## 🏗️ Project Structure

```
livness/
├── app.py                 # Main Streamlit application
├── requirements.txt       # Python dependencies
├── README.md             # This file
│
├── models/
│   ├── __init__.py
│   ├── cnn_model.py      # Custom CNN model
│   └── efficientnet_model.py  # EfficientNet model
│
├── utils/
│   ├── __init__.py
│   ├── preprocessing.py   # Image/video preprocessing
│   ├── face_detection.py  # Face detection utilities
│   └── inference.py       # Inference pipeline
│
└── weights/              # Directory for model weights
    ├── cnn_weights.pth   # CNN model weights (optional)
    └── efficientnet_weights.pth  # EfficientNet weights (optional)
```

## 🧠 Models

### Custom CNN Model
- **Input**: 300x300 RGB images
- **Architecture**: 5 Conv blocks with BatchNorm and MaxPooling
- **Output**: Binary classification (Real/Fake)
- **Advantage**: Fast, specialized for liveness detection

### EfficientNet-B0
- **Input**: 224x224 RGB images
- **Architecture**: Pre-trained on ImageNet
- **Output**: Binary classification (Real/Fake)
- **Advantage**: Better generalization, robust features

## 🔧 Configuration

### Model Selection
Use the sidebar to switch between CNN and EfficientNet models.

### Number of Frames
For video processing, adjust the number of frames to analyze:
- **5-10 frames**: Fast analysis
- **15-20 frames**: Balanced
- **25-30 frames**: Thorough analysis

## 📊 Interpretation

### Prediction Results
- **✅ Live**: Real face detected
- **❌ Fake**: Spoofed/fake face detected

### Confidence Score
- **High (>80%)**: Strong prediction
- **Medium (50-80%)**: Moderate confidence
- **Low (<50%)**: Uncertain, consider with caution

### Video Aggregation
- **Majority Voting**: Overall result based on frame predictions
- **Consistency**: Higher agreement = higher confidence

## 🎓 Technical Details

### Preprocessing Pipeline
1. **Face Detection**: MediaPipe FaceDetection
2. **Face Cropping**: Extract face region with padding
3. **Normalization**: ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
4. **Resizing**: Resize to model input size

### Inference Pipeline
1. Preprocess input (image/frame)
2. Run model forward pass
3. Compute softmax probabilities
4. Extract prediction and confidence

### Video Processing
1. Extract temporal frames
2. Detect faces in reference frame
3. Process all frames with same ROI
4. Aggregate predictions
5. Compute overall score

## 💡 Tips for Best Results

1. **Good Lighting**: Avoid harsh shadows
2. **Clear Face**: Ensure face is not obscured
3. **Still Position**: Keep head steady
4. **Quality Input**: Use clear, high-resolution sources
5. **Multiple Frames**: Videos provide more robust results
6. **Face Centered**: Keep face in center of frame

## ⚠️ Limitations

- Performance depends on face quality and size
- May struggle with extreme head angles (>45°)
- Lighting conditions significantly affect accuracy
- Very high-quality deepfakes might bypass detection
- Performance varies with different face ethnicities (bias exists)

## 🔒 Privacy & Security

- Images are processed locally (no cloud upload)
- No data is stored after processing
- Supports GPU acceleration for faster processing
- Can run completely offline

## 🐛 Troubleshooting

### Model Loading Issues
- Ensure PyTorch is installed correctly
- Check CUDA compatibility if using GPU
- Try CPU mode if GPU causes issues

### Face Detection Issues
- Ensure face is clearly visible
- Improve lighting conditions
- Keep face at reasonable distance from camera
- Avoid extreme angles

### Webcam Access Issues
- Check browser/system camera permissions
- Try different browser
- Ensure camera is not in use by other apps

## 📝 Model Training (Optional)

To train custom models, prepare dataset with:
- Real face images
- Spoofed face images (printed, screen, etc.)

Use the preprocessing transforms for data augmentation.

## 🤝 Contributing

Feel free to:
- Report issues
- Suggest improvements
- Submit enhancements
- Share feedback

## 📜 License

This project is provided as-is for educational and research purposes.

## 📧 Contact & Support

For issues or questions, please check the troubleshooting section or review the code comments.

## 🔄 Updates

**Version 1.0.0** (January 2026)
- Initial release
- CNN and EfficientNet models
- Image, video, and webcam support
- Batch processing
- Multi-face detection

---

**Enjoy using the Liveness Detection Application! 🎥✨**
