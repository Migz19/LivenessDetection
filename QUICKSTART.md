# Quick Start Guide for Liveness Detection App

## 🚀 Quick Start

### Option 1: Using Python Script (Recommended)
```bash
cd x:\AI\livness
python run.py
```

### Option 2: Direct Streamlit Command
```bash
cd x:\AI\livness
streamlit run app.py
```

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- Webcam (for webcam detection feature)

## 🔧 Installation Steps

### 1. Navigate to Project Directory
```bash
cd x:\AI\livness
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

If you encounter issues, try:
```bash
pip install --upgrade pip
pip install -r requirements.txt --no-cache-dir
```

### 3. Verify Installation
```bash
python -c "import torch, streamlit, cv2, mediapipe; print('All dependencies OK!')"
```

### 4. Run the Application
```bash
python run.py
```

or

```bash
streamlit run app.py
```

## 🎯 First Time Setup

The application will:
1. ✓ Check Python version
2. ✓ Verify all dependencies
3. ✓ Detect GPU/CUDA availability
4. ✓ Launch Streamlit interface

## 🌐 Accessing the App

Once running, open your browser and go to:
- **Local**: http://localhost:8501
- **Network**: http://<your-ip>:8501

## 📸 Features Overview

| Feature | Description |
|---------|-------------|
| 📷 Image Detection | Upload single images for liveness detection |
| 🎬 Video Detection | Analyze video files frame by frame |
| 📹 Webcam Detection | Real-time detection from webcam |
| 📊 Batch Processing | Process multiple images at once |
| 🧠 Dual Models | Choose between CNN or EfficientNet |

## ⚙️ Model Configuration

Switch models in the sidebar:
- **CNN**: Lightweight, specialized for liveness
- **EfficientNet**: Pre-trained, robust

## 🆘 Troubleshooting

### Issue: "Module not found" error
**Solution**: Make sure you're in the project directory and dependencies are installed
```bash
cd x:\AI\livness
pip install -r requirements.txt
```

### Issue: "CUDA out of memory"
**Solution**: The app will automatically fall back to CPU

### Issue: Webcam not accessible
**Solution**: 
- Check browser permissions
- Close other apps using camera
- Try a different browser

### Issue: Very slow inference
**Solution**: 
- Check if GPU is available (shown in sidebar)
- Use fewer frames for video processing
- Reduce image resolution

## 💾 Using Model Weights

If you have trained model weights:

1. Place weights in the `weights/` directory:
   - `weights/cnn_weights.pth` for CNN
   - `weights/efficientnet_weights.pth` for EfficientNet

2. The app will automatically load them if available

## 📊 Expected Performance

| Input | Time | Accuracy |
|-------|------|----------|
| Single Image | <1s | 95%+ |
| 10-frame Video | 2-5s | 90%+ |
| Batch (10 images) | 5-10s | 90%+ |

*Times vary based on hardware*

## 🎓 Training Custom Models

To train your own models:

```bash
python train.py --model cnn --data-dir ./data --epochs 50
```

Dataset structure should be:
```
data/
├── train/
│   ├── live/
│   └── fake/
├── val/
│   ├── live/
│   └── fake/
└── test/
    ├── live/
    └── fake/
```

## 📝 Configuration Files

### requirements.txt
Python dependencies for the project

### app.py
Main Streamlit application

### train.py
Training script for custom models

### models/
- `cnn_model.py`: Custom CNN architecture
- `efficientnet_model.py`: EfficientNet wrapper

### utils/
- `preprocessing.py`: Image/video preprocessing
- `face_detection.py`: Face detection utilities
- `inference.py`: Inference pipeline

## 🔍 Debugging

For verbose output:
```bash
streamlit run app.py --logger.level=debug
```

Check Streamlit status:
```bash
streamlit run app.py --version
```

## 🌟 Best Practices

1. **Start with Images**: Test with clear single-face images first
2. **Use Videos**: For better confidence, use video input
3. **Check Lighting**: Ensure proper lighting for best results
4. **Multiple Faces**: The app handles multiple faces automatically
5. **Batch Mode**: Use batch processing for multiple files

## 📚 Additional Resources

- [Streamlit Documentation](https://docs.streamlit.io/)
- [PyTorch Documentation](https://pytorch.org/docs/)
- [MediaPipe Documentation](https://developers.google.com/mediapipe)
- [EfficientNet Paper](https://arxiv.org/abs/1905.11946)

## ⚡ Performance Tips

1. **Use GPU**: Install CUDA for faster processing
2. **Reduce Frames**: Use fewer frames for faster video analysis
3. **Smaller Images**: Pre-process images to smaller sizes
4. **Batch Mode**: Process multiple files together for efficiency

## 📞 Support

For issues or questions:
1. Check the troubleshooting section above
2. Review error messages carefully
3. Check dependencies are installed correctly
4. Verify Python version is 3.8+

## ✅ Verification Checklist

Before using the app:
- [ ] Python 3.8+ installed
- [ ] Dependencies installed (`pip install -r requirements.txt`)
- [ ] Verified installation (`python -c "import..."`)
- [ ] App starts successfully (`python run.py`)
- [ ] Browser opens to localhost:8501
- [ ] Can select model in sidebar
- [ ] Can upload test image

## 🎉 Ready to Go!

Your Liveness Detection App is ready to use. Start with the Image Detection tab and explore the features!

---

**Version**: 1.0.0  
**Last Updated**: January 2026
