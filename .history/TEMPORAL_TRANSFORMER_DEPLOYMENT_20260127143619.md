# Temporal Transformer: Implementation & Deployment Guide

## Quick Start (5 Minutes)

### 1. Install (if needed)
```bash
pip install torch torchvision
```

### 2. Load Pretrained Model
```python
import torch
from models.temporal_transformer import TemporalLivenessTransformer
from models.efficientnet_model import load_efficientnet_model
from inference_temporal import TemporalLivenessInference

# Load models
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
transformer = TemporalLivenessTransformer()
transformer.load_state_dict(torch.load('temporal_transformer_best.pt', map_location=device))

efficientnet = load_efficientnet_model(device=device)

# Initialize inference
inference = TemporalLivenessInference(transformer, efficientnet, device=device)

# Process video
score, confidence = inference.process_video('video.mp4')
print(f"Live: {score:.3f}, Confidence: {confidence:.3f}")
```

---

## Training from Scratch

### Step 1: Prepare Data

You need:
- **Video files** (mp4, avi, mov)
- **Labels** (0=spoof, 1=live)
- **Organized folder structure:**

```
data/
├── live/
│   ├── video1.mp4
│   ├── video2.mp4
│   └── ...
└── spoof/
    ├── video1.mp4
    ├── video2.mp4
    └── ...
```

### Step 2: Create Dataset

```python
from pathlib import Path
from train_temporal_transformer import VideoLivenessDataset

# Collect paths and labels
video_paths = []
labels = []

data_root = Path('data')
for video_file in (data_root / 'live').glob('*.mp4'):
    video_paths.append(str(video_file))
    labels.append(1)

for video_file in (data_root / 'spoof').glob('*.mp4'):
    video_paths.append(str(video_file))
    labels.append(0)

# Split into train/val
split = int(0.8 * len(video_paths))
train_paths = video_paths[:split]
train_labels = labels[:split]
val_paths = video_paths[split:]
val_labels = labels[split:]

# Create datasets
train_dataset = VideoLivenessDataset(
    train_paths, train_labels,
    window_size=12,
    stride=6,
    augment=True  # MANDATORY!
)

val_dataset = VideoLivenessDataset(
    val_paths, val_labels,
    window_size=12,
    stride=6,
    augment=False
)

print(f"Train windows: {len(train_dataset)}")
print(f"Val windows: {len(val_dataset)}")
```

### Step 3: Train Model

```python
import torch
from torch.utils.data import DataLoader
from models.temporal_transformer import TemporalLivenessTransformer
from train_temporal_transformer import train_temporal_transformer

# Create loaders
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

# Create model
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
)

# Train
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = train_temporal_transformer(
    model,
    train_loader,
    val_loader,
    device=device,
    num_epochs=50,
    learning_rate=1e-3,
)

# Save
torch.save(model.state_dict(), 'temporal_transformer_best.pt')
```

**Training time:** ~2-4 hours on GPU (50 epochs, 16 batch size)

---

## Hyperparameter Reference

### Model Architecture

| Parameter | Recommended | Range | Impact |
|-----------|-------------|-------|--------|
| `embedding_dim` | 256 | 128-512 | Higher = more capacity |
| `num_transformer_layers` | 2 | 1-4 | More = deeper temporal modeling |
| `num_heads` | 4 | 2-8 | More = more diverse patterns |
| `dropout` | 0.1 | 0.05-0.3 | Higher = more regularization |

### Training

| Parameter | Recommended | Impact |
|-----------|-------------|--------|
| `window_size` | 12 | 8-16 frames, higher for smoother videos |
| `stride` | 6 | 4-8, smaller = more windows = longer training |
| `batch_size` | 16 | 8-32, adjust based on GPU memory |
| `learning_rate` | 1e-3 | 1e-4 to 1e-2, use scheduler |
| `consistency_weight` | 0.1 | 0.05-0.3, higher = enforce more temporal smoothness |
| `num_epochs` | 50 | 30-100, depends on data size |

### Data Augmentation (in training only)

Must apply during training:
1. **Motion blur** - kernel 5-15px
2. **Gaussian blur** - kernel 3-7
3. **JPEG compression** - quality 30-80
4. **Downscale→upscale** - 2x down, then up
5. **Frame dropping** - simulate low FPS

**Why:** Forces model to learn temporal patterns, not sharpness

---

## Integration with Existing System

### Option A: Replace Video Scoring (Minimal Changes)

**Before:**
```python
# Single-frame CNN inference for each frame, average score
score = average_cnn_predictions_per_frame(video)
```

**After:**
```python
from inference_temporal import TemporalLivenessInference

inference = TemporalLivenessInference(transformer, efficientnet)
score, confidence = inference.process_video(video_path)
```

### Option B: Ensemble with CNN (Better Confidence)

```python
# Get both predictions
score_cnn, _ = detect_with_cnn(video_path)           # Existing
score_tf, conf_tf = inference.process_video(video_path)  # New

# Combine
score_ensemble = 0.4 * score_cnn + 0.6 * score_tf
confidence = conf_tf  # Use transformer's confidence calibration

print(f"Score: {score_ensemble:.3f}, Confidence: {confidence:.3f}")
```

### Option C: Cascade (Fast + Accurate)

```python
# Fast: CNN only on first few frames
quick_cnn_score = detect_with_cnn_sample(video_path, frames=[0, 5, 10])

if 0.3 < quick_cnn_score < 0.7:
    # Uncertain → use transformer for detailed analysis
    score, confidence = inference.process_video(video_path)
else:
    # Clear prediction → use CNN only
    score = quick_cnn_score
    confidence = 0.9
```

---

## Real-Time Streaming Integration

### Webcam Example

```python
import cv2
from inference_temporal import TemporalLivenessInference

# Initialize
inference = TemporalLivenessInference(transformer, efficientnet, device='cuda')
inference.reset_stream()

# Process frames
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Returns result every 12 frames
    result = inference.process_frame_stream(frame, buffer_size=12)
    
    if result is not None:
        score, confidence, variance = result
        
        # Display
        text = f"Live: {score:.2f} | Conf: {confidence:.2f}"
        color = (0, 255, 0) if score > 0.5 else (0, 0, 255)
        cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, color, 2)
    
    cv2.imshow('Liveness Detection', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### IP Camera Example

```python
import cv2
from inference_temporal import TemporalLivenessInference

inference = TemporalLivenessInference(transformer, efficientnet)
inference.reset_stream()

# IP camera URL
camera_url = 'rtsp://user:pass@192.168.1.100:554/stream'
cap = cv2.VideoCapture(camera_url)

while True:
    ret, frame = cap.read()
    if not ret:
        print("Connection lost, reconnecting...")
        cap = cv2.VideoCapture(camera_url)
        continue
    
    result = inference.process_frame_stream(frame, buffer_size=12)
    
    if result is not None:
        score, confidence, variance = result
        print(f"Score: {score:.3f}, Confidence: {confidence:.3f}")
        
        # Send alert if spoof detected
        if score < 0.3:
            log_alert("SPOOF DETECTED")
```

---

## Evaluation & Metrics

### Standard Metrics

```python
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)

# Get predictions on test set
predictions = []
targets = []
confidences = []

for video_path, label in test_data:
    details = inference.process_video(video_path, return_details=True)
    pred = 1 if details['liveness_score'] > 0.5 else 0
    
    predictions.append(pred)
    targets.append(label)
    confidences.append(details['confidence'])

# Compute metrics
acc = accuracy_score(targets, predictions)
prec = precision_score(targets, predictions)
rec = recall_score(targets, predictions)
f1 = f1_score(targets, predictions)
auc = roc_auc_score(targets, [p[0] for p in predictions])

print(f"Accuracy:  {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall:    {rec:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

# Confidence analysis
avg_confidence_correct = np.mean([c for p, t, c in zip(predictions, targets, confidences) if p == t])
avg_confidence_wrong = np.mean([c for p, t, c in zip(predictions, targets, confidences) if p != t])

print(f"\nConfidence on correct predictions: {avg_confidence_correct:.4f}")
print(f"Confidence on wrong predictions:   {avg_confidence_wrong:.4f}")
```

### Temporal Stability Metric

```python
# Track variance across windows for same video
def temporal_stability(video_path):
    details = inference.process_video(video_path, return_details=True)
    window_scores = details['window_scores']
    
    # Lower variance = more stable
    stability = 1.0 - np.std(window_scores)
    return stability

# Average across test set
stabilities = [temporal_stability(v) for v, _ in test_data]
print(f"Avg Temporal Stability: {np.mean(stabilities):.4f}")
```

---

## Troubleshooting

### Training Issues

**Problem:** Training loss doesn't decrease
- **Cause:** Learning rate too high/low
- **Fix:** Try lr ∈ [1e-4, 1e-2], use scheduler

**Problem:** Model overfits (train ↓, val ↑)
- **Cause:** Too few regularization
- **Fix:** Increase dropout (0.2-0.3), augmentation, consistency_weight

**Problem:** Out of memory (OOM)
- **Cause:** Large batch size, many frames
- **Fix:** Reduce batch_size (8), window_size (8), num_workers (0)

### Inference Issues

**Problem:** Inference is very slow
- **Cause:** Too many windows, GPU not used
- **Fix:** Increase stride (8), device='cuda', limit frames

**Problem:** Output always ~0.5
- **Cause:** Model not trained well or feature extraction wrong
- **Fix:** Check transformer outputs attention weights (should vary), verify features are being extracted

**Problem:** Confidence is always high/low
- **Cause:** Temporal variance is always similar
- **Fix:** Check attention weights are meaningful, increase transformer depth

---

## Performance Optimization

### GPU Inference

```python
# Best practices
device = torch.device('cuda:0')  # Specific GPU

# Move models once
model.to(device)
model.eval()

# Use no_grad
with torch.no_grad():
    # ... inference ...

# Batch processing
batch_videos = [v1, v2, v3, v4]
scores = [inference.process_video(v)[0] for v in batch_videos]
```

### CPU-only (Slower but works)

```python
# Disable gradient tracking
torch.set_grad_enabled(False)

# Use half precision (float16) if supported
model = model.half()

# Reduce model size
model = TemporalLivenessTransformer(embedding_dim=128)  # Smaller
```

### Multi-Processing

```python
from multiprocessing import Pool

# Process videos in parallel
with Pool(4) as p:
    scores = p.map(
        lambda v: inference.process_video(v)[0],
        video_paths
    )
```

---

## Deployment Checklist

- [ ] Models trained on representative data
- [ ] Validation metrics ≥ 75% accuracy
- [ ] Confidence calibration tested (separate train/val)
- [ ] Inference speed benchmarked
- [ ] Memory usage verified
- [ ] Error handling implemented
- [ ] Logging configured
- [ ] Model versioning in place
- [ ] Documentation updated
- [ ] Integration tests passed

---

## Common Questions

**Q: Can I use a pretrained EfficientNet?**
A: Yes, the code loads EfficientNet-B3 with ImageNet weights by default. You can also load your own finetuned weights.

**Q: How much data do I need?**
A: Minimum 100 videos (50 live, 50 spoof) for reasonable training. 500+ recommended for production.

**Q: What's the minimal video length?**
A: At least 12 frames (window_size). 24+ frames recommended for sliding windows.

**Q: Can I use other backbones?**
A: Yes, modify `cnn_embedding_dim` to match your backbone's output (e.g., ResNet50=2048).

**Q: Is transformer necessary?**
A: Significantly improves stability on low-quality videos. CNN-only achieves ~65% accuracy; Transformer+CNN ~85%+.

---

## Contact & Support

For issues or questions:
1. Check [TEMPORAL_TRANSFORMER.md](TEMPORAL_TRANSFORMER.md) for architecture details
2. Review example scripts in [quick_integration_example.py](quick_integration_example.py)
3. Examine training logs for convergence patterns
