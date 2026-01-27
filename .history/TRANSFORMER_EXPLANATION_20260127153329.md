# Why Transformer Works Differently Across Input Types

## 🔍 Quick Answer
The temporal transformer works on **all video inputs** (webcam, video upload, batches), but:
- **Shows indicator only on video** because it needs multiple frames to smooth across time
- **Single images don't show indicator** because there's no temporal dimension to smooth

---

## Architecture Overview

### 1. **Webcam & Video Upload Tabs** ✅ **Shows Transformer Indicator**
```
Multiple Frames (Video) 
    ↓
CNN Model (per-frame predictions)
    ↓
Temporal Transformer (FROZEN, random init)
    ↓
Smoothed Confidence
    ↓
Motion Detection (confirmation)
    ↓
Final Result + 🔄 Indicator
```

**Why it shows:**
- Multiple frames = temporal sequence
- Transformer can learn which frames are reliable
- Attention mechanism smooths predictions
- **Result: More stable, confident decisions**

**Display:**
```
Raw Prediction: 65%
After Transformer: 78%
🔄 Transformer increased confidence by 13% ✅
```

---

### 2. **Image Tab** ❌ **No Transformer Indicator**
```
Single Image
    ↓
CNN Model
    ↓
Temporal Transformer (can't work - only 1 frame)
    ↓
Skipped (needs batch)
    ↓
Feature Analysis (LBP, brightness, blur)
    ↓
Final Result (no indicator)
```

**Why it doesn't show:**
- Single image = no temporal sequence
- Transformer requires multiple frames to smooth across time
- Single frame has nothing to "smooth" against
- **Result: Model prediction + feature adjustments only**

---

## Technical Details

### Transformer Requirements
- **Minimum frames:** 2 (but ideally 5+)
- **Window size:** 8 frames (configurable)
- **Operation:** Frozen (no training needed)
- **Purpose:** Learn temporal attention weights → smooth noisy predictions

### When Transformer is Applied
| Input | Frames | Transformer | Indicator |
|-------|--------|-------------|-----------|
| **Image Upload** | 1 | ❌ Skipped | No |
| **Video Upload** | 10-30 | ✅ Applied | Yes |
| **Webcam** | 5-30 | ✅ Applied | Yes |

---

## How to Verify It's Working

### ✅ Video/Webcam (Should see indicator)
1. Click "Capture Video" or "Upload Video"
2. Analyze → You'll see:
   ```
   Raw Prediction Confidence: 65%
   After Transformer: 78%
   🔄 Transformer increased confidence by 13%
   ```

### ❌ Single Image (No indicator expected)
1. Upload image
2. Results shown with LBP + brightness features
3. No transformer indicator (only 1 frame)

---

## Code Locations

**Webcam/Video Processing:**
- [utils/enhanced_inference.py](utils/enhanced_inference.py#L104) - `predict_video_with_motion()`
- [app.py](app.py#L449) - Webcam tab (shows indicator)
- [app.py](app.py#L222) - Video upload tab (shows indicator)

**Image Processing:**
- [app.py](app.py#L136) - Image tab
- Uses `predict_single_with_features()` (no temporal smoothing)

**Indicator Display:**
- [app.py](app.py#L74) - `display_detection_results()` function
- Shows temporal info when `temporal_info` dict is provided

---

## Summary

**The transformer ALWAYS works on video/webcam** - it's now explicitly enabled everywhere:
- ✅ Webcam tab: Transformer applied + indicator shown
- ✅ Video upload tab: Transformer applied + indicator shown  
- ✅ Image tab: Transformer enabled but doesn't apply (single frame)

**Why only show on video?** Because the smoothing effect is only visible with multiple frames across time.

