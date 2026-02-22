# Liveness Detection: Improvements & Validation Guide

## Overview

This guide covers the comprehensive improvements made to the liveness detection system to **increase accuracy**, **improve explainability**, and provide **validation mechanisms** before backend integration.

---

## 🎯 Key Improvements

### 1. **Enhanced Response Schema**

#### Before:
```json
{
  "is_live": true,
  "status": "live",
  "message": "Liveness detected successfully"
}
```

#### After:
```json
{
  "is_live": true,
  "status": "live",
  "message": "Liveness detected successfully",
  "confidence": {
    "model_confidence": 0.92,
    "motion_confidence": 0.85,
    "temporal_confidence": 0.88,
    "texture_confidence": 0.79,
    "final_confidence": 0.78
  },
  "decision_factors": {
    "primary_factor": "Strong model agreement (28/30 frames)",
    "supporting_factors": [
      "Motion analysis confirms (0.85)",
      "Good video quality"
    ],
    "warning_flags": [],
    "model_frame_predictions": {
      "live_frames": 28,
      "spoof_frames": 2,
      "total_frames": 30
    }
  },
  "video_metrics": {
    "total_frames": 300,
    "processed_frames": 30,
    "frame_rate": 30.0,
    "video_duration": 10.0,
    "video_quality": "good",
    "blur_detected": false,
    "low_light_detected": false,
    "face_detected_frames": 30
  },
  "processing_time_ms": 2450.5,
  "model_version": "v1.0"
}
```

**Benefits:**
- ✅ Confidence breakdown explains which components agree/disagree
- ✅ Decision factors show WHY a decision was made
- ✅ Video metrics reveal quality issues
- ✅ Warning flags highlight potential accuracy concerns

---

### 2. **Video Quality Validation**

All videos are analyzed for quality issues BEFORE processing:

#### Quality Checks:
```
✓ File validation (format, size)
✓ Frame extraction (duration, FPS)
✓ Blur detection (Laplacian variance)
✓ Lighting analysis (brightness levels)
✓ Contrast analysis (texture visibility)
✓ Face detection (at least 1, max 1)
✓ Frame consistency (face detected in all frames)
```

#### Quality Penalties Applied:
- **Blur > 30%**: Reduces confidence up to 10%
- **Low light > 30%**: Reduces confidence up to 20%
- **Poor overall quality**: 15% confidence reduction

---

### 3. **Confidence Calibration**

The final confidence is now **calibrated** by:

1. **Model Prediction** (base confidence)
2. **Quality Score** (video quality penalty)
3. **Feature Score** (liveness feature quality)
4. **Model Agreement** (frame-to-frame consistency)

```
Final Confidence = Base × Quality × Features × Agreement
```

This ensures confidence scores reflect actual accuracy.

---

### 4. **Two API Endpoints for Different Use Cases**

#### Endpoint 1: Simple Response (Fast Integration)
```
POST /api/v1/liveness/detect
→ Returns: {is_live, status, message}
→ Use: Simple yes/no decisions
→ Response time: ~2-3 seconds
```

#### Endpoint 2: Detailed Response (Validation & Analysis)
```
POST /api/v1/liveness/detect-detailed
→ Returns: Full diagnostic response
→ Use: Before integration, accuracy analysis
→ Response time: ~2-3 seconds (same as simple)
```

---

### 5. **Warning Flags for Known Issues**

The system now detects and reports:

```
"warning_flags": [
  "Low frame agreement - model uncertain",
  "High blur in video - may affect accuracy",
  "Poor lighting conditions",
  "Very short video - recommend 2-5 seconds",
  "Low motion confidence - static video unusual"
]
```

**Action Items:**
- Flag ⚠️ = Recommendation to review or ask for new video
- No flags ✓ = Model confident in decision

---

## 🧪 Testing & Validation Framework

### Quick Start

#### 1. **Single Video Test**
```python
from utils.testing_framework import LivenessDetectionTester

tester = LivenessDetectionTester(base_url="http://localhost:8000")

# Test one video
result = tester.test_video(
    "path/to/video.mp4",
    expected_label="live"
)

print(result)
# {
#   'status': 'success',
#   'predicted': 'live',
#   'is_correct': True,
#   'confidence': 0.92,
#   'warnings': []
# }
```

#### 2. **Batch Testing**
```python
# Create test configuration
test_config = {
    'real_person_1.mp4': 'live',
    'real_person_2.mp4': 'live',
    'photo_attack.mp4': 'spoof',
    'screen_attack.mp4': 'spoof',
    'mask_attack.mp4': 'spoof'
}

# Run batch tests
results = tester.batch_test(
    "path/to/test_videos",
    test_config
)

# Get summary
metrics = tester.print_summary()
# Output:
# ============================================================
# VALIDATION SUMMARY
# ============================================================
# Total tests: 50
# Successful: 50
# With labels: 50
# Correct predictions: 48/50
# Overall Accuracy: 96.0%
# ============================================================
```

#### 3. **Confidence Calibration Analysis**
```python
# Check if confidence scores match reality
calibration = tester.validate_confidence_calibration(results)

# Output:
# {
#   '0.8-1.0': {
#     'sample_count': 35,
#     'accuracy': 0.94,      # 94% of high-confidence predictions correct
#     'calibrated': True
#   },
#   '0.6-0.8': {
#     'sample_count': 10,
#     'accuracy': 0.80,
#     'calibrated': True
#   },
#   ...
# }
```

#### 4. **Warning Flag Analysis**
```python
# See which warnings correlate with errors
warnings = tester.analyze_warning_flags(results)

# Output:
# {
#   'High blur in video': {
#     'sample_count': 8,
#     'error_rate': 0.25,    # 25% error when blur detected
#     'recommendation': 'REVIEW'
#   },
#   'Low frame agreement': {
#     'sample_count': 3,
#     'error_rate': 0.67,    # 67% error - high risk!
#     'recommendation': 'REVIEW'
#   }
# }
```

---

## 📊 Decision-Making Logic

### Primary Decision (Model-Based)
```
Frame Predictions:
  - Count Live frames
  - Count Spoof frames
  - Majority wins

Confidence = (max_count / total_frames)
```

### Secondary Factor (Motion Analysis)
```
If motion confirms prediction:
  Confidence boosted (up to 98%)

If motion contradicts prediction:
  Confidence reduced (±20%)

If motion uncertain:
  Uses model confidence as-is
```

### Tertiary Factor (Temporal Smoothing)
```
If enabled:
  Smooths confidence across frames
  Reduces frame-to-frame jitter
  Produces more stable predictions
```

### Final Decision
```
Final Confidence = Base × Quality × Motion × Temporal

if Final Confidence >= THRESHOLD (0.5):
  Decision = "LIVE"
else:
  Decision = "SPOOF"
```

---

## 🔧 Configuration

### Video Processing Settings
```python
# backend/core/config.py

# Thresholds
LIVENESS_THRESHOLD: float = 0.5          # Decision threshold
FACE_CONFIDENCE_THRESHOLD: float = 0.7   # Face detection confidence

# Video Limits
MAX_VIDEO_SIZE_MB: int = 50
MAX_VIDEO_DURATION_SECONDS: int = 300
MAX_FRAMES_TO_PROCESS: int = 30          # Process ~30 frames
FRAME_SAMPLE_RATE: int = 5               # Process every 5th frame

# Device
DEVICE: str = "cuda"  # or "cpu"
```

### Recommended Settings for Validation
```python
# Before Integration Testing
LIVENESS_THRESHOLD = 0.6          # Higher = stricter
MAX_FRAMES_TO_PROCESS = 50        # Process more frames for analysis
FRAME_SAMPLE_RATE = 2             # More frequent sampling
```

---

## ✅ Pre-Integration Checklist

### Phase 1: Single Model Testing
- [ ] Test 10 live videos → ≥90% recognized as "live"
- [ ] Test 10 spoof videos → ≥90% recognized as "spoof"
- [ ] Check warning flags → All expected issues flagged
- [ ] Review confidence scores → Are they ~same as accuracy?

### Phase 2: Edge Case Testing
- [ ] Poor lighting (low brightness)
- [ ] Motion blur
- [ ] Multiple attacks (photo, screen, mask)
- [ ] Different ethnicities
- [ ] Different age groups
- [ ] Various video qualities (480p, 720p, 1080p)

### Phase 3: Production Readiness
- [ ] Accuracy ≥95% on diverse test set
- [ ] No unexpected warning flags
- [ ] Processing time <5 seconds
- [ ] Confidence calibration confirmed
- [ ] Load test (10+ concurrent requests)

### Phase 4: Integration
- [ ] Deploy detailed endpoint first
- [ ] Monitor for 24-48 hours
- [ ] Analyze real-world accuracy
- [ ] Switch to simple endpoint (optional)
- [ ] Set up monitoring/alarms

---

## 🚀 Integration Best Practices

### 1. **Start with Detailed Endpoint**
```python
# Week 1-2: Use detailed endpoint for analysis
response = requests.post(
    "http://your-api/api/v1/liveness/detect-detailed",
    files={'file': video_file}
)
result = response.json()

# Check warnings before making decision
if result['decision_factors']['warning_flags']:
    log_warning(result)  # Review later

# Make decision based on confidence + warnings
confidence = result['confidence']['final_confidence']
if confidence >= 0.7 and not result['warning_flags']:
    is_live = result['is_live']
```

### 2. **Monitor Real-World Performance**
```python
# Log all predictions for analysis
{
    'timestamp': now,
    'user_id': user_id,
    'video_duration': duration,
    'is_live': result['is_live'],
    'confidence': result['confidence']['final_confidence'],
    'warnings': result['decision_factors']['warning_flags'],
    'quality': result['video_metrics']['video_quality']
}

# Analysis later:
# - Accuracy by confidence bin
# - False positive/negative patterns
# - Warning flag effectiveness
```

### 3. **Gradual Rollout**
```
Week 1: 10% of users, detailed endpoint
Week 2: 25% of users, if >95% accuracy
Week 3: 50% of users, if still >95%
Week 4: 100% of users
Switch to simple endpoint once stable
```

---

## 📈 Success Metrics

### Primary Metrics
| Metric | Target | How to Measure |
|--------|--------|---------------|
| **Accuracy** | ≥95% | Batch test with labeled dataset |
| **Confidence Calibration** | ±5% | Compare confidence vs. actual accuracy |
| **False Positive Rate** | <2% | Spoof videos incorrectly marked "live" |
| **False Negative Rate** | <3% | Live videos incorrectly marked "spoof" |

### Secondary Metrics
| Metric | Target | How to Measure |
|--------|--------|---------------|
| **Processing Time** | <5s | Average from real requests |
| **Warning Flag Accuracy** | ≥90% | Do warnings predict errors? |
| **Model Agreement** | >80% | How consistent are frame predictions? |
| **Video Quality Issues** | <10% | Percentage of videos with poor quality |

---

## 🐛 Troubleshooting

### Issue: Low Accuracy on Your Videos

**Checklist:**
```
1. Check video quality
   - Is duration 2-5 seconds? (too short/long affects results)
   - Is lighting sufficient? (>50 brightness)
   - Is motion smooth? (blur <30%)

2. Check warning flags
   - "Low frame agreement" → Model uncertain (retry with better video)
   - "High blur" → Ask user to film slower/steadier
   - "Poor lighting" → Request better lighting

3. Analyze confidence scores
   - If confidence 0.50-0.60 and multiple warnings
   - Consider asking for new video instead
```

### Issue: Too Many Warnings

**Solutions:**
```
1. Update LIVENESS_THRESHOLD (0.5 → 0.6)
   - Stricter decision boundary

2. Increase MAX_FRAMES_TO_PROCESS
   - Process more frames = better statistics

3. Adjust quality penalties
   - Reduce blur penalty if too strict
   - Reduce lighting penalty if too sensitive
```

### Issue: Inconsistent Predictions

**Diagnosis:**
```
1. Check model_frame_predictions
   - If ["live", "spoof", "live", "spoof"] → Jittery predictions
   - Solution: Enable temporal smoothing

2. Check motion_confidence
   - If <0.3 → Motion detection failing
   - Solution: Use static face detection instead

3. Check for warnings
   - Review all warning flags
   - May indicate video quality problem
```

---

## 🔐 Security Considerations

### Input Validation
```python
✓ File type validation (mp4, avi, mov, mkv only)
✓ File size limits (max 50MB)
✓ Video duration limits (max 300 seconds)
✓ Frame count limits (max 30 processed frames)
```

### Model Security
```python
✓ Model inference runs in isolated process
✓ Temporary files automatically cleaned up
✓ No sensitive data logged
✓ API key recommended for production
```

### Best Practices
```python
1. Add authentication to API endpoints
2. Rate limit: 10 requests/minute per user
3. Log all predictions for audit trail
4. Monitor for abuse (repeated failures)
5. Regular model retraining with new data
```

---

## 📚 File Structure

```
.
├── backend/
│   ├── routers/
│   │   └── liveness.py              # Two endpoints (simple + detailed)
│   ├── services/
│   │   └── liveness_services.py     # Enhanced with diagnostics
│   ├── schema/
│   │   └── model_schema.py          # New detailed response schemas
│   └── core/
│       └── config.py                # Configuration
├── utils/
│   ├── enhanced_inference.py        # Inference with motion + temporal
│   ├── face_detection.py            # Face detection
│   ├── liveness_features.py         # Feature extraction
│   ├── validation.py                # NEW: Quality validation & calibration
│   └── testing_framework.py         # NEW: Testing & analysis tools
└── README.md                        # This guide
```

---

## 🎓 Next Steps

1. **Run quick validation**: `python -m utils.testing_framework`
2. **Prepare test dataset**: Collect 50+ labeled videos
3. **Run batch tests**: Use `batch_test()` function
4. **Analyze results**: Use `print_summary()` and calibration checks
5. **Adjust thresholds**: Based on your accuracy needs
6. **Deploy carefully**: Start with detailed endpoint, monitor, then switch

---

## 📞 Support

For issues or questions:
1. Check warning flags in response
2. Review this guide's troubleshooting section
3. Analyze confidence breakdown
4. Run batch tests on similar videos
5. Review model decision factors

---

**Version**: v1.0  
**Last Updated**: February 2026  
**Status**: Ready for Production Testing
