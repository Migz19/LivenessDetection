# 🚀 Quick Examples: Using the Improved Liveness Detection

This file contains practical examples showing how to use the enhanced liveness detection system.

---

## Example 1: Simple Single Video Test

### Python
```python
import requests

# Test one video
video_path = "path/to/test_video.mp4"

with open(video_path, 'rb') as f:
    response = requests.post(
        "http://localhost:8000/api/v1/liveness/detect-detailed",
        files={'file': f}
    )

result = response.json()

# Print results
print(f"Is Live: {result['is_live']}")
print(f"Confidence: {result['confidence']['final_confidence']:.2%}")
print(f"Message: {result['message']}")

if result['decision_factors']['warning_flags']:
    print("⚠️ Warnings:")
    for warning in result['decision_factors']['warning_flags']:
        print(f"   - {warning}")
```

**Output:**
```
Is Live: True
Confidence: 92.00%
Message: Liveness detected successfully
```

---

## Example 2: Batch Testing (Most Common Use Case)

### Python
```python
from utils.testing_framework import LivenessDetectionTester
import json

# Initialize tester
tester = LivenessDetectionTester(base_url="http://localhost:8000")

# Your test videos with expected labels
test_config = {
    "person_1_recording.mp4": "live",
    "person_2_recording.mp4": "live",
    "person_3_recording.mp4": "live",
    "photo_held_up.mp4": "spoof",
    "phone_screen_attack.mp4": "spoof",
    "mask_attack.mp4": "spoof"
}

# Run batch tests
print("Running batch tests...")
results = tester.batch_test("./test_videos", test_config)

# Get summary
metrics = tester.print_summary()

# Analyze calibration
calibration = tester.validate_confidence_calibration(results)
print("\nConfidence Calibration:")
for bin_name, metrics in calibration.items():
    print(f"  {bin_name}: {metrics['accuracy']*100:.1f}% accuracy")

# Analyze warnings
print("\nWarning Analysis:")
warnings = tester.analyze_warning_flags(results)
for warning, data in warnings.items():
    print(f"  {warning}: {data['error_rate']*100:.1f}% error rate")

# Export for later analysis
tester.export_results("batch_test_results.json")
```

**Output:**
```
Running batch tests...
person_1_recording.mp4 ✓ [live] confidence: 0.94
person_2_recording.mp4 ✓ [live] confidence: 0.91
person_3_recording.mp4 ✓ [live] confidence: 0.87
photo_held_up.mp4 ✓ [spoof] confidence: 0.12
phone_screen_attack.mp4 ✓ [spoof] confidence: 0.08
mask_attack.mp4 ✓ [spoof] confidence: 0.31

============================================================
VALIDATION SUMMARY
============================================================
Total tests: 6
Successful: 6
With labels: 6
Correct predictions: 6/6
Overall Accuracy: 100.0%
============================================================

Confidence Scores:
  Min: 0.081
  Max: 0.944
  Avg: 0.537

Confidence Calibration:
  0.0-0.2: 66.7% accuracy
  0.2-0.4: 100.0% accuracy
  0.8-1.0: 100.0% accuracy

Warning Analysis:
  High blur in video: 12.5% error rate
  Poor lighting conditions: 25.0% error rate
```

---

## Example 3: Real-World Integration

### Flask/FastAPI Application
```python
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import requests
import tempfile
import os

app = FastAPI()

LIVENESS_API = "http://liveness-service:8000"

@app.post("/verify-user")
async def verify_user(video: UploadFile = File(...)):
    """Verify user with liveness detection"""
    
    try:
        # Forward to liveness API
        with tempfile.NamedTemporaryFile(delete=False) as tmp:
            tmp.write(await video.read())
            tmp_path = tmp.name
        
        with open(tmp_path, 'rb') as f:
            response = requests.post(
                f"{LIVENESS_API}/api/v1/liveness/detect-detailed",
                files={'file': f},
                timeout=60
            )
        
        result = response.json()
        
        # Clean up
        os.unlink(tmp_path)
        video.file.close()
        
        # Decision logic
        confidence = result['confidence']['final_confidence']
        warnings = result['decision_factors']['warning_flags']
        is_live = result['is_live']
        
        # Tier 1: High confidence
        if confidence >= 0.85 and not warnings:
            return {
                "verification": "approved",
                "confidence_level": "high",
                "details": "Identity verified"
            }
        
        # Tier 2: Medium confidence
        elif confidence >= 0.70 and len(warnings) <= 1:
            if is_live:
                return {
                    "verification": "approved",
                    "confidence_level": "medium",
                    "details": f"Identity verified (confidence: {confidence:.0%})"
                }
            else:
                return {
                    "verification": "rejected",
                    "confidence_level": "medium",
                    "details": "Liveness check failed. Please try again.",
                    "next_steps": "retry"
                }
        
        # Tier 3: Low confidence
        else:
            return {
                "verification": "review_required",
                "confidence_level": "low",
                "details": "Unable to verify. Please submit a clear video.",
                "warnings": warnings,
                "next_steps": "retry"
            }
    
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "verification": "error",
                "error": str(e)
            }
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
```

**Usage:**
```bash
# Test endpoint
curl -X POST -F "video=@user_video.mp4" http://localhost:8001/verify-user

# Response - High Confidence
{
  "verification": "approved",
  "confidence_level": "high",
  "details": "Identity verified"
}

# Response - Low Quality Video
{
  "verification": "review_required",
  "confidence_level": "low",
  "details": "Unable to verify. Please submit a clear video.",
  "warnings": [
    "High blur in video - may affect accuracy",
    "Poor lighting conditions"
  ],
  "next_steps": "retry"
}
```

---

## Example 4: Testing Framework with Detailed Analysis

### Data Analysis
```python
from utils.testing_framework import LivenessDetectionTester
import pandas as pd
import json

# Run tests
tester = LivenessDetectionTester()
results = tester.batch_test("./test_videos", test_config)

# Convert to DataFrame for analysis
import pandas as pd

data = []
for r in results:
    if r['status'] == 'success':
        data.append({
            'video': r['video'],
            'expected': r['expected'],
            'predicted': r['predicted'],
            'is_correct': r['is_correct'],
            'confidence': r['confidence'],
            'num_warnings': len(r['warnings']),
            'has_blur': any('blur' in w.lower() for w in r['warnings']),
            'has_lighting': any('light' in w.lower() for w in r['warnings']),
            'processing_time_ms': r['processing_time_ms']
        })

df = pd.DataFrame(data)

# Analysis
print("Overall Accuracy:", df['is_correct'].mean() * 100)
print("\nAccuracy by prediction:")
print(df.groupby('predicted')['is_correct'].mean() * 100)

print("\nAccuracy by confidence range:")
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    mask = df['confidence'] >= threshold
    if mask.sum() > 0:
        acc = df[mask]['is_correct'].mean()
        count = mask.sum()
        print(f"  ≥ {threshold}: {acc*100:.1f}% ({count} videos)")

print("\nImpact of warnings:")
print(f"  Without warnings: {df[~df['has_blur'] & ~df['has_lighting']]['is_correct'].mean()*100:.1f}% accuracy")
print(f"  With blur warning: {df[df['has_blur']]['is_correct'].mean()*100:.1f}% accuracy")
print(f"  With lighting warning: {df[df['has_lighting']]['is_correct'].mean()*100:.1f}% accuracy")

# Export to CSV for reporting
df.to_csv("liveness_test_results.csv", index=False)
print("\nResults exported to: liveness_test_results.csv")
```

**Output:**
```
Overall Accuracy: 95.8%

Accuracy by prediction:
predicted
live    96.6%
spoof   94.1%

Accuracy by confidence range:
  ≥ 0.5: 95.8% (50 videos)
  ≥ 0.6: 97.1% (45 videos)
  ≥ 0.7: 99.1% (31 videos)
  ≥ 0.8: 100.0% (15 videos)
  ≥ 0.9: 100.0% (5 videos)

Impact of warnings:
  Without warnings: 98.2% accuracy
  With blur warning: 91.3% accuracy
  With lighting warning: 87.5% accuracy

Results exported to: liveness_test_results.csv
```

---

## Example 5: Command-Line Quick Test

### Using the Validation Script
```bash
# Create test config file
python validate_liveness.py --create-config

# Test single video
python validate_liveness.py --video sample.mp4 --expected live

# Batch test with detailed output
python validate_liveness.py --batch ./test_videos --config test_config.json

# Test with custom API
python validate_liveness.py --batch ./test_videos --api http://192.168.1.100:8000
```

---

## Example 6: Production Deployment

### Monitoring & Logging
```python
import requests
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(
    filename='liveness_predictions.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def verify_liveness(video_path, user_id):
    """Check liveness and log for monitoring"""
    
    try:
        with open(video_path, 'rb') as f:
            response = requests.post(
                "http://localhost:8000/api/v1/liveness/detect-detailed",
                files={'file': f},
                timeout=60
            )
        
        result = response.json()
        
        # Log prediction
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'is_live': result['is_live'],
            'confidence': result['confidence']['final_confidence'],
            'warnings': result['decision_factors']['warning_flags'],
            'video_quality': result['video_metrics']['video_quality'],
            'processing_time_ms': result['processing_time_ms']
        }
        
        logging.info(json.dumps(log_entry))
        
        # Make decision
        if result['is_live']:
            return {
                'approved': True,
                'reason': 'Liveness verified'
            }
        else:
            return {
                'approved': False,
                'reason': f"Liveness check failed (confidence: {result['confidence']['final_confidence']:.0%})"
            }
    
    except Exception as e:
        logging.error(f"Error processing video for user {user_id}: {str(e)}")
        return {
            'approved': None,
            'reason': 'Unable to process video'
        }

# Later: Analyze logs
# python analysis.py liveness_predictions.log
```

---

## Example 7: Threshold Tuning

### Finding Optimal Threshold
```python
from utils.testing_framework import LivenessDetectionTester
from sklearn.metrics import roc_curve, auc

# Get all predictions and actual labels
tester = LivenessDetectionTester()
results = tester.batch_test("./test_videos", test_config)

# Extract data
y_true = []
y_score = []

for r in results:
    if r['status'] == 'success' and r['expected'] is not None:
        y_true.append(1 if r['expected'].lower() == 'live' else 0)
        y_score.append(r['confidence'])

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_true, y_score)

# Find best threshold (Youden's J statistic)
j_scores = tpr - fpr
best_idx = j_scores.argmax()
best_threshold = thresholds[best_idx]

print(f"Recommended threshold: {best_threshold:.3f}")
print(f"  True Positive Rate: {tpr[best_idx]:.1%}")
print(f"  False Positive Rate: {fpr[best_idx]:.1%}")

# Update config
# LIVENESS_THRESHOLD = best_threshold
```

---

## Example 8: A/B Testing

### Comparing Model Versions
```python
# Test with current model
results_v1 = tester.batch_test("./test_videos", test_config)
accuracy_v1 = tester.print_summary()['accuracy']

# Deploy new model version
# (Update model weights, restart API)

# Test with new model  
results_v2 = tester.batch_test("./test_videos", test_config)
accuracy_v2 = tester.print_summary()['accuracy']

# Compare
print(f"Model v1 Accuracy: {accuracy_v1*100:.1f}%")
print(f"Model v2 Accuracy: {accuracy_v2*100:.1f}%")
print(f"Improvement: {(accuracy_v2 - accuracy_v1)*100:.1f}%")

if accuracy_v2 > accuracy_v1:
    print("✓ New model is better - proceed with deployment")
else:
    print("✗ Rollback to previous model")
```

---

## Tips & Tricks

### 1. **Handling Low Confidence Cases**
```python
confidence = result['confidence']['final_confidence']
warnings = result['decision_factors']['warning_flags']

if confidence >= 0.85:
    # High confidence - approve
    decision = "approved"
elif confidence >= 0.65:
    # Medium confidence - approve if no major warnings
    major_warnings = [w for w in warnings if any(
        x in w.lower() for x in ['frame agreement', 'blur']
    )]
    decision = "approved" if not major_warnings else "try_again"
else:
    # Low confidence - always request new video
    decision = "try_again"
```

### 2. **Improving Video Quality**
```python
video_quality = result['video_metrics']

if video_quality['blur_detected']:
    feedback = "Please film more slowly and steadily"
if video_quality['low_light_detected']:
    feedback = "Please ensure good lighting"
if video_quality['video_duration'] < 2:
    feedback = "Please film for at least 2 seconds"
```

### 3. **Batch Processing with Progress**
```python
from tqdm import tqdm

videos = list(Path("./test_videos").glob("*.mp4"))
results = []

for video_path in tqdm(videos, desc="Processing"):
    result = tester.test_video(str(video_path))
    results.append(result)

# Process results
accuracy = sum(r['is_correct'] for r in results) / len(results)
print(f"Final Accuracy: {accuracy*100:.1f}%")
```

---

**These examples should cover 90% of use cases. For advanced scenarios, refer to the comprehensive guides!**
