# Liveness Detection - Quick Reference

## Setup
```bash
pip install -r requirements.txt
uvicorn backend.main:app --reload
```

## API Usage

### Endpoint 1: Simple Response
```bash
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/v1/liveness/detect
```

Response:
```json
{
  "is_live": true,
  "status": "live",
  "message": "Liveness detected"
}
```

### Endpoint 2: Detailed Response (with diagnostics)
```bash
curl -X POST -F "file=@video.mp4" http://localhost:8000/api/v1/liveness/detect-detailed
```

Response includes:
- `confidence`: breakdown by model, motion, temporal components
- `decision_factors`: why decision was made
- `video_metrics`: blur, lighting, quality info
- `warning_flags`: issues that might affect accuracy

## Python Testing
```python
from utils.testing_framework import LivenessTester

tester = LivenessTester()

# Single video
result = tester.test_video("video.mp4", expected="live")

# Batch test
results = tester.test_batch("./test_videos", {
    "person.mp4": "live",
    "photo.mp4": "spoof"
})

# Accuracy
print(f"Accuracy: {tester.accuracy()*100:.1f}%")
```

## Configuration
Edit `backend/core/config.py`:
- `LIVENESS_THRESHOLD`: 0.5 (decision threshold, increase for stricter)
- `DEVICE`: "cuda" or "cpu"
- `MAX_VIDEO_SIZE_MB`: 50
- `MAX_FRAMES_TO_PROCESS`: 30

## Tips
- Use detailed endpoint to validate before deployment
- Monitor warning flags - they indicate risky predictions
- Videos should be 2-5 seconds with clear face and good lighting
- Accuracy target: ≥95%
