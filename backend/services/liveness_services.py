import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import tempfile
import os
import cv2
import time
import numpy as np

from fastapi import UploadFile
from backend.schema.model_schema import DetailedLivenessResponse, LivenessStatus, ConfidenceBreakdown, VideoQualityMetrics, DecisionFactors
from backend.core.config import settings
from utils.enhanced_inference import EnhancedLivenessInference
from utils.face_detection import FaceDetector
from utils.validation import check_video_quality, adjust_confidence_for_quality
from models.efficientnet_model import load_efficientnet_model

inference_engine = None
face_detector = None


def load_models():
    """Load models once at startup"""
    global inference_engine, face_detector
    model = load_efficientnet_model(
        weights_path=str(settings.EFFICIENTNET_WEIGHTS_PATH),
        device=settings.DEVICE,
        pretrained=True
    )
    inference_engine = EnhancedLivenessInference(model=model, device=settings.DEVICE)
    face_detector = FaceDetector(detector_backend='opencv')


async def predict_liveness(file: UploadFile, detailed: bool = True) -> DetailedLivenessResponse:
    """Enhanced liveness detection with diagnostics"""
    start_time = time.time()
    tmp_path = None
    
    try:
        # Validate file size
        contents = await file.read()
        size_mb = len(contents) / (1024 * 1024)
        if size_mb > settings.MAX_VIDEO_SIZE_MB:
            return _error_response(f"File too large ({size_mb:.1f}MB)", start_time)

        # Save to temp file
        suffix = Path(file.filename).suffix or ".mp4"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name

        # Extract frames
        frames = extract_frames(tmp_path)
        if not frames:
            return _error_response("Video too short or unreadable", start_time)

        # Check video quality
        quality = check_video_quality(frames)
        
        # Detect faces
        faces = face_detector.detect_faces(frames[0])
        if len(faces) != 1:
            msg = "No face detected" if len(faces) == 0 else f"{len(faces)} faces detected"
            return _error_response(msg, start_time)

        # Run inference
        face_bboxes = [faces[0]['bbox']] * len(frames)
        result = inference_engine.predict_video_with_motion(
            frames=frames,
            face_bboxes=face_bboxes
        )

        # Adjust confidence for quality
        final_conf = adjust_confidence_for_quality(result['final_confidence'], quality)
        is_live = final_conf >= settings.LIVENESS_THRESHOLD
        
        # Build response
        live_count = result.get('live_count', 0)
        fake_count = result.get('fake_count', 0)
        warnings = quality['issues'] if quality['quality'] != 'good' else []
        
        return DetailedLivenessResponse(
            is_live=is_live,
            status=LivenessStatus.LIVE if is_live else LivenessStatus.SPOOF,
            message="Liveness detected" if is_live else "Spoof detected",
            confidence=ConfidenceBreakdown(
                model_confidence=result['final_confidence'],
                motion_confidence=result.get('motion_confidence'),
                temporal_confidence=result.get('temporal_confidence'),
                texture_confidence=None,
                final_confidence=final_conf
            ),
            decision_factors=DecisionFactors(
                primary_factor=f"Frame agreement ({live_count}/{len(frames)})",
                supporting_factors=[],
                warning_flags=warnings,
                model_frame_predictions={'live': live_count, 'spoof': fake_count, 'total': len(frames)}
            ),
            video_metrics=VideoQualityMetrics(
                total_frames=len(frames),
                processed_frames=len(frames),
                frame_rate=30.0,
                video_duration=len(frames) / 30.0,
                video_quality=quality['quality'],
                blur_detected=quality['blur_percent'] > 30,
                low_light_detected=quality['low_light_percent'] > 30,
                face_detected_frames=len(frames)
            ),
            processing_time_ms=(time.time() - start_time) * 1000
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        return _error_response(f"Error: {str(e)}", start_time)

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)


def _error_response(message: str, start_time: float) -> DetailedLivenessResponse:
    """Helper to create error responses"""
    return DetailedLivenessResponse(
        is_live=False,
        status=LivenessStatus.ERROR,
        message=message,
        confidence=ConfidenceBreakdown(model_confidence=0, final_confidence=0),
        decision_factors=DecisionFactors(primary_factor="Error", warning_flags=[message]),
        processing_time_ms=(time.time() - start_time) * 1000
    )


def extract_frames(video_path: str) -> list:
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    frame_idx = 0

    while len(frames) < settings.MAX_FRAMES_TO_PROCESS:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % settings.FRAME_SAMPLE_RATE == 0:
            frames.append(frame)
        frame_idx += 1

    cap.release()
    return frames