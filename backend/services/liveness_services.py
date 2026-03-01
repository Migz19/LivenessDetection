import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))

import tempfile
import os
import cv2
import time
import numpy as np
import torch

from fastapi import UploadFile
from backend.schema.model_schema import DetailedLivenessResponse, LivenessStatus, ConfidenceBreakdown, VideoQualityMetrics, DecisionFactors
from backend.core.config import settings
from utils.enhanced_inference import EnhancedLivenessInference
from utils.face_detection import FaceDetector
from utils.validation import check_video_quality, adjust_confidence_for_quality
from utils.liveness_features import LivenessPreprocessor, MotionBasedLivenessDetector
from models.efficientnet_model import load_efficientnet_model
from models.cnn_model import load_cnn_model

inference_engine_efficientnet = None
inference_engine_cnn = None
face_detector = None
motion_detector = None


def load_models():
    """Load both models for ensemble predictions"""
    global inference_engine_efficientnet, inference_engine_cnn, face_detector, motion_detector
    
    efficientnet_model = load_efficientnet_model(
        weights_path=str(settings.EFFICIENTNET_WEIGHTS_PATH),
        device=settings.DEVICE,
        pretrained=True
    )
    inference_engine_efficientnet = EnhancedLivenessInference(model=efficientnet_model, device=settings.DEVICE)
    
    cnn_model = load_cnn_model(
        weights_path=str(Path(settings.EFFICIENTNET_WEIGHTS_PATH).parent / 'cnn_livness.pt'),
        device=settings.DEVICE
    )
    inference_engine_cnn = cnn_model
    
    face_detector = FaceDetector(detector_backend='opencv')
    motion_detector = MotionBasedLivenessDetector(threshold=0.05)


async def predict_liveness(file: UploadFile, detailed: bool = True) -> DetailedLivenessResponse:
    """Enhanced liveness detection with dual-model ensemble"""
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

        # Run ensemble inference
        face_bboxes = [faces[0]['bbox']] * len(frames)
        ensemble_result = ensemble_predict(frames, face_bboxes)

        # Calculate frame voting ratio
        live_count = ensemble_result.get('live_count', 0)
        fake_count = ensemble_result.get('fake_count', 0)
        total_frames = live_count + fake_count
        frame_live_ratio = live_count / total_frames if total_frames > 0 else 0
        
        # Adjust confidence for quality
        final_conf = adjust_confidence_for_quality(ensemble_result['final_confidence'], quality)
        
        # FIXED: Both frame voting AND confidence must agree
        # Frame voting rules:
        # - < 40%: Strong spoof signal
        # - 40-60%: Uncertain
        # - >= 60%: Strong live signal (if confidence also >= 0.65)
        
        if frame_live_ratio < 0.4:
            is_live = False
            status = LivenessStatus.SPOOF
            confidence_used = "Frame ratio < 40%"
        elif frame_live_ratio >= 0.6 and final_conf >= 0.65:
            is_live = True
            status = LivenessStatus.LIVE
            confidence_used = f"Frame ratio >= 60% AND confidence >= 0.65"
        elif 0.4 <= frame_live_ratio < 0.6:
            # Uncertain: use confidence as tiebreaker
            is_live = final_conf >= 0.65
            status = LivenessStatus.LIVE if is_live else LivenessStatus.SPOOF
            confidence_used = "Uncertain (40-60% frames) - confidence tiebreaker"
        else:
            # frame_live_ratio >= 0.6 but confidence < 0.65
            is_live = False
            status = LivenessStatus.SPOOF
            confidence_used = "High frame ratio but low confidence (< 0.65)"
        
        # Build response
        warnings = quality['issues'] if quality['quality'] != 'good' else []
        
        # Add motion info if significantly low
        if ensemble_result.get('motion_confidence', 0) < 0.001:
            warnings.append("No motion detected - static frame")
        
        return DetailedLivenessResponse(
            is_live=is_live,
            status=status,
            message="Liveness detected" if is_live else "Spoof detected",
            confidence=ConfidenceBreakdown(
                model_confidence=ensemble_result['final_confidence'],
                motion_confidence=ensemble_result.get('motion_confidence'),
                temporal_confidence=ensemble_result.get('temporal_confidence'),
                texture_confidence=None,
                final_confidence=final_conf
            ),
            decision_factors=DecisionFactors(
                primary_factor=f"Frame agreement: {live_count}/{total_frames} ({frame_live_ratio*100:.1f}%)",
                supporting_factors=[
                    f"EfficientNet conf: {ensemble_result.get('efficientnet_conf', 0):.3f}",
                    f"CNN conf: {ensemble_result.get('cnn_conf', 0):.3f}",
                    f"Quality-adjusted confidence: {final_conf:.3f}",
                    f"Decision rule: {confidence_used}"
                ],
                warning_flags=warnings,
                model_frame_predictions={'live': live_count, 'spoof': fake_count, 'total': total_frames}
            ),
            video_metrics=VideoQualityMetrics(
                total_frames=total_frames,
                processed_frames=total_frames,
                frame_rate=30.0,
                video_duration=total_frames / 30.0,
                video_quality=quality['quality'],
                blur_detected=quality['blur_percent'] > 30,
                low_light_detected=quality['low_light_percent'] > 30,
                face_detected_frames=total_frames
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


def ensemble_predict(frames: list, face_bboxes: list) -> dict:
    """Ensemble prediction combining EfficientNet and CNN models with motion as supporting signal"""
    preprocessor = LivenessPreprocessor()
    live_count = 0
    fake_count = 0
    efficientnet_confs = []
    cnn_confs = []
    
    for frame, bbox in zip(frames, face_bboxes):
        # Preprocess frame
        cropped = preprocessor._crop_face(frame, bbox)
        tensor = preprocessor._preprocess_face(cropped).to(settings.DEVICE)
        
        with torch.no_grad():
            # EfficientNet prediction
            eff_output = inference_engine_efficientnet.model(tensor)
            eff_probs = torch.softmax(eff_output, dim=1)
            eff_pred = torch.argmax(eff_probs, dim=1).item()
            eff_conf = eff_probs[0, 1].item()  # confidence for "live" class
            efficientnet_confs.append(eff_conf)
            
            # CNN prediction
            cnn_output = inference_engine_cnn(tensor)
            cnn_probs = torch.softmax(cnn_output, dim=1)
            cnn_pred = torch.argmax(cnn_probs, dim=1).item()
            cnn_conf = cnn_probs[0, 1].item()  # confidence for "live" class
            cnn_confs.append(cnn_conf)
        
        # Ensemble vote (majority + confidence averaging)
        ensemble_conf = (eff_conf + cnn_conf) / 2
        ensemble_pred = 1 if ensemble_conf >= 0.5 else 0
        
        if ensemble_pred == 1:
            live_count += 1
        else:
            fake_count += 1
    
    # Motion-based analysis as supporting signal (not dominant)
    motion_label, motion_conf, motion_features = motion_detector.detect_from_frames(frames, face_bboxes)
    
    # Motion signal: slight adjustment based on motion presence
    motion_adjustment = 1.0
    if motion_features:
        mean_motion = motion_features.get('mean_motion', 0)
        
        # Slight boost for good motion presence
        if mean_motion > 0.05:
            motion_adjustment = 1.05  # Small bonus
        elif mean_motion > 0.02:
            motion_adjustment = 1.02  # Tiny bonus
        elif mean_motion < 0.001:
            motion_adjustment = 0.95  # Small penalty only for completely static
        # Moderate static motion (0.001-0.02) gets no adjustment
    
    avg_efficientnet = np.mean(efficientnet_confs)
    avg_cnn = np.mean(cnn_confs)
    final_confidence = (avg_efficientnet + avg_cnn) / 2
    
    # Apply small motion adjustment (not a hard penalty)
    final_confidence_with_motion = np.clip(final_confidence * motion_adjustment, 0, 1)
    
    return {
        'live_count': live_count,
        'fake_count': fake_count,
        'final_confidence': final_confidence_with_motion,
        'efficientnet_conf': avg_efficientnet,
        'cnn_conf': avg_cnn,
        'motion_score': motion_adjustment,
        'motion_confidence': motion_features.get('mean_motion', 0) if motion_features else 0,
        'temporal_confidence': None
    }


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