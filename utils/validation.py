"""Simple video validation utilities"""

import cv2
import numpy as np
from typing import Tuple, List, Dict


def check_video_quality(frames: List[np.ndarray]) -> Dict:
    """Analyze video quality - blur, brightness"""
    if not frames:
        return {'quality': 'poor', 'blur_percent': 100, 'issues': []}
    
    blur_scores = []
    brightness = []
    
    for frame in frames:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blur = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_scores.append(blur)
        brightness.append(np.mean(gray))
    
    blur_pct = sum(1 for b in blur_scores if b < 100) / len(blur_scores) * 100
    low_light_pct = sum(1 for b in brightness if b < 50) / len(brightness) * 100
    
    issues = []
    if blur_pct > 30:
        issues.append("High blur")
    if low_light_pct > 30:
        issues.append("Poor lighting")
    
    return {
        'quality': 'good' if not issues else 'poor',
        'blur_percent': blur_pct,
        'low_light_percent': low_light_pct,
        'issues': issues
    }


def adjust_confidence_for_quality(confidence: float, quality_info: Dict) -> float:
    """Reduce confidence if video quality is poor"""
    penalty = 1.0
    
    if quality_info['blur_percent'] > 30:
        penalty *= 0.9
    if quality_info['low_light_percent'] > 30:
        penalty *= 0.85
    
    return max(0.5, confidence * penalty)
