from pydantic import BaseModel, Field
from enum import Enum
from typing import Optional, Dict, List

class LivenessStatus(str, Enum):
    LIVE = "live"
    SPOOF = "spoof"
    ERROR = "error"

class ConfidenceBreakdown(BaseModel):
    """Confidence from different components"""
    model_confidence: float = Field(description="Raw model confidence")
    motion_confidence: Optional[float] = None
    temporal_confidence: Optional[float] = None
    texture_confidence: Optional[float] = None
    final_confidence: float = Field(description="Final confidence used for decision")

class VideoQualityMetrics(BaseModel):
    """Video quality metrics"""
    total_frames: int
    processed_frames: int
    frame_rate: float
    video_duration: float
    video_quality: str  # "good", "fair", "poor"
    blur_detected: bool
    low_light_detected: bool
    face_detected_frames: int

class DecisionFactors(BaseModel):
    """What affected the decision"""
    primary_factor: str
    supporting_factors: List[str] = []
    warning_flags: List[str] = []
    model_frame_predictions: Dict = {}

class DetailedLivenessResponse(BaseModel):
    """Full response with diagnostics"""
    is_live: bool
    status: LivenessStatus
    message: str
    confidence: ConfidenceBreakdown
    decision_factors: DecisionFactors
    video_metrics: Optional[VideoQualityMetrics] = None
    processing_time_ms: float = 0
    
    class Config:
        use_enum_values = True

class LivenessResponse(BaseModel):
    """Simple response for backward compatibility"""
    is_live: bool
    status: LivenessStatus
    message: str

    class Config:
        use_enum_values = True