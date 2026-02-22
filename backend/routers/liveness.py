from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.schema.model_schema import LivenessResponse, DetailedLivenessResponse
from backend.services.liveness_services import predict_liveness

router = APIRouter(prefix="/api/v1", tags=["Liveness"])

ALLOWED_CONTENT_TYPES = {
    "video/mp4",
    "video/avi",
    "video/quicktime",   # .mov
    "video/x-matroska",  # .mkv
}


@router.post("/liveness/detect")
async def detect_liveness_simple(file: UploadFile = File(...)):
    """Simple liveness detection - returns basic response"""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: mp4, avi, mov, mkv"
        )
    
    result = await predict_liveness(file, detailed=False)
    return LivenessResponse(
        is_live=result.is_live,
        status=result.status,
        message=result.message
    )


@router.post("/liveness/detect-detailed", response_model=DetailedLivenessResponse)
async def detect_liveness_detailed(file: UploadFile = File(...)):
    """Detailed liveness detection with confidence breakdown and diagnostics"""
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{file.content_type}'. Allowed: mp4, avi, mov, mkv"
        )
    
    return await predict_liveness(file, detailed=True)