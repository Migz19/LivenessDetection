from pydantic_settings  import BaseSettings
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent.parent

class Settings(BaseSettings):
    # Model weights
    CNN_WEIGHTS_PATH: Path = BASE_DIR / "weights" / "cnn_livness.pt"
    EFFICIENTNET_WEIGHTS_PATH: Path = BASE_DIR / "weights" / "efficientnet.pt"

    # Device
    DEVICE: str = "cpu"  # or "cuda"

    # Inference thresholds
    LIVENESS_THRESHOLD: float = 0.8
    FACE_CONFIDENCE_THRESHOLD: float = 0.7

    # Video processing
    MAX_VIDEO_SIZE_MB: int = 50
    MAX_VIDEO_DURATION_SECONDS: int = 300
    ALLOWED_VIDEO_FORMATS: str = "mp4,avi,mov"
    MAX_FRAMES_TO_PROCESS: int = 30
    FRAME_SAMPLE_RATE: int = 5  # process every 5th frame

    # API Security
    # API_KEY: str = "change-me-in-env"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()