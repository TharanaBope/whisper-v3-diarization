"""Backend configuration for GUI application."""
import os
from pathlib import Path
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    """Application settings."""

    # API Configuration
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    API_RELOAD: bool = True

    # CORS Configuration
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]

    # Storage paths
    BASE_DIR: Path = Path(__file__).parent
    UPLOAD_DIR: Path = BASE_DIR / "storage" / "uploads"
    RESULTS_DIR: Path = BASE_DIR / "storage" / "results"

    # Processing Configuration
    MAX_FILE_SIZE: int = 500 * 1024 * 1024  # 500MB
    ALLOWED_EXTENSIONS: set[str] = {".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"}

    # Whisper Configuration
    DEFAULT_MODEL: str = "large-v3"
    DEFAULT_DEVICE: str = "auto"

    # HuggingFace Token (for diarization)
    HF_TOKEN: str = os.getenv("HF_TOKEN", "")

    # CUDA Configuration (optional)
    CUDA_VISIBLE_DEVICES: str = "0"

    class Config:
        env_file = ".env"
        extra = "ignore"  # Ignore extra fields from .env

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Ensure storage directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

settings = Settings()
