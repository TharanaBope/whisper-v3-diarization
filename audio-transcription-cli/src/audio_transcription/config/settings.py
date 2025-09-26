"""Application configuration settings."""

import os
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class AppConfig:
    """Application configuration."""
    
    # HuggingFace token for speaker diarization
    hf_token: Optional[str] = None
    
    # Default directories
    audio_dir: Path = Path("audio")
    transcription_dir: Path = Path("transcriptions") 
    diarization_dir: Path = Path("diarizations")
    logs_dir: Path = Path("logs")
    
    # Audio processing settings
    target_sample_rate: int = 16000
    max_audio_duration: float = 600.0  # 10 minutes
    chunk_duration: float = 30.0
    
    # Model settings
    default_model: str = "large-v3"
    compute_type: str = "float16"
    batch_size: int = 16
    
    def __post_init__(self):
        """Initialize configuration after creation."""
        # Load HF token from environment if not provided
        if self.hf_token is None:
            self.hf_token = os.getenv("HF_TOKEN")
        
        # Create directories
        for directory in [self.audio_dir, self.transcription_dir, 
                         self.diarization_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)