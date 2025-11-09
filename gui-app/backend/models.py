"""Pydantic models for request/response validation."""
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Literal

class UploadResponse(BaseModel):
    """Response after file upload."""
    session_id: str
    filename: str
    file_size: int
    message: str

class ProcessingOptions(BaseModel):
    """Options for audio processing."""
    model_config = ConfigDict(protected_namespaces=())

    model_size: str = Field(default="large-v3", description="Whisper model size")
    language: Optional[str] = Field(default=None, description="Audio language code")
    enable_diarization: bool = Field(default=True, description="Enable speaker diarization")
    min_speakers: Optional[int] = Field(default=None, description="Minimum speakers")
    max_speakers: Optional[int] = Field(default=None, description="Maximum speakers")
    use_assistant: bool = Field(default=False, description="Use Distil-Whisper assistant")
    device: str = Field(default="auto", description="Processing device")

class ProgressEvent(BaseModel):
    """Progress event for SSE streaming."""
    event_type: Literal["progress", "complete", "error"]
    session_id: str
    stage: Optional[str] = None
    progress: Optional[int] = None  # 0-100
    message: Optional[str] = None
    error: Optional[str] = None
    results_available: bool = False

class ProcessingResult(BaseModel):
    """Final processing results."""
    session_id: str
    success: bool
    filename: str
    transcription: Optional[dict] = None
    diarization: Optional[dict] = None
    transcription_txt_url: Optional[str] = None
    transcription_json_url: Optional[str] = None
    diarization_txt_url: Optional[str] = None
    diarization_json_url: Optional[str] = None
    error: Optional[str] = None
