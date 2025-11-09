"""File upload endpoint."""
import logging
from pathlib import Path
from uuid import uuid4
from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from typing import Optional

from models import UploadResponse, ProcessingOptions
from config import settings
from services.processor import processor_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["upload"])

@router.post("/upload", response_model=UploadResponse)
async def upload_audio(
    file: UploadFile = File(...),
    model_size: str = Form(default="large-v3"),
    language: Optional[str] = Form(default=None),
    enable_diarization: bool = Form(default=True),
    min_speakers: Optional[int] = Form(default=None),
    max_speakers: Optional[int] = Form(default=None),
    use_assistant: bool = Form(default=False),
    device: str = Form(default="auto")
):
    """
    Upload audio file for processing.

    Args:
        file: Audio file to upload
        model_size: Whisper model size
        language: Audio language code
        enable_diarization: Enable speaker diarization
        min_speakers: Minimum number of speakers
        max_speakers: Maximum number of speakers
        use_assistant: Use Distil-Whisper assistant
        device: Processing device (auto/cpu/cuda)

    Returns:
        UploadResponse with session ID and file info
    """
    try:
        # Validate file extension
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in settings.ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file format. Allowed: {', '.join(settings.ALLOWED_EXTENSIONS)}"
            )

        # Generate session ID
        session_id = str(uuid4())

        # Create session directory
        session_dir = settings.UPLOAD_DIR / session_id
        session_dir.mkdir(parents=True, exist_ok=True)

        # Save uploaded file
        file_path = session_dir / file.filename
        content = await file.read()

        # Validate file size
        if len(content) > settings.MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {settings.MAX_FILE_SIZE / (1024*1024):.0f}MB"
            )

        with open(file_path, "wb") as f:
            f.write(content)

        logger.info(f"File uploaded: {file.filename} ({len(content)} bytes) - Session: {session_id}")

        # Create processing options
        options = ProcessingOptions(
            model_size=model_size,
            language=language,
            enable_diarization=enable_diarization,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
            use_assistant=use_assistant,
            device=device
        )

        # Store options in session
        processor_service.active_sessions[session_id] = {
            "status": "uploaded",
            "audio_path": file_path,
            "filename": file.filename,
            "options": options
        }

        return UploadResponse(
            session_id=session_id,
            filename=file.filename,
            file_size=len(content),
            message="File uploaded successfully. Use session_id to start processing."
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Upload failed")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
