"""Results retrieval endpoints."""
import logging
from pathlib import Path
from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse

from models import ProcessingResult
from config import settings
from services.processor import processor_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["results"])

@router.get("/results/{session_id}", response_model=ProcessingResult)
async def get_results(session_id: str):
    """
    Get processing results for a session.

    Args:
        session_id: Session ID

    Returns:
        ProcessingResult with transcription and diarization data
    """
    session_info = processor_service.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")

    if session_info.get("status") != "complete":
        raise HTTPException(
            status_code=400,
            detail=f"Session not complete. Current status: {session_info.get('status')}"
        )

    results_dir = settings.RESULTS_DIR / session_id

    # Build file URLs
    transcription_txt = results_dir / f"{Path(session_info['filename']).stem}_transcription.txt"
    transcription_json = results_dir / f"{Path(session_info['filename']).stem}_transcription.json"
    diarization_txt = results_dir / f"{Path(session_info['filename']).stem}_diarized_transcript.txt"
    diarization_json = results_dir / f"{Path(session_info['filename']).stem}_diarization.json"

    return ProcessingResult(
        session_id=session_id,
        success=True,
        filename=session_info["filename"],
        transcription=session_info.get("transcription"),
        diarization=session_info.get("diarization"),
        transcription_txt_url=f"/api/download/{session_id}/transcription.txt" if transcription_txt.exists() else None,
        transcription_json_url=f"/api/download/{session_id}/transcription.json" if transcription_json.exists() else None,
        diarization_txt_url=f"/api/download/{session_id}/diarization.txt" if diarization_txt.exists() else None,
        diarization_json_url=f"/api/download/{session_id}/diarization.json" if diarization_json.exists() else None
    )

@router.get("/download/{session_id}/{file_type}")
async def download_file(session_id: str, file_type: str):
    """
    Download result file.

    Args:
        session_id: Session ID
        file_type: File type (transcription.txt, transcription.json, etc.)

    Returns:
        FileResponse with the requested file
    """
    session_info = processor_service.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")

    results_dir = settings.RESULTS_DIR / session_id
    filename_stem = Path(session_info["filename"]).stem

    # Map file types to actual filenames
    file_mapping = {
        "transcription.txt": f"{filename_stem}_transcription.txt",
        "transcription.json": f"{filename_stem}_transcription.json",
        "diarization.txt": f"{filename_stem}_diarized_transcript.txt",
        "diarization.json": f"{filename_stem}_diarization.json"
    }

    if file_type not in file_mapping:
        raise HTTPException(status_code=400, detail="Invalid file type")

    file_path = results_dir / file_mapping[file_type]
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(
        path=file_path,
        filename=file_mapping[file_type],
        media_type="application/octet-stream"
    )
