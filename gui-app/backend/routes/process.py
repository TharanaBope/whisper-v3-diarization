"""Processing endpoint with SSE progress streaming."""
import logging
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from typing import AsyncGenerator

from services.processor import processor_service

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["process"])

@router.get("/progress/{session_id}")
async def stream_progress(session_id: str):
    """
    Stream processing progress via Server-Sent Events (SSE).

    Args:
        session_id: Session ID from upload response

    Returns:
        StreamingResponse with SSE events
    """
    # Check if session exists
    session_info = processor_service.get_session_info(session_id)
    if not session_info:
        raise HTTPException(status_code=404, detail="Session not found")

    # Check if already processing or complete
    if session_info.get("status") == "processing":
        raise HTTPException(status_code=409, detail="Session already processing")

    if session_info.get("status") == "complete":
        raise HTTPException(status_code=410, detail="Session already completed")

    async def event_generator() -> AsyncGenerator[str, None]:
        """Generate SSE events for processing progress."""
        try:
            audio_path = session_info["audio_path"]
            options = session_info["options"]

            async for progress_event in processor_service.process_audio(
                session_id=session_id,
                audio_path=audio_path,
                options=options
            ):
                # Format as SSE event
                event_data = progress_event.model_dump_json()
                yield f"data: {event_data}\n\n"

        except Exception as e:
            logger.exception(f"Error in progress stream for {session_id}")
            error_event = {
                "event_type": "error",
                "session_id": session_id,
                "message": "Processing error",
                "error": str(e)
            }
            yield f"data: {error_event}\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )
