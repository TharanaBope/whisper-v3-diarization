"""Audio processing service with progress tracking."""
import asyncio
import logging
from pathlib import Path
from typing import AsyncGenerator, Optional
from uuid import uuid4
import sys

# Add CLI module to Python path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent / "audio-transcription-cli" / "src"))

from audio_transcription.core.audio_processor import AudioProcessor
from audio_transcription.core.diarization import WhisperXDiarizer
from models import ProgressEvent, ProcessingOptions
from config import settings

logger = logging.getLogger(__name__)

class ProcessorService:
    """Service for processing audio files with progress tracking."""

    def __init__(self):
        """Initialize processor service."""
        self.active_sessions: dict[str, dict] = {}

    async def process_audio(
        self,
        session_id: str,
        audio_path: Path,
        options: ProcessingOptions
    ) -> AsyncGenerator[ProgressEvent, None]:
        """
        Process audio file and yield progress events.

        Args:
            session_id: Unique session identifier
            audio_path: Path to uploaded audio file
            options: Processing options

        Yields:
            ProgressEvent objects with progress updates
        """
        try:
            # Store session info
            self.active_sessions[session_id] = {
                "status": "processing",
                "audio_path": audio_path,
                "filename": audio_path.name
            }

            # Stage 1: Initialize processor
            yield ProgressEvent(
                event_type="progress",
                session_id=session_id,
                stage="initializing",
                progress=5,
                message="Initializing audio processor..."
            )

            processor = AudioProcessor(
                model_size=options.model_size,
                device=options.device,
                hf_token=settings.HF_TOKEN if options.enable_diarization else None,
                use_assistant=options.use_assistant
            )

            await asyncio.sleep(0.1)  # Allow event to be sent

            # Stage 2: Load audio
            yield ProgressEvent(
                event_type="progress",
                session_id=session_id,
                stage="loading_audio",
                progress=10,
                message=f"Loading audio file: {audio_path.name}"
            )

            await asyncio.sleep(0.1)

            # Stage 3: Transcription
            yield ProgressEvent(
                event_type="progress",
                session_id=session_id,
                stage="transcribing",
                progress=30,
                message="Transcribing audio with Whisper..."
            )

            # Run transcription in thread pool to avoid blocking
            transcription_result = await asyncio.to_thread(
                processor.transcribe_file,
                file_path=audio_path,
                language=options.language,
                output_dir=settings.RESULTS_DIR / session_id
            )

            if not transcription_result["success"]:
                raise Exception(f"Transcription failed: {transcription_result.get('error')}")

            yield ProgressEvent(
                event_type="progress",
                session_id=session_id,
                stage="transcription_complete",
                progress=60,
                message="Transcription completed successfully"
            )

            # Stage 4: Diarization (if enabled)
            diarization_result = None
            if options.enable_diarization and settings.HF_TOKEN:
                yield ProgressEvent(
                    event_type="progress",
                    session_id=session_id,
                    stage="diarizing",
                    progress=70,
                    message="Performing speaker diarization..."
                )

                diarization_result = await asyncio.to_thread(
                    processor.diarize_file,
                    file_path=audio_path,
                    language=options.language,
                    min_speakers=options.min_speakers,
                    max_speakers=options.max_speakers,
                    output_dir=settings.RESULTS_DIR / session_id
                )

                if diarization_result["success"]:
                    yield ProgressEvent(
                        event_type="progress",
                        session_id=session_id,
                        stage="diarization_complete",
                        progress=95,
                        message=f"Diarization complete: {diarization_result.get('num_speakers', 'Unknown')} speakers detected"
                    )

            # Stage 5: Complete
            yield ProgressEvent(
                event_type="complete",
                session_id=session_id,
                stage="complete",
                progress=100,
                message="Processing completed successfully",
                results_available=True
            )

            # Update session status
            self.active_sessions[session_id]["status"] = "complete"
            self.active_sessions[session_id]["transcription"] = transcription_result
            self.active_sessions[session_id]["diarization"] = diarization_result

        except Exception as e:
            logger.exception(f"Processing failed for session {session_id}")
            yield ProgressEvent(
                event_type="error",
                session_id=session_id,
                stage="error",
                progress=0,
                message="Processing failed",
                error=str(e)
            )
            self.active_sessions[session_id]["status"] = "error"
            self.active_sessions[session_id]["error"] = str(e)

    def get_session_info(self, session_id: str) -> Optional[dict]:
        """Get information about a processing session."""
        return self.active_sessions.get(session_id)

# Global processor service instance
processor_service = ProcessorService()
