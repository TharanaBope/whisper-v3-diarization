"""Main audio processing engine."""

import torch
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

from .transcription import WhisperTranscriber
from .diarization import WhisperXDiarizer
from ..utils.file_handler import AudioFileHandler

logger = logging.getLogger(__name__)


class AudioProcessor:
    """Main audio processing coordinator."""
    
    def __init__(
        self,
        model_size: str = "large-v3",
        device: str = "auto",
        hf_token: Optional[str] = None
    ):
        """Initialize audio processor.
        
        Args:
            model_size: Whisper model size
            device: Processing device (auto, cpu, cuda)
            hf_token: HuggingFace token for diarization
        """
        self.model_size = model_size
        self.device = self._setup_device(device)
        self.hf_token = hf_token
        
        # Initialize components
        self.file_handler = AudioFileHandler()
        self.transcriber = WhisperTranscriber(model_size, self.device)
        self.diarizer = WhisperXDiarizer(self.device, hf_token) if hf_token else None
        
        logger.info(f"AudioProcessor initialized - Model: {model_size}, Device: {self.device}")
    
    def _setup_device(self, device: str) -> str:
        """Setup processing device."""
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA requested but not available, falling back to CPU")
            device = "cpu"
        
        return device
    
    def transcribe_file(
        self,
        file_path: Path,
        language: Optional[str] = None,
        output_dir: Path = Path("transcriptions")
    ) -> Dict[str, Any]:
        """Transcribe a single audio file.
        
        Args:
            file_path: Path to audio file
            language: Audio language (auto-detect if None)
            output_dir: Directory to save results
        
        Returns:
            Dict with success status and results
        """
        try:
            # Validate and prepare audio
            validation_result = self.file_handler.validate_audio_file(file_path)
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": f"Invalid audio file: {validation_result['errors']}"
                }
            
            # Load and preprocess audio
            audio_data = self.file_handler.load_audio(file_path)
            if not audio_data["success"]:
                return {
                    "success": False,
                    "error": f"Failed to load audio: {audio_data['error']}"
                }
            
            # Transcribe
            result = self.transcriber.transcribe(
                audio_path=file_path,
                language=language
            )
            
            if result["success"]:
                # Save results
                output_file = output_dir / f"{file_path.stem}_transcription.json"
                self.file_handler.save_transcription_result(result, output_file)
                
                # Also save as plain text
                text_file = output_dir / f"{file_path.stem}_transcription.txt"
                self.file_handler.save_text_file(result["text"], text_file)
                
                result["output_files"] = {
                    "json": str(output_file),
                    "text": str(text_file)
                }
            
            return result
            
        except Exception as e:
            logger.exception(f"Error transcribing {file_path}")
            return {"success": False, "error": str(e)}
    
    def diarize_file(
        self,
        file_path: Path,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
        output_dir: Path = Path("diarizations")
    ) -> Dict[str, Any]:
        """Process audio file with speaker diarization.
        
        Args:
            file_path: Path to audio file
            language: Audio language (auto-detect if None)
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
            output_dir: Directory to save results
        
        Returns:
            Dict with success status and results
        """
        if not self.diarizer:
            return {
                "success": False,
                "error": "Diarization not available - no HuggingFace token provided"
            }
        
        try:
            # Validate audio
            validation_result = self.file_handler.validate_audio_file(file_path)
            if not validation_result["is_valid"]:
                return {
                    "success": False,
                    "error": f"Invalid audio file: {validation_result['errors']}"
                }
            
            # Load audio
            audio_data = self.file_handler.load_audio(file_path)
            if not audio_data["success"]:
                return {
                    "success": False,
                    "error": f"Failed to load audio: {audio_data['error']}"
                }
            
            # Process with diarization
            result = self.diarizer.diarize(
                audio_path=file_path,
                language=language,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            if result["success"]:
                # Save results
                output_file = output_dir / f"{file_path.stem}_diarization.json"
                self.file_handler.save_diarization_result(result, output_file)
                
                # Save formatted transcript
                transcript_file = output_dir / f"{file_path.stem}_diarized_transcript.txt"
                self.file_handler.save_diarized_transcript(result, transcript_file)
                
                result["output_files"] = {
                    "json": str(output_file),
                    "transcript": str(transcript_file)
                }
            
            return result
            
        except Exception as e:
            logger.exception(f"Error diarizing {file_path}")
            return {"success": False, "error": str(e)}
    
    def cleanup_memory(self):
        """Clean up GPU memory."""
        if self.transcriber:
            self.transcriber.cleanup()
        if self.diarizer:
            self.diarizer.cleanup()
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def __del__(self):
        """Cleanup on deletion."""
        self.cleanup_memory()