"""WhisperX diarization implementation."""

import whisperx
import torch
import gc
import logging
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


class WhisperXDiarizer:
    """WhisperX-based speaker diarization service."""
    
    def __init__(self, device: str = "cuda", hf_token: Optional[str] = None):
        """Initialize WhisperX diarizer.
        
        Args:
            device: Processing device
            hf_token: HuggingFace token for speaker models
        """
        self.device = device
        self.hf_token = hf_token
        self.compute_type = "float16" if device == "cuda" else "float32"
        
        # Models will be loaded on first use
        self.model = None
        self.align_model = None
        self.diarize_model = None
        
        if not hf_token:
            raise ValueError("HuggingFace token required for speaker diarization")
        
        logger.info(f"WhisperXDiarizer initialized - Device: {device}")
    
    def _load_models(self, language_code: str = "en"):
        """Lazy load WhisperX models."""
        try:
            # Load Whisper model if not already loaded
            if self.model is None:
                logger.info("Loading WhisperX transcription model")
                self.model = whisperx.load_model(
                    "large-v2",  # WhisperX works best with large-v2
                    self.device,
                    compute_type=self.compute_type
                )
            
            # Load alignment model if not already loaded
            if self.align_model is None:
                logger.info(f"Loading alignment model for language: {language_code}")
                self.align_model, self.metadata = whisperx.load_align_model(
                    language_code=language_code,
                    device=self.device
                )
            
            # Load diarization model if not already loaded
            if self.diarize_model is None:
                logger.info("Loading speaker diarization model")
                self.diarize_model = whisperx.DiarizationPipeline(
                    use_auth_token=self.hf_token,
                    device=self.device
                )
            
        except Exception as e:
            logger.error(f"Failed to load WhisperX models: {e}")
            raise
    
    def diarize(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform transcription with speaker diarization.
        
        Args:
            audio_path: Path to audio file
            language: Language code
            min_speakers: Minimum number of speakers
            max_speakers: Maximum number of speakers
        
        Returns:
            Diarization result
        """
        try:
            # Load audio
            logger.info(f"Loading audio: {audio_path}")
            audio = whisperx.load_audio(str(audio_path))
            
            # Auto-detect language if not provided
            if not language:
                logger.info("Auto-detecting language")
                # Quick transcription to detect language
                temp_model = whisperx.load_model("base", self.device, compute_type=self.compute_type)
                temp_result = temp_model.transcribe(audio, batch_size=16)
                language = temp_result.get("language", "en")
                del temp_model
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info(f"Detected language: {language}")
            
            # Load models for detected/specified language
            self._load_models(language)
            
            # Step 1: Transcribe
            logger.info("Step 1: Transcribing audio")
            result = self.model.transcribe(audio, batch_size=16)
            
            # Step 2: Align whisper output
            logger.info("Step 2: Aligning transcription")
            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            
            # Step 3: Assign speaker labels
            logger.info("Step 3: Performing speaker diarization")
            diarize_segments = self.diarize_model(
                audio,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Step 4: Assign word speakers
            logger.info("Step 4: Assigning speakers to words")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            
            # Process results
            segments_with_speakers = []
            speakers_found = set()
            
            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                speakers_found.add(speaker)
                
                segments_with_speakers.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": segment.get("text", ""),
                    "speaker": speaker,
                    "words": segment.get("words", [])
                })
            
            # Generate formatted transcript
            formatted_transcript = self._format_transcript(segments_with_speakers)
            
            return {
                "success": True,
                "segments": segments_with_speakers,
                "num_speakers": len(speakers_found),
                "speakers": sorted(list(speakers_found)),
                "language": language,
                "formatted_transcript": formatted_transcript,
                "model": "whisperx-large-v2"
            }
            
        except Exception as e:
            logger.exception(f"Diarization failed for {audio_path}")
            return {
                "success": False,
                "error": str(e)
            }
        finally:
            # Clean up memory after processing
            self._cleanup_memory()
    
    def _format_transcript(self, segments) -> str:
        """Format segments into readable transcript."""
        transcript_lines = []
        
        for segment in segments:
            start_time = self._format_time(segment["start"])
            speaker = segment["speaker"]
            text = segment["text"].strip()
            
            if text:
                transcript_lines.append(f"[{start_time}] {speaker}: {text}")
        
        return "\n".join(transcript_lines)
    
    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format."""
        minutes = int(seconds // 60)
        seconds = int(seconds % 60)
        return f"{minutes:02d}:{seconds:02d}"
    
    def _cleanup_memory(self):
        """Clean up memory between operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def cleanup(self):
        """Clean up all models."""
        models = [
            (self.model, "transcription model"),
            (self.align_model, "alignment model"),
            (self.diarize_model, "diarization model")
        ]
        
        for model, name in models:
            if model:
                try:
                    del model
                    logger.info(f"Cleaned up {name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {e}")
        
        self.model = None
        self.align_model = None
        self.diarize_model = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("WhisperXDiarizer cleaned up")