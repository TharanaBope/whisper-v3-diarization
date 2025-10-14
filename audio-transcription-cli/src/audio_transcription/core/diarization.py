"""WhisperX diarization implementation with real speaker detection."""

import whisperx
import torch
import gc
import logging
import librosa
import subprocess
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)


def load_audio_with_static_ffmpeg(file_path: str, sr: int = 16000) -> np.ndarray:
    """Load audio using static-ffmpeg if available, otherwise fall back to librosa.

    Args:
        file_path: Path to audio file
        sr: Target sample rate (default: 16000)

    Returns:
        Audio as numpy array (float32)
    """
    try:
        # Try to use static-ffmpeg from venv
        import static_ffmpeg
        ffmpeg_path, _ = static_ffmpeg.run.get_or_fetch_platform_executables_else_raise()

        logger.info(f"Using static-ffmpeg from: {ffmpeg_path}")

        # Use FFmpeg to load and convert audio (same as WhisperX)
        cmd = [
            ffmpeg_path,
            "-nostdin",
            "-threads", "0",
            "-i", file_path,
            "-f", "s16le",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            "-ar", str(sr),
            "-"
        ]

        result = subprocess.run(cmd, capture_output=True, check=True)
        audio = np.frombuffer(result.stdout, np.int16).flatten().astype(np.float32) / 32768.0
        logger.info(f"Audio loaded with static-ffmpeg: {len(audio)} samples at {sr}Hz")
        return audio

    except (ImportError, Exception) as e:
        # Fall back to librosa if static-ffmpeg fails
        logger.info(f"static-ffmpeg not available or failed ({e}), using librosa instead")
        audio, sample_rate = librosa.load(file_path, sr=sr)
        logger.info(f"Audio loaded with librosa: {len(audio)} samples at {sample_rate}Hz")
        return audio


class WhisperXDiarizer:
    """WhisperX-based speaker diarization service with real speaker detection."""

    def __init__(self, device: str = "cuda", hf_token: Optional[str] = None, model_size: str = "large-v2"):
        """Initialize WhisperX diarizer.

        Args:
            device: Processing device (cuda or cpu)
            hf_token: HuggingFace token for speaker diarization models
            model_size: WhisperX model size to use
        """
        self.device = device
        self.hf_token = hf_token
        self.model_size = model_size
        self.compute_type = "float16" if device == "cuda" else "float32"

        # Models will be loaded on first use (lazy loading)
        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

        if not hf_token:
            raise ValueError("HuggingFace token required for speaker diarization")

        logger.info(f"WhisperXDiarizer initialized - Device: {device}, Model: {model_size}")

    def diarize(
        self,
        audio_path: Path,
        language: Optional[str] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None
    ) -> Dict[str, Any]:
        """Perform transcription with real WhisperX speaker diarization.

        This implements the full WhisperX pipeline:
        1. Transcribe with WhisperX (batched inference)
        2. Align whisper output for word-level timestamps
        3. Perform speaker diarization with pyannote
        4. Assign speaker labels to words/segments

        Args:
            audio_path: Path to audio file
            language: Language code (e.g., 'en', 'es'). Auto-detected if None
            min_speakers: Minimum number of speakers (optional)
            max_speakers: Maximum number of speakers (optional)

        Returns:
            Dictionary with diarization results including segments with speaker labels
        """
        try:
            logger.info(f"Starting WhisperX diarization pipeline for: {audio_path}")

            # Step 1: Load WhisperX model
            logger.info(f"Step 1/5: Loading WhisperX model ({self.model_size})")
            if self.model is None:
                self.model = whisperx.load_model(
                    self.model_size,
                    self.device,
                    compute_type=self.compute_type
                )
                logger.info("WhisperX model loaded successfully")

            # Step 2: Load and transcribe audio
            logger.info(f"Step 2/5: Loading audio and transcribing")
            # Use our custom loader that tries static-ffmpeg first, then librosa
            audio = load_audio_with_static_ffmpeg(str(audio_path), sr=16000)

            # Transcribe with batched inference
            batch_size = 16 if self.device == "cuda" else 8
            transcribe_options = {"batch_size": batch_size}
            if language:
                transcribe_options["language"] = language

            result = self.model.transcribe(audio, **transcribe_options)
            logger.info(f"Transcription complete. Language: {result.get('language', 'unknown')}")

            # Get detected language if not provided
            detected_language = language or result.get("language", "en")

            # Clean up transcription model if memory is tight
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Step 3: Align whisper output for word-level timestamps
            logger.info(f"Step 3/5: Aligning transcription for word-level timestamps")
            if self.align_model is None or self.align_metadata is None:
                self.align_model, self.align_metadata = whisperx.load_align_model(
                    language_code=detected_language,
                    device=self.device
                )
                logger.info(f"Alignment model loaded for language: {detected_language}")

            result = whisperx.align(
                result["segments"],
                self.align_model,
                self.align_metadata,
                audio,
                self.device,
                return_char_alignments=False
            )
            logger.info("Alignment complete - word-level timestamps generated")

            # Clean up alignment model if memory is tight
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Step 4: Perform speaker diarization
            logger.info(f"Step 4/5: Performing speaker diarization")
            if self.diarize_model is None:
                # Try to load the diarization model with proper error handling
                # Use speaker-diarization-3.1 which is designed for pyannote.audio 3.x
                # This version removes problematic onnxruntime usage and runs in pure PyTorch
                try:
                    self.diarize_model = whisperx.DiarizationPipeline(
                        model_name="pyannote/speaker-diarization-3.1",  # Native v3.1 model for pyannote.audio 3.x
                        use_auth_token=self.hf_token,
                        device=self.device
                    )
                    logger.info("Diarization pipeline loaded successfully (pyannote/speaker-diarization-3.1)")
                except Exception as e:
                    logger.error(f"Failed to load diarization pipeline: {e}")
                    # Try alternative import path with explicit model version
                    try:
                        from whisperx.diarize import DiarizationPipeline
                        self.diarize_model = DiarizationPipeline(
                            model_name="pyannote/speaker-diarization-3.1",  # Native v3.1 model
                            use_auth_token=self.hf_token,
                            device=self.device
                        )
                        logger.info("Diarization pipeline loaded via alternative import (pyannote/speaker-diarization-3.1)")
                    except Exception as e2:
                        logger.error(f"Alternative import also failed: {e2}")
                        raise RuntimeError(
                            "Failed to load diarization pipeline. "
                            "Please ensure pyannote.audio is installed and you have accepted "
                            "the terms at: https://huggingface.co/pyannote/speaker-diarization-3.1 "
                            "and https://huggingface.co/pyannote/segmentation-3.0"
                        )

            # Run diarization with optional speaker count constraints
            diarize_options = {}
            if min_speakers is not None:
                diarize_options["min_speakers"] = min_speakers
            if max_speakers is not None:
                diarize_options["max_speakers"] = max_speakers

            logger.info(f"Running diarization with options: {diarize_options}")
            diarize_segments = self.diarize_model(audio, **diarize_options)
            logger.info("Speaker diarization complete")

            # Clean up diarization model if memory is tight
            if self.device == "cuda":
                torch.cuda.empty_cache()

            # Step 5: Assign speaker labels to words and segments
            logger.info(f"Step 5/5: Assigning speaker labels to transcript")
            result = whisperx.assign_word_speakers(diarize_segments, result)
            logger.info("Speaker assignment complete")

            # Extract speaker information
            speakers_found = set()
            segments_with_speakers = []

            for segment in result["segments"]:
                speaker = segment.get("speaker", "UNKNOWN")
                speakers_found.add(speaker)

                segments_with_speakers.append({
                    "start": segment.get("start", 0.0),
                    "end": segment.get("end", 0.0),
                    "text": segment.get("text", "").strip(),
                    "speaker": speaker,
                    "words": segment.get("words", [])
                })

            # Generate formatted transcript
            formatted_transcript = self._format_transcript(segments_with_speakers)

            logger.info(f"Diarization complete: {len(segments_with_speakers)} segments, {len(speakers_found)} speakers")

            return {
                "success": True,
                "segments": segments_with_speakers,
                "num_speakers": len(speakers_found),
                "speakers": sorted(list(speakers_found)),
                "language": detected_language,
                "formatted_transcript": formatted_transcript,
                "model": f"whisperx-{self.model_size}",
                "diarization_method": "pyannote-audio"
            }

        except Exception as e:
            logger.exception(f"WhisperX diarization failed for {audio_path}")
            return {
                "success": False,
                "error": str(e),
                "error_type": type(e).__name__
            }
        finally:
            # Always clean up memory after processing
            self._cleanup_memory()

    def _format_transcript(self, segments) -> str:
        """Format segments into readable transcript with speaker labels.

        Args:
            segments: List of segment dictionaries with speaker labels

        Returns:
            Formatted transcript string
        """
        transcript_lines = []

        for segment in segments:
            start_time = self._format_time(segment["start"])
            speaker = segment["speaker"]
            text = segment["text"].strip()

            if text:
                transcript_lines.append(f"[{start_time}] {speaker}: {text}")

        return "\n".join(transcript_lines)

    def _format_time(self, seconds: float) -> str:
        """Format time in seconds to MM:SS format.

        Args:
            seconds: Time in seconds

        Returns:
            Formatted time string (MM:SS)
        """
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes:02d}:{secs:02d}"

    def _cleanup_memory(self):
        """Clean up GPU/CPU memory between operations."""
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.debug("Memory cleanup performed")

    def cleanup(self):
        """Clean up all loaded models and free memory.

        This should be called when done with the diarizer to free resources.
        """
        models = [
            (self.model, "transcription model"),
            (self.align_model, "alignment model"),
            (self.diarize_model, "diarization model")
        ]

        for model, name in models:
            if model is not None:
                try:
                    del model
                    logger.info(f"Cleaned up {name}")
                except Exception as e:
                    logger.warning(f"Error cleaning up {name}: {e}")

        self.model = None
        self.align_model = None
        self.align_metadata = None
        self.diarize_model = None

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info("WhisperXDiarizer fully cleaned up")
